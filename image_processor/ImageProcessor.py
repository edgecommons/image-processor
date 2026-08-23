"""ImageProcessor: the component that turns an image into a durable, confirmed decision.

This module is the wiring. Every subsystem it builds is specified elsewhere -- configuration in
``config/``, bundles in ``bundles/``, the ledger and completion in ``ledger/`` and
``completion/``, decode and inference in ``engine/``, discovery in ``sources/``, and the outputs
in ``outputs/`` -- and what happens here is that they are assembled in the one order DESIGN.md
allows, and that the result pipeline runs in the one sequence DESIGN.md 7 fixes.

The pipeline, in the order it must happen and no other::

    executor result
      -> build the result body and validate it against the shipped schema
      -> install the evidence sidecar (temp, flush, atomic install)
      -> prepare the app message, freezing its exact bytes
      -> one ledger transaction: result, sidecar digest, outbox row, RESULT_COMMITTED
      -> the outbox drains with publish_confirmed, and only then PUBLISHED
      -> the decision mirror, best effort, and the correlated reply when one was asked for
      -> the completion action under a persisted intent: archive, delete, retain, quarantine

Nothing in that chain is reordered for convenience. The sidecar exists before the transaction so
a crash cannot leave a committed result with no evidence; the transaction precedes publication so
a published result is always recoverable; publication precedes cleanup so an image is never moved
before the fleet has been told what it says.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from edgecommons.config.manager.configuration_change_listener import (
    ConfigurationChangeListener,
)
from edgecommons.facades.app_facade import PreparedAppMessage
from edgecommons.messaging.message import Message
from edgecommons.messaging.qos import Qos

from image_processor.artifacts import ArtifactManager
from image_processor.bundles import BundleCache
from image_processor.commands import ProcessorCommands
from image_processor.completion import CleanupError, Completer
from image_processor.config import CompletionPolicy, ConfigError, parse_component_config
from image_processor.connectivity import ConnectivityProvider, RouteStatus
from image_processor.engine.protocol import is_input_error
from image_processor.engine.residency import ResidencyPolicy
from image_processor.engine.scheduler import Scheduler
from image_processor.engine.supervisor import Supervisor, SupervisorError
from image_processor.health import Health
from image_processor.ledger import Ledger, OutboxRow
from image_processor.metrics import ProcessorMetrics
from image_processor.outputs import (
    RESULT_CHANNEL,
    RESULT_MESSAGE_NAME,
    DecisionMirror,
    OutboxPublisher,
    ResultLimits,
    RouteEvents,
    body_bytes,
    build_result_body,
    fits_budget,
    sidecar_document,
    sidecar_path_for,
    write_sidecar,
)
from image_processor.outputs.result import split_error
from image_processor.sources import (
    CaptureStatusReconciler,
    SpoolSource,
    TriggerSource,
    capture_status_topic,
    image_captured_topic,
)
from image_processor.types import (
    CompletionAction,
    Job,
    JobState,
    SourceKind,
    derive_inference_id,
)

logger = logging.getLogger("ImageProcessor")

#: How often readiness, the queue conditions, and the backlog condition are evaluated.
HEALTH_INTERVAL_SECS = 5.0

#: The states a job passes through while it is still work.
IN_FLIGHT_STATES = (
    JobState.DISCOVERED,
    JobState.READY,
    JobState.CLAIMED,
    JobState.WAITING_MODEL,
    JobState.INFERENCING,
    JobState.RETRY_WAIT,
)

#: How many ledger pages one spool priming walk reads. Admission is idempotent on the inference
#: id, so priming is a latency optimization and never the thing that makes deduplication correct.
PRIME_PAGES = 40

#: The page size of that walk.
PRIME_PAGE_SIZE = 500

#: How many failed completions one supervision pass retries.
CLEANUP_RETRY_LIMIT = 50


@dataclass
class RouteRuntime:
    """The live objects one configured route owns.

    Attributes:
        route: The parsed route configuration.
        source: Its :class:`~image_processor.sources.spool.SpoolSource` or
            :class:`~image_processor.sources.trigger.TriggerSource`.
        reconciler: Its capture-status reconciler, when it is camera-bound.
        topics: The topic filters subscribed for it, so they are unsubscribed on the way out.
        paused: Whether an operator has paused it.
        override: An operator activation override, or ``None`` when configuration decides.
        last_error: The most recent condition, for the connectivity detail line.
    """

    route: Any
    source: Any = None
    reconciler: Any = None
    topics: tuple = ()
    paused: bool = False
    override: Optional[bool] = None
    last_error: Optional[str] = None

    @property
    def enabled(self) -> bool:
        """Whether the route claims new work, after any operator override."""
        if self.override is not None:
            return self.override
        return bool(self.route.enabled)


class ImageProcessor(ConfigurationChangeListener):
    """One component process hosting every configured route.

    Args:
        gg: The EdgeCommons handle, already built and initialized.
        limits: The result message budget, or ``None`` for the default.
    """

    def __init__(self, gg: Any, limits: Optional[ResultLimits] = None) -> None:
        """Build every subsystem from configuration. Nothing runs until :meth:`run`."""
        self._gg = gg
        self._cm = gg.get_config_manager()
        self._limits = limits or ResultLimits()
        self._config = self._parse_config()
        self._generation = self._read_generation()
        self._prepare_directories()

        self._ledger = Ledger(
            Path(self._config.paths.state_db),
            reserve_budget_bytes=self._config.publish.outbox_reserve_budget_bytes,
        )
        self._cache = BundleCache(Path(self._config.paths.model_cache))
        self._supervisor = Supervisor(runtime=self._config.runtime, gpu=self._config.gpu)
        self._residency = ResidencyPolicy(gpu=self._config.gpu, scheduler=self._config.scheduler)
        self._events = RouteEvents(gg)
        self._metrics = ProcessorMetrics(gg, gauges=self._gauges)
        self._mirror = DecisionMirror(gg, on_error=self._on_mirror_error)
        self._completer = Completer(
            self._ledger, on_collision=str(self._config.completion_defaults.on_collision)
        )
        self._scheduler = Scheduler(
            self._ledger,
            self._supervisor,
            self._cache,
            self._residency,
            cfg=self._config.scheduler,
            on_result=self._on_result,
            route_priorities={route.id: route.priority for route in self._config.routes},
        )
        self._publisher = OutboxPublisher(
            self._ledger,
            self._publish_confirmed,
            timeout_secs=self._config.publish.confirmation_timeout_secs,
            max_attempts=self._config.publish.max_attempts,
            on_published=self._on_published,
            on_exhausted=self._on_publish_exhausted,
        )
        self._artifacts = ArtifactManager(
            self._config,
            self._cache,
            self._ledger,
            self._supervisor,
            trusted_keys=self._trusted_keys(),
            credentials=self._model_credentials(),
            events=self._events,
            metrics=self._metrics,
            on_activated=self._on_model_activated,
        )
        self._health = Health(
            statuses=self.route_statuses,
            state_writable=self._state_writable,
            cache_verified=self._cache_verified,
            outbox_pending=self._publisher.pending,
            outbox_capacity=self._config.publish.outbox_capacity,
            requires_executor=lambda: bool(self._config.enabled_routes),
            executor_healthy=self._supervisor.healthy,
        )
        self.commands = ProcessorCommands(self)

        self._routes: dict = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._stopped = False
        self._started_at = time.time()

        self._cm.add_config_change_listener(self)
        gg.set_instance_connectivity_provider(ConnectivityProvider(self.route_statuses))

    # -- configuration -----------------------------------------------------------------

    def _parse_config(self) -> Any:
        """Parse the component configuration into frozen values.

        Returns:
            The :class:`~image_processor.config.models.ComponentConfig`.

        Raises:
            ConfigError: The configuration cannot be used. The candidate validator rejects a bad
                reload before it reaches here, so this is the initial generation failing.
        """
        instances = [
            self._cm.get_instance_config(instance_id) or {}
            for instance_id in (self._cm.get_instance_ids() or [])
        ]
        return parse_component_config(self._cm.get_global_config() or {}, instances)

    def _read_generation(self) -> int:
        """Return the configuration generation, or ``0`` when the manager reports none."""
        accessor = getattr(self._cm, "get_generation", None)
        if not callable(accessor):
            return 0
        try:
            return int(accessor() or 0)
        except Exception:  # noqa: BLE001 - a bring-up without a generation is generation zero
            return 0

    def _prepare_directories(self) -> None:
        """Create the durable directories this component owns.

        A completion directory is created here too: an archive that fails because its target
        directory does not exist is an evidence failure, and the deployment already declared the
        directory by configuring it.
        """
        paths = self._config.paths
        for directory in (Path(paths.state_db).parent, Path(paths.model_cache), Path(paths.staging)):
            directory.mkdir(parents=True, exist_ok=True)
        for route in self._config.routes:
            for directory in route.output_dirs:
                Path(directory).mkdir(parents=True, exist_ok=True)
            if route.is_trigger:
                Path(route.source.inline_staging).mkdir(parents=True, exist_ok=True)

    def _credentials(self) -> Any:
        """Return the credentials vault, or ``None`` when the deployment configured none."""
        accessor = getattr(self._gg, "get_credentials", None)
        return accessor() if callable(accessor) else None

    def _resolve_secret(self, reference: Any) -> Optional[str]:
        """Resolve one ``$secret`` reference through the vault (DESIGN.md 9).

        The core resolves ``$secret`` automatically only under ``streaming``, so this component
        is the single reader of its own references: model source credentials and trusted signing
        keys.

        Args:
            reference: A :class:`~image_processor.types.SecretRef`.

        Returns:
            The secret value, or ``None`` when there is no vault or the secret is missing.
        """
        from edgecommons.credentials import resolve_secret_refs

        vault = self._credentials()
        if vault is None:
            logger.warning(
                "configuration references the secret '%s' but no credentials vault is configured",
                reference.name,
            )
            return None
        document = {"value": reference.to_config()}
        try:
            resolve_secret_refs(document, vault)
        except Exception as exc:  # noqa: BLE001 - a missing secret degrades, it does not crash
            logger.error("the secret '%s' could not be resolved: %s", reference.name, exc)
            return None
        value = document.get("value")
        return value if isinstance(value, str) else None

    def _trusted_keys(self) -> dict:
        """Resolve the configured signing keys into ``keyId`` to key bytes."""
        keys: dict = {}
        for trusted in self._config.signing.trusted_keys:
            material = trusted.public_key
            if not isinstance(material, str):
                material = self._resolve_secret(material)
            if not material:
                logger.error("trusted key '%s' has no usable material", trusted.key_id)
                continue
            keys[trusted.key_id] = material.encode("utf-8")
        return keys

    def _model_credentials(self) -> dict:
        """Resolve the configured model source credentials into ``modelId`` to a mapping."""
        credentials: dict = {}
        for entry in self._config.models:
            if entry.credentials_ref is None:
                continue
            value = self._resolve_secret(entry.credentials_ref)
            if value is None:
                continue
            credentials[entry.id] = _credentials_document(value)
        return credentials

    # -- lifecycle ---------------------------------------------------------------------

    def run(self) -> None:
        """Bring the component up, then keep it healthy until it is stopped.

        Readiness is claimed last and only once the whole chain exists: the executor boundary,
        the recovered ledger, every source, the scheduler, the outbox publisher, and at least one
        activated model generation. A component that reported ready before that would be telling
        the fleet it can answer for an image while it is still deciding what to run.
        """
        self._metrics.define()
        self._supervisor.start()
        self._recover()
        self._artifacts.reconcile()
        self._start_routes()
        self._scheduler.start()
        self._publisher.start()
        self._artifacts.start()
        self._metrics.start()
        self._resubmit()
        self._health.apply(self._gg)
        logger.info(
            "ImageProcessor is running: %d route(s), %d model(s)",
            len(self._routes),
            len(self._config.models),
        )
        while not self._stop.wait(HEALTH_INTERVAL_SECS):
            self._tick()

    def stop(self) -> None:
        """Stop everything in the order that cannot lose work. Idempotent.

        Sources first, so nothing new is admitted; then the scheduler, so the executors drain;
        then the publisher, so anything already committed gets its chance to be confirmed; then
        the cells; then the ledger. A job that was in flight is durable in the ledger and comes
        back as ``READY`` on the next start.
        """
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
        self._stop.set()
        logger.info("stopping ImageProcessor")
        for runtime in list(self._routes.values()):
            self._stop_route(runtime)
        self._artifacts.stop()
        self._scheduler.stop()
        self._publisher.stop()
        self._metrics.stop()
        self._supervisor.stop()
        try:
            self._ledger.close()
        except Exception:  # noqa: BLE001 - shutdown must not raise
            logger.warning("the ledger did not close cleanly", exc_info=True)
        logger.info("ImageProcessor stopped")

    def _tick(self) -> None:
        """One supervision pass: readiness, route conditions, and the backlog condition."""
        try:
            report = self._health.apply(self._gg)
            self._report_conditions(report)
            self._retry_cleanups()
        except Exception:  # noqa: BLE001 - the supervision loop outlives one bad pass
            logger.warning("a supervision pass failed", exc_info=True)

    def _retry_cleanups(self) -> int:
        """Retry the completions that failed (DESIGN.md 7).

        A cleanup failure is never success and never final: the intent stays on record and the job
        sits in ``CLEANUP_FAILED`` until the thing that blocked it -- a file another process still
        had open, a directory that was not there yet -- goes away. Retrying it on the supervision
        pass is what "retried by policy" means; ``retry-cleanup`` is the same thing on demand.

        Returns:
            How many jobs left ``CLEANUP_FAILED`` in this pass.
        """
        jobs, _cursor = self._ledger.by_state([JobState.CLEANUP_FAILED], None, None, CLEANUP_RETRY_LIMIT)
        repaired = 0
        for job in jobs:
            intent = self._ledger.cleanup_intent(job.inference_id)
            if intent is None:
                continue
            if self._completer.reconcile(intent, self._collision_for(job.inference_id)) is not (
                JobState.CLEANUP_FAILED
            ):
                repaired += 1
        if repaired:
            logger.info("retried %d failed completion(s)", repaired)
        return repaired

    def _report_conditions(self, report: Any) -> None:
        """Raise and clear the operator conditions the current state implies."""
        self._events.executor_unavailable(
            not self._supervisor.healthy(), "no executor cell is alive"
        )
        pending = self._publisher.pending()
        capacity = self._config.publish.outbox_capacity
        self._events.publish_backlog(pending >= capacity * 0.8, pending, capacity)
        threshold = float(self._config.scheduler.queue_age_warning_secs)
        for status in self.route_statuses():
            # A paused route is an operator decision, not a fault: it is reported as paused in the
            # connectivity sample and does not raise a condition.
            self._events.route_degraded(
                status.route_id,
                status.enabled and not status.paused and not status.connected,
                status.last_error or "",
            )
            self._events.queue_age_exceeded(
                status.route_id,
                status.oldest_age_secs > threshold,
                status.oldest_age_secs,
                threshold,
                status.queued,
            )

    # -- recovery ----------------------------------------------------------------------

    def _recover(self) -> Any:
        """Reconcile the ledger and the filesystem before anything publishes (DESIGN.md 7).

        Two reconciliations happen here and they are different. The sidecar reconciliation decides
        what a committed result means when the evidence beside it is gone: a committed record
        whose sidecar is absent returns to re-inference, because publishing a result whose
        evidence no longer exists would put an unverifiable decision on the bus. The cleanup
        reconciliation decides an interrupted file mutation from observed state, which is the only
        way to tell a completed move from one that never started.

        Returns:
            The :class:`~image_processor.ledger.recovery.RecoveryReport`.
        """
        report = self._ledger.recover()
        if report.total:
            logger.info("recovery restarted %d job(s): %s", report.total, report.transitions)
        for record in report.sidecars:
            self._reconcile_sidecar(record)
        for intent in report.cleanup_pending:
            self._reconcile_cleanup(intent)
        return report

    def _reconcile_sidecar(self, record: Any) -> None:
        """Adopt, remove, or re-infer around one committed job evidence sidecar."""
        path = Path(record.sidecar_path)
        if path.is_file():
            import hashlib

            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as exc:
                logger.warning("evidence %s could not be read: %s", path, exc)
                digest = None
            if digest == record.sidecar_sha256:
                return
            logger.error(
                "evidence %s does not match the committed digest; removing it and re-inferring",
                path,
            )
            try:
                path.unlink()
            except OSError:  # pragma: no cover - the file vanished under us
                pass
        else:
            logger.warning("evidence %s is missing; re-inferring %s", path, record.inference_id)
        try:
            self._ledger.requeue_for_reinference(record.inference_id, "SIDECAR_MISSING")
        except Exception:  # noqa: BLE001 - the job may have moved on already
            logger.warning(
                "could not requeue %s for re-inference", record.inference_id, exc_info=True
            )

    def _reconcile_cleanup(self, intent: Any) -> None:
        """Decide one interrupted completion from what the filesystem now shows."""
        try:
            state = self._completer.reconcile(intent, self._collision_for(intent.inference_id))
        except Exception:  # noqa: BLE001 - a failed reconciliation is reported, not fatal
            logger.warning("cleanup of %s could not be reconciled", intent.inference_id, exc_info=True)
            return
        logger.info("cleanup of %s reconciled to %s", intent.inference_id, state.value)
        if state is JobState.CLEANUP_FAILED:
            job = self._ledger.get(intent.inference_id)
            route_id = job.route_id if job is not None else ""
            self._events.cleanup_failed(
                route_id,
                intent.inference_id,
                intent.action.value,
                self._ledger.last_error(intent.inference_id) or "",
            )

    def _resubmit(self) -> None:
        """Hand every claimable job back to the scheduler after a restart."""
        submitted = 0
        for state in (JobState.READY,):
            cursor = None
            for _page in range(PRIME_PAGES):
                jobs, cursor = self._ledger.by_state([state], None, cursor, PRIME_PAGE_SIZE)
                for job in jobs:
                    route = self._config.route(job.route_id)
                    if route is None:
                        continue
                    self._scheduler.submit(job, route.priority)
                    submitted += 1
                if cursor is None:
                    break
        if submitted:
            logger.info("resubmitted %d recovered job(s)", submitted)

    # -- routes ------------------------------------------------------------------------

    def _start_routes(self) -> None:
        """Build and start a runtime for every configured route."""
        for route in self._config.routes:
            try:
                self._routes[route.id] = self._start_route(route)
            except Exception as exc:  # noqa: BLE001 - one bad route degrades itself only
                logger.exception("route %s could not start", route.id)
                self._routes[route.id] = RouteRuntime(route=route, last_error=str(exc))

    def _start_route(self, route: Any) -> RouteRuntime:
        """Build one route source, prime it, and subscribe what it listens to."""
        runtime = RouteRuntime(route=route)
        if route.is_spool:
            reconciler: Any = None

            def _lookup(relative_path: str) -> Any:
                return reconciler.lookup(relative_path) if reconciler is not None else None

            camera = route.source.camera
            source = SpoolSource(
                route,
                self,
                status_lookup=_lookup if camera is not None else None,
                debounce_secs=self._config.discovery.debounce_secs,
                rescan_interval_secs=float(self._config.discovery.rescan_secs),
            )
            source.prime(self._known_pairs(route.id))
            reconciler = self._build_reconciler(route, source)
            runtime.source = source
            runtime.reconciler = reconciler
            if route.enabled:
                source.start()
                if reconciler is not None and camera.reconcile_capture_status_secs > 0:
                    reconciler.start()
                runtime.topics = self._subscribe_hint(route)
        else:
            source = TriggerSource(route, self)
            runtime.source = source
            if route.enabled:
                runtime.topics = self._subscribe_trigger(route, source)
        logger.info(
            "route %s ready: %s source%s",
            route.id,
            route.source.kind,
            "" if route.enabled else " (disabled)",
        )
        return runtime

    def _stop_route(self, runtime: RouteRuntime) -> None:
        """Stop one route source and drop every subscription it made."""
        messaging = self._gg.get_messaging()
        for topic in runtime.topics:
            try:
                messaging.unsubscribe(topic)
            except Exception as exc:  # noqa: BLE001 - shutdown must not raise
                logger.debug("unsubscribing %s failed: %s", topic, exc)
        runtime.topics = ()
        for component in (runtime.reconciler, runtime.source):
            stop = getattr(component, "stop", None)
            if stop is None:
                continue
            try:
                stop()
            except Exception:  # noqa: BLE001 - shutdown must not raise
                logger.debug("stopping a route component failed", exc_info=True)

    def _build_reconciler(self, route: Any, source: Any) -> Any:
        """Build the capture-status reconciler of a camera-bound route.

        The announcement is a hint and the status verb is the authority (DESIGN.md 4.1), so a
        camera-bound route polls the camera paged ``sb/capture-status`` whatever its readiness
        mode is: under ``cameraStatus`` the record is the proof of finalization, and under
        ``cameraSidecar`` it is what recovers a capture whose announcement was lost.
        """
        camera = route.source.camera
        if camera is None:
            return None
        device = self._device()
        return CaptureStatusReconciler(
            route_id=route.id,
            root=Path(route.source.root),
            topic=capture_status_topic(device, camera.component, camera.instance),
            request=self._request,
            kv_get=self._ledger.kv_get,
            kv_set=self._ledger.kv_set,
            instance=camera.instance,
            interval_secs=float(camera.reconcile_capture_status_secs or 30),
            on_verified=lambda record: source.nudge(),
        )

    def _subscribe_hint(self, route: Any) -> tuple:
        """Subscribe the camera announcement channel of a camera-bound route.

        The announcement is a low-latency hint carrying its own proof: it declares the image size
        and digest, so a hint that verifies against the file admits the job without waiting for
        the next walk. It is not a queue, and a lost hint costs latency rather than a job.
        """
        camera = route.source.camera
        if camera is None or not camera.subscribe_announcements:
            return ()
        topic = image_captured_topic(self._device(), camera.component, camera.instance)

        def _on_hint(_topic: str, message: Any) -> None:
            source = self._source_of(route.id)
            if source is None:
                return
            try:
                source.on_hint(_body_of(message))
            except Exception:  # noqa: BLE001 - a bad hint is never a lost job
                logger.warning("route %s could not use a camera hint", route.id, exc_info=True)

        try:
            self._gg.get_messaging().subscribe(topic, _on_hint, 1, 64)
        except Exception as exc:  # noqa: BLE001 - the walk still covers the route
            logger.warning("route %s could not subscribe %s: %s", route.id, topic, exc)
            return ()
        logger.info("route %s subscribed camera hints on %s", route.id, topic)
        return (topic,)

    def _subscribe_trigger(self, route: Any, source: Any) -> tuple:
        """Subscribe every topic filter a trigger route consumes."""
        topics = []
        messaging = self._gg.get_messaging()
        for topic in source.subscribe:

            def _on_message(_topic: str, message: Any, _source=source) -> None:
                try:
                    _source.on_message(message)
                except Exception:  # noqa: BLE001 - one bad message never kills the route
                    logger.warning("route %s could not admit a trigger", route.id, exc_info=True)

            try:
                messaging.subscribe(topic, _on_message, 1, 64)
            except Exception as exc:  # noqa: BLE001 - report it and keep the other filters
                logger.error("route %s could not subscribe %s: %s", route.id, topic, exc)
                continue
            topics.append(topic)
            logger.info("route %s subscribed %s", route.id, topic)
        return tuple(topics)

    def _known_pairs(self, route_id: str) -> list:
        """Return the inputs this route has already admitted, for the spool announced set.

        Admission is idempotent on the inference id, so this is a latency optimization: without
        it a restart re-hashes and re-announces finished work, and the ledger refuses it.
        """
        pairs = []
        cursor = None
        for _page in range(PRIME_PAGES):
            jobs, cursor = self._ledger.by_state(
                list(JobState), route_id, cursor, PRIME_PAGE_SIZE
            )
            pairs.extend((job.source.relative_path, job.source.sha256) for job in jobs)
            if cursor is None:
                break
        return pairs

    def _source_of(self, route_id: str) -> Any:
        """Return one route live source, or ``None`` when it has none."""
        runtime = self._routes.get(route_id)
        return runtime.source if runtime is not None else None

    def _device(self) -> str:
        """Return this device (thing) identity, which is the ``{device}`` topic level."""
        try:
            return self._cm.get_thing_name()
        except Exception:  # noqa: BLE001 - a bring-up without identity still builds topics
            return "unknown"

    def _request(self, topic: str, body: dict, timeout_secs: float = 10.0) -> Any:
        """Send one command request and return the answer another component gave.

        The library wraps every command reply as ``{"ok": ..., "result": ...}`` or
        ``{"ok": false, "error": {"code", "message"}}``. What a caller wants is the verb answer,
        so the envelope is unwrapped here and an error is rendered as the ``errorCode`` /
        ``errorMessage`` pair the capture-status reconciler reads. This is the only place in the
        component that speaks to another component command inbox, so it is the right place for
        that translation.

        Args:
            topic: The target command topic.
            body: The request body.
            timeout_secs: The deadline.

        Returns:
            The answer as a mapping, or ``None`` when the deadline passed or the reply was
            unreadable.
        """
        verb = topic.rsplit("/cmd/", 1)[-1].split("/")[-1]
        builder = self._gg.instance(self._request_instance()).new_message(verb, "1.0")
        request = builder.with_payload(body).build()
        iou = self._gg.get_messaging().request(topic, request, timeout_secs)
        done, reply = iou.get(timeout_secs)
        return _command_answer(reply) if done else None

    def _request_instance(self) -> str:
        """Return the instance token this component uses when it asks another component."""
        routes = self._config.routes
        return routes[0].id if routes else "main"

    # -- discovery (the SourceEvents contract) -----------------------------------------

    def discovered(self, route_id: str, source: Any, staged_path: Optional[Path]) -> None:
        """Admit one verified, finished input (``sources.SourceEvents``).

        The identity is derived rather than generated, so re-discovering the same image under the
        same model yields the same job and the ledger refuses the duplicate. That is what makes a
        replayed camera hint, a periodic walk, and a restart all harmless.

        Args:
            route_id: The route that found it.
            source: Its verified :class:`~image_processor.types.SourceIdentity`.
            staged_path: The processor-owned copy, when the source made one. A spool input has
                none: the component owns the spool and the cell reads the file in place.
        """
        route = self._config.route(route_id)
        if route is None:
            logger.warning("an input arrived for the unknown route %s", route_id)
            return
        model = route.model_ref
        inference_id = derive_inference_id(
            route_id, source.capture_id, source.sha256, source.relative_path, model.digest
        )
        path = staged_path if staged_path is not None else self._input_path(route, source)
        job = Job(
            inference_id=inference_id,
            route_id=route_id,
            source=source,
            model=model,
            transform_version=self._transform_version(model.digest),
            state=JobState.READY,
            staged_path=str(path),
            config_generation=self._generation,
        )
        try:
            admitted = self._ledger.admit(job, self._reserve_bytes())
        except Exception as exc:  # noqa: BLE001 - a refused admission is never a silent drop
            logger.exception("route %s could not admit %s", route_id, source.relative_path)
            self._note(route_id, f"ADMISSION_FAILED: {exc}")
            return
        if not admitted:
            logger.debug("route %s already knows %s", route_id, inference_id)
            return
        self._metrics.incr("ImageProcessorQueue", "admitted")
        self._metrics.incr("ImageProcessorDiscovery", "discovered")
        if source.kind is not SourceKind.SPOOL:
            self._metrics.incr("ImageProcessorDiscovery", "triggersAccepted")
        self._scheduler.submit(job, route.priority)

    def invalid(self, route_id: str, relative_path: str, reason: str) -> None:
        """Refuse one input that can never be admitted as it stands.

        A refused spool file is quarantined under the route ``onInvalidInput`` policy so it stops
        being rediscovered and an operator can see what was rejected; a refused trigger message
        has no file to move, and is reported and counted.

        Args:
            route_id: The route that refused it.
            relative_path: The path as configured or as the message declared it.
            reason: A stable SCREAMING_SNAKE token, carrying no path and no digest.
        """
        self._metrics.incr("ImageProcessorDiscovery", "rejected")
        route = self._config.route(route_id)
        if route is not None and route.is_trigger:
            self._metrics.incr("ImageProcessorDiscovery", "triggersRejected")
        self._note(route_id, f"INPUT_REJECTED: {reason}")
        self._events.input_rejected(route_id, relative_path, reason)
        if route is None or not route.is_spool or not relative_path:
            return
        try:
            self._quarantine_invalid(route, relative_path, reason)
        except Exception:  # noqa: BLE001 - a refused input must never crash discovery
            logger.warning(
                "route %s could not quarantine %s", route_id, relative_path, exc_info=True
            )

    def _quarantine_invalid(self, route: Any, relative_path: str, reason: str) -> None:
        """Admit a permanently bad input just far enough to complete it (DESIGN.md 7).

        The lifecycle for an input that can never succeed is ``DISCOVERED -> INPUT_INVALID ->
        QUARANTINED``, and taking it needs a durable row, which needs the digest of the bytes on
        disk. The file is hashed here for exactly that reason: what is quarantined is recorded by
        content, so the completion can verify that it moved what it meant to move.
        """
        from image_processor.sources.staging import (
            classify_path,
            resolve_under_root,
            sha256_file,
        )

        root = Path(route.source.root)
        try:
            path = resolve_under_root(root, relative_path)
        except Exception:  # noqa: BLE001 - a path we cannot resolve is not ours to move
            return
        if classify_path(path) is not None:
            return
        digest = sha256_file(path)
        identity = _invalid_identity(route.id, relative_path, path.stat().st_size, digest)
        inference_id = derive_inference_id(
            route.id, None, digest, relative_path, route.model_ref.digest
        )
        job = Job(
            inference_id=inference_id,
            route_id=route.id,
            source=identity,
            model=route.model_ref,
            transform_version="",
            state=JobState.DISCOVERED,
            staged_path=str(path),
            config_generation=self._generation,
        )
        if not self._ledger.admit(job, self._reserve_bytes()):
            return
        job = self._ledger.transition(
            inference_id, JobState.DISCOVERED, JobState.INPUT_INVALID, last_error=reason
        )
        self._complete(job, route)

    def _input_path(self, route: Any, source: Any) -> Path:
        """Return where an input actually lives, which is what the executor cell reads."""
        from image_processor.sources.staging import resolve_under_root

        if source.kind is SourceKind.SPOOL:
            return resolve_under_root(Path(route.source.root), source.relative_path)
        if source.kind is SourceKind.REFERENCE:
            return resolve_under_root(Path(route.source.file_root), source.relative_path)
        return Path(route.source.inline_staging) / source.relative_path

    def _transform_version(self, digest: str) -> str:
        """Return the transform contract of a staged generation, or an empty pin.

        A job pinned to a transform version the resident bundle does not declare is refused by
        the executor cell, which is the check that keeps two results comparable. Until the bundle
        is staged there is nothing to pin to, and the cell reads the version off the manifest it
        loads instead.
        """
        manifest = self._manifest(digest)
        return str(getattr(manifest, "transform_version", "") or "") if manifest else ""

    def _manifest(self, digest: str) -> Any:
        """Return the manifest of a cached generation, or ``None`` when it is not staged."""
        try:
            cached = self._cache.get(digest, verify=False)
        except Exception:  # noqa: BLE001 - an unreadable cache entry is reported elsewhere
            logger.warning("the cached bundle %s could not be read", digest, exc_info=True)
            return None
        return cached.manifest if cached is not None else None

    def _reserve_bytes(self) -> int:
        """Return the capacity one job reserves at admission (DESIGN.md 7).

        A job reserves what its maximum configured result can cost: the published message and the
        evidence sidecar that holds the full version of it. Reserving at admission is what stops
        a finished job from being stranded by a full outbox.
        """
        return int(self._limits.max_body_bytes) * 2

    # -- the result pipeline -----------------------------------------------------------

    def _on_result(self, job: Any, result: Any) -> None:
        """Take one answer from the executor boundary and make it durable and visible.

        This is the head of the ordered chain in the module docstring. It runs on the scheduler
        thread, and it must not raise: an exception here would leave a job the scheduler has
        already accounted for with nothing recorded against it, which is the one shape recovery
        cannot tell from a crash.
        """
        route = self._config.route(job.route_id)
        if route is None:
            logger.warning("a result arrived for the unknown route %s", job.route_id)
            return
        self._observe(result)
        try:
            if result.status == "SUCCEEDED" and job.state is JobState.INFERENCING:
                self._commit(job, route, result)
            else:
                self._fail(job, route, result)
        except Exception as exc:  # noqa: BLE001 - the scheduler thread survives a bad result
            logger.exception("the result pipeline failed for %s", job.inference_id)
            self._note(job.route_id, f"RESULT_PIPELINE_FAILED: {exc}")
            self._events.inference_failed(
                job.route_id, job.inference_id, "RESULT_PIPELINE_FAILED", str(exc)
            )

    def _observe(self, result: Any) -> None:
        """Record the timings and the outcome of one attempt."""
        group = "ImageProcessorInference"
        self._metrics.incr(group, "succeeded" if result.status == "SUCCEEDED" else "failed")
        timings = result.timings
        self._metrics.observe(group, "queueMs", timings.queue_ms)
        self._metrics.observe(group, "inferenceMs", timings.inference_ms)
        self._metrics.observe(group, "totalMs", timings.total_ms)

    def _commit(self, job: Any, route: Any, result: Any) -> None:
        """Install the evidence, freeze the message, and commit them in one transaction."""
        manifest = self._manifest(job.model.digest)
        full = build_result_body(job, result, manifest)
        oversize = not fits_budget(full, self._limits)
        installed = None
        artifacts = None
        if route.outputs.write_result_sidecar or oversize:
            # A body over the budget is published as a summary, and a summary is only honest
            # when the full result exists somewhere. So an oversized result writes its evidence
            # even on a route that configured none, rather than shortening itself in silence.
            installed = self._install_evidence(job, route, full, oversize)
            if installed is None:
                return
            artifacts = {
                "evidenceId": job.inference_id,
                "localRelativePath": self._evidence_relative(route, installed.path),
                "sha256": installed.sha256,
                "bytes": installed.bytes,
            }
        published = build_result_body(
            job, result, manifest, artifacts=artifacts, limits=self._limits
        )
        prepared = self._gg.instance(route.id).app().prepare(
            RESULT_MESSAGE_NAME, RESULT_CHANNEL, published
        )
        rows = [
            OutboxRow(
                id=None,
                inference_id=job.inference_id,
                topic=prepared.topic,
                encoded_bytes=prepared.encoded_bytes,
                # Confirmation always gates cleanup: the result is the only durable record
                # that a decision was made, so nothing moves before the broker has it (D-IP-6,
                # D-IP-20).
                gating=True,
            )
        ]
        sidecar = (str(installed.path), installed.sha256) if installed is not None else None
        self._ledger.commit_result(job.inference_id, body_bytes(published), sidecar, rows)
        logger.info(
            "committed %s: %s on %s",
            job.inference_id,
            published.get("decision", {}).get("outcome"),
            prepared.topic,
        )
        self._publisher.wake()
        mirrored = self._mirror.publish(route.id, route.outputs.decision_signals, published)
        if mirrored:
            self._metrics.incr("ImageProcessorCompletion", "mirrored", mirrored)
        self._reply(job, route, published)

    def _install_evidence(self, job: Any, route: Any, full: dict, oversize: bool) -> Any:
        """Write the evidence sidecar, or report why nothing was committed."""
        path = sidecar_path_for(self._input_path(route, job.source))
        document = sidecar_document(
            job,
            full,
            evidence_id=job.inference_id,
            config_generation=job.config_generation,
            manifest=self._manifest(job.model.digest),
        )
        try:
            installed = write_sidecar(path, document)
        except Exception as exc:  # noqa: BLE001 - no evidence means no commit
            logger.error("evidence for %s could not be installed: %s", job.inference_id, exc)
            self._note(route.id, f"EVIDENCE_FAILED: {exc}")
            self._events.evidence_failed(route.id, job.inference_id, str(exc))
            return None
        if oversize:
            logger.info(
                "result for %s exceeds the message budget; publishing a summary against %s",
                job.inference_id,
                installed.path,
            )
        return installed

    def _evidence_relative(self, route: Any, path: Path) -> str:
        """Render an installed sidecar path relative to the route own root."""
        for root in (route.input_root, Path(self._config.paths.staging)):
            try:
                return Path(path).relative_to(Path(root)).as_posix()
            except ValueError:
                continue
        return Path(path).name

    def _fail(self, job: Any, route: Any, result: Any) -> None:
        """Publish a failed result, then take the route failure completion.

        A failed inference is still an answer the fleet needs: its decision is ``HOLD``, and a
        consumer that never hears anything cannot tell a held image from a component that
        stopped. The failure path cannot use the outbox, because the ledger only accepts a
        committed result from an in-flight job, so the message is published directly and its
        failure reported rather than retried forever.
        """
        manifest = self._manifest(job.model.digest)
        code, message = split_error(result.error)
        body = build_result_body(job, result, manifest, limits=self._limits)
        self._events.inference_failed(route.id, job.inference_id, code, message)
        self._note(route.id, f"{code}: {message}")
        try:
            prepared = self._gg.instance(route.id).app().prepare(
                RESULT_MESSAGE_NAME, RESULT_CHANNEL, body
            )
            self._gg.app().publish_confirmed(
                prepared, self._config.publish.confirmation_timeout_secs
            )
        except Exception as exc:  # noqa: BLE001 - the completion still has to run
            logger.warning(
                "the failed result for %s could not be published: %s", job.inference_id, exc
            )
        self._reply(job, route, body)
        self._mirror.publish(route.id, route.outputs.decision_signals, body)
        current = self._ledger.get(job.inference_id) or job
        if current.state in (JobState.PROCESSING_EXHAUSTED, JobState.INPUT_INVALID):
            self._complete(current, route, error_class=result.error_class, error_code=code)

    def _reply(self, job: Any, route: Any, body: dict) -> None:
        """Answer the requester of a trigger job on its ``reply_to`` topic (DESIGN.md 4.2)."""
        reply_to = getattr(job.source, "reply_to", None)
        if not reply_to:
            return
        try:
            builder = self._gg.instance(route.id).new_message(RESULT_MESSAGE_NAME, "1.0")
            builder = builder.with_payload(body)
            correlation_id = job.source.correlation_id
            if correlation_id:
                builder = builder.with_correlation_id(correlation_id)
            self._gg.get_messaging().publish(reply_to, builder.build())
        except Exception as exc:  # noqa: BLE001 - a lost reply never fails the job
            logger.warning("could not answer %s for %s: %s", reply_to, job.inference_id, exc)

    # -- publication and completion ----------------------------------------------------

    def _publish_confirmed(self, topic: str, encoded: bytes, timeout_secs: float) -> None:
        """Publish exact stored bytes and wait for positive transport acceptance.

        The bytes are what ``app().prepare()`` froze, so a retry carries the same envelope UUID
        and a consumer can deduplicate on it. They are handed back to the ``app()`` facade as a
        prepared message; when the stored envelope cannot be reconstituted into one, the same
        bytes go through the messaging client confirmed publish, which is what the facade calls
        anyway. Either way the exact bytes are what reaches the broker.
        """
        prepared = _prepared_message(topic, encoded)
        if prepared is not None:
            self._gg.app().publish_confirmed(prepared, timeout_secs)
            return
        self._gg.get_messaging().publish_confirmed(
            topic, encoded, Qos.AT_LEAST_ONCE, timeout_secs
        )

    def _on_published(self, job: Any) -> None:
        """Release the completion of a job whose gating publications are all confirmed."""
        route = self._config.route(job.route_id)
        if route is None:
            logger.warning("%s published on a route that no longer exists", job.inference_id)
            return
        self._metrics.incr("ImageProcessorOutbox", "published")
        self._complete(job, route)

    def _on_publish_exhausted(self, inference_id: str, error: str) -> None:
        """Report a result that spent its publication budget.

        The input stays where it is. The job is not terminal -- ``retry-publication`` returns it
        to the outbox -- and archiving or deleting the image now would move the evidence out from
        under a publication that is still expected to happen.
        """
        job = self._ledger.get(inference_id)
        route_id = job.route_id if job is not None else ""
        route = self._config.route(route_id) if route_id else None
        action = (
            route.completion_for(job.source.kind).on_publish_failure.value
            if route is not None and job is not None
            else "retain"
        )
        self._metrics.incr("ImageProcessorOutbox", "exhausted")
        self._note(route_id, f"PUBLISH_EXHAUSTED: {error}")
        self._events.publish_exhausted(route_id, inference_id, error, action)

    def _complete(
        self,
        job: Any,
        route: Any,
        error_class: Optional[str] = None,
        error_code: Optional[str] = None,
    ) -> None:
        """Take the route completion action under a persisted intent (DESIGN.md 7)."""
        policy = self._effective_policy(job, route, error_class, error_code)
        members = self._members(job, route)
        try:
            intent = self._completer.plan(job, policy, members)
        except CleanupError as exc:
            logger.error("no completion for %s: %s", job.inference_id, exc)
            self._events.cleanup_failed(route.id, job.inference_id, "plan", str(exc))
            self._metrics.incr("ImageProcessorCompletion", "failed")
            return
        try:
            self._completer.apply(intent, str(policy.on_collision))
        except CleanupError as exc:
            logger.error("the completion of %s failed: %s", job.inference_id, exc)
            self._note(route.id, f"CLEANUP_FAILED: {exc}")
            self._events.cleanup_failed(
                route.id, job.inference_id, intent.action.value, str(exc)
            )
            self._metrics.incr("ImageProcessorCompletion", "failed")
            return
        self._metrics.incr("ImageProcessorCompletion", "completed")
        self._metrics.incr("ImageProcessorCompletion", _completion_measure(intent.action))
        self._discard_staged(job)
        logger.info("completed %s: %s", job.inference_id, intent.action.value)

    def _discard_staged(self, job: Any) -> None:
        """Remove the processor-owned copy of a referenced input once its job is over.

        A file reference is copied into staging because the producer still owns the original and
        may rewrite or delete it while the job waits for a GPU. Once the job is complete nothing
        reads that copy again, and the completion already decided what happens to the original.
        An inline input needs none of this: its staged copy is the only copy, so it is the object
        the completion acts on.
        """
        if job.source.kind is not SourceKind.REFERENCE or not job.staged_path:
            return
        staged = Path(job.staged_path)
        try:
            if staged.is_relative_to(Path(self._config.paths.staging)):
                staged.unlink(missing_ok=True)
        except OSError as exc:  # noqa: BLE001 - a staged copy left behind is not a failed job
            logger.debug("the staged copy of %s could not be removed: %s", job.inference_id, exc)

    def _effective_policy(
        self,
        job: Any,
        route: Any,
        error_class: Optional[str],
        error_code: Optional[str] = None,
    ) -> Any:
        """Resolve the completion policy this job actually runs under.

        Two configured behaviors are resolved here rather than in the completer, because both
        depend on what happened to this job.

        The first is which failure this was. Only a permanent failure the *input* caused takes
        the invalid-input action, because that action may quarantine and only bad evidence may be
        quarantined (DESIGN.md 15.2). A permanent model, bundle, provider, GPU, runtime, or
        postprocess-schema failure says nothing about the image, so it takes the operational
        action -- retain, by default -- and the event raised on the way here is what asks an
        operator to repair the deployment. ``engine/protocol.is_input_error`` owns that split.

        The second is that a route with no ``failedDir`` quarantines in place, which is a retain
        of the file with the terminal state recorded against it (DESIGN.md 11).
        """
        policy = route.completion_for(job.source.kind)
        if error_class == "permanent" and is_input_error(error_code):
            policy = replace(policy, on_operational_failure=policy.on_invalid_input)
        if policy.failed_dir is None:
            changes = {
                field_name: CompletionAction.RETAIN
                for field_name in (
                    "on_success",
                    "on_invalid_input",
                    "on_operational_failure",
                    "on_publish_failure",
                )
                if getattr(policy, field_name) is CompletionAction.QUARANTINE
            }
            if changes:
                policy = replace(policy, **changes)
        return policy

    def _members(self, job: Any, route: Any) -> list:
        """Return the companion files that move with an image.

        The camera metadata sidecar and this component evidence sidecar are part of the record,
        so they travel with the image rather than being left behind in a spool the component has
        just emptied.
        """
        if job.source.kind is not SourceKind.SPOOL:
            return []
        image = self._input_path(route, job.source)
        members = []
        for candidate in (image.with_name(image.name + ".json"), sidecar_path_for(image)):
            if candidate.is_file():
                members.append(candidate)
        return members

    def _collision_for(self, inference_id: str) -> Optional[str]:
        """Return the collision policy of the route that owns one job."""
        job = self._ledger.get(inference_id)
        route = self._config.route(job.route_id) if job is not None else None
        return str(route.completion.on_collision) if route is not None else None

    # -- configuration reload ----------------------------------------------------------

    def on_configuration_change(self, configuration: Any) -> bool:
        """Adopt a new configuration generation without dropping admitted work.

        The core has already run the candidate validator, so a configuration that reaches here
        parses and satisfies the cross-field rules. What remains is reconciliation: a route that
        appeared starts, a route that disappeared stops, and a route whose source or model changed
        is rebuilt. Nothing here touches the ledger, so a job admitted under the previous
        generation keeps its pinned model and finishes under it.

        Args:
            configuration: The applied configuration document.

        Returns:
            Whether the change was adopted.
        """
        try:
            new = self._parse_config()
        except ConfigError as exc:
            logger.error("the applied configuration cannot be used: %s", exc)
            return False
        with self._lock:
            previous = self._config
            self._config = new
            self._generation = self._read_generation()
            self._scheduler.route_priorities = {
                route.id: route.priority for route in new.routes
            }
            self._health.outbox_capacity = max(1, int(new.publish.outbox_capacity))
            self._publisher.timeout_secs = float(new.publish.confirmation_timeout_secs)
            self._publisher.max_attempts = max(1, int(new.publish.max_attempts))
            self._prepare_directories()
            self._reconcile_routes(previous, new)
        self._artifacts.adopt(new)
        logger.info(
            "adopted configuration generation %d: %d route(s)", self._generation, len(new.routes)
        )
        return True

    def _reconcile_routes(self, previous: Any, new: Any) -> None:
        """Start, stop, or rebuild each route to match the new configuration."""
        wanted = {route.id: route for route in new.routes}
        for route_id in list(self._routes):
            if route_id not in wanted:
                logger.info("route %s was removed", route_id)
                self._stop_route(self._routes.pop(route_id))
        for route_id, route in wanted.items():
            runtime = self._routes.get(route_id)
            if runtime is None:
                self._routes[route_id] = self._start_route(route)
                continue
            if _route_changed(runtime.route, route):
                logger.info("route %s changed; rebuilding it", route_id)
                self._stop_route(runtime)
                rebuilt = self._start_route(route)
                rebuilt.override = runtime.override
                rebuilt.paused = runtime.paused
                self._routes[route_id] = rebuilt
                continue
            runtime.route = route

    def _on_model_activated(self, route_id: str, previous: Optional[str], digest: str) -> None:
        """React to a route switching model generation (DESIGN.md 9).

        A switch does not replay terminal inputs unless the route asks for it. When it does, the
        route forgets what it announced, and the next authoritative walk rediscovers everything
        still in the spool -- as new jobs, because a new model digest is a new identity.
        """
        route = self._config.route(route_id)
        source = self._source_of(route_id)
        if route is None or source is None:
            return
        if previous and route.reprocess_existing_on_model_change and route.is_spool:
            forgotten = source.forget()
            source.nudge()
            logger.info(
                "route %s will reprocess %d input(s) under %s", route_id, forgotten, digest
            )

    # -- status ------------------------------------------------------------------------

    def route_statuses(self) -> list:
        """Describe every configured route, for connectivity, health, and ``status``."""
        now_ms = int(time.time() * 1000)
        statuses = []
        for route in self._config.routes:
            runtime = self._routes.get(route.id)
            desired, active = self._ledger.route_generation(route.id)
            counts = self._ledger.counts_by_state(route.id)
            queued = sum(counts.get(state, 0) for state in IN_FLIGHT_STATES)
            oldest = self._ledger.oldest_created_at_ms(IN_FLIGHT_STATES, route.id)
            statuses.append(
                RouteStatus(
                    route_id=route.id,
                    enabled=runtime.enabled if runtime is not None else route.enabled,
                    paused=bool(runtime.paused) if runtime is not None else False,
                    source_reachable=self._source_reachable(route, runtime),
                    source_detail=_source_detail(route),
                    desired_generation=desired or route.model_ref.digest,
                    active_generation=active,
                    executor_healthy=self._supervisor.healthy(),
                    queued=queued,
                    oldest_age_secs=(now_ms - oldest) / 1000.0 if oldest else 0.0,
                    last_error=runtime.last_error if runtime is not None else None,
                )
            )
        return statuses

    def _source_reachable(self, route: Any, runtime: Optional[RouteRuntime]) -> bool:
        """Whether a route can actually see its input right now."""
        if route.is_spool:
            return Path(route.source.root).is_dir()
        return bool(runtime is not None and runtime.topics)

    def _state_writable(self) -> bool:
        """Whether the durable state still accepts writes."""
        try:
            self._ledger.kv_set("image-processor/health", str(int(time.time())))
        except Exception:  # noqa: BLE001 - an unwritable ledger is a failed component
            logger.warning("the ledger is not writable", exc_info=True)
            return False
        return True

    def _cache_verified(self) -> bool:
        """Whether the model cache metadata reads back."""
        try:
            self._cache.list()
        except Exception:  # noqa: BLE001 - an unreadable cache degrades readiness
            logger.warning("the model cache could not be read", exc_info=True)
            return False
        return True

    def _gauges(self) -> dict:
        """Sample the live subsystems for the metric flush."""
        counts = self._ledger.counts_by_state()
        scheduler = self._scheduler.status()
        statuses = self.route_statuses()
        oldest = self._ledger.oldest_created_at_ms(IN_FLIGHT_STATES)
        now_ms = int(time.time() * 1000)
        resident = sum(len(cell.get("resident", ())) for cell in scheduler.get("cells", ()))
        resident_mib = sum(
            sum(cell.get("residentMib", {}).values()) for cell in scheduler.get("cells", ())
        )
        discovery = _sum_counters(self._routes.values())
        return {
            "ImageProcessorDiscovery": discovery,
            "ImageProcessorQueue": {
                "queued": sum(counts.get(state, 0) for state in IN_FLIGHT_STATES),
                "retryWaiting": counts.get(JobState.RETRY_WAIT, 0),
                "dispatched": scheduler.get("counters", {}).get("dispatched", 0),
                "oldestAgeSecs": (now_ms - oldest) / 1000.0 if oldest else 0.0,
                "pausedRoutes": sum(1 for status in statuses if status.paused),
            },
            "ImageProcessorModelCache": {
                "cachedBundles": len(self._cache.list()),
                "routesStaging": sum(1 for status in statuses if status.staging),
            },
            "ImageProcessorGpu": {
                "residentModels": resident,
                "residentMiB": resident_mib,
                "loads": scheduler.get("counters", {}).get("loads", 0),
                "evictions": scheduler.get("counters", {}).get("evictions", 0),
                "recycles": scheduler.get("recycleCount", 0),
                "healthyCells": sum(
                    1 for cell in scheduler.get("cells", ()) if cell.get("alive")
                ),
            },
            "ImageProcessorOutbox": {
                "pending": counts.get(JobState.PUBLISH_PENDING, 0),
                "reservedBytes": self._ledger.reserved_bytes(),
            },
            "ImageProcessorDisk": self._disk_gauges(),
        }

    def _disk_gauges(self) -> dict:
        """Sample the sizes and the free space of the directories this component owns."""
        paths = self._config.paths
        free_mib = 0.0
        try:
            free_mib = shutil.disk_usage(str(Path(paths.state_db).parent)).free / (1024 * 1024)
        except OSError:  # pragma: no cover - an unreadable mount is reported by readiness
            pass
        return {
            "stateDbBytes": _file_size(Path(paths.state_db)),
            "modelCacheBytes": _tree_size(Path(paths.model_cache)),
            "stagingBytes": _tree_size(Path(paths.staging)),
            "freeMiB": free_mib,
        }

    def _note(self, route_id: str, error: str) -> None:
        """Record the most recent condition of a route, for the connectivity detail line."""
        runtime = self._routes.get(route_id)
        if runtime is not None:
            runtime.last_error = error[:512]

    def _on_mirror_error(self, route_id: str, signal_id: str, error: str) -> None:
        """Count a decision-mirror failure. It never fails a job."""
        self._metrics.incr("ImageProcessorCompletion", "failed", 0)
        logger.debug("route %s could not mirror %s: %s", route_id, signal_id, error)

    # -- the command surface -----------------------------------------------------------

    def route_ids(self) -> list:
        """Return every configured route id."""
        return [route.id for route in self._config.routes]

    def list_models(self) -> list:
        """Return the staged, active, and rollback state of every configured model."""
        return self._artifacts.status()

    def list_jobs(
        self,
        route_id: Optional[str],
        states: Optional[Iterable],
        cursor: Optional[str],
        limit: int,
    ) -> tuple:
        """Page through jobs by state and age.

        Args:
            route_id: One route, or ``None`` for every route.
            states: The state names to include, or ``None`` for the states that are still work.
            cursor: The opaque cursor from a previous page.
            limit: The page size.

        Returns:
            ``(jobs, next_cursor)``.

        Raises:
            ValueError: A state name or the cursor is unusable.
        """
        wanted = list(IN_FLIGHT_STATES)
        if states:
            wanted = []
            for name in states:
                try:
                    wanted.append(JobState(str(name)))
                except ValueError as exc:
                    raise ValueError(f"'{name}' is not a job state") from exc
        jobs, next_cursor = self._ledger.by_state(wanted, route_id, cursor, limit)
        return [self._job_document(job) for job in jobs], next_cursor

    def _job_document(self, job: Any) -> dict:
        """Render one job for a command reply, bounded and without a local path."""
        return {
            "inferenceId": job.inference_id,
            "route": job.route_id,
            "state": job.state.value,
            "attempts": job.attempts,
            "nextAttemptAtMs": job.next_attempt_at_ms,
            "configGeneration": job.config_generation,
            "source": {
                "kind": job.source.kind.value,
                "relativePath": job.source.relative_path,
                "bytes": job.source.bytes,
                "sha256": job.source.sha256,
                "captureId": job.source.capture_id,
            },
            "model": {
                "id": job.model.id,
                "version": job.model.version,
                "digest": job.model.digest,
            },
            "lastError": self._ledger.last_error(job.inference_id),
        }

    def job_counts(self, route_id: Optional[str] = None) -> dict:
        """Return how many jobs are in each state."""
        return {
            state.value: count
            for state, count in sorted(
                self._ledger.counts_by_state(route_id).items(), key=lambda item: item[0].value
            )
        }

    def scheduler_summary(self) -> dict:
        """Return the scheduler view of the lanes, the cells, and its counters."""
        return self._scheduler.status()

    def rescan(self, route_id: Optional[str] = None) -> int:
        """Walk one route root now, or every spool route.

        Args:
            route_id: The route, or ``None`` for all of them.

        Returns:
            How many inputs the walk announced.
        """
        found = 0
        for runtime in self._runtimes(route_id):
            source = runtime.source
            if runtime.route.is_spool and source is not None:
                found += source.rescan()
        return found

    def _runtimes(self, route_id: Optional[str]) -> list:
        """Return the runtimes a command addresses."""
        if route_id is None:
            return list(self._routes.values())
        runtime = self._routes.get(route_id)
        return [runtime] if runtime is not None else []

    def preload_model(self, model_id: Optional[str], digest: Optional[str]) -> dict:
        """Stage and warm one model generation before a route needs it.

        Args:
            model_id: The configured model id, or ``None`` to select by digest.
            digest: The bundle digest, or ``None`` to select by id.

        Returns:
            What the generation now looks like.

        Raises:
            CommandException: The configuration names no such model, or it could not be staged.
        """
        from edgecommons.command_inbox import CommandException

        from image_processor.artifacts import ArtifactError
        from image_processor.commands import ERR_FAILED, ERR_NOT_FOUND

        entry = self._model_entry(model_id, digest)
        if entry is None:
            raise CommandException(ERR_NOT_FOUND, "no configured model matches")
        try:
            bundle = self._artifacts.stage(entry)
            if entry.activation.require_warmup:
                self._artifacts.warm(entry, bundle)
        except ArtifactError as exc:
            raise CommandException(ERR_FAILED, f"{exc.code}: {exc.message}") from exc
        self._scheduler.reset_lane(entry.digest)
        switched = self._artifacts.reconcile(force=True)
        return {
            "id": entry.id,
            "version": entry.version,
            "digest": entry.digest,
            "staged": True,
            "warmed": bool(entry.activation.require_warmup),
            "routesSwitched": switched,
        }

    def _model_entry(self, model_id: Optional[str], digest: Optional[str]) -> Any:
        """Find the configured model a command names."""
        for entry in self._config.models:
            if digest and entry.digest != digest:
                continue
            if model_id and entry.id != model_id:
                continue
            return entry
        return None

    def evict_model(self, digest: str) -> dict:
        """Release an idle resident session. A leased generation is refused."""
        return self._scheduler.evict(digest)

    def reload_model_catalog(self) -> dict:
        """Re-evaluate the configured models against the cache, retrying failed generations."""
        for entry in self._config.models:
            self._scheduler.reset_lane(entry.digest)
        switched = self._artifacts.reconcile(force=True)
        collected = self._artifacts.collect()
        return {
            "routesSwitched": switched,
            "collected": list(collected),
            "models": self._artifacts.status(),
        }

    def set_activation_override(self, route_id: str, enabled: Optional[bool]) -> dict:
        """Persist an operator override of a route configured activation."""
        runtime = self._routes.get(route_id)
        if runtime is None:
            from edgecommons.command_inbox import CommandException

            from image_processor.commands import ERR_NOT_FOUND

            raise CommandException(ERR_NOT_FOUND, f"no route '{route_id}'")
        runtime.override = enabled
        self._ledger.kv_set(
            f"image-processor/activation-override/{route_id}",
            None if enabled is None else ("true" if enabled else "false"),
        )
        if runtime.enabled:
            self._resume_route(runtime)
        else:
            self._pause_route(runtime)
        return {
            "route": route_id,
            "configured": bool(runtime.route.enabled),
            "override": enabled,
            "effective": runtime.enabled,
        }

    def retry_publication(self, route_id: Optional[str], inference_id: Optional[str]) -> dict:
        """Return exhausted publications to the outbox and drain it."""
        returned = []
        for job in self._jobs_in(JobState.PUBLISH_EXHAUSTED, route_id, inference_id):
            try:
                self._ledger.retry_publication(job.inference_id)
                returned.append(job.inference_id)
            except Exception as exc:  # noqa: BLE001 - report what actually happened
                logger.warning("could not retry %s: %s", job.inference_id, exc)
        published = self._publisher.drain_once() if returned else 0
        return {"returned": returned, "published": published}

    def retry_cleanup(self, route_id: Optional[str], inference_id: Optional[str]) -> dict:
        """Retry the completion actions that failed."""
        repaired = []
        failed = []
        for job in self._jobs_in(JobState.CLEANUP_FAILED, route_id, inference_id):
            intent = self._ledger.cleanup_intent(job.inference_id)
            if intent is None:
                continue
            state = self._completer.reconcile(intent, self._collision_for(job.inference_id))
            (repaired if state is not JobState.CLEANUP_FAILED else failed).append(
                job.inference_id
            )
        return {"repaired": repaired, "stillFailed": failed}

    def reconcile(self, route_id: Optional[str] = None) -> dict:
        """Re-decide every open cleanup intent against observed filesystem state."""
        outcomes: dict = {}
        for intent in self._ledger.pending_cleanup(PRIME_PAGE_SIZE):
            job = self._ledger.get(intent.inference_id)
            if job is None or (route_id is not None and job.route_id != route_id):
                continue
            state = self._completer.reconcile(intent, self._collision_for(intent.inference_id))
            outcomes[intent.inference_id] = state.value
        return {"reconciled": outcomes, "counts": self.job_counts(route_id)}

    def _jobs_in(
        self, state: JobState, route_id: Optional[str], inference_id: Optional[str]
    ) -> list:
        """Return the jobs a repair verb addresses."""
        if inference_id:
            job = self._ledger.get(inference_id)
            if job is None or job.state is not state:
                return []
            if route_id is not None and job.route_id != route_id:
                return []
            return [job]
        jobs, _cursor = self._ledger.by_state([state], route_id, None, PRIME_PAGE_SIZE)
        return jobs

    def pause(self, route_id: Optional[str] = None) -> dict:
        """Stop claiming new work. In-flight jobs finish."""
        if route_id is None:
            self._scheduler.pause()
        for runtime in self._runtimes(route_id):
            runtime.paused = True
            self._pause_route(runtime)
        return {"paused": True, "routes": [runtime.route.id for runtime in self._runtimes(route_id)]}

    def resume(self, route_id: Optional[str] = None) -> dict:
        """Start claiming work again."""
        if route_id is None:
            self._scheduler.resume()
        for runtime in self._runtimes(route_id):
            runtime.paused = False
            self._resume_route(runtime)
        return {
            "paused": False,
            "routes": [runtime.route.id for runtime in self._runtimes(route_id)],
        }

    def _pause_route(self, runtime: RouteRuntime) -> None:
        """Stop one route from claiming work, leaving its admitted jobs alone."""
        source = runtime.source
        if source is None:
            return
        if runtime.route.is_spool:
            source.stop()
            if runtime.reconciler is not None:
                runtime.reconciler.stop()
            return
        messaging = self._gg.get_messaging()
        for topic in runtime.topics:
            try:
                messaging.unsubscribe(topic)
            except Exception as exc:  # noqa: BLE001 - pausing must not raise
                logger.debug("unsubscribing %s failed: %s", topic, exc)
        runtime.topics = ()

    def _resume_route(self, runtime: RouteRuntime) -> None:
        """Let one route claim work again."""
        source = runtime.source
        if source is None or runtime.paused or not runtime.enabled:
            return
        if runtime.route.is_spool:
            source.start()
            camera = runtime.route.source.camera
            if runtime.reconciler is not None and camera.reconcile_capture_status_secs > 0:
                runtime.reconciler.start()
            if not runtime.topics:
                runtime.topics = self._subscribe_hint(runtime.route)
            source.nudge()
            return
        if not runtime.topics:
            runtime.topics = self._subscribe_trigger(runtime.route, source)


# -- module helpers --------------------------------------------------------------------


def _body_of(message: Any) -> Any:
    """Return a message body, accepting a core ``Message``, a mapping, or raw bytes."""
    accessor = getattr(message, "get_body", None)
    if callable(accessor):
        return accessor()
    if isinstance(message, dict) and "body" in message:
        return message["body"]
    return message


def _prepared_message(topic: str, encoded: bytes) -> Optional[PreparedAppMessage]:
    """Rebuild a prepared app message around stored envelope bytes.

    Returns:
        The prepared message, or ``None`` when the stored bytes cannot be reconstituted into one.
        The caller then publishes the same bytes through the messaging client instead, which is
        what the facade does with them anyway.
    """
    try:
        return PreparedAppMessage(topic, Message.from_bytes(encoded), encoded)
    except Exception:  # noqa: BLE001 - the bytes are still published, just not through the facade
        logger.debug("stored envelope for %s is not a prepared app message", topic, exc_info=True)
        return None


def _command_answer(reply: Any) -> Optional[dict]:
    """Unwrap a command reply into the answer its verb produced.

    Args:
        reply: The reply message, a mapping, or ``None``.

    Returns:
        The verb answer, an ``errorCode`` / ``errorMessage`` pair when the verb refused, or
        ``None`` when there is nothing readable to unwrap.
    """
    body = _body_of(reply)
    if not isinstance(body, dict):
        return None
    if "ok" not in body:
        return body
    if body.get("ok"):
        result = body.get("result")
        return result if isinstance(result, dict) else {}
    error = body.get("error") if isinstance(body.get("error"), dict) else {}
    return {
        "errorCode": error.get("code", "COMMAND_FAILED"),
        "errorMessage": error.get("message", ""),
    }


def _credentials_document(value: str) -> dict:
    """Turn a resolved secret into the mapping a fetcher reads.

    A secret holding a JSON object is used as it stands, which is how an S3 key pair or a set of
    request headers is carried. Anything else is a bearer token, which is the only unambiguous
    reading of a bare string.
    """
    import json

    try:
        document = json.loads(value)
    except ValueError:
        return {"bearerToken": value}
    return document if isinstance(document, dict) else {"bearerToken": value}


def _invalid_identity(route_id: str, relative_path: str, size: int, digest: str) -> Any:
    """Build the identity of an input that is being quarantined rather than inferred."""
    from image_processor.types import SourceIdentity

    return SourceIdentity(
        kind=SourceKind.SPOOL,
        route_id=route_id,
        relative_path=relative_path,
        bytes=int(size),
        sha256=digest,
    )


def _completion_measure(action: Any) -> str:
    """Return the completion metric measure one action increments."""
    return {
        CompletionAction.ARCHIVE: "archived",
        CompletionAction.DELETE: "deleted",
        CompletionAction.QUARANTINE: "quarantined",
        CompletionAction.RETAIN: "retained",
    }.get(action, "retained")


def _route_changed(previous: Any, current: Any) -> bool:
    """Whether a route changed in a way that means rebuilding its source."""
    return (
        previous.source != current.source
        or previous.model_ref != current.model_ref
        or previous.enabled != current.enabled
    )


def _source_detail(route: Any) -> str:
    """Return the operator detail line for a route input."""
    if route.is_spool:
        return str(route.source.root)
    return ", ".join(route.source.subscribe)


def _sum_counters(runtimes: Iterable) -> dict:
    """Total the discovery counters of every route source."""
    names = {
        "discovered_count": "discovered",
        "rejected_count": "rejected",
        "rescans": "rescans",
        "nudges": "nudges",
        "hints_accepted": "hintsAccepted",
        "hints_rejected": "hintsRejected",
        "hints_unmapped": "hintsUnmapped",
    }
    totals: dict = {measure: 0.0 for measure in names.values()}
    records = 0.0
    for runtime in runtimes:
        source = getattr(runtime, "source", None)
        for attribute, measure in names.items():
            totals[measure] += float(getattr(source, attribute, 0) or 0)
        reconciler = getattr(runtime, "reconciler", None)
        records += float(getattr(reconciler, "verified_count", 0) or 0)
    totals["captureRecords"] = records
    return totals


def _file_size(path: Path) -> float:
    """Return a file size in bytes, or zero when it is not there."""
    try:
        return float(path.stat().st_size)
    except OSError:
        return 0.0


def _tree_size(root: Path) -> float:
    """Return how many bytes a directory tree holds."""
    total = 0.0
    try:
        for current, _directories, files in os.walk(str(root)):
            for name in files:
                total += _file_size(Path(current) / name)
    except OSError:  # pragma: no cover - an unreadable tree is reported by readiness
        return total
    return total


__all__ = ["HEALTH_INTERVAL_SECS", "IN_FLIGHT_STATES", "ImageProcessor", "RouteRuntime"]
