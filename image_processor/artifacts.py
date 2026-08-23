"""The model artifact manager: desired generation to active generation (DESIGN.md 9, LLD 8).

Configuration names the model set a route should be running. This is what makes that true, in the
order DESIGN.md 9 fixes and no other: fetch, verify the tarball digest, verify the signature when
one is required, extract under bounds, verify every declared file, check the manifest against the
schema and the task family, promote into the content-addressed cache, run golden warmup on the
target provider, and only then switch the route generation.

Two properties follow from that order and both are load-bearing.

*Nothing serves before it warms.* A bundle that fetched and extracted but failed its warmup never
becomes a route active generation, so a model whose weights are wrong degrades its routes instead
of answering wrongly.

*The switch is atomic and the previous generation is kept.* The route desired and active
generations are two separate durable fields; while they differ the route reports ``STAGING`` and
keeps serving the last known good model. Rollback is therefore a fact about the cache rather than
a download.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

from image_processor.bundles import BundleError, stage_bundle
from image_processor.engine.families import FamilyError, family_for
from image_processor.engine.protocol import LoadFailed, Loaded, LoadModel, Unload

logger = logging.getLogger(__name__)

#: How often the manager re-evaluates desired against active.
DEFAULT_INTERVAL_SECS = 30.0

#: How long a failed generation is left alone before it is tried again.
DEFAULT_RETRY_BACKOFF_SECS = 60.0

#: The deadline for one warmup load.
DEFAULT_WARMUP_TIMEOUT_S = 600.0


class ArtifactError(Exception):
    """A model generation could not be made active.

    Attributes:
        code: Stable SCREAMING_SNAKE code.
        message: Operator-readable detail.
    """

    def __init__(self, code: str, message: str = "") -> None:
        """Initialize the error.

        Args:
            code: Stable SCREAMING_SNAKE code.
            message: Operator-readable detail.
        """
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}" if message else code)


@dataclass
class GenerationState:
    """What the manager knows about one model generation.

    Attributes:
        model_id: The model id.
        version: The model version.
        digest: The bundle digest.
        staged: Whether a verified bundle is in the cache.
        warmed: Whether golden warmup succeeded on this device.
        warmup_samples: Golden samples the warmup compared. ``0`` means the bundle carries none
            and the session was primed instead.
        load_ms: Wall-clock milliseconds the warmup load took, measured on this device. It is the
            reload cost the residency policy prices an eviction against, and the number
            ``get-models`` reports.
        device_mib: Device memory this model occupies, measured across the warmup load from a
            baseline taken after the cell device context exists, or ``0`` when no device probe is
            available. It is the model alone, which is what admission budgets against before the
            scheduler loads this generation for real; the context is fixed per-cell overhead the
            scheduler carries once (DESIGN.md 10.2).
        error: The last failure, or ``None``.
        failed_at: When that failure happened, on the monotonic clock.
        routes: The routes that want this generation.
    """

    model_id: str
    version: str
    digest: str
    staged: bool = False
    warmed: bool = False
    warmup_samples: int = 0
    load_ms: float = 0.0
    device_mib: int = 0
    error: Optional[str] = None
    failed_at: float = 0.0
    routes: tuple = ()


def family_validator(manifest: Any) -> None:
    """Refuse a bundle whose head no task family can interpret (DESIGN.md 9 step 4).

    Args:
        manifest: The parsed bundle manifest.

    Raises:
        FamilyError: The family is not implemented, or its parameters do not describe the head.
    """
    family_for(manifest).validate_manifest(manifest)


class ArtifactManager:
    """Makes the configured model set active, and keeps it that way.

    Args:
        config: The parsed :class:`~image_processor.config.models.ComponentConfig`.
        cache: The content-addressed :class:`~image_processor.bundles.cache.BundleCache`.
        ledger: The durable ledger, which holds each route desired and active generation.
        supervisor: The executor supervisor, for the warmup load.
        trusted_keys: ``keyId`` to public key bytes, already resolved from the vault.
        credentials: ``modelId`` to the credentials mapping for its source, already resolved.
        events: The :class:`~image_processor.outputs.events.RouteEvents` helper, or ``None``.
        metrics: The :class:`~image_processor.metrics.ProcessorMetrics` accumulator, or ``None``.
        on_activated: Called with ``(route_id, previous_digest, digest)`` after a switch.
        on_requeued: Called with ``(route_id, count)`` after configuration-blocked jobs were
            returned to ``READY``, so the caller can hand them back to the scheduler.
        interval_secs: How often the background thread reconciles.
        retry_backoff_secs: How long a failed generation is left alone.
        warmup_timeout_s: The deadline for one warmup load.
        clock: Monotonic clock, injected by tests.
    """

    def __init__(
        self,
        config: Any,
        cache: Any,
        ledger: Any,
        supervisor: Any = None,
        *,
        trusted_keys: Optional[Mapping] = None,
        credentials: Optional[Mapping] = None,
        events: Any = None,
        metrics: Any = None,
        on_activated: Optional[Callable[[str, Optional[str], str], None]] = None,
        on_requeued: Optional[Callable[[str, int], None]] = None,
        interval_secs: float = DEFAULT_INTERVAL_SECS,
        retry_backoff_secs: float = DEFAULT_RETRY_BACKOFF_SECS,
        warmup_timeout_s: float = DEFAULT_WARMUP_TIMEOUT_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Build the manager. Nothing is fetched until :meth:`reconcile` runs."""
        self.config = config
        self.cache = cache
        self.ledger = ledger
        self.supervisor = supervisor
        self.trusted_keys = dict(trusted_keys or {})
        self.credentials = dict(credentials or {})
        self.interval_secs = float(interval_secs)
        self.retry_backoff_secs = float(retry_backoff_secs)
        self.warmup_timeout_s = float(warmup_timeout_s)
        self._events = events
        self._metrics = metrics
        self._on_activated = on_activated
        self._on_requeued = on_requeued
        self._clock = clock
        self._lock = threading.RLock()
        self._generations: dict = {}
        self._rollback: dict = {}
        self._route_errors: dict = {}
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.counters = {
            "staged": 0,
            "activated": 0,
            "stagingFailures": 0,
            "warmupFailures": 0,
            "rollbacks": 0,
            "requeued": 0,
        }

    # -- configuration -----------------------------------------------------------------

    def adopt(self, config: Any) -> None:
        """Take a new configuration generation, keeping what is already staged.

        Args:
            config: The newly applied configuration.
        """
        with self._lock:
            self.config = config
        self.wake()

    # -- staging -----------------------------------------------------------------------

    def stage(self, entry: Any) -> Any:
        """Fetch and verify one model bundle into the cache (DESIGN.md 9 steps 2 to 6).

        A bundle already in the cache is verified and reused rather than fetched again, which is
        what makes a restart cheap and a redeployment of the same digest a no-op.

        Args:
            entry: The :class:`~image_processor.config.models.ModelEntry` to stage.

        Returns:
            The :class:`~image_processor.types.CachedBundle`.

        Raises:
            ArtifactError: The bundle could not be fetched, verified, extracted, or interpreted.
        """
        cached = self.cache.get(entry.digest, verify=False)
        if cached is not None:
            return cached
        signing = self.config.signing
        sources = self.config.model_sources
        try:
            cached = stage_bundle(
                entry.uri,
                entry.digest,
                Path(self.config.paths.staging),
                self.cache,
                credentials=self.credentials.get(entry.id),
                signing_required=bool(signing.required),
                trusted_keys=self.trusted_keys,
                allowed_prefixes=list(sources.allowed_uri_prefixes) or None,
                model_id=entry.id,
                version=entry.version,
                available_providers=list(self.config.runtime.providers),
                validators=(family_validator,),
            )
        except (BundleError, FamilyError) as exc:
            raise ArtifactError(getattr(exc, "code", "STAGING_FAILED"), str(exc)) from exc
        except OSError as exc:
            raise ArtifactError("STAGING_IO_FAILED", str(exc)) from exc
        self.counters["staged"] += 1
        if self._metrics is not None:
            self._metrics.incr("ImageProcessorModelCache", "staged")
        logger.info("staged %s %s (%s)", entry.id, entry.version, entry.digest)
        return cached

    def warm(self, entry: Any, bundle: Any) -> Loaded:
        """Run golden warmup on the target provider (DESIGN.md 9 step 5).

        The session is released again as soon as it has proved itself. Activation asks whether
        this bundle loads and reproduces its goldens on this device; it does not ask for a
        resident session, and the scheduler is the only component that decides what stays on the
        GPU. A session left behind here would hold device memory the scheduler's residency map
        knows nothing about, so every later admission would budget against a number that is
        already wrong (DESIGN.md 10.2). The measurements survive the unload: what the load cost
        and what it occupied are carried back to the caller. What it occupied is the model alone,
        because the cell establishes and measures its device context separately, so a warmup that
        happens to be the first load on a cell does not report the context as model memory.

        Args:
            entry: The model entry being activated.
            bundle: The staged :class:`~image_processor.types.CachedBundle`.

        Returns:
            The :class:`~image_processor.engine.protocol.Loaded` reply the warmup produced.

        Raises:
            ArtifactError: No executor can warm it, or the warmup failed.
        """
        if self.supervisor is None:
            raise ArtifactError("NO_EXECUTOR", "no executor boundary is configured")
        cells = [cell for cell in self.supervisor.cells() if cell.is_alive()]
        if not cells:
            raise ArtifactError("NO_EXECUTOR", "no executor cell is alive")
        runtime = self.config.runtime
        manifest = bundle.manifest
        message = LoadModel(
            digest=bundle.digest,
            bundle_root=str(bundle.root),
            providers=tuple(runtime.providers),
            provider_policy=manifest.provider_policy,
            providers_permitted=tuple(manifest.providers_permitted),
            warmup=True,
            required_provider=runtime.required_provider,
            allow_cpu_only=bool(runtime.allow_cpu_only),
        )
        last: Optional[str] = None
        for cell in cells:
            try:
                reply = self.supervisor.call(cell, message, self.warmup_timeout_s)
            except Exception as exc:  # noqa: BLE001 - a dead cell is one cell, not the answer
                last = f"{type(exc).__name__}: {exc}"
                logger.warning("cell %s could not warm %s: %s", cell.cell_id, entry.id, exc)
                continue
            if isinstance(reply, Loaded):
                logger.info(
                    "warmed %s on %s in %.0f ms (%d golden sample(s), providers %s)",
                    entry.id,
                    cell.cell_id,
                    reply.load_ms,
                    reply.warmup_samples,
                    ",".join(reply.providers_assigned),
                )
                self._release(cell, entry, bundle.digest)
                return reply
            if isinstance(reply, LoadFailed):
                last = f"{reply.code or 'LOAD_FAILED'}: {reply.error}"
                if reply.error_class == "permanent":
                    break
                continue
            last = f"UNEXPECTED_REPLY: {type(reply).__name__}"
        raise ArtifactError("WARMUP_FAILED", last or "no executor accepted the model")

    def _release(self, cell: Any, entry: Any, digest: str) -> None:
        """Give back the session a warmup built, on the cell that built it.

        A failed unload is logged rather than raised: the bundle has already proved that it
        loads, which is what activation asked, and a session the cell could not release is the
        supervisor's recycle path to deal with (DESIGN.md 10.4).

        Args:
            cell: The cell that answered the warmup.
            entry: The model entry being activated, for the message.
            digest: The bundle digest to release.
        """
        try:
            self.supervisor.call(cell, Unload(digest), self.warmup_timeout_s)
        except Exception as exc:  # noqa: BLE001 - the warmup answered; the unload is hygiene
            logger.warning(
                "cell %s could not release %s after its warmup: %s", cell.cell_id, entry.id, exc
            )
            return
        logger.debug("released %s on %s after its warmup", entry.id, cell.cell_id)

    # -- activation --------------------------------------------------------------------

    def reconcile(self, route_ids: Optional[Iterable] = None, *, force: bool = False) -> int:
        """Bring every route active generation up to its desired one.

        Args:
            route_ids: Restrict the pass to these routes, or ``None`` for every enabled route.
            force: Retry a generation whose backoff has not elapsed. The ``preload-model`` and
                ``reload-model-catalog`` commands set it.

        Returns:
            How many routes switched generation in this pass.
        """
        with self._lock:
            config = self.config
        wanted = set(route_ids) if route_ids is not None else None
        switched = 0
        for route in config.enabled_routes:
            if wanted is not None and route.id not in wanted:
                continue
            try:
                if self._reconcile_route(route, config, force=force):
                    switched += 1
            except Exception:  # noqa: BLE001 - one bad model degrades its route, not the pass
                logger.exception("route %s could not reconcile its model", route.id)
        return switched

    def _reconcile_route(self, route: Any, config: Any, *, force: bool) -> bool:
        """Reconcile one route. Returns whether it switched generation."""
        desired = route.model_ref.digest
        _, active = self.ledger.route_generation(route.id)
        if active != desired:
            # Record the desire first: while desired and active differ the route reports STAGING
            # and keeps serving the generation it already has.
            self.ledger.set_route_generation(route.id, desired, active)
        elif not force:
            return False
        entry = config.model_entry(route.model_ref)
        if entry is None:
            self._fail(route, desired, "UNRESOLVED_MODEL_REF", "no models[] entry names it")
            return False
        state = self._state(entry, route)
        if not force and state.error and self._clock() - state.failed_at < self.retry_backoff_secs:
            return False
        try:
            bundle = self.stage(entry)
            state.staged = True
            if entry.activation.require_warmup:
                reply = self.warm(entry, bundle)
                state.warmed = True
                state.warmup_samples = int(reply.warmup_samples)
                state.load_ms = float(reply.load_ms)
                state.device_mib = int(reply.device_mib)
        except ArtifactError as exc:
            state.error = f"{exc.code}: {exc.message}"
            state.failed_at = self._clock()
            self._fail(route, desired, exc.code, exc.message)
            return False
        state.error = None
        self._route_errors.pop(route.id, None)
        if active == desired:
            return False
        self.ledger.set_route_generation(route.id, desired, desired)
        if active and entry.activation.retain_for_rollback:
            self._rollback[route.id] = active
        self.counters["activated"] += 1
        if self._metrics is not None:
            self._metrics.incr("ImageProcessorModelCache", "activated")
        logger.info("route %s is now running %s (%s)", route.id, entry.id, desired)
        if self._events is not None:
            self._events.model_activated(route.id, f"{entry.id} {entry.version}", desired)
        if self._on_activated is not None:
            try:
                self._on_activated(route.id, active, desired)
            except Exception:  # noqa: BLE001 - a listener failure is not an activation failure
                logger.exception("the activation listener failed for %s", route.id)
        self.requeue_blocked(route.id)
        self.collect()
        return True

    def _fail(self, route: Any, digest: str, code: str, message: str) -> None:
        """Report a generation that could not be made active, keeping the last known good.

        The failure is recorded against the route as well as counted, because a route whose model
        will not stage, warm, or activate is a degraded route and an operator asking why is asking
        about this: :meth:`route_error` is what the ``route-degraded`` condition and the
        per-instance connectivity detail line carry (DESIGN.md 12.3, 14).
        """
        self._route_errors[route.id] = f"{code}: {message}" if message else code
        which = "warmupFailures" if code == "WARMUP_FAILED" else "stagingFailures"
        self.counters[which] += 1
        if self._metrics is not None:
            self._metrics.incr("ImageProcessorModelCache", which)
        logger.error("route %s cannot activate %s: %s: %s", route.id, digest, code, message)
        if self._events is None:
            return
        if code == "WARMUP_FAILED":
            self._events.model_warmup_failed(route.id, digest, message)
        else:
            self._events.model_staging_failed(route.id, digest, f"{code}: {message}")

    def route_error(self, route_id: str) -> Optional[str]:
        """Return why one route's model could not be made active, if it could not.

        Args:
            route_id: The route.

        Returns:
            The ``CODE: detail`` of the last staging, warmup, or activation failure, or ``None``
            once the route has activated a generation. A successful pass clears it, so a value
            here always describes the model the route is failing on now.
        """
        return self._route_errors.get(route_id)

    def requeue_blocked(self, route_id: str) -> int:
        """Give one route's configuration-blocked jobs another chance (DESIGN.md 7).

        A job at ``BLOCKED_CONFIGURATION`` is pinned to a generation that would not load. An
        activation that succeeded is the configuration change that unblocks it: the route is now
        running a generation that did load, so the jobs go back to ``READY`` and the scheduler
        runs them on its next pass.

        A ledger that refuses is logged rather than raised. The activation itself succeeded, and a
        route that is serving again must not be reported as failed because the jobs it stranded
        earlier could not be moved.

        Args:
            route_id: The route whose blocked jobs are returned to ``READY``.

        Returns:
            How many jobs moved.
        """
        try:
            moved = int(self.ledger.requeue_blocked(route_id))
        except Exception:  # noqa: BLE001 - a failed requeue is not a failed activation
            logger.warning(
                "route %s could not requeue its configuration-blocked jobs", route_id,
                exc_info=True,
            )
            return 0
        if not moved:
            return 0
        self.counters["requeued"] += moved
        logger.info("route %s returned %d configuration-blocked job(s) to READY", route_id, moved)
        if self._on_requeued is not None:
            try:
                self._on_requeued(route_id, moved)
            except Exception:  # noqa: BLE001 - a listener failure is not a requeue failure
                logger.exception("the requeue listener failed for %s", route_id)
        return moved

    def verified(self, route_id: str) -> bool:
        """Whether one route's catalog entry is staged and carries no activation failure.

        Args:
            route_id: The route.

        Returns:
            ``True`` when the route's desired generation is in the cache and the last pass over it
            recorded no failure. ``reload-model-catalog`` reads it to decide whose
            configuration-blocked jobs may go back to ``READY`` (DESIGN.md 7).
        """
        with self._lock:
            config = self.config
        route = config.route(route_id)
        if route is None:
            return False
        entry = config.model_entry(route.model_ref)
        if entry is None:
            return False
        state = self._generations.get(entry.digest)
        return bool(state is not None and state.staged and not state.error)

    def rollback(self, route_id: str) -> Optional[str]:
        """Return one route retained rollback generation, if it has one.

        Args:
            route_id: The route.

        Returns:
            The digest of the previous generation, or ``None``.
        """
        return self._rollback.get(route_id)

    def _state(self, entry: Any, route: Any) -> GenerationState:
        """Return the tracked state of one generation, creating it on first sight."""
        state = self._generations.get(entry.digest)
        if state is None:
            state = GenerationState(
                model_id=entry.id, version=entry.version, digest=entry.digest
            )
            self._generations[entry.digest] = state
        if route.id not in state.routes:
            state.routes = tuple(sorted(set(state.routes) | {route.id}))
        return state

    # -- cache -------------------------------------------------------------------------

    def pinned(self) -> tuple:
        """Return every digest garbage collection must keep (DESIGN.md 9).

        Returns:
            The active, desired, and rollback generation of every route, plus the digest of every
            job the ledger has not finished with.
        """
        with self._lock:
            config = self.config
        keep = set(self._rollback.values())
        for route in config.routes:
            desired, active = self.ledger.route_generation(route.id)
            keep.update(digest for digest in (desired, active) if digest)
            keep.add(route.model_ref.digest)
        for entry in config.models:
            keep.add(entry.digest)
        return tuple(sorted(keep))

    def collect(self) -> tuple:
        """Remove cached bundles nothing pins.

        Returns:
            The digests removed.
        """
        try:
            removed = tuple(self.cache.gc(self.pinned()))
        except Exception:  # noqa: BLE001 - a failed collection is not a failed activation
            logger.warning("the model cache could not be collected", exc_info=True)
            return ()
        if removed:
            logger.info("collected %d unpinned bundle(s)", len(removed))
        return removed

    # -- reporting ---------------------------------------------------------------------

    def status(self) -> list:
        """Describe every configured generation, for the ``get-models`` verb (DESIGN.md 13).

        Returns:
            One entry per model, naming its staged, active, and rollback state.
        """
        with self._lock:
            config = self.config
        active: dict = {}
        staging: dict = {}
        for route in config.routes:
            desired, current = self.ledger.route_generation(route.id)
            if current:
                active.setdefault(current, []).append(route.id)
            if desired and desired != current:
                staging.setdefault(desired, []).append(route.id)
        rollbacks = set(self._rollback.values())
        out = []
        for entry in config.models:
            state = self._generations.get(entry.digest)
            out.append(
                {
                    "id": entry.id,
                    "version": entry.version,
                    "digest": entry.digest,
                    "uri": entry.uri,
                    "staged": bool(self.cache.get(entry.digest, verify=False)),
                    "warmed": bool(state.warmed) if state else False,
                    "warmupSamples": int(state.warmup_samples) if state else 0,
                    "loadMs": round(state.load_ms, 3) if state else 0.0,
                    "deviceMiB": int(state.device_mib) if state else 0,
                    "activeRoutes": sorted(active.get(entry.digest, ())),
                    "stagingRoutes": sorted(staging.get(entry.digest, ())),
                    "rollback": entry.digest in rollbacks,
                    "error": state.error if state else None,
                }
            )
        return out

    # -- lifecycle ---------------------------------------------------------------------

    def start(self) -> "ArtifactManager":
        """Start the reconciliation thread.

        Returns:
            This manager.
        """
        if self._thread is not None:
            return self
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="image-processor-artifacts", daemon=True
        )
        self._thread.start()
        return self

    def stop(self, timeout_s: float = 10.0) -> None:
        """Stop the reconciliation thread.

        Args:
            timeout_s: How long to wait for the thread.
        """
        self._stop.set()
        self._wake.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout_s)

    def wake(self) -> None:
        """Ask for a reconciliation pass soon."""
        self._wake.set()

    def _loop(self) -> None:
        """Reconcile on the interval, and whenever something asks."""
        while not self._stop.is_set():
            try:
                self.reconcile()
            except Exception:  # noqa: BLE001 - the thread outlives one bad pass
                logger.warning("a model reconciliation pass failed", exc_info=True)
            self._wake.wait(self.interval_secs)
            self._wake.clear()


__all__ = [
    "DEFAULT_INTERVAL_SECS",
    "DEFAULT_RETRY_BACKOFF_SECS",
    "ArtifactError",
    "ArtifactManager",
    "GenerationState",
    "family_validator",
]
