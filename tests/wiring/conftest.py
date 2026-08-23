"""Fakes for the application wiring suites (WP6).

The fakes here are deliberately close to the real thing rather than convenient. The app facade
returns a real prepared message over real envelope bytes, so the outbox retries the same bytes a
broker would have seen; the command inbox enforces scope and pagination the way the library does;
and the messaging fake records exactly what was published on which topic, so a test asserts about
the wire rather than about a mock call.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pytest
from edgecommons.command_inbox import CommandScope, DeferredReply
from edgecommons.facades.app_facade import PreparedAppMessage
from edgecommons.messaging.message import Message, MessageHeader

DEVICE = "smoke-device"
COMPONENT = "image-processor"


class FakeConfigManager:
    """The configuration surface :class:`ImageProcessor` reads."""

    def __init__(self, document: dict, thing_name: str = DEVICE) -> None:
        self.document = document
        self.thing_name = thing_name
        self.generation = 1
        self.listeners: list = []

    def component(self) -> dict:
        """Return the ``component`` section."""
        return self.document.get("component") or {}

    def get_global_config(self) -> dict:
        """Return ``component.global``."""
        return self.component().get("global") or {}

    def get_instance_ids(self) -> list:
        """Return the configured route ids."""
        return [entry["id"] for entry in self.component().get("instances") or []]

    def get_instance_config(self, instance_id: str) -> dict:
        """Return one route configuration."""
        for entry in self.component().get("instances") or []:
            if entry["id"] == instance_id:
                return entry
        return {}

    def get_generation(self) -> int:
        """Return the configuration generation."""
        return self.generation

    def get_thing_name(self) -> str:
        """Return the device identity."""
        return self.thing_name

    def get_component_name(self) -> str:
        """Return the component name."""
        return COMPONENT

    def add_config_change_listener(self, listener: Any) -> None:
        """Record a listener."""
        self.listeners.append(listener)

    def apply(self, document: dict) -> bool:
        """Swap the configuration and drive the listeners, as the core does on a reload."""
        self.document = document
        self.generation += 1
        return all(listener.on_configuration_change(document) for listener in self.listeners)


@dataclass
class Published:
    """One thing that reached the fake bus."""

    topic: str
    body: Any
    confirmed: bool = False
    kind: str = "app"


class FakeMessaging:
    """Records publications and dispatches subscriptions."""

    def __init__(self) -> None:
        self.published: list = []
        self.subscriptions: dict = {}
        self.unsubscribed: list = []
        self.fail_confirmed: Optional[Exception] = None
        self.confirm_delay_calls = 0
        self.requests: list = []
        self.reply_factory = None

    def subscribe(self, topic, callback, max_concurrency=None, max_messages=None) -> None:
        """Register a subscription."""
        self.subscriptions[topic] = callback

    def unsubscribe(self, topic) -> None:
        """Drop a subscription."""
        self.subscriptions.pop(topic, None)
        self.unsubscribed.append(topic)

    def publish(self, topic, message) -> None:
        """Record an ordinary publication."""
        self.published.append(Published(topic, _body(message), False, "publish"))

    def publish_confirmed(self, topic, message, qos=None, timeout_secs=None) -> None:
        """Record a confirmed publication, failing when the test asked for a failure."""
        self.confirm_delay_calls += 1
        if self.fail_confirmed is not None:
            raise self.fail_confirmed
        self.published.append(Published(topic, _body(message), True, "confirmed"))

    def request(self, topic, message, timeout_secs=None):
        """Answer a request with whatever the test configured."""
        self.requests.append((topic, _body(message)))
        return _Iou(self.reply_factory(topic, message) if self.reply_factory else None)

    def topics(self, suffix: str = "") -> list:
        """Return the topics published on, optionally filtered by suffix."""
        return [item.topic for item in self.published if item.topic.endswith(suffix)]

    def bodies(self, marker: str) -> list:
        """Return the bodies published on topics containing ``marker``."""
        return [item.body for item in self.published if marker in item.topic]


class _Iou:
    """The one-shot future the messaging client returns from a request."""

    def __init__(self, reply: Any) -> None:
        self._reply = reply

    def get(self, timeout=None) -> tuple:
        """Return ``(done, reply)``."""
        return (self._reply is not None, self._reply)


def _body(message: Any) -> Any:
    """Return a published message body, whatever form it arrived in."""
    if isinstance(message, bytes):
        try:
            return Message.from_bytes(message).get_body()
        except Exception:  # noqa: BLE001 - a test double keeps the bytes
            return message
    accessor = getattr(message, "get_body", None)
    return accessor() if callable(accessor) else message


class FakeAppFacade:
    """The ``app()`` facade: prepares real envelopes and publishes them confirmed."""

    def __init__(self, gg, instance_id: Optional[str]) -> None:
        self._gg = gg
        self._instance = instance_id

    def prepare(self, name: str, channel: str, body: dict) -> PreparedAppMessage:
        """Freeze one application envelope, exactly as the library does."""
        instance = f"{self._instance}/" if self._instance else ""
        topic = f"ecv1/{self._gg.device}/{COMPONENT}/{instance}app/{channel}"
        message = Message(header=MessageHeader(name, "1.0"), body=body)
        return PreparedAppMessage(topic, message, message.to_bytes())

    def publish_confirmed(self, prepared, timeout_secs, routing=None) -> None:
        """Publish exact prepared bytes, failing when the test asked for a failure."""
        self._gg.messaging.publish_confirmed(prepared.topic, prepared.encoded_bytes, None, timeout_secs)

    def publish(self, name, channel, body, routing=None) -> None:
        """Publish without confirmation."""
        self._gg.messaging.publish(self.prepare(name, channel, body).topic, body)


class FakeDataFacade:
    """The ``data()`` facade: one topic per signal id."""

    def __init__(self, gg, instance_id: Optional[str]) -> None:
        self._gg = gg
        self._instance = instance_id

    def publish(self, signal_path: str, value: Any, quality: Any = None) -> None:
        """Record one mirrored reading."""
        if self._gg.mirror_error is not None:
            raise self._gg.mirror_error
        instance = f"{self._instance}/" if self._instance else ""
        topic = f"ecv1/{self._gg.device}/{COMPONENT}/{instance}data/{signal_path}"
        self._gg.messaging.published.append(
            Published(
                topic,
                {
                    "signal": {"id": signal_path},
                    "samples": [{"value": value, "quality": str(quality or "GOOD")}],
                },
                False,
                "data",
            )
        )


class FakeEventsFacade:
    """The ``events()`` facade: severity derives the channel."""

    def __init__(self, gg, instance_id: Optional[str]) -> None:
        self._gg = gg
        self._instance = instance_id

    def _emit(self, severity, event_type, message, context, alarm, active) -> None:
        instance = f"{self._instance}/" if self._instance else ""
        topic = f"ecv1/{self._gg.device}/{COMPONENT}/{instance}evt/{severity}/{event_type}"
        self._gg.emitted_events.append(
            {
                "topic": topic,
                "type": event_type,
                "severity": str(severity),
                "message": message,
                "context": context or {},
                "alarm": alarm,
                "active": active,
            }
        )

    def emit(self, event_type, message=None, context=None, severity=None) -> None:
        """Record a one-shot event."""
        self._emit(severity or "info", event_type, message, context, None, None)

    def raise_alarm(self, event_type, message=None, context=None, severity=None) -> None:
        """Record an alarm raise."""
        self._emit(severity or "critical", event_type, message, context, True, True)

    def clear_alarm(self, event_type, context=None, severity=None) -> None:
        """Record an alarm clear."""
        self._emit(severity or "critical", event_type, None, context, True, False)


class FakeMessageBuilder:
    """The envelope builder ``instance(id).new_message(...)`` returns."""

    def __init__(self, name: str, version: str) -> None:
        self._header = MessageHeader(name, version)
        self._body: Any = None

    def with_payload(self, body: Any) -> "FakeMessageBuilder":
        """Set the body."""
        self._body = body
        return self

    def with_correlation_id(self, correlation_id: str) -> "FakeMessageBuilder":
        """Set the correlation id."""
        self._header.correlation_id = correlation_id
        return self

    def build(self) -> Message:
        """Return the envelope."""
        return Message(header=self._header, body=self._body)


class FakeInstance:
    """One instance-scoped handle."""

    def __init__(self, gg, instance_id: Optional[str]) -> None:
        self._gg = gg
        self.instance_id = instance_id

    def app(self) -> FakeAppFacade:
        """The ``app`` facade."""
        return FakeAppFacade(self._gg, self.instance_id)

    def data(self) -> FakeDataFacade:
        """The ``data`` facade."""
        return FakeDataFacade(self._gg, self.instance_id)

    def events(self) -> FakeEventsFacade:
        """The ``evt`` facade."""
        return FakeEventsFacade(self._gg, self.instance_id)

    def new_message(self, name: str, version: str) -> FakeMessageBuilder:
        """A new envelope stamped with this instance token."""
        return FakeMessageBuilder(name, version)


class FakeMetricEmitter:
    """The metric service: definitions and emissions, both recorded."""

    def __init__(self) -> None:
        self.defined: dict = {}
        self.emitted: list = []

    def define_metric(self, metric: Any) -> None:
        """Record one definition."""
        self.defined[metric.get_name()] = metric

    def emit_metric(self, name: str, values: dict) -> None:
        """Record one emission."""
        self.emitted.append((name, dict(values)))

    def last(self, name: str) -> dict:
        """Return the most recent emission of one group."""
        for group, values in reversed(self.emitted):
            if group == name:
                return values
        return {}


class FakeCommandInbox:
    """The command inbox: scope enforcement and dispatch, as the library does it."""

    def __init__(self) -> None:
        self.handlers: dict = {}
        self.scopes: dict = {}
        self.deferred: list = []

    def register_outcome(self, verb: str, scope: CommandScope, handler: Any) -> None:
        """Register one verb."""
        if verb in self.handlers:
            raise ValueError(f"verb '{verb}' is already registered")
        self.handlers[verb] = handler
        self.scopes[verb] = scope

    def defer(self, request: Any, lifetime_secs: float) -> "FakeDeferredReply":
        """Hand out a guarded deferred-reply token."""
        token = FakeDeferredReply(lifetime_secs)
        self.deferred.append(token)
        return token

    def dispatch(self, verb: str, body: Optional[dict] = None, instance: Optional[str] = None):
        """Dispatch one request the way the inbox does, enforcing the declared scope."""
        from edgecommons.command_inbox import CommandException, Deferred, ImmediateError

        scope = self.scopes[verb]
        if scope is CommandScope.COMPONENT and instance:
            return ImmediateError("BAD_ARGS", f"{verb} is a component verb")
        request = Message(header=MessageHeader(verb, "1.0"), body=dict(body or {}))
        request.get_header().reply_to = "edgecommons/reply-test"
        try:
            outcome = self.handlers[verb](request, instance)
        except CommandException as exc:
            return ImmediateError(exc.code, str(exc))
        if isinstance(outcome, Deferred) and outcome.post_accept_continuation is not None:
            outcome.post_accept_continuation()
        return outcome


class FakeDeferredReply(DeferredReply):
    """A deferred reply token that records how it settled.

    It subclasses the library token rather than standing in for it, because
    :class:`edgecommons.command_inbox.Deferred` refuses an outcome carrying anything else -- and
    that refusal is part of what the verbs are being tested against.
    """

    def __init__(self, lifetime_secs: float) -> None:
        self.lifetime_secs = lifetime_secs
        self.state = "PROVISIONAL"
        self.result: Any = None
        self.error: Any = None

    def activate(self) -> bool:
        """Open the token."""
        self.state = "OPEN"
        return True

    def discard(self) -> bool:
        """Abandon the token."""
        self.state = "DISCARDED"
        return True

    def settle_success(self, result: Optional[dict] = None) -> str:
        """Settle with a result."""
        self.state = "SETTLED"
        self.result = result
        return "SETTLED"

    def settle_error(self, code: str, message: Optional[str] = None) -> str:
        """Settle with a coded error."""
        self.state = "SETTLED"
        self.error = (code, message)
        return "SETTLED"


class FakeVault:
    """The credentials vault, holding whatever the test put in it."""

    def __init__(self, secrets: Optional[dict] = None) -> None:
        self.secrets = dict(secrets or {})

    def exists(self, name: str) -> bool:
        """Whether the secret is there."""
        return name in self.secrets

    def get_string(self, name: str) -> Optional[str]:
        """The secret value."""
        return self.secrets.get(name)

    def get_json(self, name: str) -> Optional[dict]:
        """The secret value parsed as JSON."""
        raw = self.secrets.get(name)
        return json.loads(raw) if raw else None


class FakeGg:
    """The EdgeCommons handle, with every surface :class:`ImageProcessor` uses."""

    def __init__(self, document: dict, device: str = DEVICE, vault: Any = None) -> None:
        self.config_manager = FakeConfigManager(document, device)
        self.messaging = FakeMessaging()
        self.metrics = FakeMetricEmitter()
        self.inbox = FakeCommandInbox()
        self.device = device
        self.vault = vault
        self.emitted_events: list = []
        self.ready: list = []
        self.connectivity_provider = None
        self.mirror_error: Optional[Exception] = None
        self._instances: dict = {}

    def get_config_manager(self) -> FakeConfigManager:
        """The configuration manager."""
        return self.config_manager

    def get_messaging(self) -> FakeMessaging:
        """The messaging handle."""
        return self.messaging

    def get_metrics(self) -> FakeMetricEmitter:
        """The metric service."""
        return self.metrics

    def get_commands(self) -> FakeCommandInbox:
        """The command inbox."""
        return self.inbox

    def get_credentials(self) -> Any:
        """The credentials vault, or ``None``."""
        return self.vault

    def instance(self, instance_id: str) -> FakeInstance:
        """One instance-scoped handle, cached per id as the library caches it."""
        handle = self._instances.get(instance_id)
        if handle is None:
            handle = FakeInstance(self, instance_id)
            self._instances[instance_id] = handle
        return handle

    def app(self) -> FakeAppFacade:
        """The component-scope ``app`` facade."""
        return FakeAppFacade(self, None)

    def data(self) -> FakeDataFacade:
        """The component-scope ``data`` facade."""
        return FakeDataFacade(self, None)

    def events(self) -> FakeEventsFacade:
        """The component-scope ``evt`` facade."""
        return FakeEventsFacade(self, None)

    def set_ready(self, ready: bool) -> None:
        """Record a readiness transition."""
        self.ready.append(bool(ready))

    def set_instance_connectivity_provider(self, provider: Any) -> None:
        """Record the connectivity provider."""
        self.connectivity_provider = provider

    def event_types(self) -> list:
        """Return the event types emitted, in order."""
        return [event["type"] for event in self.emitted_events]

    def find_event(self, event_type: str) -> Optional[dict]:
        """Return the first event of a type."""
        for event in self.emitted_events:
            if event["type"] == event_type:
                return event
        return None


# -- the corpus and the configuration ---------------------------------------------------

BUNDLE = "synthetic-anomaly-scalar-1.0.0"


class Corpus:
    """The generated tier-1 corpus, packed into the bundle the wiring suites run.

    Attributes:
        root: Where the corpus was built.
        expected: The oracle document.
        tarball: The packed anomaly bundle.
        digest: Its bundle digest.
    """

    def __init__(self, root: Path, expected: dict, tarball: Path, digest: str) -> None:
        """Initialize the handle."""
        self.root = root
        self.expected = expected
        self.tarball = tarball
        self.digest = digest

    def image(self, name: str) -> bytes:
        """Return the bytes of one corpus image."""
        return (self.root / self.expected["images"][name]["path"]).read_bytes()

    def case(self, image: str) -> dict:
        """Return the oracle case for one image under the packed bundle."""
        for entry in self.expected["bundles"][BUNDLE]["cases"]:
            if entry["image"].endswith(image):
                return entry
        raise KeyError(image)


@pytest.fixture(scope="session")
def corpus(tmp_path_factory) -> Corpus:
    """Build and pack the corpus once for the whole session."""
    from tests.fixtures.build import build
    from tools.make_bundle import make_bundle

    root = tmp_path_factory.mktemp("wiring-corpus")
    expected = build(root)
    tarball = root / f"{BUNDLE}.tar"
    digest = make_bundle(root / "bundles" / BUNDLE, tarball)
    return Corpus(root, expected, tarball, digest)


def config_document(
    home: Path,
    corpus: Corpus,
    *,
    spool_root: Optional[Path] = None,
    write_sidecar: bool = True,
    on_success: str = "archive",
    reprocess: bool = False,
    trigger: bool = True,
) -> dict:
    """Build a configuration document for the wiring suites.

    Args:
        home: The directory the component owns.
        corpus: The packed corpus.
        spool_root: The spool route root, or ``None`` for ``home/spool``.
        write_sidecar: Whether the spool route writes evidence.
        on_success: The spool route success action.
        reprocess: Whether a model change replays terminal inputs.
        trigger: Whether to configure the trigger route.

    Returns:
        The whole configuration document, as a config source would deliver it.
    """
    spool = Path(spool_root) if spool_root is not None else home / "spool"
    instances = [
        {
            "id": "clearance-cam-01",
            "priority": 100,
            "source": {
                "kind": "spool",
                "root": str(spool),
                "include": ["**/*.png", "**/*.jpg"],
                "readiness": {"mode": "cameraSidecar"},
                "camera": {
                    "component": "camera-adapter",
                    "instance": "cam-01",
                    "subscribeAnnouncements": True,
                    "reconcileCaptureStatusSecs": 0,
                },
            },
            "modelRef": {"id": "synthetic-anomaly-scalar", "version": "1.0.0", "digest": corpus.digest},
            "outputs": {
                "writeResultSidecar": write_sidecar,
                "decisionSignals": [
                    {"id": "line-clearance/pass", "value": "$.decision.pass"},
                    {"id": "line-clearance/status", "value": "$.status"},
                ],
            },
            "completion": {
                "onSuccess": on_success,
                "archiveDir": str(home / "processed"),
                "failedDir": str(home / "failed"),
            },
            "reprocessExistingOnModelChange": reprocess,
        }
    ]
    if trigger:
        instances.append(
            {
                "id": "adhoc-inspect",
                "source": {
                    "kind": "trigger",
                    "subscribe": ["ecv1/+/inspection-ui/+/app/inspect/request"],
                    "fileRoot": str(home / "inbox"),
                    "inlineStaging": str(home / "staging" / "adhoc"),
                },
                "modelRef": {
                    "id": "synthetic-anomaly-scalar",
                    "version": "1.0.0",
                    "digest": corpus.digest,
                },
                "outputs": {"writeResultSidecar": False, "decisionSignals": []},
                "completion": {"onSuccess": "delete"},
            }
        )
    return {
        "component": {
            "token": COMPONENT,
            "global": {
                "paths": {
                    "stateDb": str(home / "state.db"),
                    "modelCache": str(home / "models"),
                    "staging": str(home / "staging"),
                },
                "runtime": {
                    "providers": ["CPUExecutionProvider"],
                    "requiredProvider": "CPUExecutionProvider",
                    "allowCpuOnly": True,
                },
                "gpu": {"devices": [], "reserveMiB": 0},
                "scheduler": {"maxAttempts": 2, "retryBackoffSecs": 0.01, "minResidencySecs": 0},
                "discovery": {"rescanSecs": 3600, "debounceMs": 10},
                "publish": {"confirmationTimeoutSecs": 1, "maxAttempts": 3},
                "modelSources": {"allowedSchemes": ["file"]},
                "models": [
                    {
                        "id": "synthetic-anomaly-scalar",
                        "version": "1.0.0",
                        "digest": corpus.digest,
                        "uri": str(corpus.tarball),
                        "activation": {"requireWarmup": True, "retainForRollback": True},
                    }
                ],
                "completionDefaults": {
                    "onSuccess": "archive",
                    "onInvalidInput": "quarantine",
                    "onOperationalFailure": "retainInPlace",
                    "onPublishFailure": "retainInPlace",
                    "onCollision": "fail",
                },
            },
            "instances": instances,
        }
    }


def write_capture(root: Path, relative: str, data: bytes, capture_id: str = "cap-0001") -> dict:
    """Write one camera-shaped capture, sidecar first, as camera-adapter does."""
    import hashlib

    target = Path(root) / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "schemaVersion": 1,
        "eventId": f"evt-{capture_id}",
        "captureId": capture_id,
        "cameraId": "cam-01",
        "correlationId": f"corr-{capture_id}",
        "timestamps": {"persistedAt": "2026-08-22T10:15:04.512Z"},
        "image": {
            "absolutePath": str(target),
            "relativePath": relative,
            "contentType": "image/png",
            "encoding": "png",
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "metadataSidecarRelativePath": relative + ".json",
        },
        "camera": {"backend": "sim"},
    }
    target.with_name(target.name + ".json").write_text(json.dumps(body), encoding="utf-8")
    target.write_bytes(data)
    return body


@pytest.fixture
def home(tmp_path) -> Path:
    """The directory tree one component instance owns."""
    root = tmp_path / "home"
    (root / "spool").mkdir(parents=True)
    (root / "inbox").mkdir(parents=True)
    return root


@pytest.fixture
def document(home, corpus) -> dict:
    """A configuration document for the wiring suites."""
    return config_document(home, corpus)


@pytest.fixture
def gg(document) -> FakeGg:
    """A fake EdgeCommons handle over that configuration."""
    return FakeGg(document)


@pytest.fixture
def app(gg):
    """A built, unstarted component. The test decides what to run."""
    from image_processor.ImageProcessor import ImageProcessor

    processor = ImageProcessor(gg)
    try:
        yield processor
    finally:
        processor.stop()


@pytest.fixture
def running(app, gg):
    """A component with its executor, its model, and its sources up.

    The filesystem observers are stopped again immediately: a test drives the authoritative walk
    itself, and a background walk racing it would decide which of the two announced the image.
    Everything the walk feeds -- admission, scheduling, publication, completion -- is the same
    either way, because the walk is the only thing that admits work.
    """
    app._metrics.define()
    app._supervisor.start()
    app._recover()
    app._artifacts.reconcile()
    app._start_routes()
    for runtime in app._routes.values():
        if runtime.route.is_spool and runtime.source is not None:
            runtime.source.stop()
    yield app
