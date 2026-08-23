"""Spool discovery: the authoritative walk, the OS-notification nudge, and the camera hint.

Filesystem state is authoritative (DESIGN.md 4.1). ``rescan`` is the only thing that admits work:
it walks the configured root deterministically, applies the route's include and exclude patterns,
refuses anything that is not a plain regular file inside the root, asks the route's readiness rule
whether each candidate is finished, hashes the exact bytes, and re-stats to prove the bytes did not
move while it read them.

Everything else feeds that walk rather than replacing it. A ``watchdog`` observer coalesces OS
notifications into debounced nudges, and a periodic interval nudges it anyway, so a missed or
dropped notification costs latency and never a job. The camera's ``ImageCaptured`` announcement is
the same kind of nudge, except that it arrives carrying proof: it declares the image's size and
digest, so a hint that verifies against the file is admitted immediately without waiting for the
next walk.

Each ``(relative_path, sha256)`` pair is announced once. The set of pairs already announced lives
in memory and the application primes it from the ledger at startup, so a restart does not
rediscover finished work and the same file rewritten with new bytes is a new input.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional

from image_processor.types import SourceIdentity, SourceKind

from image_processor.sources.readiness import (
    Readiness,
    identity_from_capture,
    verify_declared_image,
)
from image_processor.sources.staging import (
    PathError,
    classify_path,
    config_field,
    normalize_relative,
    real_root,
    relative_to_root,
    resolve_under_root,
    sha256_file,
    stat_signature,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from image_processor.sources import SourceEvents

logger = logging.getLogger(__name__)

#: Include patterns a route inherits when it names none.
DEFAULT_INCLUDE = ("**/*",)

#: camera-adapter's hidden in-progress names (``camera-adapter/src/storage.rs``). They are never
#: inputs: the image only becomes visible under its final name once it is complete.
CAMERA_PARTIAL_PREFIX = ".camera-adapter-"
CAMERA_PARTIAL_SUFFIX = ".partial"

#: This component's own evidence sidecar. Reading it back would be an output feedback loop.
RESULT_SIDECAR_SUFFIX = ".inference.json"

#: How long the watcher waits for notifications to stop arriving before it walks.
DEFAULT_DEBOUNCE_SECS = 0.5

#: How often the watcher walks even when no notification arrives.
DEFAULT_RESCAN_INTERVAL_SECS = 60.0


class SpoolError(Exception):
    """The spool source cannot be built from this route configuration."""

    def __init__(self, code: str, message: str = "") -> None:
        self.code = code
        super().__init__(f"{code}: {message}" if message else code)


def compile_glob(pattern: str) -> "re.Pattern":
    """Compile one configuration glob into a regular expression over a relative path.

    The grammar is the one the configuration examples use: ``**`` crosses directory separators,
    ``*`` and ``?`` do not, and ``[...]`` is a character class. ``**/`` also matches zero
    directories, so ``**/*.jpg`` matches ``a.jpg`` as well as ``2026/08/a.jpg``.
    """
    index = 0
    out = ["(?s:"]
    length = len(pattern)
    while index < length:
        character = pattern[index]
        if character == "*":
            if pattern.startswith("**/", index):
                out.append("(?:.*/)?")
                index += 3
                continue
            if pattern.startswith("**", index):
                out.append(".*")
                index += 2
                continue
            out.append("[^/]*")
            index += 1
            continue
        if character == "?":
            out.append("[^/]")
            index += 1
            continue
        if character == "[":
            close = pattern.find("]", index + 1)
            if close == -1:
                out.append(r"\[")
                index += 1
                continue
            body = pattern[index + 1 : close]
            if body.startswith("!"):
                body = "^" + body[1:]
            out.append("[" + body.replace("\\", "\\\\") + "]")
            index = close + 1
            continue
        out.append(re.escape(character))
        index += 1
    out.append(r")\Z")
    return re.compile("".join(out))


def _default_observer_factory():  # pragma: no cover - real OS notification seam
    """Return a live ``watchdog`` observer. Tests inject a fake in its place."""
    from watchdog.observers import Observer

    return Observer()


class _NudgeHandler:
    """A ``watchdog`` event handler that does nothing but nudge the walk.

    It deliberately carries no per-event logic. An OS notification says only that something below
    the root changed; what changed is answered by walking, so the handler's whole job is to say
    "soon" without saying "what".
    """

    def __init__(self, nudge: Callable[[], None]) -> None:
        self._nudge = nudge

    def dispatch(self, event: Any) -> None:
        """Handle any filesystem event by scheduling a walk."""
        self._nudge()

    # ``watchdog`` calls ``dispatch``; these keep the handler usable as a plain event handler too.
    on_any_event = dispatch


class SpoolSource:
    """Discovers finished images under one route's owned spool root.

    Args:
        route: The route configuration. ``id`` names the route; ``source`` carries ``root``,
            ``include``, ``exclude``, ``readiness``, and ``camera``.
        events: The application's sink for discovered inputs and invalid ones.
        clock: Monotonic clock, shared with the ``stability`` readiness mode.
        status_lookup: Resolves a relative path to a verified capture record, for ``cameraStatus``.
        observer_factory: Builds the filesystem observer. Tests substitute a fake.
        debounce_secs: How long notifications must stop before a nudged walk runs.
        rescan_interval_secs: How often the watcher walks with no notification at all.

    Raises:
        SpoolError: The route names no id or no root.
        ReadinessError: The route's readiness mode cannot be built.
    """

    def __init__(
        self,
        route: Any,
        events: "SourceEvents",
        clock: Callable[[], float] = time.monotonic,
        *,
        status_lookup: Optional[Callable[[str], Any]] = None,
        observer_factory: Optional[Callable[[], Any]] = None,
        debounce_secs: float = DEFAULT_DEBOUNCE_SECS,
        rescan_interval_secs: float = DEFAULT_RESCAN_INTERVAL_SECS,
    ) -> None:
        try:
            self.route_id = str(config_field(route, "id", "route_id"))
            source = config_field(route, "source")
            root = config_field(source, "root")
        except Exception as exc:
            raise SpoolError("SPOOL_ROUTE_INCOMPLETE", str(exc)) from exc
        if not self.route_id:
            raise SpoolError("SPOOL_ROUTE_INCOMPLETE", "a route needs an id")
        self.root = real_root(Path(root))
        self.include = tuple(config_field(source, "include", default=DEFAULT_INCLUDE))
        self.exclude = tuple(config_field(source, "exclude", default=()))
        self._include = [compile_glob(pattern) for pattern in self.include]
        self._exclude = [compile_glob(pattern) for pattern in self.exclude]
        self.readiness = Readiness.for_route(route, status_lookup=status_lookup, clock=clock)
        self._events = events
        self._clock = clock
        self._observer_factory = observer_factory or _default_observer_factory
        self.debounce_secs = float(debounce_secs)
        self.rescan_interval_secs = float(rescan_interval_secs)
        self._seen: set = set()
        self._rejected: set = set()
        self._lock = threading.Lock()
        self._nudged = threading.Event()
        self._stop = threading.Event()
        self._observer: Any = None
        self._worker: Optional[threading.Thread] = None
        self.discovered_count = 0
        self.rejected_count = 0
        self.rescans = 0
        self.nudges = 0
        self.hints_accepted = 0
        self.hints_rejected = 0
        self.hints_unmapped = 0

    # -- lifecycle ---------------------------------------------------------------------------

    def start(self) -> None:
        """Start the filesystem observer and the debounced walk thread."""
        if self._worker is not None:
            return
        self._stop.clear()
        self._nudged.clear()
        try:
            observer = self._observer_factory()
            observer.schedule(_NudgeHandler(self.nudge), str(self.root), recursive=True)
            observer.start()
            self._observer = observer
        except Exception:
            logger.warning(
                "route %s could not watch %s; the periodic walk still covers it",
                self.route_id,
                self.root,
                exc_info=True,
            )
            self._observer = None
        self._worker = threading.Thread(
            target=self._loop, name=f"spool-{self.route_id}", daemon=True
        )
        self._worker.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        """Stop the observer and the walk thread."""
        self._stop.set()
        self._nudged.set()
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.join(timeout_s)
        observer, self._observer = self._observer, None
        if observer is not None:
            try:
                observer.stop()
                observer.join(timeout_s)
            except Exception:
                logger.debug("the filesystem observer did not stop cleanly", exc_info=True)

    def nudge(self) -> None:
        """Ask for a walk soon. Repeated nudges coalesce into one."""
        self.nudges += 1
        self._nudged.set()

    def _loop(self) -> None:
        """Walk when nudges settle, and walk anyway on the interval."""
        last_walk = self._clock()
        while not self._stop.is_set():
            nudged = self._nudged.wait(self.debounce_secs)
            if self._stop.is_set():
                return
            if nudged:
                self._nudged.clear()
                while self._nudged.wait(self.debounce_secs):
                    self._nudged.clear()
                    if self._stop.is_set():
                        return
            elif self._clock() - last_walk < self.rescan_interval_secs:
                continue
            self._safe_rescan()
            last_walk = self._clock()

    def _safe_rescan(self) -> None:
        """Walk, keeping the thread alive through a failing walk."""
        try:
            self.rescan()
        except Exception:
            logger.warning("route %s failed a spool walk", self.route_id, exc_info=True)

    # -- the seen set ------------------------------------------------------------------------

    def prime(self, pairs: Iterable) -> None:
        """Seed the announced set from the ledger so a restart rediscovers nothing.

        Args:
            pairs: ``(relative_path, sha256)`` pairs already durable in the ledger.
        """
        with self._lock:
            for relative_path, digest in pairs:
                self._seen.add((str(relative_path), str(digest)))

    # WP6 -- `reprocessExistingOnModelChange` (DESIGN.md §4.3) replays inputs that already
    # reached a terminal state when a route switches model generation. The walk is what admits
    # work, so forgetting what was announced is what makes the next walk see it again.
    def forget(self, pairs: Optional[Iterable] = None) -> int:
        """Forget announced inputs so the next walk rediscovers them.

        A rediscovered input under a new model digest is a new job; under the same digest the
        ledger recognizes the identity and refuses the duplicate, so this is never a way to
        process the same image twice.

        Args:
            pairs: The ``(relative_path, sha256)`` pairs to forget, or ``None`` for all of them.

        Returns:
            How many pairs were forgotten.
        """
        with self._lock:
            if pairs is None:
                count = len(self._seen)
                self._seen.clear()
            else:
                count = 0
                for relative_path, digest in pairs:
                    if (str(relative_path), str(digest)) in self._seen:
                        self._seen.discard((str(relative_path), str(digest)))
                        count += 1
            self._rejected.clear()
        return count

    def seen(self) -> set:
        """Return a copy of the announced ``(relative_path, sha256)`` pairs."""
        with self._lock:
            return set(self._seen)

    def _claim(self, relative_path: str, digest: str) -> bool:
        """Claim one input, returning False when it was already announced."""
        key = (relative_path, digest)
        with self._lock:
            if key in self._seen:
                return False
            self._seen.add(key)
            return True

    def _reject(self, relative_path: str, reason: str) -> None:
        """Report an unusable path once, not on every walk."""
        with self._lock:
            if relative_path in self._rejected:
                return
            self._rejected.add(relative_path)
        self.rejected_count += 1
        logger.warning("route %s refuses %s: %s", self.route_id, relative_path, reason)
        self._events.invalid(self.route_id, relative_path, reason)

    # -- matching ----------------------------------------------------------------------------

    def matches(self, relative_path: str) -> bool:
        """Report whether a relative path passes the route's include and exclude patterns."""
        if not any(pattern.match(relative_path) for pattern in self._include):
            return False
        return not any(pattern.match(relative_path) for pattern in self._exclude)

    def is_companion(self, path: Path) -> bool:
        """Report whether a file is metadata about an input rather than an input.

        Companions sit in the same directory as the image they describe: the camera's hidden
        in-progress names, its ``<image>.json`` sidecar, this component's own evidence sidecar, and
        a ``marker`` route's completion marker.
        """
        name = path.name
        if name.startswith(CAMERA_PARTIAL_PREFIX) and name.endswith(CAMERA_PARTIAL_SUFFIX):
            return True
        if name.endswith(RESULT_SIDECAR_SUFFIX):
            return True
        for suffix in self.readiness.companion_suffixes():
            if name.endswith(suffix) and path.with_name(name[: -len(suffix)]).exists():
                return True
        return False

    # -- the authoritative walk --------------------------------------------------------------

    def rescan(self) -> int:
        """Walk the root and announce every newly ready input.

        This is the authoritative path: the observer, the periodic interval, the camera hint, and
        the ``trigger-rescan`` command all end here, and nothing admits work any other way.

        Returns:
            The number of inputs announced by this walk.
        """
        self.rescans += 1
        if not self.root.is_dir():
            logger.debug("route %s has no spool root at %s yet", self.route_id, self.root)
            return 0
        announced = 0
        observed: set = set()
        for path, relative_path in self._walk(observed):
            if self._admit(path, relative_path):
                announced += 1
        self.readiness.prune(observed)
        with self._lock:
            self._rejected &= observed
        return announced

    def _walk(self, observed: set):
        """Yield every candidate ``(path, relative_path)`` under the root.

        Directories are visited in name order so two walks of the same tree see the same files in
        the same order, and a symlinked or reparse-point directory is refused rather than
        descended: following one is how a walk leaves the root it was told to stay inside.

        Args:
            observed: Filled with every relative path the walk examined, candidate or not, so the
                caller can forget the state of files that are gone.
        """
        stack = [self.root]
        while stack:
            directory = stack.pop()
            try:
                with os.scandir(directory) as entries:
                    listing = sorted(entries, key=lambda entry: entry.name)
            except OSError:
                logger.debug("route %s could not list %s", self.route_id, directory)
                continue
            for entry in listing:
                path = Path(entry.path)
                try:
                    relative_path = relative_to_root(self.root, path)
                except ValueError:
                    continue
                observed.add(relative_path)
                reason = classify_path(path)
                if reason == "DIRECTORY":
                    stack.append(path)
                    continue
                if reason in ("SYMLINK", "REPARSE_POINT"):
                    if self.matches(relative_path):
                        self._reject(relative_path, reason)
                    continue
                if reason == "MISSING":
                    continue
                if self.is_companion(path):
                    continue
                if not self.matches(relative_path):
                    continue
                if reason is not None:
                    self._reject(relative_path, reason)
                    continue
                yield path, relative_path

    def _admit(self, path: Path, relative_path: str) -> bool:
        """Judge, hash, and announce one candidate file.

        Returns:
            True when this call announced the file.
        """
        try:
            contained = resolve_under_root(self.root, relative_path)
        except PathError as exc:
            self._reject(relative_path, exc.code)
            return False
        signature_before = stat_signature(contained)
        if signature_before is None:
            return False
        verdict = self.readiness.ready(contained, relative_path)
        if not verdict.ready:
            logger.debug("route %s holds %s: %s", self.route_id, relative_path, verdict.reason)
            return False
        identity = verdict.identity
        if identity is None:
            try:
                digest = sha256_file(contained)
            except OSError:
                return False
            identity = SourceIdentity(
                kind=SourceKind.SPOOL,
                route_id=self.route_id,
                relative_path=relative_path,
                bytes=signature_before[0],
                sha256=digest,
            )
        if stat_signature(contained) != signature_before:
            logger.debug(
                "route %s saw %s change while it was read; the next walk takes it",
                self.route_id,
                relative_path,
            )
            return False
        if not self._claim(relative_path, identity.sha256):
            return False
        self.discovered_count += 1
        self._events.discovered(self.route_id, identity, None)
        return True

    # -- the camera hint ---------------------------------------------------------------------

    def on_hint(self, body: Any) -> None:
        """Admit one image announced by the camera, if it verifies.

        The hint is the ``ImageCaptured`` terminal body. It declares the image's root-relative
        path, its exact byte count, and its digest, which is proof enough on its own: a hint whose
        declared size and digest match the file is ready, with no sidecar read and no wait.

        ``image.absolutePath`` is never used. It is the camera's own view of its filesystem, and
        honoring it would let a message decide which file this component reads. The path is always
        ``image.relativePath`` resolved under this route's configured root, and a path that leaves
        the root is refused.
        """
        if not isinstance(body, Mapping):
            self.hints_rejected += 1
            return
        image = body.get("image")
        if not isinstance(image, Mapping):
            self.hints_unmapped += 1
            return
        declared_path = image.get("relativePath")
        try:
            relative_path = normalize_relative(declared_path)
            path = resolve_under_root(self.root, relative_path)
        except PathError as exc:
            self.hints_rejected += 1
            self._reject(str(declared_path), exc.code)
            return
        reason = classify_path(path)
        if reason == "MISSING":
            self.hints_unmapped += 1
            logger.debug("route %s has no file for hinted %s yet", self.route_id, relative_path)
            return
        if reason is not None:
            self.hints_rejected += 1
            self._reject(relative_path, reason)
            return
        signature_before = stat_signature(path)
        size, digest, failure = verify_declared_image(path, dict(image))
        if failure is not None:
            self.hints_rejected += 1
            logger.warning(
                "route %s refuses a hint for %s: %s", self.route_id, relative_path, failure
            )
            return
        if stat_signature(path) != signature_before:
            self.hints_rejected += 1
            return
        identity = identity_from_capture(
            self.route_id, relative_path, dict(body), int(size), str(digest)
        )
        self.hints_accepted += 1
        if not self._claim(relative_path, identity.sha256):
            return
        self.discovered_count += 1
        self._events.discovered(self.route_id, identity, None)
