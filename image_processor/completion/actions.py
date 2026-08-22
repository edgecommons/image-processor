"""Completion actions: archive, delete, retain, quarantine, with write-ahead intents.

DESIGN.md §7 fixes the protocol. Before any mutation the ledger stores a cleanup intent naming the
action, the deterministic target, the source digest, and the bundle members. The mutation then runs
as an atomic same-filesystem rename, or as a cross-filesystem copy that is verified by digest
before the source is removed. Recovery reads the intent, looks at what the filesystem actually
shows, and applies the observed-state rules in :meth:`Completer.reconcile`.

Cleanup failure is ``CLEANUP_FAILED``, retried by policy and repairable by command, and never
recorded as success.

All filesystem work goes through the :class:`FsOps` seam so the rules can be tested against
injected collisions, cross-filesystem moves, and failures without a real disk.
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path, PurePosixPath
from typing import Callable, Optional, Protocol, runtime_checkable

from image_processor.ledger import Ledger
from image_processor.types import (
    TERMINAL_STATES,
    CleanupIntent,
    CompletionAction,
    Job,
    JobState,
)

log = logging.getLogger(__name__)

#: Config spells ``retainInPlace``; the durable enum spells ``retain`` (DESIGN.md §11, LLD §2).
_ACTION_ALIASES = {
    "retaininplace": CompletionAction.RETAIN,
    "retain_in_place": CompletionAction.RETAIN,
}

#: Collision policies (DESIGN.md §11 ``completionDefaults.onCollision``).
COLLISION_FAIL = "fail"
COLLISION_SUFFIX = "suffix"

#: Suffix appended to a moved image to name its bundle manifest (DESIGN.md §7).
BUNDLE_MANIFEST_SUFFIX = ".bundle.json"

#: Suffix of the quarantine error record bundled with a rejected input.
ERROR_RECORD_SUFFIX = ".error.json"


class CleanupError(Exception):
    """A cleanup action could not be completed.

    Args:
        code: A SCREAMING_SNAKE code naming the failure.
        message: Detail for the operator.
    """

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code
        self.message = message


@runtime_checkable
class CompletionPolicy(Protocol):
    """The completion settings one route runs under (DESIGN.md §11 ``completion``).

    ``config.models.CompletionPolicy`` (WP1) satisfies this structurally; the protocol is what
    ``completion/`` depends on so the two packages stay decoupled.
    """

    source_root: str
    archive_dir: Optional[str]
    failed_dir: Optional[str]
    on_success: object
    on_invalid_input: object
    on_operational_failure: object
    on_publish_failure: object
    on_collision: str


class FsOps(Protocol):
    """The filesystem operations completion needs, as a seam tests can replace."""

    def exists(self, path: Path) -> bool: ...

    def makedirs(self, path: Path) -> None: ...

    def replace(self, src: Path, dst: Path) -> None: ...

    def copy(self, src: Path, dst: Path) -> None: ...

    def remove(self, path: Path) -> None: ...

    def sha256(self, path: Path) -> str: ...

    def size(self, path: Path) -> int: ...

    def write_bytes(self, path: Path, data: bytes) -> None: ...

    def fsync_dir(self, path: Path) -> None: ...


class RealFs:
    """:class:`FsOps` against the real filesystem."""

    def exists(self, path: Path) -> bool:
        """Report whether ``path`` exists."""
        return Path(path).exists()

    def makedirs(self, path: Path) -> None:
        """Create ``path`` and every missing parent."""
        Path(path).mkdir(parents=True, exist_ok=True)

    def replace(self, src: Path, dst: Path) -> None:
        """Atomically rename ``src`` onto ``dst``.

        Raises:
            OSError: With ``errno.EXDEV`` when the two paths are on different filesystems.
        """
        os.replace(str(src), str(dst))

    def copy(self, src: Path, dst: Path) -> None:
        """Copy ``src`` to ``dst``, preserving mtime and mode."""
        shutil.copy2(str(src), str(dst))

    def remove(self, path: Path) -> None:
        """Delete ``path``."""
        os.remove(str(path))

    def sha256(self, path: Path) -> str:
        """Return the SHA-256 of ``path`` as lowercase hex."""
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def size(self, path: Path) -> int:
        """Return the size of ``path`` in bytes."""
        return Path(path).stat().st_size

    def write_bytes(self, path: Path, data: bytes) -> None:
        """Write ``data`` to ``path`` through a temporary file and an atomic rename."""
        target = Path(path)
        tmp = target.with_name(target.name + ".tmp")
        with open(tmp, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(tmp), str(target))

    def fsync_dir(self, path: Path) -> None:
        """Flush a directory entry where the platform supports it.

        Windows has no directory handle to fsync, so this is a no-op there, matching
        camera-adapter and file-replicator.
        """
        if os.name == "nt":
            return
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _now_ms() -> int:
    """Return the current wall clock in milliseconds."""
    return int(time.time() * 1000)


def coerce_action(value) -> CompletionAction:
    """Normalize a configured completion action into the durable enum.

    Args:
        value: A :class:`~image_processor.types.CompletionAction`, or a config string such as
            ``"archive"``, ``"delete"``, ``"retainInPlace"``, or ``"quarantine"``.

    Returns:
        The matching :class:`~image_processor.types.CompletionAction`.

    Raises:
        CleanupError: The value names no known action.
    """
    if isinstance(value, CompletionAction):
        return value
    text = str(value)
    alias = _ACTION_ALIASES.get(text.replace("-", "_").lower())
    if alias is not None:
        return alias
    try:
        return CompletionAction(text.lower())
    except ValueError as exc:
        raise CleanupError("UNKNOWN_COMPLETION_ACTION", text) from exc


def safe_relative(relative_path: str) -> PurePosixPath:
    """Validate a route-relative path and return it in POSIX form.

    Args:
        relative_path: The path of an input relative to its route root.

    Returns:
        The path as a :class:`~pathlib.PurePosixPath`.

    Raises:
        CleanupError: The path is empty, absolute, drive-qualified, or escapes its root.
    """
    text = str(relative_path).replace("\\", "/").strip()
    if not text:
        raise CleanupError("UNSAFE_RELATIVE_PATH", "empty")
    candidate = PurePosixPath(text)
    if candidate.is_absolute() or text.startswith("/") or ":" in candidate.parts[0]:
        raise CleanupError("UNSAFE_RELATIVE_PATH", text)
    if any(part in ("..", "") for part in candidate.parts):
        raise CleanupError("UNSAFE_RELATIVE_PATH", text)
    return candidate


def suffixed(target: Path, source_sha256: str) -> Path:
    """Return the deterministic collision-suffixed sibling of ``target``.

    The suffix is the first eight hex characters of the source digest, so the same input always
    resolves to the same name and a retry after a crash lands on the file it already wrote.

    Args:
        target: The colliding target path.
        source_sha256: The source digest, with or without a ``sha256:`` prefix.

    Returns:
        The suffixed path.
    """
    short = source_sha256.split(":")[-1][:8]
    return target.with_name(f"{target.stem}.{short}{target.suffix}")


class _Installed:
    """The outcome of installing one file at its target.

    Attributes:
        path: Where the file now is.
        sha256: Its digest.
        how: ``"renamed"``, ``"copied"``, ``"resumed"`` (a cross-filesystem copy finished by
            removing the surviving source), or ``"already"`` (the target was already correct).
    """

    __slots__ = ("path", "sha256", "how")

    def __init__(self, path: Path, sha256: str, how: str) -> None:
        self.path = path
        self.sha256 = sha256
        self.how = how


class Completer:
    """Runs archive, delete, retain, and quarantine under write-ahead intents (DESIGN.md §7).

    Args:
        ledger: The durable ledger. Intents are committed through it before any mutation.
        fs: The filesystem seam. Defaults to :class:`RealFs`.
        on_collision: The component-wide collision policy, ``"fail"`` (the default) or
            ``"suffix"``. :meth:`apply` and :meth:`reconcile` take a per-route override.
        clock: Returns the current wall clock in milliseconds. Injected by tests.
    """

    def __init__(
        self,
        ledger: Ledger,
        fs: Optional[FsOps] = None,
        on_collision: str = COLLISION_FAIL,
        clock: Callable[[], int] = _now_ms,
    ) -> None:
        self._ledger = ledger
        self._fs: FsOps = fs if fs is not None else RealFs()
        self._on_collision = on_collision
        self._clock = clock

    # -- planning --------------------------------------------------------------------------

    def plan(self, job: Job, policy: CompletionPolicy, members: list) -> CleanupIntent:
        """Compute the write-ahead intent for a job's completion.

        The action follows the job's state: a published job takes ``onSuccess``, a rejected input
        ``onInvalidInput``, an exhausted job ``onOperationalFailure``, and a job whose publication
        gave up ``onPublishFailure``. The target preserves the input's relative subtree under
        ``archiveDir`` (archive) or ``failedDir`` (quarantine), so the same input always resolves
        to the same path.

        Args:
            job: The job to complete.
            policy: The route's completion settings.
            members: Companion files that move with the image, such as the camera sidecar and the
                evidence sidecar.

        Returns:
            The :class:`~image_processor.types.CleanupIntent` to persist and then execute.

        Raises:
            CleanupError: The job's state has no configured completion, the action names no known
                completion, the required directory is not configured, or the relative path is
                unsafe.
        """
        action = coerce_action(_policy_action(job.state, policy))
        relative = safe_relative(job.source.relative_path)
        source_path = Path(policy.source_root) / relative
        target_path: Optional[str] = None
        if action is CompletionAction.ARCHIVE:
            target_path = str(Path(_required_dir(policy, "archive_dir", action)) / relative)
        elif action is CompletionAction.QUARANTINE:
            target_path = str(Path(_required_dir(policy, "failed_dir", action)) / relative)
        return CleanupIntent(
            inference_id=job.inference_id,
            action=action,
            source_path=str(source_path),
            source_sha256=job.source.sha256,
            target_path=target_path,
            members=tuple(str(Path(m)) for m in members),
        )

    # -- execution -------------------------------------------------------------------------

    def apply(self, intent: CleanupIntent, on_collision: Optional[str] = None) -> None:
        """Persist the intent, then perform the mutation it describes.

        The intent is committed before the first byte moves, so a crash at any point leaves a
        durable record recovery can evaluate. A same-filesystem move is an atomic rename; a
        cross-filesystem move copies, verifies the copy by digest, and only then removes the
        source. A multi-file move installs its bundle manifest last. Any failure records
        ``fail_cleanup`` and raises; nothing is ever recorded as success.

        Args:
            intent: The planned mutation.
            on_collision: Override the component-wide collision policy for this route.

        Raises:
            CleanupError: The mutation failed. The job is left non-complete and retryable.
        """
        self._ledger.record_cleanup_intent(intent)
        try:
            observed = self._execute(intent, on_collision or self._on_collision)
        except CleanupError as exc:
            self._ledger.fail_cleanup(intent.inference_id, f"{exc.code}: {exc.message}")
            raise
        except OSError as exc:
            self._ledger.fail_cleanup(intent.inference_id, f"FS_ERROR: {exc}")
            raise CleanupError("FS_ERROR", str(exc)) from exc
        self._ledger.complete_cleanup(intent.inference_id, observed)

    def reconcile(self, intent: CleanupIntent, on_collision: Optional[str] = None) -> JobState:
        """Decide a stored intent against observed filesystem state (DESIGN.md §7).

        Recovery re-records the intent (which is what returns a ``CLEANUP_FAILED`` job to
        ``CLEANUP_PENDING``) and then evaluates what the filesystem shows:

        * source present and target absent retries the move;
        * source absent and target present with a matching digest completes;
        * both present after a cross-filesystem copy verifies the target and removes the source;
        * a target holding a different digest is a collision failure;
        * a source absent under a delete intent completes, because the intent names the object
          that was to be removed;
        * a retain intent whose source is gone, or an archive intent with neither source nor
          target, is a failure — the evidence is lost, and that is never success.

        A job that already reached a terminal state is reported as it stands and its files are not
        touched again, so an operator ``reconcile`` command is safe to run over anything.

        Args:
            intent: The stored intent, typically from
                :meth:`~image_processor.ledger.ledger.Ledger.pending_cleanup`.
            on_collision: Override the component-wide collision policy for this route.

        Returns:
            The job's :class:`~image_processor.types.JobState` after reconciliation.
        """
        settled = self._ledger.get(intent.inference_id)
        if settled is not None and settled.state in TERMINAL_STATES:
            return settled.state
        job = self._ledger.record_cleanup_intent(intent)
        try:
            observed = self._execute(intent, on_collision or self._on_collision)
        except CleanupError as exc:
            return self._ledger.fail_cleanup(
                intent.inference_id, f"{exc.code}: {exc.message}"
            ).state
        except OSError as exc:
            return self._ledger.fail_cleanup(intent.inference_id, f"FS_ERROR: {exc}").state
        log.info("cleanup reconciled %s from %s: %s", intent.inference_id, job.state.value, observed)
        return self._ledger.complete_cleanup(intent.inference_id, observed).state

    # -- internals -------------------------------------------------------------------------

    def _execute(self, intent: CleanupIntent, on_collision: str) -> str:
        """Perform the intent's mutation and return what was observed.

        Args:
            intent: The persisted intent.
            on_collision: The collision policy in force.

        Returns:
            A short observed-state string retained against the intent.

        Raises:
            CleanupError: The mutation could not be completed.
        """
        source = Path(intent.source_path)
        if intent.action is CompletionAction.RETAIN:
            if not self._fs.exists(source):
                raise CleanupError("SOURCE_MISSING", str(source))
            self._verify_source(source, intent.source_sha256)
            return "retained"
        if intent.action is CompletionAction.DELETE:
            if not self._fs.exists(source):
                return "already-deleted"
            self._verify_source(source, intent.source_sha256)
            for member in intent.members:
                if self._fs.exists(Path(member)):
                    self._fs.remove(Path(member))
            self._fs.remove(source)
            return "deleted"
        return self._move_bundle(intent, on_collision)

    def _verify_source(self, source: Path, expected: str) -> None:
        """Confirm the file at ``source`` is still the object the intent named.

        Args:
            source: The file to check.
            expected: The digest recorded in the intent.

        Raises:
            CleanupError: The file holds a different object.
        """
        expected_hex = expected.split(":")[-1]
        actual = self._fs.sha256(source)
        if actual != expected_hex:
            raise CleanupError("SOURCE_REPLACED", f"{source} is no longer {expected_hex}")

    def _move_bundle(self, intent: CleanupIntent, on_collision: str) -> str:
        """Move the image and its members to their targets, manifest last.

        Args:
            intent: The persisted intent; its ``target_path`` must be set.
            on_collision: The collision policy in force.

        Returns:
            ``"archived"`` or ``"quarantined"``.

        Raises:
            CleanupError: A target is missing, a collision is unresolvable, or the evidence is
                gone from both the source and the target.
        """
        if not intent.target_path:
            raise CleanupError("NO_TARGET", intent.action.value)
        source = Path(intent.source_path)
        target = Path(intent.target_path)
        if not self._fs.exists(source) and not self._fs.exists(target):
            raise CleanupError("EVIDENCE_LOST", f"neither {source} nor {target} exists")
        self._fs.makedirs(target.parent)
        installed = [self._install(source, target, intent.source_sha256, on_collision)]
        image_path = installed[0].path
        for member in intent.members:
            member_target = _member_target(source, image_path, Path(member))
            self._fs.makedirs(member_target.parent)
            installed.append(self._install(Path(member), member_target, None, on_collision))
        if intent.action is CompletionAction.QUARANTINE:
            installed.append(self._write_error_record(intent, image_path))
        if len(installed) > 1:
            self._write_bundle_manifest(intent, image_path, installed)
        return "quarantined" if intent.action is CompletionAction.QUARANTINE else "archived"

    def _install(
        self, src: Path, dst: Path, expected: Optional[str], on_collision: str
    ) -> _Installed:
        """Install one file at ``dst``, applying the DESIGN.md §7 observed-state rules.

        Args:
            src: The file to move.
            dst: Its deterministic target.
            expected: The digest the intent recorded, or ``None`` for a companion member.
            on_collision: The collision policy in force.

        Returns:
            The :class:`_Installed` outcome.

        Raises:
            CleanupError: The source is gone, its digest changed, the copy did not verify, or the
                target holds a different object.
        """
        expected_hex = expected.split(":")[-1] if expected else None
        if self._fs.exists(dst):
            return self._resolve_existing_target(src, dst, expected_hex, on_collision)
        if not self._fs.exists(src):
            raise CleanupError("SOURCE_MISSING", str(src))
        digest = self._fs.sha256(src)
        if expected_hex is not None and digest != expected_hex:
            raise CleanupError("SOURCE_DIGEST_MISMATCH", f"{src} is not {expected_hex}")
        try:
            self._fs.replace(src, dst)
            how = "renamed"
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            self._fs.copy(src, dst)
            if self._fs.sha256(dst) != digest:
                self._fs.remove(dst)
                raise CleanupError("COPY_VERIFY_FAILED", str(dst)) from exc
            self._fs.remove(src)
            how = "copied"
        self._fs.fsync_dir(dst.parent)
        return _Installed(dst, digest, how)

    def _resolve_existing_target(
        self, src: Path, dst: Path, expected_hex: Optional[str], on_collision: str
    ) -> _Installed:
        """Decide what an already-occupied target means.

        Args:
            src: The file the intent wanted to move.
            dst: The occupied target.
            expected_hex: The digest the intent recorded, or ``None`` for a member.
            on_collision: The collision policy in force.

        Returns:
            The :class:`_Installed` outcome when the target is the right object.

        Raises:
            CleanupError: The target holds a different object and no suffix is configured.
        """
        target_digest = self._fs.sha256(dst)
        source_exists = self._fs.exists(src)
        source_digest = self._fs.sha256(src) if source_exists else expected_hex
        if source_exists and target_digest == source_digest:
            self._fs.remove(src)
            return _Installed(dst, target_digest, "resumed")
        if not source_exists and (expected_hex is None or target_digest == expected_hex):
            return _Installed(dst, target_digest, "already")
        if on_collision == COLLISION_SUFFIX:
            alternate = suffixed(dst, source_digest or target_digest)
            if alternate != dst:
                return self._install(src, alternate, expected_hex, COLLISION_FAIL)
        raise CleanupError("COLLISION", f"{dst} holds a different object")

    def _write_error_record(self, intent: CleanupIntent, image_path: Path) -> _Installed:
        """Write the quarantine error record beside the quarantined image.

        Args:
            intent: The persisted intent.
            image_path: Where the image now is.

        Returns:
            The :class:`_Installed` outcome for the record.
        """
        job = self._ledger.get(intent.inference_id)
        record = {
            "schemaVersion": "1.0",
            "inferenceId": intent.inference_id,
            "action": intent.action.value,
            "recordedAtMs": self._clock(),
            "source": {
                "path": intent.source_path,
                "sha256": intent.source_sha256,
                "relativePath": job.source.relative_path if job else None,
            },
            "job": {
                "routeId": job.route_id if job else None,
                "state": job.state.value if job else None,
                "attempts": job.attempts if job else None,
                "model": {
                    "id": job.model.id,
                    "version": job.model.version,
                    "digest": job.model.digest,
                }
                if job
                else None,
            },
            "error": self._ledger.last_error(intent.inference_id),
        }
        path = image_path.with_name(image_path.name + ERROR_RECORD_SUFFIX)
        data = json.dumps(record, indent=2, sort_keys=True).encode("utf-8")
        self._fs.write_bytes(path, data)
        return _Installed(path, hashlib.sha256(data).hexdigest(), "written")

    def _write_bundle_manifest(
        self, intent: CleanupIntent, image_path: Path, installed: list
    ) -> None:
        """Install the bundle manifest for a multi-file move, last (DESIGN.md §7).

        Its presence is what tells a later reader that every member of the bundle arrived.

        Args:
            intent: The persisted intent.
            image_path: Where the image now is.
            installed: Every :class:`_Installed` member of the bundle.
        """
        manifest = {
            "schemaVersion": "1.0",
            "inferenceId": intent.inference_id,
            "action": intent.action.value,
            "installedAtMs": self._clock(),
            "image": image_path.name,
            "members": [
                {"path": _relative_name(image_path, item.path), "sha256": item.sha256}
                for item in installed
            ],
        }
        path = image_path.with_name(image_path.name + BUNDLE_MANIFEST_SUFFIX)
        self._fs.write_bytes(path, json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"))
        self._fs.fsync_dir(path.parent)


def _policy_action(state: JobState, policy: CompletionPolicy):
    """Return the configured completion for a job in ``state``.

    Args:
        state: The job's state.
        policy: The route's completion settings.

    Returns:
        The configured action, still in whatever form the policy holds it.

    Raises:
        CleanupError: The state has no configured completion.
    """
    if state in (JobState.PUBLISHED, JobState.CLEANUP_PENDING, JobState.CLEANUP_FAILED):
        return policy.on_success
    if state is JobState.INPUT_INVALID:
        return policy.on_invalid_input
    if state is JobState.PROCESSING_EXHAUSTED:
        return policy.on_operational_failure
    if state is JobState.PUBLISH_EXHAUSTED:
        return policy.on_publish_failure
    raise CleanupError("NO_COMPLETION_FOR_STATE", state.value)


def _required_dir(policy: CompletionPolicy, attribute: str, action: CompletionAction) -> str:
    """Return a configured completion directory, refusing an unset one.

    Args:
        policy: The route's completion settings.
        attribute: ``"archive_dir"`` or ``"failed_dir"``.
        action: The action that needs it, for the error message.

    Returns:
        The directory path.

    Raises:
        CleanupError: The directory is not configured.
    """
    value = getattr(policy, attribute, None)
    if not value:
        raise CleanupError("COMPLETION_DIR_MISSING", f"{action.value} needs {attribute}")
    return str(value)


def _member_target(source: Path, image_target: Path, member: Path) -> Path:
    """Return the deterministic target of one companion member.

    Members sit beside the image, so the member keeps its position relative to the image's own
    directory. A member from elsewhere lands beside the image under its own name.

    Args:
        source: The image's source path.
        image_target: Where the image was installed.
        member: The companion's source path.

    Returns:
        The member's target path.
    """
    try:
        relative = member.relative_to(source.parent)
    except ValueError:
        return image_target.parent / member.name
    return image_target.parent / relative


def _relative_name(image_path: Path, member_path: Path) -> str:
    """Render a bundle member's path relative to the image's directory, in POSIX form.

    Every member of a bundle is installed under the image's directory by
    :func:`_member_target`, so the relative form always exists.
    """
    return member_path.relative_to(image_path.parent).as_posix()
