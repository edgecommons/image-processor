"""Shared support for the WP3 completion suite: a fake filesystem, a route policy, and jobs."""

import errno
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from image_processor.completion import COLLISION_FAIL  # noqa: E402
from image_processor.ledger import Ledger  # noqa: E402
from image_processor.types import (  # noqa: E402
    CompletionAction,
    Job,
    JobState,
    ModelRef,
    SourceIdentity,
    SourceKind,
)

MODEL = ModelRef(id="line-clearance", version="2026.08.20", digest="sha256:" + "1" * 64)
SPOOL = "/spool/cam-01"
ARCHIVE = "/evidence/processed/cam-01"
FAILED = "/evidence/failed/cam-01"
RELATIVE = "2026/08/22/capture-1.jpg"


def key(path) -> str:
    """Normalize a path into the fake filesystem's key form."""
    return str(Path(path))


class FakeFs:
    """An in-memory :class:`~image_processor.completion.FsOps` with fault injection.

    Args:
        devices: Path prefixes that name separate filesystems. Two paths under different prefixes
            make :meth:`replace` raise ``OSError(EXDEV)``, which is how a cross-filesystem move is
            simulated. Anything outside every prefix is device zero.
    """

    def __init__(self, devices=()) -> None:
        self.files: dict = {}
        self.dirs: set = set()
        self.devices = [key(d) for d in devices]
        self.failures: dict = {}
        self.corrupt_copies: set = set()
        self.calls: list = []

    # -- test control ----------------------------------------------------------------------

    def write(self, path, data: bytes) -> str:
        """Seed a file and return its digest."""
        self.files[key(path)] = data
        self.dirs.add(key(Path(path).parent))
        return hashlib.sha256(data).hexdigest()

    def fail(self, op: str, path, exc: Exception) -> None:
        """Make the next ``op`` on ``path`` raise ``exc``."""
        self.failures[(op, key(path))] = exc

    def corrupt_copy(self, path) -> None:
        """Make a copy landing at ``path`` arrive with different bytes."""
        self.corrupt_copies.add(key(path))

    def _check(self, op: str, path) -> None:
        exc = self.failures.pop((op, key(path)), None)
        self.calls.append((op, key(path)))
        if exc is not None:
            raise exc

    def _device(self, path) -> int:
        text = key(path)
        for index, prefix in enumerate(self.devices, start=1):
            if text == prefix or text.startswith(prefix + "\\") or text.startswith(prefix + "/"):
                return index
        return 0

    # -- FsOps -----------------------------------------------------------------------------

    def exists(self, path) -> bool:
        """Report whether the path holds a file or a created directory."""
        return key(path) in self.files or key(path) in self.dirs

    def makedirs(self, path) -> None:
        """Record the directory and every parent."""
        self._check("makedirs", path)
        current = Path(path)
        while True:
            self.dirs.add(key(current))
            if current.parent == current:
                return
            current = current.parent

    def replace(self, src, dst) -> None:
        """Move a file, refusing a cross-device move the way ``os.replace`` does."""
        self._check("replace", src)
        if self._device(src) != self._device(dst):
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        self.files[key(dst)] = self.files.pop(key(src))

    def copy(self, src, dst) -> None:
        """Copy a file, optionally corrupting the copy."""
        self._check("copy", src)
        data = self.files[key(src)]
        self.files[key(dst)] = b"corrupted" if key(dst) in self.corrupt_copies else data

    def remove(self, path) -> None:
        """Delete a file."""
        self._check("remove", path)
        del self.files[key(path)]

    def sha256(self, path) -> str:
        """Return the digest of a file."""
        self._check("sha256", path)
        return hashlib.sha256(self.files[key(path)]).hexdigest()

    def size(self, path) -> int:
        """Return the size of a file."""
        return len(self.files[key(path)])

    def write_bytes(self, path, data: bytes) -> None:
        """Install generated content."""
        self._check("write_bytes", path)
        self.files[key(path)] = data

    def fsync_dir(self, path) -> None:
        """Record a directory flush."""
        self.calls.append(("fsync_dir", key(path)))


@dataclass(frozen=True)
class Policy:
    """A route's completion settings, shaped like WP1's ``config.models.CompletionPolicy``."""

    source_root: str = SPOOL
    archive_dir: Optional[str] = ARCHIVE
    failed_dir: Optional[str] = FAILED
    on_success: object = CompletionAction.ARCHIVE
    on_invalid_input: object = CompletionAction.QUARANTINE
    on_operational_failure: object = "retainInPlace"
    on_publish_failure: object = "retainInPlace"
    on_collision: str = COLLISION_FAIL


class StepClock:
    """A monotonic millisecond clock that advances one tick per read."""

    def __init__(self, start: int = 1_700_000_000_000) -> None:
        self.now = start

    def __call__(self) -> int:
        self.now += 1
        return self.now


def build_job(
    inference_id: str = "job-1",
    state: JobState = JobState.DISCOVERED,
    relative_path: str = RELATIVE,
    sha256: str = "a" * 64,
) -> Job:
    """Build a job value for a test."""
    return Job(
        inference_id=inference_id,
        route_id="cam-01",
        source=SourceIdentity(
            kind=SourceKind.SPOOL,
            route_id="cam-01",
            relative_path=relative_path,
            bytes=9,
            sha256=sha256,
            capture_id="capture-1",
            camera_id="cam-01",
        ),
        model=MODEL,
        transform_version="t1",
        state=state,
    )


PATHS = {
    JobState.PUBLISHED: [
        JobState.READY,
        JobState.CLAIMED,
        JobState.WAITING_MODEL,
        JobState.INFERENCING,
        JobState.RESULT_COMMITTED,
        JobState.PUBLISH_PENDING,
        JobState.PUBLISHED,
    ],
    JobState.INPUT_INVALID: [JobState.INPUT_INVALID],
    JobState.PROCESSING_EXHAUSTED: [
        JobState.READY,
        JobState.CLAIMED,
        JobState.WAITING_MODEL,
        JobState.INFERENCING,
        JobState.PROCESSING_EXHAUSTED,
    ],
    JobState.PUBLISH_EXHAUSTED: [
        JobState.READY,
        JobState.CLAIMED,
        JobState.WAITING_MODEL,
        JobState.INFERENCING,
        JobState.RESULT_COMMITTED,
        JobState.PUBLISH_PENDING,
        JobState.PUBLISH_EXHAUSTED,
    ],
}


def admitted(ledger: Ledger, state: JobState = JobState.PUBLISHED, **kwargs) -> Job:
    """Admit a job and drive it along legal edges to ``state``."""
    job = build_job(**kwargs)
    assert ledger.admit(job, 1024) is True
    for step in PATHS[state]:
        job = ledger.transition(job.inference_id, job.state, step)
    return job
