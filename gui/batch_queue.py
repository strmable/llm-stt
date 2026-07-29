"""STT batch queue state machine (design.md SS7.1, v3.4).

Qt-free on purpose -- MainWindow owns all QThread/dialog/widget wiring
around this; this module only tracks queue contents and the small set of
decisions SS7.1 documents, so those rules are unit-testable without
QApplication:

  - idle + a single dropped file behaves exactly like pre-v3.4 (just the
    plain "selected file", no visible queue) UNLESS a queue is already in
    progress, in which case it's appended like everything else.
  - idle + multiple dropped files, or *any* drop while a job is running,
    always appends to the queue tail.
  - [Stop] halts auto-advance to the next queued item without clearing the
    remaining queue -- the user has to press [Transcript] again to resume.
  - [완전 취소]/remove only ever touches a still-"waiting" item.

What SS7.1 does NOT put in this class: the "first manually-started job shows
the Resume 3-button dialog, but an auto-advanced item resumes silently"
rule. That's purely about *how* MainWindow starts a job (dialog vs no
dialog), not queue state, so it lives in MainWindow's two call paths
instead.
"""

from pathlib import Path


class BatchQueueItem:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.status = "waiting"  # "waiting" | "running" | "done" | "failed"

    def __repr__(self):
        return f"BatchQueueItem({self.path.name!r}, status={self.status!r})"


class BatchQueue:
    def __init__(self):
        self.items: list[BatchQueueItem] = []
        self.stopped = False  # [Stop] halts auto-advance (design.md SS7.1)

    def add(self, paths) -> list[BatchQueueItem]:
        added = [BatchQueueItem(p) for p in paths]
        self.items.extend(added)
        return added

    def handle_drop(self, paths, running: bool) -> str:
        """Returns "single" if the caller should treat this as the plain
        pre-v3.4 single-file selection (idle, exactly one path, and no
        queue already in progress) -- nothing is added to the queue in that
        case. Returns "queued" otherwise, having already appended `paths`
        to the queue tail."""
        if not running and len(paths) == 1 and not self.items:
            return "single"
        self.add(paths)
        return "queued"

    def remove(self, item: BatchQueueItem):
        """No-op for a non-"waiting" item -- a running/finished job can't be
        pulled out from under itself (design.md SS7.1 개별 [제거])."""
        if item.status == "waiting":
            self.items.remove(item)

    def next_waiting(self) -> BatchQueueItem | None:
        return next((it for it in self.items if it.status == "waiting"), None)

    def start(self, item: BatchQueueItem):
        item.status = "running"

    def finish(self, item: BatchQueueItem, success: bool = True):
        item.status = "done" if success else "failed"

    def stop(self):
        self.stopped = True

    def clear_stopped(self):
        self.stopped = False

    def should_auto_advance(self) -> bool:
        return not self.stopped and self.next_waiting() is not None
