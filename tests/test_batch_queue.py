"""Unit tests for gui/batch_queue.py (design.md SS7.1 STT batch queue)."""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gui.batch_queue import BatchQueue  # noqa: E402


class HandleDropTests(unittest.TestCase):
    def test_idle_single_file_with_empty_queue_is_not_enqueued(self):
        q = BatchQueue()
        result = q.handle_drop([Path("a.mp4")], running=False)
        self.assertEqual(result, "single")
        self.assertEqual(q.items, [])

    def test_idle_multiple_files_are_enqueued_without_auto_start(self):
        q = BatchQueue()
        result = q.handle_drop([Path("a.mp4"), Path("b.mp4")], running=False)
        self.assertEqual(result, "queued")
        self.assertEqual([it.path.name for it in q.items], ["a.mp4", "b.mp4"])
        self.assertTrue(all(it.status == "waiting" for it in q.items))

    def test_drop_while_running_is_always_enqueued_even_if_single(self):
        q = BatchQueue()
        result = q.handle_drop([Path("a.mp4")], running=True)
        self.assertEqual(result, "queued")
        self.assertEqual(len(q.items), 1)

    def test_idle_single_file_with_queue_already_in_progress_is_enqueued(self):
        q = BatchQueue()
        q.add([Path("existing.mp4")])
        result = q.handle_drop([Path("a.mp4")], running=False)
        self.assertEqual(result, "queued")
        self.assertEqual(len(q.items), 2)


class RemoveTests(unittest.TestCase):
    def test_remove_waiting_item(self):
        q = BatchQueue()
        added = q.add([Path("a.mp4"), Path("b.mp4"), Path("c.mp4")])
        q.remove(added[1])
        self.assertEqual([it.path.name for it in q.items], ["a.mp4", "c.mp4"])

    def test_remove_is_noop_for_running_item(self):
        q = BatchQueue()
        added = q.add([Path("a.mp4")])
        q.start(added[0])
        q.remove(added[0])
        self.assertEqual(len(q.items), 1)

    def test_remove_is_noop_for_done_item(self):
        q = BatchQueue()
        added = q.add([Path("a.mp4")])
        q.start(added[0])
        q.finish(added[0])
        q.remove(added[0])
        self.assertEqual(len(q.items), 1)


class AdvanceTests(unittest.TestCase):
    def test_next_waiting_skips_removed_and_finished_items(self):
        q = BatchQueue()
        a, b, c = q.add([Path("a.mp4"), Path("b.mp4"), Path("c.mp4")])
        q.start(a)
        q.finish(a)
        q.remove(b)
        self.assertIs(q.next_waiting(), c)

    def test_should_auto_advance_true_while_items_waiting_and_not_stopped(self):
        q = BatchQueue()
        q.add([Path("a.mp4")])
        self.assertTrue(q.should_auto_advance())

    def test_should_auto_advance_false_when_queue_empty(self):
        q = BatchQueue()
        self.assertFalse(q.should_auto_advance())

    def test_stop_halts_auto_advance_without_clearing_queue(self):
        q = BatchQueue()
        q.add([Path("a.mp4"), Path("b.mp4")])
        q.stop()
        self.assertFalse(q.should_auto_advance())
        self.assertEqual(len(q.items), 2)  # nothing removed by stop()

    def test_clear_stopped_re_enables_auto_advance(self):
        q = BatchQueue()
        q.add([Path("a.mp4")])
        q.stop()
        q.clear_stopped()
        self.assertTrue(q.should_auto_advance())

    def test_start_and_finish_transition_status(self):
        q = BatchQueue()
        (a,) = q.add([Path("a.mp4")])
        q.start(a)
        self.assertEqual(a.status, "running")
        q.finish(a, success=False)
        self.assertEqual(a.status, "failed")


if __name__ == "__main__":
    unittest.main()
