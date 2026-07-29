"""Unit tests for pipeline/build_srt.py's parse_srt(preserve_newlines=...)
(design.md SS5C.3 needs multi-line cues intact for the marker protocol)."""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import build_srt  # noqa: E402

SAMPLE_SRT = (
    "1\n00:00:00,000 --> 00:00:02,000\n안녕하세요, 오늘은\n날씨가 참 좋네요.\n"
    "\n"
    "2\n00:00:02,500 --> 00:00:04,000\n다음 안건으로 넘어가겠습니다.\n"
)


class ParseSrtDefaultBehaviorTests(unittest.TestCase):
    def test_default_collapses_multiline_cue_to_single_line(self):
        cues = build_srt.parse_srt(SAMPLE_SRT)
        self.assertEqual(cues[0]["text"], "안녕하세요, 오늘은 날씨가 참 좋네요.")


class ParseSrtPreserveNewlinesTests(unittest.TestCase):
    def test_preserves_internal_newline(self):
        cues = build_srt.parse_srt(SAMPLE_SRT, preserve_newlines=True)
        self.assertEqual(cues[0]["text"], "안녕하세요, 오늘은\n날씨가 참 좋네요.")
        self.assertEqual(cues[1]["text"], "다음 안건으로 넘어가겠습니다.")

    def test_timing_unaffected(self):
        cues = build_srt.parse_srt(SAMPLE_SRT, preserve_newlines=True)
        self.assertEqual(cues[0]["start_sec"], 0.0)
        self.assertEqual(cues[0]["end_sec"], 2.0)


if __name__ == "__main__":
    unittest.main()
