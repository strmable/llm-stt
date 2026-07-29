"""Unit tests for pipeline/translation.py (design.md SS5C marker protocol,
batching, and merge logic)."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import translation  # noqa: E402


def cue(text, start=0.0, end=1.0):
    return {"start_sec": start, "end_sec": end, "text": text}


class MarkerRoundTripTests(unittest.TestCase):
    def test_build_then_parse_recovers_single_line_cues(self):
        cues = [cue("첫 번째 문장"), cue("두 번째 문장"), cue("세 번째 문장")]
        text = translation.build_markers(cues)
        parsed = translation.parse_markers(text)
        self.assertEqual(parsed, {1: "첫 번째 문장", 2: "두 번째 문장", 3: "세 번째 문장"})

    def test_build_then_parse_recovers_multiline_cue(self):
        cues = [cue("안녕하세요, 오늘은\n날씨가 참 좋네요."), cue("다음 안건으로 넘어가겠습니다.")]
        text = translation.build_markers(cues)
        self.assertEqual(
            text,
            "[0001]|안녕하세요, 오늘은\n날씨가 참 좋네요.\n[0002]|다음 안건으로 넘어가겠습니다.",
        )
        parsed = translation.parse_markers(text)
        self.assertEqual(parsed[1], "안녕하세요, 오늘은\n날씨가 참 좋네요.")
        self.assertEqual(parsed[2], "다음 안건으로 넘어가겠습니다.")


class RenderTranslatedMapTests(unittest.TestCase):
    def test_renders_in_serial_order_regardless_of_dict_insertion_order(self):
        text = translation.render_translated_map({2: "two", 1: "one"})
        self.assertEqual(text, "[0001]|one\n[0002]|two")

    def test_round_trips_through_parse_markers(self):
        rendered = translation.render_translated_map({1: "a", 3: "c"})
        self.assertEqual(translation.parse_markers(rendered), {1: "a", 3: "c"})


class RobustParsingTests(unittest.TestCase):
    def test_tolerates_spaces_inserted_inside_marker_digits(self):
        text = "[00 07]|hello world\n[0008]|goodbye"
        parsed = translation.parse_markers(text)
        self.assertEqual(parsed, {7: "hello world", 8: "goodbye"})

    def test_tolerates_missing_blank_line_between_blocks(self):
        # blocks glued together with no separator at all between them
        text = "[0001]|first cue[0002]|second cue[0003]|third cue"
        parsed = translation.parse_markers(text)
        self.assertEqual(parsed, {1: "first cue", 2: "second cue", 3: "third cue"})

    def test_internal_newlines_preserved_but_edges_trimmed(self):
        text = "[0001]|  line one\nline two  \n[0002]|next"
        parsed = translation.parse_markers(text)
        self.assertEqual(parsed[1], "line one\nline two")


class MakeBatchesTests(unittest.TestCase):
    def test_never_splits_a_cue_and_respects_budget(self):
        cues = [cue("a" * 10) for _ in range(5)]  # each renders to ~15 chars incl marker
        batches = translation.make_batches(cues, max_chars=32)
        # every cue must appear exactly once, across all batches, in order
        flat = [serial for batch in batches for serial, _ in batch]
        self.assertEqual(flat, [1, 2, 3, 4, 5])
        for batch in batches:
            rendered_len = len(translation.render_markers(batch))
            self.assertLessEqual(rendered_len, 32)

    def test_single_over_budget_cue_gets_its_own_batch(self):
        cues = [cue("short"), cue("x" * 500), cue("short2")]
        batches = translation.make_batches(cues, max_chars=50)
        # the oversized cue (serial 2) must be alone in its batch
        oversized_batch = next(b for b in batches if any(s == 2 for s, _ in b))
        self.assertEqual(len(oversized_batch), 1)

    def test_single_cue_below_budget_not_split(self):
        cues = [cue("hello")]
        batches = translation.make_batches(cues, max_chars=1000)
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0]), 1)


class MergeTranslationTests(unittest.TestCase):
    def test_replaces_matched_cues(self):
        cues = [cue("one"), cue("two"), cue("three")]
        merged, missing = translation.merge_translation(cues, {1: "ONE", 2: "TWO", 3: "THREE"})
        self.assertEqual([c["text"] for c in merged], ["ONE", "TWO", "THREE"])
        self.assertEqual(missing, [])

    def test_missing_serials_leave_original_text_and_are_reported(self):
        cues = [cue("one"), cue("two"), cue("three")]
        merged, missing = translation.merge_translation(cues, {1: "ONE", 3: "THREE"})
        self.assertEqual([c["text"] for c in merged], ["ONE", "two", "THREE"])
        self.assertEqual(missing, [2])

    def test_timing_fields_preserved(self):
        cues = [cue("one", start=1.5, end=2.5)]
        merged, _ = translation.merge_translation(cues, {1: "ONE"})
        self.assertEqual(merged[0]["start_sec"], 1.5)
        self.assertEqual(merged[0]["end_sec"], 2.5)


class BuildTranslationPromptTests(unittest.TestCase):
    def test_substitutes_target_language_and_vocabulary(self):
        template = "Translate to {{target_language}}.\n{{vocabulary}}Go."
        result = translation.build_translation_prompt(template, "en", ["Anthropic", "Claude"])
        self.assertIn("Translate to en.", result)
        self.assertIn("- Anthropic", result)
        self.assertIn("- Claude", result)

    def test_empty_vocabulary_yields_empty_block(self):
        template = "{{vocabulary}}Go."
        result = translation.build_translation_prompt(template, "en", [])
        self.assertEqual(result, "Go.")


class RunBatchesTests(unittest.TestCase):
    def test_orchestrates_batches_and_merges_responses(self):
        cues = [cue("a"), cue("b")]

        def fake_post(url, json, timeout):
            resp = MagicMock()
            resp.raise_for_status.return_value = None
            # echo back translated markers for whatever was sent
            sent = json["messages"][1]["content"]
            translated = sent.replace("a", "A").replace("b", "B")
            resp.json.return_value = {"choices": [{"message": {"content": translated}}]}
            return resp

        with patch("translation.requests.post", side_effect=fake_post):
            result = translation.run_batches(
                cues, prompt_template="translate", max_chars=1000, url="http://fake/v1/chat/completions",
            )
        self.assertEqual(result, {1: "A", 2: "B"})

    def test_should_stop_halts_before_remaining_batches(self):
        cues = [cue("a" * 5), cue("b" * 5)]
        calls = []

        def fake_post(url, json, timeout):
            calls.append(json)
            resp = MagicMock()
            resp.raise_for_status.return_value = None
            resp.json.return_value = {"choices": [{"message": {"content": "[0001]|A"}}]}
            return resp

        with patch("translation.requests.post", side_effect=fake_post):
            translation.run_batches(
                cues, prompt_template="translate", max_chars=5, url="http://fake",
                should_stop=lambda: len(calls) >= 1,
            )
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
