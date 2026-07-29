"""Unit tests for the design.md SS9 (v3.4) config-file split: legacy
config.json -> config-stt.json/config-translate.json/config-postprocessing.json.

Runs entirely against temp-directory paths -- never touches this checkout's
real config.json/config-*.json.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import common  # noqa: E402


class SplitLegacyConfigTests(unittest.TestCase):
    def test_srt_postprocess_moves_to_postprocessing_domain(self):
        legacy = {
            "provider": "local_api",
            "language": "ko",
            "srt_postprocess": {"cps_threshold": 15},
        }
        split = common.split_legacy_config(legacy)
        self.assertEqual(split["postprocessing"], {"srt_postprocess": {"cps_threshold": 15}})
        self.assertEqual(split["stt"], {"provider": "local_api", "language": "ko"})
        self.assertEqual(split["translate"], {})

    def test_no_srt_postprocess_key_yields_empty_postprocessing(self):
        legacy = {"provider": "local_api"}
        split = common.split_legacy_config(legacy)
        self.assertEqual(split["postprocessing"], {})
        self.assertEqual(split["stt"], {"provider": "local_api"})


class MigrateLegacyConfigTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        self.legacy_path = self.tmp_dir / "config.json"
        self.domain_paths = {
            "stt": self.tmp_dir / "config-stt.json",
            "translate": self.tmp_dir / "config-translate.json",
            "postprocessing": self.tmp_dir / "config-postprocessing.json",
        }

    def tearDown(self):
        self._tmp.cleanup()

    def _write_legacy(self, data: dict):
        self.legacy_path.write_text(json.dumps(data), encoding="utf-8")

    def test_splits_into_files_and_keeps_legacy_file(self):
        self._write_legacy({
            "provider": "local_api",
            "vad": {"threshold": 0.3},
            "srt_postprocess": {"cps_threshold": 20},
        })
        common.migrate_legacy_config_if_needed(self.legacy_path, self.domain_paths)

        stt = json.loads(self.domain_paths["stt"].read_text(encoding="utf-8"))
        postprocessing = json.loads(self.domain_paths["postprocessing"].read_text(encoding="utf-8"))

        self.assertEqual(stt, {"provider": "local_api", "vad": {"threshold": 0.3}})
        self.assertEqual(postprocessing, {"srt_postprocess": {"cps_threshold": 20}})
        # no legacy data maps to translation -- left unwritten so the normal
        # real->example->{} fallback chain still reaches the example file
        # instead of being shadowed by an empty real one
        self.assertFalse(self.domain_paths["translate"].exists())
        # legacy file itself is never touched/deleted
        self.assertTrue(self.legacy_path.exists())
        self.assertEqual(json.loads(self.legacy_path.read_text(encoding="utf-8"))["provider"], "local_api")

    def test_domain_with_no_legacy_data_is_left_unwritten(self):
        self._write_legacy({"provider": "local_api"})
        common.migrate_legacy_config_if_needed(self.legacy_path, self.domain_paths)
        self.assertFalse(self.domain_paths["translate"].exists())
        self.assertFalse(self.domain_paths["postprocessing"].exists())
        self.assertTrue(self.domain_paths["stt"].exists())

    def test_noop_if_any_domain_file_already_exists(self):
        self._write_legacy({"provider": "local_api"})
        self.domain_paths["stt"].write_text(json.dumps({"provider": "already_migrated"}), encoding="utf-8")

        common.migrate_legacy_config_if_needed(self.legacy_path, self.domain_paths)

        self.assertFalse(self.domain_paths["translate"].exists())
        self.assertFalse(self.domain_paths["postprocessing"].exists())
        stt = json.loads(self.domain_paths["stt"].read_text(encoding="utf-8"))
        self.assertEqual(stt, {"provider": "already_migrated"})

    def test_noop_if_no_legacy_file(self):
        common.migrate_legacy_config_if_needed(self.legacy_path, self.domain_paths)
        for path in self.domain_paths.values():
            self.assertFalse(path.exists())


class LoadConfigChainTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_real_file_wins_over_example(self):
        real = self.tmp_dir / "real.json"
        example = self.tmp_dir / "example.json"
        real.write_text(json.dumps({"source": "real"}), encoding="utf-8")
        example.write_text(json.dumps({"source": "example"}), encoding="utf-8")
        self.assertEqual(common._load_config_chain([real, example]), {"source": "real"})

    def test_falls_back_to_example_when_real_missing(self):
        real = self.tmp_dir / "real.json"
        example = self.tmp_dir / "example.json"
        example.write_text(json.dumps({"source": "example"}), encoding="utf-8")
        self.assertEqual(common._load_config_chain([real, example]), {"source": "example"})

    def test_falls_back_to_empty_dict_when_neither_exists(self):
        real = self.tmp_dir / "real.json"
        example = self.tmp_dir / "example.json"
        self.assertEqual(common._load_config_chain([real, example]), {})


if __name__ == "__main__":
    unittest.main()
