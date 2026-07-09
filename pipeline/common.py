"""Shared helpers for the Phase A/B pipeline stage scripts (see phase_a_roadmap.md).

Each stage (extract_audio.py, vad_raw_test.py, ...) is an independently
runnable script, but they all need to agree on where a given input file's
working directory lives, so job_id/temp-dir resolution (design.md SS13/SS14.1)
lives here instead of being duplicated per stage.
"""

import datetime
import hashlib
import json
from pathlib import Path

# Temp dir root is relative to the repo/install root (parent of this file's
# pipeline/ dir), not the input file's location, per design.md SS13 (input
# may be read-only/network; the future GUI main.py lives at the repo root).
REPO_ROOT = Path(__file__).resolve().parent.parent
TEMP_ROOT = REPO_ROOT / "temp"

SOURCE_INFO_FILENAME = "source_info.json"

CONFIG_PATH = REPO_ROOT / "config.json"
CONFIG_EXAMPLE_PATH = REPO_ROOT / "config.example.json"

# Fallback if neither config.json nor config.example.json exists (e.g. a
# fresh checkout before either was ever created) -- keeps the CLI scripts
# working standalone rather than erroring on a missing file.
_VAD_DEFAULTS_FALLBACK = {
    "threshold": 0.5,
    "min_silence": 0.7,
    "min_speech": 1.0,
    "max_absorb_gap": 3.0,
    "max_chunk": 30.0,
}


def load_config() -> dict:
    """config.json (design.md SS9) if present, else config.example.json, else {}.

    config.json is gitignored (may hold a real Gemini API key/local paths once
    someone fills in Settings) -- config.example.json is the tracked template
    with safe defaults, so a fresh checkout without a local config.json still
    gets the same VAD/etc. defaults instead of silently falling back further.
    """
    for path in (CONFIG_PATH, CONFIG_EXAMPLE_PATH):
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def vad_defaults() -> dict:
    """VAD section of the config, with _VAD_DEFAULTS_FALLBACK filling in any
    key missing from the file (partial/edited configs shouldn't crash CLI
    scripts that expect all five keys to exist)."""
    return {**_VAD_DEFAULTS_FALLBACK, **load_config().get("vad", {})}


def compute_job_id(source: Path) -> str:
    """job_id = f(source abs path, mtime, size) -- design.md SS14.1."""
    st = source.stat()
    key = f"{source.resolve()}|{st.st_mtime}|{st.st_size}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]


def job_dir(source: Path) -> Path:
    d = TEMP_ROOT / compute_job_id(source)
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_source_info(job_dir_path: Path, source: Path) -> None:
    """Persist the original source file's path/mtime/size next to its job_id
    (design.md SS14.1's triple) so later stage scripts can recover it even
    when invoked directly against an already-extracted audio_16k_mono.wav
    instead of the original media file -- without this, Stage 2c's manifest
    would record the WAV as "source_file", and Stage 4's SRT would land next
    to the WAV in temp/ instead of next to the real source (design.md SS13:
    output SRT belongs beside the original input, named after it).
    """
    st = source.stat()
    info = {
        "source_file": str(source.resolve()),
        "source_mtime": datetime.datetime.fromtimestamp(st.st_mtime).isoformat(),
        "source_size": st.st_size,
        "job_id": job_dir_path.name,
    }
    (job_dir_path / SOURCE_INFO_FILENAME).write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def read_source_info(job_dir_path: Path) -> dict | None:
    path = job_dir_path / SOURCE_INFO_FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
