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

# design.md SS9 (v3.4): the pre-v3.4 single config.json is split into three
# domain files, each following the same tracked-example + gitignored-real
# fallback pattern config.json/config.example.json used before. The legacy
# file is never deleted -- see migrate_legacy_config_if_needed().
LEGACY_CONFIG_PATH = REPO_ROOT / "config.json"

CONFIG_STT_PATH = REPO_ROOT / "config-stt.json"
CONFIG_STT_EXAMPLE_PATH = REPO_ROOT / "config-stt.example.json"

CONFIG_TRANSLATE_PATH = REPO_ROOT / "config-translate.json"
CONFIG_TRANSLATE_EXAMPLE_PATH = REPO_ROOT / "config-translate.example.json"

CONFIG_POSTPROCESSING_PATH = REPO_ROOT / "config-postprocessing.json"
CONFIG_POSTPROCESSING_EXAMPLE_PATH = REPO_ROOT / "config-postprocessing.example.json"

_DOMAIN_REAL_PATHS = {
    "stt": CONFIG_STT_PATH,
    "translate": CONFIG_TRANSLATE_PATH,
    "postprocessing": CONFIG_POSTPROCESSING_PATH,
}
_DOMAIN_EXAMPLE_PATHS = {
    "stt": CONFIG_STT_EXAMPLE_PATH,
    "translate": CONFIG_TRANSLATE_EXAMPLE_PATH,
    "postprocessing": CONFIG_POSTPROCESSING_EXAMPLE_PATH,
}

# Everything in a legacy config.json is STT-domain except srt_postprocess
# (-> postprocessing domain). Nothing legacy maps to translation -- it's new
# in v3.4, so a migrated setup just gets translation's defaults.
_LEGACY_POSTPROCESSING_KEY = "srt_postprocess"

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


def split_legacy_config(legacy: dict) -> dict:
    """Pure split of a pre-v3.4 single config.json's top-level keys into the
    three v3.4 domains (design.md SS9). Returns {"stt": ..., "translate": {},
    "postprocessing": ...}."""
    postprocessing = {}
    if _LEGACY_POSTPROCESSING_KEY in legacy:
        postprocessing[_LEGACY_POSTPROCESSING_KEY] = legacy[_LEGACY_POSTPROCESSING_KEY]
    stt = {k: v for k, v in legacy.items() if k != _LEGACY_POSTPROCESSING_KEY}
    return {"stt": stt, "translate": {}, "postprocessing": postprocessing}


def migrate_legacy_config_if_needed(legacy_path: Path, domain_paths: dict[str, Path]) -> None:
    """One-time migration: if none of `domain_paths` exist yet but
    `legacy_path` does, split it into them. No-op once any domain file
    exists (including a fresh install with none of them, including no
    legacy file -- callers fall through to their .example.json/{} as
    before). `legacy_path` itself is never modified or deleted."""
    if any(path.exists() for path in domain_paths.values()):
        return
    if not legacy_path.exists():
        return
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    split = split_legacy_config(legacy)
    for domain, path in domain_paths.items():
        if not split[domain]:
            # No legacy data for this domain (e.g. translation, all-new in
            # v3.4) -- leave it unwritten so the normal real->example->{}
            # fallback chain still reaches config-{domain}.example.json,
            # instead of shadowing it with an empty real file.
            continue
        path.write_text(json.dumps(split[domain], ensure_ascii=False, indent=2), encoding="utf-8")


def _load_config_chain(paths: list[Path]) -> dict:
    """First existing file in `paths`, parsed as JSON, else {}."""
    for path in paths:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _load_domain_config(domain: str) -> dict:
    migrate_legacy_config_if_needed(LEGACY_CONFIG_PATH, _DOMAIN_REAL_PATHS)
    return _load_config_chain([_DOMAIN_REAL_PATHS[domain], _DOMAIN_EXAMPLE_PATHS[domain]])


def load_stt_config() -> dict:
    """config-stt.json if present, else config-stt.example.json, else {}
    (design.md SS9). Covers provider/local_api/gemini/llm/prompt/
    text_enhancement/vad/cleanup/logging -- everything the STT tab owns."""
    return _load_domain_config("stt")


def load_translate_config() -> dict:
    """config-translate.json if present, else config-translate.example.json,
    else {} (design.md SS9/SS5C.8) -- the 번역 tab's own `translation` section."""
    return _load_domain_config("translate")


def load_postprocessing_config() -> dict:
    """config-postprocessing.json if present, else
    config-postprocessing.example.json, else {} (design.md SS9) -- the
    후처리 탭's `srt_postprocess` section."""
    return _load_domain_config("postprocessing")


def vad_defaults() -> dict:
    """VAD section of the STT config, with _VAD_DEFAULTS_FALLBACK filling in
    any key missing from the file (partial/edited configs shouldn't crash
    CLI scripts that expect all five keys to exist)."""
    return {**_VAD_DEFAULTS_FALLBACK, **load_stt_config().get("vad", {})}


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
