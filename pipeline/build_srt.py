"""Stage 4 (phase_a_roadmap.md): build an SRT from Stage 2c's manifest.json +
Stage 3's chunk_NNNN.txt transcripts.

Output path defaults to manifest["output_srt"] (design.md SS13: same folder
as the source file, "{input filename}.srt" -- fixed, not user-configurable).

Usage:
    python pipeline/build_srt.py temp/3c6527d5
    python pipeline/build_srt.py temp/3c6527d5/manifest.json --output out.srt
"""

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json


def resolve_manifest_path(job_arg: Path) -> Path:
    if job_arg.is_dir():
        return job_arg / "manifest.json"
    return job_arg


def srt_timestamp(seconds: float) -> str:
    total_ms = round(seconds * 1000)
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def collect_entries(manifest: dict, job_dir: Path) -> tuple[list[dict], int, int]:
    """Returns (entries with non-empty transcribed text, n_failed, n_empty)."""
    entries = []
    n_failed = 0
    n_empty = 0
    for chunk in manifest["chunks"]:
        if chunk["status"] != "transcribed":
            n_failed += 1
            continue
        txt_path = job_dir / chunk["file"]
        txt_path = txt_path.with_suffix(".txt")
        text = txt_path.read_text(encoding="utf-8").strip() if txt_path.exists() else ""
        text = " ".join(text.split())  # collapse embedded newlines/whitespace to one line
        if not text:
            n_empty += 1
            continue
        entries.append({"start_sec": chunk["start_sec"], "end_sec": chunk["end_sec"], "text": text})
    return entries, n_failed, n_empty


def validate_entries(entries: list[dict]) -> None:
    """Automated check (roadmap Stage 4): timestamps must be monotonic and non-overlapping."""
    prev_end = -1.0
    for i, e in enumerate(entries, 1):
        assert e["end_sec"] >= e["start_sec"], f"entry {i}: end < start ({e})"
        assert e["start_sec"] >= prev_end - 1e-6, (
            f"entry {i} starts at {e['start_sec']}s, before/inside previous entry's end ({prev_end}s)"
        )
        prev_end = e["end_sec"]


def render_srt(entries: list[dict]) -> str:
    blocks = []
    for i, e in enumerate(entries, 1):
        blocks.append(
            f"{i}\n{srt_timestamp(e['start_sec'])} --> {srt_timestamp(e['end_sec'])}\n{e['text']}\n"
        )
    return "\n".join(blocks)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("job", type=Path, help="job directory (temp/{job_id}) or its manifest.json")
    ap.add_argument("--output", type=Path, default=None,
                     help="override output .srt path (default: manifest['output_srt'], design.md SS13)")
    args = ap.parse_args()

    manifest_path = resolve_manifest_path(args.job)
    if not manifest_path.exists():
        sys.exit(f"manifest.json not found: {manifest_path} (run Stage 2c/3 first)")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    job_dir = manifest_path.parent

    entries, n_failed, n_empty = collect_entries(manifest, job_dir)
    validate_entries(entries)

    out_path = args.output or Path(manifest["output_srt"])
    out_path.write_text(render_srt(entries), encoding="utf-8")

    covered = sum(e["end_sec"] - e["start_sec"] for e in entries)
    total = manifest["source_duration_sec"]
    coverage_pct = 100 * covered / total if total > 0 else 0.0

    print(f"[srt] {len(entries)} subtitle(s) written to {out_path}")
    print(f"[srt] skipped: {n_failed} not-transcribed/failed chunk(s), {n_empty} empty-text chunk(s)")
    print(f"[srt] coverage: {covered:.3f}s / {total:.3f}s total ({coverage_pct:.1f}%)")
    print("[srt] manual check still needed: load this file in a real player (VLC etc.) alongside "
          "the source video and confirm subtitle timing doesn't visibly drift (roadmap Stage 4 "
          "pass criterion: no gap >=0.5s felt by eye)")


if __name__ == "__main__":
    main()
