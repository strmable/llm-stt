"""End-to-end runner: source media file -> {source}.srt (phase_a_roadmap.md Stage 1-4).

Runs the pipeline stage scripts in sequence as subprocesses (each is also
independently runnable -- see pipeline/README.md):

    extract_audio.py -> chunk_export.py (VAD raw+merge+export internally)
    -> transcribe_chunks.py -> build_srt.py

On success, the SRT lands next to the source file (design.md SS13) and
temp/{job_id}/ is deleted (design.md SS8 cleanup option, default ON --
config.json's cleanup.remove_temp_on_success controls this, --keep-temp
forces keeping it regardless). On failure, temp/{job_id}/ is always left in
place for debugging (design.md SS14.4: only delete on a clean finish).

Usage:
    python run_transcript.py path/to/source_video.mp4
    python run_transcript.py path/to/source_video.mp4 --language ko --keep-temp
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PIPELINE_DIR = REPO_ROOT / "pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

from common import compute_job_id, job_dir as get_job_dir, load_config  # noqa: E402
from transcribe_chunks import check_server  # noqa: E402

# line_buffering=True: without it, stdout is fully block-buffered whenever
# it isn't a real terminal (redirected to a file, captured by a wrapper,
# etc.), so this script's own print()s can sit unflushed until exit while
# each subprocess's own (separate fd) writes appear immediately -- garbling
# the interleaving between our banners and the child stage's own logging.
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)


def run_stage(script: str, stage_args: list[str]) -> None:
    cmd = [sys.executable, str(PIPELINE_DIR / script), *stage_args]
    print(f"\n{'=' * 70}\n[run_transcript] {' '.join(cmd)}\n{'=' * 70}")
    # Flush before handing stdout to the child -- otherwise our own print()
    # above can sit in Python's buffer while the subprocess's writes (a
    # separate OS-level fd) reach the terminal/log first, garbling the order.
    sys.stdout.flush()
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"[run_transcript] {script} failed (exit {result.returncode}) -- "
                 f"temp dir preserved for debugging, see output above")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path, help="source media file (any ffmpeg-readable format)")

    ap.add_argument("--threshold", type=float, default=None, help="Stage 2 VAD arg (default: config.json)")
    ap.add_argument("--min-silence", type=float, default=None, help="Stage 2 VAD arg (default: config.json)")
    ap.add_argument("--min-speech", type=float, default=None, help="Stage 2 VAD arg (default: config.json)")
    ap.add_argument("--max-absorb-gap", type=float, default=None, help="Stage 2 VAD arg (default: config.json)")
    ap.add_argument("--max-chunk", type=float, default=None, help="Stage 2 VAD arg (default: config.json)")

    ap.add_argument("--language", default="auto", help="Stage 3 arg, see pipeline/README.md")
    ap.add_argument("--server", default="http://localhost:8080", help="llama-server base URL")
    ap.add_argument("--model", default="qwen3-asr")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=120.0)

    ap.add_argument("--output", type=Path, default=None,
                     help="override output .srt path (default: next to source, design.md SS13)")
    ap.add_argument("--keep-temp", action="store_true",
                     help="don't delete temp/{job_id}/ on success (default: delete, design.md SS8)")
    args = ap.parse_args()

    if not args.source.exists():
        sys.exit(f"File not found: {args.source}")

    # Fail fast before spending time on extraction/VAD/chunking if the STT
    # backend isn't even reachable -- transcribe_chunks.py checks this too,
    # but only after stages 1-2 have already run.
    check_server(args.server)

    job_id = compute_job_id(args.source)
    job_dir_path = get_job_dir(args.source)
    print(f"[run_transcript] source={args.source}  job_id={job_id}  dir={job_dir_path}")

    run_stage("extract_audio.py", [str(args.source)])

    vad_args = []
    for flag, value in [("--threshold", args.threshold), ("--min-silence", args.min_silence),
                         ("--min-speech", args.min_speech), ("--max-absorb-gap", args.max_absorb_gap),
                         ("--max-chunk", args.max_chunk)]:
        if value is not None:
            vad_args += [flag, str(value)]
    run_stage("chunk_export.py", [str(args.source), *vad_args])

    run_stage("transcribe_chunks.py", [
        str(job_dir_path), "--language", args.language, "--server", args.server,
        "--model", args.model, "--temperature", str(args.temperature), "--timeout", str(args.timeout),
    ])

    build_args = [str(job_dir_path)]
    if args.output:
        build_args += ["--output", str(args.output)]
    run_stage("build_srt.py", build_args)

    remove_on_success = load_config().get("cleanup", {}).get("remove_temp_on_success", True)
    if args.keep_temp:
        print(f"\n[run_transcript] --keep-temp set, leaving {job_dir_path} in place")
    elif not remove_on_success:
        print(f"\n[run_transcript] config cleanup.remove_temp_on_success=false, leaving {job_dir_path} in place")
    else:
        shutil.rmtree(job_dir_path)
        print(f"\n[run_transcript] done -- removed {job_dir_path}")


if __name__ == "__main__":
    main()
