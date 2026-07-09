"""Stage 1 (phase_a_roadmap.md): extract a 16kHz mono WAV from any media file via ffmpeg.

Usage:
    python pipeline/extract_audio.py path/to/input.mp4
    python pipeline/extract_audio.py path/to/input.mp4 --job-dir temp/custom

Validates the output (design.md SS13 output format policy) by reopening it with
soundfile.info() and asserting sample_rate==16000, channels==1, and duration
within 0.1s of the source -- failures raise instead of failing silently.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import soundfile as sf

from common import compute_job_id, job_dir, write_source_info

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def probe(path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr}")
    return json.loads(result.stdout)


def extract(source: Path, out_dir: Path) -> Path:
    out_path = out_dir / "audio_16k_mono.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(source),
        "-ac", "1", "-ar", "16000", "-vn",
        str(out_path),
    ]
    print(f"[ffmpeg] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"[ffmpeg] exit code: {result.returncode}")
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"ffmpeg failed for {source} (exit {result.returncode})")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path, help="input media file (any ffmpeg-readable format)")
    ap.add_argument("--job-dir", type=Path, default=None,
                     help="override temp/{job_id}/ output dir (default: derived from "
                          "input path/mtime/size, design.md SS14.1)")
    args = ap.parse_args()

    if not args.input.exists():
        sys.exit(f"File not found: {args.input}")

    src_info = probe(args.input)
    src_format = src_info["format"]
    src_duration = float(src_format["duration"])
    print(f"[input] {args.input}")
    print(f"  format: {src_format.get('format_name')}  duration: {src_duration:.3f}s  "
          f"size: {src_format.get('size')} bytes")
    for stream in src_info["streams"]:
        if stream["codec_type"] == "audio":
            print(f"  audio stream: codec={stream.get('codec_name')} "
                  f"sample_rate={stream.get('sample_rate')} channels={stream.get('channels')}")
            break

    out_dir = args.job_dir or job_dir(args.input)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[job] job_id={compute_job_id(args.input)}  dir={out_dir}")

    out_path = extract(args.input, out_dir)

    info = sf.info(str(out_path))
    print(f"[output] {out_path}")
    print(f"  sample_rate={info.samplerate}  channels={info.channels}  duration={info.duration:.3f}s")

    errors = []
    if info.samplerate != 16000:
        errors.append(f"sample_rate {info.samplerate} != 16000")
    if info.channels != 1:
        errors.append(f"channels {info.channels} != 1")
    if abs(info.duration - src_duration) > 0.1:
        errors.append(f"duration mismatch: output {info.duration:.3f}s vs source {src_duration:.3f}s (>0.1s)")

    if errors:
        raise AssertionError("Stage 1 validation failed: " + "; ".join(errors))

    print("[PASS] output WAV matches expected format and duration")

    # So later stages can recover the true original path/name even if invoked
    # directly against audio_16k_mono.wav instead of args.input (see
    # common.write_source_info's docstring for why this matters).
    write_source_info(out_dir, args.input)


if __name__ == "__main__":
    main()
