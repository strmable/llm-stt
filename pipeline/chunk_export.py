"""Stage 2c (phase_a_roadmap.md): export Stage 2b's final segments as chunk WAV
files + manifest.json (design.md SS14.2 schema).

The one thing this stage has to get right is time-offset tracking: each
chunk_NNNN.wav is its own standalone file starting at local t=0, so once it's
handed to Stage 3 for transcription, the ONLY place its original position in
the source audio survives is manifest.json. Two things back that up here:

  - Chunks are sliced by sample index directly out of the single in-memory
    array read from audio_16k_mono.wav (itself already verified in Stage 1 to
    match the source's duration within 0.1s), rather than by re-invoking
    ffmpeg per chunk with float seconds -- slicing a numpy array by integer
    index is exact, so there's no per-chunk seek/rounding drift to accumulate
    across dozens of chunks the way repeated ffmpeg -ss/-to calls could have.
  - Each chunk's manifest entry records the offset three ways: start_sec/
    end_sec (float seconds, for arithmetic), start/end (HH:MM:SS.mmm strings,
    for human review and direct reuse when Stage 4 builds SRT timestamps),
    and start_sample/end_sample (the exact indices used to slice, for
    byte-exact reproducibility/debugging).

Usage:
    python pipeline/chunk_export.py path/to/input.mp4
    python pipeline/chunk_export.py path/to/input.mp4 --min-silence 0.7 --min-speech 1.0
"""

import argparse
import datetime
import json
import random
import sys
from pathlib import Path

import soundfile as sf

from common import read_source_info
from vad_merge import add_merge_cli_args, merge_pipeline, print_merge_stats, resolve_max_absorb_gap
from vad_raw_test import SAMPLE_RATE, WINDOW_SAMPLES, ensure_extracted_wav, print_stats, raw_segments, run_vad

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

Segment = tuple[float, float]


def format_hms(seconds: float) -> str:
    total_ms = round(seconds * 1000)
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def export_chunks(samples, final_segments: list[Segment], chunks_dir: Path) -> list[dict]:
    """Slice each final segment directly out of `samples` by sample index and
    write it as its own WAV. Returns the manifest "chunks" entries."""
    chunks_dir.mkdir(parents=True, exist_ok=True)
    n_samples = len(samples)
    entries = []

    for i, (start_sec, end_sec) in enumerate(final_segments, 1):
        start_sample = max(0, round(start_sec * SAMPLE_RATE))
        end_sample = min(n_samples, round(end_sec * SAMPLE_RATE))

        chunk_name = f"chunk_{i:04d}.wav"
        chunk_path = chunks_dir / chunk_name
        sf.write(str(chunk_path), samples[start_sample:end_sample], SAMPLE_RATE, subtype="PCM_16")

        # Re-open what was actually written rather than trusting the slice
        # math -- catches any writer/subtype surprise instead of silently
        # trusting a mismatched file (Stage 1's "no silent failure" rule).
        info = sf.info(str(chunk_path))
        expected_duration = (end_sample - start_sample) / SAMPLE_RATE
        assert abs(info.duration - expected_duration) < 1e-3, (
            f"{chunk_name}: written duration {info.duration:.3f}s != expected {expected_duration:.3f}s"
        )

        entries.append({
            "id": i,
            "file": f"chunks/{chunk_name}",
            "start": format_hms(start_sample / SAMPLE_RATE),
            "end": format_hms(end_sample / SAMPLE_RATE),
            "start_sec": round(start_sample / SAMPLE_RATE, 3),
            "end_sec": round(end_sample / SAMPLE_RATE, 3),
            "duration_sec": round(expected_duration, 3),
            "start_sample": start_sample,
            "end_sample": end_sample,
            "status": "vad_extracted",
        })
        print(f"  {chunk_name}  {format_hms(start_sample / SAMPLE_RATE)} -> "
              f"{format_hms(end_sample / SAMPLE_RATE)}  "
              f"len={expected_duration:6.3f}s  size={chunk_path.stat().st_size / 1024:7.1f} KB")

    return entries


def preserve_prior_transcriptions(job_dir_path: Path, chunk_entries: list[dict]) -> int:
    """Re-running this stage (e.g. after fixing source_file tracking, or
    re-exporting with unchanged VAD params) overwrites manifest.json from
    scratch, which would otherwise silently reset every chunk's status back
    to "vad_extracted" even though Stage 3 already transcribed them and their
    chunk_NNNN.txt files are still sitting right there on disk -- discovered
    by hand while verifying the source_file fix above. Match old chunks to
    new ones by identical (start_sec, end_sec) and carry the status/metadata
    forward when the transcript file still exists.
    """
    old_manifest_path = job_dir_path / "manifest.json"
    if not old_manifest_path.exists():
        return 0
    old_chunks = json.loads(old_manifest_path.read_text(encoding="utf-8")).get("chunks", [])
    old_by_offset = {(c["start_sec"], c["end_sec"]): c for c in old_chunks}

    n_restored = 0
    for entry in chunk_entries:
        # The .txt file actually existing is the ground truth -- not the old
        # manifest's status field, which may itself already be stale/wrong
        # (e.g. from a prior buggy re-export that already clobbered it back
        # to "vad_extracted" while the .txt was left untouched on disk).
        txt_path = (job_dir_path / entry["file"]).with_suffix(".txt")
        if not txt_path.exists() or not txt_path.read_text(encoding="utf-8").strip():
            continue
        entry["status"] = "transcribed"
        old = old_by_offset.get((entry["start_sec"], entry["end_sec"]))
        if old:
            for key in ("language_detected", "transcribe_elapsed_sec", "flags"):
                if key in old:
                    entry[key] = old[key]
        n_restored += 1
    return n_restored


def validate_manifest(chunk_entries: list[dict]) -> None:
    """Automated check (roadmap Stage 2c): chunks must stay in non-decreasing,
    non-overlapping order -- Stage 4's SRT timestamps depend on it."""
    prev_end = -1.0
    for entry in chunk_entries:
        assert entry["start_sec"] >= prev_end - 1e-6, (
            f"chunk {entry['id']} starts at {entry['start_sec']}s, before/inside "
            f"the previous chunk's end ({prev_end}s) -- overlap bug"
        )
        assert entry["end_sec"] >= entry["start_sec"], f"chunk {entry['id']} has end < start"
        prev_end = entry["end_sec"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path, help="input media file (any ffmpeg-readable format)")
    add_merge_cli_args(ap)
    ap.add_argument("--provider", default="local_api", help="design.md SS9 config.json provider id, "
                                                             "recorded as manifest metadata only at this stage")
    args = ap.parse_args()
    max_absorb_gap = resolve_max_absorb_gap(args)

    if not args.input.exists():
        sys.exit(f"File not found: {args.input}")

    wav_path = ensure_extracted_wav(args.input)
    job_dir_path = wav_path.parent
    job_id = job_dir_path.name

    samples, sr = sf.read(str(wav_path), dtype="float32")
    assert sr == SAMPLE_RATE, f"expected {SAMPLE_RATE}Hz wav from stage 1, got {sr}"
    total_duration = len(samples) / SAMPLE_RATE

    print(f"[vad] running Silero VAD over {total_duration:.3f}s of audio "
          f"({len(samples)} samples, {WINDOW_SAMPLES}-sample frames, threshold={args.threshold})")
    probs = run_vad(samples)
    raw = raw_segments(probs, args.threshold)
    print_stats(raw, total_duration)

    stages = merge_pipeline(raw, args.min_silence, args.min_speech, args.max_chunk, max_absorb_gap)
    print_merge_stats(stages, total_duration)

    print(f"\n[export] writing {len(stages['final'])} chunk(s) to {job_dir_path / 'chunks'}")
    chunk_entries = export_chunks(samples, stages["final"], job_dir_path / "chunks")
    validate_manifest(chunk_entries)

    n_restored = preserve_prior_transcriptions(job_dir_path, chunk_entries)
    if n_restored:
        print(f"\n[export] restored 'transcribed' status for {n_restored} chunk(s) unchanged "
              f"since the previous manifest (their .txt transcripts are still on disk)")

    total_chunk_duration = sum(e["duration_sec"] for e in chunk_entries)
    total_speech_duration = sum(e - s for s, e in stages["final"])
    print(f"\n[stats] {len(chunk_entries)} chunk(s) exported, "
          f"total chunk duration={total_chunk_duration:.3f}s "
          f"(final-segment speech total={total_speech_duration:.3f}s)")

    # Prefer the original source file's path/mtime/size recorded by Stage 1
    # (extract_audio.py) at extraction time. Without this, pointing this
    # script directly at an already-extracted audio_16k_mono.wav (a supported
    # shortcut -- see ensure_extracted_wav) would record the WAV itself as
    # "source_file", and Stage 4's SRT would land next to the WAV in temp/
    # instead of next to the real source video under its real name.
    source_info = read_source_info(job_dir_path)
    if source_info:
        source_file, source_mtime, source_size = (
            source_info["source_file"], source_info["source_mtime"], source_info["source_size"],
        )
        output_srt = str(Path(source_file).parent / f"{Path(source_file).stem}.srt")
    else:
        print(f"\n[WARNING] no {job_dir_path / 'source_info.json'} found -- this job's true original "
              f"source file is unknown (probably because chunk_export.py was pointed directly at an "
              f"extracted WAV rather than the original media). Falling back to {args.input} itself; "
              f"output_srt below will NOT point at the real source video.")
        input_stat = args.input.stat()
        source_file = str(args.input.resolve())
        source_mtime = datetime.datetime.fromtimestamp(input_stat.st_mtime).isoformat()
        source_size = input_stat.st_size
        output_srt = str(args.input.resolve().parent / f"{args.input.stem}.srt")

    manifest = {
        "job_id": job_id,
        "source_file": source_file,
        "source_mtime": source_mtime,
        "source_size": source_size,
        "provider": args.provider,
        "created_at": datetime.datetime.now().isoformat(),
        "output_srt": output_srt,
        "audio_sample_rate": SAMPLE_RATE,
        "source_duration_sec": round(total_duration, 3),
        "vad_params": {
            "threshold": args.threshold,
            "min_silence": args.min_silence,
            "min_speech": args.min_speech,
            "max_absorb_gap": max_absorb_gap,
            "max_chunk": args.max_chunk,
        },
        "chunks": chunk_entries,
    }
    manifest_path = job_dir_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[manifest] saved {manifest_path}")

    sample_size = min(5, len(chunk_entries))
    if sample_size:
        sample = sorted(random.sample(chunk_entries, sample_size), key=lambda e: e["id"])
        names = ", ".join(e["file"] for e in sample)
        print(f"\n[manual check] listen to a random sample and confirm no sentence is cut mid-way: {names}")


if __name__ == "__main__":
    main()
