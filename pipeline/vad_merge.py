"""Stage 2b (phase_a_roadmap.md): merge Stage 2a's raw VAD segments (design.md SS12.2/12.3).

Takes the same input as Stage 2a (vad_raw_test.py), runs the same raw VAD
pass, then applies the merge policy in the exact order design.md SS12.3
specifies:

    1. VAD raw segments (Stage 2a)
    2. merge short silences (gap <= --min-silence) -- connects adjacent speech
    3. absorb short speech segments (duration < --min-speech) into a neighbor
    4. force-split anything still > --max-chunk seconds, cut mechanically
       at the boundary (no attempt to find a low-energy cut point yet --
       design.md SS12.2 explicitly defers that to a later version)

Usage:
    python pipeline/vad_merge.py path/to/input.mp4
    python pipeline/vad_merge.py path/to/input.mp4 --min-silence 0.7 --min-speech 1.0 --max-chunk 30
"""

import argparse
import sys
from pathlib import Path

import soundfile as sf

from common import vad_defaults
from vad_raw_test import (
    SAMPLE_RATE, WINDOW_SAMPLES, ensure_extracted_wav, plot, print_stats, raw_segments, run_vad,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

Segment = tuple[float, float]


def merge_short_silences(segments: list[Segment], min_silence: float) -> list[Segment]:
    """Step 2: merge adjacent segments separated by a gap <= min_silence."""
    if not segments:
        return []
    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= min_silence:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def absorb_short_speech(segments: list[Segment], min_speech: float,
                        max_absorb_gap: float | None = None) -> list[Segment]:
    """Step 3: any segment shorter than min_speech is merged into whichever
    neighbor is closer (by gap) -- this is deliberately a separate rule from
    merge_short_silences: a short interjection surrounded by long silence
    still isn't worth its own chunk.

    max_absorb_gap caps how far that merge can reach: without it, a short
    fragment gets absorbed into its nearest neighbor no matter how large the
    intervening silence is, which can drag several seconds of pure silence
    into a chunk (real case found in Stage 2b testing on temp/8bc2b06c -- a
    0.77s fragment 5s away from its neighbor pulled that whole gap into a
    10s chunk). If both neighbors are farther than max_absorb_gap, the short
    segment is left standalone rather than bridging a silence design.md
    SS12.2 itself would call a real boundary (its own "2-3s = real
    silence" cutoff is the basis for the 3.0s default).
    """
    segs = list(segments)
    while len(segs) > 1:
        # Find the shortest under-threshold segment each pass (rather than
        # the first) so a chain of short segments collapses from the most
        # clear-cut case first instead of an arbitrary index order.
        short_idxs = [i for i, (s, e) in enumerate(segs) if e - s < min_speech]
        if not short_idxs:
            break
        short_idxs.sort(key=lambda idx: segs[idx][1] - segs[idx][0])

        merged_any = False
        for i in short_idxs:
            if i == 0:
                target, gap = 1, segs[1][0] - segs[0][1]
            elif i == len(segs) - 1:
                target, gap = len(segs) - 2, segs[i][0] - segs[len(segs) - 2][1]
            else:
                gap_prev = segs[i][0] - segs[i - 1][1]
                gap_next = segs[i + 1][0] - segs[i][1]
                target, gap = (i - 1, gap_prev) if gap_prev <= gap_next else (i + 1, gap_next)

            if max_absorb_gap is not None and gap > max_absorb_gap:
                continue  # neighbor too far -- leave this fragment standalone, try the next shortest

            lo_idx, hi_idx = sorted((i, target))
            new_seg = (min(segs[lo_idx][0], segs[hi_idx][0]), max(segs[lo_idx][1], segs[hi_idx][1]))
            segs = segs[:lo_idx] + [new_seg] + segs[hi_idx + 1:]
            merged_any = True
            break

        if not merged_any:
            break  # every remaining short fragment is too far from both neighbors
    return segs


def force_split(segments: list[Segment], max_chunk: float) -> tuple[list[Segment], int]:
    """Step 4: mechanically cut any segment > max_chunk at max_chunk boundaries
    (design.md SS12.2: finding a low-energy cut point is deferred, not done here).
    """
    result = []
    n_split = 0
    for start, end in segments:
        length = end - start
        if length <= max_chunk:
            result.append((start, end))
            continue
        n_split += 1
        cut = start
        while end - cut > max_chunk:
            result.append((cut, cut + max_chunk))
            cut += max_chunk
        result.append((cut, end))
    return result, n_split


def merge_pipeline(segments: list[Segment], min_silence: float, min_speech: float,
                    max_chunk: float, max_absorb_gap: float | None = None) -> dict:
    after_silence = merge_short_silences(segments, min_silence)
    after_absorb = absorb_short_speech(after_silence, min_speech, max_absorb_gap)
    final, n_split = force_split(after_absorb, max_chunk)
    return {
        "raw": segments,
        "after_silence_merge": after_silence,
        "after_short_absorb": after_absorb,
        "final": final,
        "n_force_split": n_split,
    }


def print_merge_stats(stages: dict, total_duration: float) -> None:
    print(f"\n[merge] raw={len(stages['raw'])}  "
          f"after_silence_merge={len(stages['after_silence_merge'])}  "
          f"after_short_absorb={len(stages['after_short_absorb'])}  "
          f"final={len(stages['final'])}  "
          f"force_split_count={stages['n_force_split']}")

    final = stages["final"]
    print(f"\n[merged segments] {len(final)}")
    for i, (s, e) in enumerate(final, 1):
        print(f"  #{i:03d}  {s:7.3f}s -> {e:7.3f}s   len={e - s:6.3f}s")

    if not final:
        return

    over_limit = [e - s for s, e in final if e - s > 30.0 + 1e-6]
    assert not over_limit, f"{len(over_limit)} final segment(s) exceed the 30s hard limit: {over_limit}"

    per_minute = len(final) / (total_duration / 60.0) if total_duration > 0 else 0.0
    print(f"\n[stats] chunks per minute of audio: {per_minute:.2f}  "
          f"(design.md/roadmap rule of thumb: ~2-4/min is comfortable)")


def add_merge_cli_args(ap: argparse.ArgumentParser) -> None:
    """Shared CLI args for anything that runs the Stage 2a+2b pipeline
    (this script, and Stage 2c's chunk_export.py) -- kept in one place so
    the two stages can't silently drift to different defaults/help text.

    Defaults come from config.json's "vad" section (falling back to
    config.example.json, then a hardcoded fallback -- see common.vad_defaults),
    so editing config.json changes these scripts' defaults without needing
    a CLI flag every time. An explicit CLI flag still always wins.
    """
    d = vad_defaults()
    ap.add_argument("--threshold", type=float, default=d["threshold"],
                     help="speech probability threshold for the underlying raw VAD pass (Stage 2a) "
                          f"(default from config: {d['threshold']})")
    ap.add_argument("--min-silence", type=float, default=d["min_silence"],
                     help="silences <= this many seconds are merged across (design.md SS12.2: 0.5-1.0s) "
                          f"(default from config: {d['min_silence']})")
    ap.add_argument("--min-speech", type=float, default=d["min_speech"],
                     help="speech segments shorter than this are absorbed into a neighbor "
                          f"(default from config: {d['min_speech']})")
    ap.add_argument("--max-absorb-gap", type=float, default=d["max_absorb_gap"],
                     help="cap on how far a short-speech absorb (--min-speech) can reach across "
                          "silence; beyond this the fragment is left standalone instead of "
                          "dragging that silence into a chunk (design.md SS12.2's own 2-3s "
                          "'real silence' cutoff is the basis for the default). Pass a negative "
                          "value to disable the cap and restore unlimited-reach absorption "
                          f"(default from config: {d['max_absorb_gap']})")
    ap.add_argument("--max-chunk", type=float, default=d["max_chunk"],
                     help="hard limit -- segments longer than this are force-split (model input cap) "
                          f"(default from config: {d['max_chunk']})")


def resolve_max_absorb_gap(args: argparse.Namespace) -> float | None:
    return None if args.max_absorb_gap < 0 else args.max_absorb_gap


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path, help="input media file (any ffmpeg-readable format)")
    add_merge_cli_args(ap)
    args = ap.parse_args()
    max_absorb_gap = resolve_max_absorb_gap(args)

    if not args.input.exists():
        sys.exit(f"File not found: {args.input}")

    wav_path = ensure_extracted_wav(args.input)
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

    gap_label = "none" if max_absorb_gap is None else f"{max_absorb_gap:.1f}"
    out_path = wav_path.parent / (
        f"{args.input.stem}_vad_merged_t{args.threshold:.2f}"
        f"_s{args.min_silence:.1f}_m{args.min_speech:.1f}_g{gap_label}_c{args.max_chunk:.0f}.png"
    )
    # Merged (final) span drawn as a broad filled background first, raw
    # segments as hatched outlines on top -- so the raw "islands" stay
    # visible inside a merged block instead of being fully covered by it,
    # which is the whole point of the raw-vs-merged comparison (roadmap
    # Stage 2b: check that merging isn't papering over a real speaker/sentence
    # boundary that should have stayed separate).
    plot(samples, probs,
         [(stages["final"], "orange", 0.25, None), (stages["raw"], "black", 0.7, "//")],
         args.threshold, out_path,
         f"{args.input.name}  threshold={args.threshold}  "
         f"min_silence={args.min_silence} min_speech={args.min_speech} "
         f"max_absorb_gap={gap_label} max_chunk={args.max_chunk}")


if __name__ == "__main__":
    main()
