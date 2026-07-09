"""Stage 2a (phase_a_roadmap.md): raw Silero VAD pass over a 16kHz mono WAV.

Extends Stage 1 (extract_audio.py): reuses/produces the job's
audio_16k_mono.wav, then runs Silero VAD (design.md SS12.1 -- onnxruntime
directly, no torch/torchaudio) frame-by-frame with NO postprocessing
(merging/splitting is Stage 2b, SS12.2/12.3). The point of this stage is to
see what the raw model output looks like before any merge heuristics touch it.

Usage:
    python pipeline/vad_raw_test.py path/to/input.mp4
    python pipeline/vad_raw_test.py path/to/input.mp4 --threshold 0.3
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import matplotlib
import numpy as np
import onnxruntime as ort
import soundfile as sf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import extract_audio as stage1
from common import job_dir, vad_defaults

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL_URL = "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "silero_vad.onnx"

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 512  # Silero VAD's required chunk size at 16kHz
CONTEXT_SAMPLES = 64  # trailing samples of the previous chunk, fed back in as lookback context
FRAME_DURATION = WINDOW_SAMPLES / SAMPLE_RATE  # 32ms


def ensure_model() -> Path:
    if not MODEL_PATH.exists():
        print(f"[model] downloading Silero VAD onnx -> {MODEL_PATH}")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def ensure_extracted_wav(input_path: Path) -> Path:
    # If the input is already a 16kHz mono WAV (e.g. a Stage 1 output being
    # pointed at directly for VAD iteration), use it as-is instead of running
    # it through job_dir()/ffmpeg again under a new job_id.
    if input_path.suffix.lower() == ".wav":
        info = sf.info(str(input_path))
        if info.samplerate == SAMPLE_RATE and info.channels == 1:
            print(f"[stage1] input is already a 16kHz mono WAV, using directly: {input_path}")
            return input_path

    out_dir = job_dir(input_path)
    wav_path = out_dir / "audio_16k_mono.wav"
    if wav_path.exists():
        print(f"[stage1] reusing existing extraction: {wav_path}")
        return wav_path
    return stage1.extract(input_path, out_dir)


def run_vad(samples: np.ndarray) -> np.ndarray:
    """Per-frame speech probability, one value per WINDOW_SAMPLES chunk.

    Silero's onnx graph is stateful across two things: the GRU `state` output
    and a 64-sample trailing "context" lookback from the previous chunk that
    must be prepended to the next one (undocumented in the graph itself --
    see silero-vad's OnnxWrapper.__call__ in utils_vad.py). Skipping the
    context concatenation silently produces near-zero probabilities for
    everything instead of erroring, so it's easy to get wrong.
    """
    sess = ort.InferenceSession(str(ensure_model()), providers=["CPUExecutionProvider"])
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)
    sr = np.array(SAMPLE_RATE, dtype=np.int64)

    n_frames = int(np.ceil(len(samples) / WINDOW_SAMPLES))
    padded = np.zeros(n_frames * WINDOW_SAMPLES, dtype=np.float32)
    padded[:len(samples)] = samples

    probs = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        chunk = padded[i * WINDOW_SAMPLES:(i + 1) * WINDOW_SAMPLES].reshape(1, -1)
        x = np.concatenate([context, chunk], axis=1)
        out, state = sess.run(["output", "stateN"], {"input": x, "state": state, "sr": sr})
        probs[i] = out[0, 0]
        context = x[:, -CONTEXT_SAMPLES:]
    return probs


def raw_segments(probs: np.ndarray, threshold: float) -> list[tuple[float, float]]:
    """Contiguous runs of frames >= threshold, with NO merge/split postprocessing."""
    is_speech = probs >= threshold
    segments = []
    start = None
    for i, speech in enumerate(is_speech):
        if speech and start is None:
            start = i
        elif not speech and start is not None:
            segments.append((start * FRAME_DURATION, i * FRAME_DURATION))
            start = None
    if start is not None:
        segments.append((start * FRAME_DURATION, len(is_speech) * FRAME_DURATION))
    return segments


def print_stats(segments: list[tuple[float, float]], total_duration: float) -> None:
    print(f"\n[raw segments] {len(segments)} found (threshold pass, no merging)")
    for i, (s, e) in enumerate(segments, 1):
        print(f"  #{i:03d}  {s:7.3f}s -> {e:7.3f}s   len={e - s:6.3f}s")

    if not segments:
        print("  (none)")
        return

    lengths = [e - s for s, e in segments]
    gaps = []
    prev_end = 0.0
    for s, e in segments:
        gaps.append(s - prev_end)
        prev_end = e
    gaps.append(total_duration - prev_end)

    short_segments = sum(1 for l in lengths if l < 1.0)
    short_silences = sum(1 for g in gaps if 0 < g <= 1.0)

    print(f"\n[stats] segment length: min={min(lengths):.3f}s max={max(lengths):.3f}s "
          f"avg={sum(lengths) / len(lengths):.3f}s")
    print(f"[stats] segments < 1s: {short_segments}/{len(segments)}")
    print(f"[stats] silence gaps (leading/between/trailing): {len(gaps)}, of which <=1s: {short_silences}")


ROW_SECONDS = 30.0  # tile the timeline into fixed-width rows instead of one
                     # ever-widening strip -- a naive single-row plot scaled
                     # to duration becomes an unreadable multi-thousand-px
                     # image for anything longer than a short test clip.
MAX_POINTS_PER_ROW = 4000  # waveform envelope bins per row


def _plot_row(ax_wave, ax_prob, samples: np.ndarray, probs: np.ndarray,
              segment_layers: list[tuple[list[tuple[float, float]], str, float, str | None]],
              threshold: float, row_start: float, row_end: float) -> None:
    lo_idx = int(row_start * SAMPLE_RATE)
    hi_idx = int(row_end * SAMPLE_RATE)
    row_samples = samples[lo_idx:hi_idx]

    if len(row_samples) > MAX_POINTS_PER_ROW:
        bin_size = len(row_samples) // MAX_POINTS_PER_ROW
        n_bins = len(row_samples) // bin_size
        clipped = row_samples[:n_bins * bin_size].reshape(n_bins, bin_size)
        lo, hi = clipped.min(axis=1), clipped.max(axis=1)
        t_bins = row_start + np.linspace(0, row_end - row_start, n_bins)
        ax_wave.fill_between(t_bins, lo, hi, color="steelblue", linewidth=0, alpha=0.8)
    else:
        t_wave = row_start + np.linspace(0, row_end - row_start, len(row_samples))
        ax_wave.plot(t_wave, row_samples, color="steelblue", linewidth=0.5)

    ax_wave.set_xlim(row_start, row_end)
    ax_wave.set_ylim(-1.05, 1.05)
    ax_wave.set_ylabel(f"{row_start:.0f}s", rotation=0, ha="right", va="center", fontsize=8)
    ax_wave.tick_params(labelbottom=False)

    prob_lo, prob_hi = int(row_start / FRAME_DURATION), int(row_end / FRAME_DURATION)
    t_prob = np.arange(prob_lo, min(prob_hi, len(probs))) * FRAME_DURATION
    ax_prob.plot(t_prob, probs[prob_lo:prob_hi], color="crimson", linewidth=0.8)
    ax_prob.axhline(threshold, color="crimson", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_prob.set_xlim(row_start, row_end)
    ax_prob.set_ylim(0, 1.05)

    for ax in (ax_wave, ax_prob):
        for segments, color, alpha, hatch in segment_layers:
            for s, e in segments:
                if s < row_end and e > row_start:
                    if hatch:
                        ax.axvspan(s, e, facecolor="none", edgecolor=color,
                                   alpha=alpha, hatch=hatch, linewidth=0.8)
                    else:
                        ax.axvspan(s, e, color=color, alpha=alpha)


ROWS_PER_PAGE = 10  # split into multiple PNGs beyond this many rows -- a
                     # single image with dozens of stacked rows gets crushed
                     # down for display and becomes just as unreadable as the
                     # original ever-widening single-row plot.


def _render_page(row_range: range, samples: np.ndarray, probs: np.ndarray,
                  segment_layers: list[tuple[list[tuple[float, float]], str, float, str | None]],
                  threshold: float, duration: float, out_path: Path, title: str) -> None:
    n_rows = len(row_range)
    fig, axes = plt.subplots(
        n_rows * 2, 1, figsize=(14, 1.3 * n_rows * 2), sharex=False,
        gridspec_kw={"height_ratios": [2, 1] * n_rows, "hspace": 0.15},
    )
    if n_rows == 1:
        axes = [axes[0], axes[1]]

    for i, r in enumerate(row_range):
        row_start = r * ROW_SECONDS
        row_end = min(duration, row_start + ROW_SECONDS)
        _plot_row(axes[i * 2], axes[i * 2 + 1], samples, probs, segment_layers, threshold,
                  row_start, row_end)

    axes[-1].set_xlabel("time (s)")
    axes[0].set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[png] saved {out_path}")


def plot(samples: np.ndarray, probs: np.ndarray,
         segment_layers: list[tuple[list[tuple[float, float]], str, float, str | None]],
         threshold: float, out_path: Path, title: str) -> None:
    """segment_layers: list of (segments, color, alpha, hatch) drawn as axvspan
    highlights, in order. hatch=None means a plain filled span; a hatch string
    (e.g. "//") draws an unfilled, edge-only span instead, so callers can
    overlay e.g. a broad filled merged span with hatched raw segments on top
    without the top layer fully hiding the one underneath."""
    duration = len(samples) / SAMPLE_RATE
    n_rows = max(1, int(np.ceil(duration / ROW_SECONDS)))
    n_pages = int(np.ceil(n_rows / ROWS_PER_PAGE))

    # Two stacked panels per row (not a shared/twin axis) so the probability
    # curve can't visually collide with the waveform's zero baseline -- with
    # a twinx overlay, both axes' bottoms align, making a near-0 probability
    # look identical to a near -1.0 amplitude sample. Since this stage's main
    # check is eyeballing the PNG, that ambiguity defeats the point.
    print()
    for p in range(n_pages):
        row_range = range(p * ROWS_PER_PAGE, min(n_rows, (p + 1) * ROWS_PER_PAGE))
        page_out_path = out_path if n_pages == 1 else out_path.with_stem(f"{out_path.stem}_p{p + 1}")
        page_title = title if n_pages == 1 else f"{title}  (page {p + 1}/{n_pages})"
        _render_page(row_range, samples, probs, segment_layers, threshold, duration, page_out_path, page_title)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path, help="input media file (any ffmpeg-readable format)")
    ap.add_argument("--threshold", type=float, default=vad_defaults()["threshold"],
                     help="speech probability threshold (design.md SS12.1) "
                          f"(default from config: {vad_defaults()['threshold']})")
    args = ap.parse_args()

    if not args.input.exists():
        sys.exit(f"File not found: {args.input}")

    wav_path = ensure_extracted_wav(args.input)
    samples, sr = sf.read(str(wav_path), dtype="float32")
    assert sr == SAMPLE_RATE, f"expected {SAMPLE_RATE}Hz wav from stage 1, got {sr}"

    print(f"[vad] running Silero VAD over {len(samples) / SAMPLE_RATE:.3f}s of audio "
          f"({len(samples)} samples, {WINDOW_SAMPLES}-sample frames, threshold={args.threshold})")
    probs = run_vad(samples)
    segments = raw_segments(probs, args.threshold)
    print_stats(segments, len(samples) / SAMPLE_RATE)

    out_path = wav_path.parent / f"{args.input.stem}_vad_raw_t{args.threshold:.2f}.png"
    plot(samples, probs, [(segments, "orange", 0.3, None)], args.threshold, out_path,
         f"{args.input.name}  threshold={args.threshold}")


if __name__ == "__main__":
    main()
