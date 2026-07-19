"""Standalone SRT post-processing tool (postprocessing.md SS11 2차 분할).

Runs the CPS-based length-splitting step of the Cue Splitter against an
*already-translated* SRT file. This is deliberately decoupled from Phase C's
automatic speaker-only split (gui/worker.py's `_phase_c()`, which uses
cue_splitter.split_by_speaker_cues()): splitting by length before external
translation would hand the translator sentence fragments instead of whole
sentences, degrading translation quality. The intended workflow is:

    ASR -> Phase C (automatic, speaker split only) -> {name}.srt
        -> translate {name}.srt with an external tool
        -> this tool (manual, CPS-based length split) -> {name}.srt

Usage:
    python pipeline/srt_postprocess.py input.srt
    python pipeline/srt_postprocess.py input.srt --output out.srt --cps 15
"""

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from build_srt import parse_srt, render_srt  # noqa: E402
from common import load_config  # noqa: E402
from cue_splitter import DEFAULT_GAP_SEC, split_long_cue  # noqa: E402

DEFAULT_CUE_CFG = {
    "cps_threshold": 15,
    # postprocessing.md SS11.2 left this "미정" (to be finalized once real
    # cues surfaced the need); 7.0s matches common subtitle-authoring
    # guidance (e.g. Netflix's max-7s-on-screen rule) and is what actually
    # forces a split for cues that read fine CPS-wise but are simply long,
    # slow-paced blocks -- see cue_splitter._apply_cps_split.
    "max_cue_duration_sec": 7.0,
    "min_cue_duration_sec": None,
    # Neither CPS nor duration bounds how much text sits on screen at once --
    # a short, dense burst can read "fine" by chars/sec and still cover the
    # screen. 40 matches common Korean subtitle guidance (~2 lines of ~20
    # characters).
    "max_chars_per_cue": 40,
    "gap_sec": DEFAULT_GAP_SEC,
}


def postprocess_cues(cues: list[dict], cue_cfg: dict, log=print,
                      on_progress=lambda done, total: None) -> list[dict]:
    """Applies split_long_cue() to every input cue in order. `cue_cfg` uses
    the same field names as DEFAULT_CUE_CFG above."""
    cps_threshold = cue_cfg.get("cps_threshold", 15)
    min_dur = cue_cfg.get("min_cue_duration_sec")
    max_dur = cue_cfg.get("max_cue_duration_sec")
    max_chars = cue_cfg.get("max_chars_per_cue")
    gap_sec = cue_cfg.get("gap_sec", DEFAULT_GAP_SEC)

    out: list[dict] = []
    total = len(cues)
    for i, cue in enumerate(cues, 1):
        pieces = split_long_cue(
            cue["text"], cue["start_sec"], cue["end_sec"], cps_threshold,
            min_cue_duration_sec=min_dur, max_cue_duration_sec=max_dur,
            max_chars_per_cue=max_chars, gap_sec=gap_sec,
        )
        if len(pieces) > 1:
            duration = max(1e-9, cue["end_sec"] - cue["start_sec"])
            n_chars = len(cue["text"])
            cps = n_chars / duration
            reasons = []
            if cps > cps_threshold:
                reasons.append(f"CPS {cps:.1f} > {cps_threshold}")
            if max_dur and duration > max_dur:
                reasons.append(f"duration {duration:.1f}s > {max_dur}s")
            if max_chars and n_chars > max_chars:
                reasons.append(f"chars {n_chars} > {max_chars}")
            reason = ", ".join(reasons) if reasons else f"CPS {cps:.1f}"
            log(f"[srt_postprocess] cue {i}: {reason} -> split into {len(pieces)}")
        out.extend(pieces)
        on_progress(i, total)
    return out


def postprocess_srt_text(srt_text: str, cue_cfg: dict, log=print,
                          on_progress=lambda done, total: None) -> str:
    cues = parse_srt(srt_text)
    if not cues:
        log("[srt_postprocess] no cues parsed from input -- is this a valid SRT file?")
        return ""
    final_cues = postprocess_cues(cues, cue_cfg, log=log, on_progress=on_progress)
    log(f"[srt_postprocess] {len(cues)} cue(s) -> {len(final_cues)} cue(s)")
    return render_srt(final_cues)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="translated .srt file to length-split")
    ap.add_argument("--output", type=Path, default=None,
                     help="output path (default: overwrite input, backing up the original as {input}.bak)")
    ap.add_argument("--cps", type=float, default=None, help="CPS threshold (default: config.json srt_postprocess)")
    ap.add_argument("--max-duration", type=float, default=None, dest="max_duration")
    ap.add_argument("--min-duration", type=float, default=None, dest="min_duration")
    ap.add_argument("--max-chars", type=int, default=None, dest="max_chars",
                     help="max raw characters per cue (default: config.json srt_postprocess, else 40)")
    ap.add_argument("--gap", type=float, default=None, help="gap seconds between split cues")
    args = ap.parse_args()

    if not args.input.exists():
        sys.exit(f"input SRT not found: {args.input}")

    cue_cfg = {**DEFAULT_CUE_CFG, **load_config().get("srt_postprocess", {})}
    if args.cps is not None:
        cue_cfg["cps_threshold"] = args.cps
    if args.max_duration is not None:
        cue_cfg["max_cue_duration_sec"] = args.max_duration
    if args.min_duration is not None:
        cue_cfg["min_cue_duration_sec"] = args.min_duration
    if args.max_chars is not None:
        cue_cfg["max_chars_per_cue"] = args.max_chars
    if args.gap is not None:
        cue_cfg["gap_sec"] = args.gap

    srt_text = args.input.read_text(encoding="utf-8")
    result = postprocess_srt_text(srt_text, cue_cfg)

    output_path = args.output or args.input
    if output_path == args.input:
        backup_path = args.input.with_suffix(args.input.suffix + ".bak")
        backup_path.write_bytes(args.input.read_bytes())
        print(f"[srt_postprocess] original backed up to {backup_path}")

    output_path.write_text(result, encoding="utf-8")
    print(f"[srt_postprocess] wrote {output_path}")


if __name__ == "__main__":
    main()
