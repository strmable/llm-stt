"""Phase C step 2 (postprocessing.md SS11): deterministic cue splitting.

This module implements the two SS11 splitting stages as two *independent*
entry points, run at different times by different callers:

  1. `split_by_speaker_cues()` -- splits on [[SPEAKER]] markers left by
     text_correction.py (a merged question+answer becomes 2 cues). Runs
     automatically right after Phase C text correction (gui/worker.py's
     `_phase_c()`), before the SRT is handed off for external translation.
  2. `split_long_cue()` -- cuts a single already-timed cue further at
     punctuation/word boundaries whenever it exceeds the CPS (chars/sec)
     threshold *or* the max cue duration (never mid-word, no LLM call). Runs
     later, manually, via the
     standalone SRT post-processing tool (pipeline/srt_postprocess.py +
     gui/srt_postprocess_dialog.py) against a *translated* SRT -- splitting
     by length before translation would hand the external translator
     fragments instead of whole sentences.

Both stages approximate timestamps by each piece's share of the input
duration, carving a small gap out of each split cue's displayed end so
consecutive cues don't visually touch.

No LLM involvement anywhere in this module (design.md explicitly rules out
summarization/paraphrase risk for this step -- postprocessing.md SS11).
"""

import math
import re

from text_correction import SPEAKER_MARKER

# Boundaries preferred for mechanical splitting, checked in this order:
# sentence-ending punctuation first, then softer punctuation, then whitespace
# (CJK text without spaces falls through to a hard character-count cut).
_SENTENCE_END_RE = re.compile(r"[.!?。！？…]")
_SOFT_PUNCT_RE = re.compile(r"[,、，,·:;：；]")
_WHITESPACE_RE = re.compile(r"\s")

DEFAULT_GAP_SEC = 0.08  # ~2 frames at 24fps, common subtitle-authoring minimum gap


def split_by_speaker(text: str) -> list[str]:
    pieces = [p.strip() for p in text.split(SPEAKER_MARKER)]
    return [p for p in pieces if p]


def _find_break_point(text: str, target: int) -> int | None:
    """Best index to cut *after* (inclusive), at or before `target`, scanning
    backward for a sentence end, then soft punctuation, then whitespace.
    Returns None if nothing usable is found in [target*0.4, target]."""
    lo = max(1, int(target * 0.4))
    window = text[lo:target + 1]
    if not window:
        return None
    for pattern in (_SENTENCE_END_RE, _SOFT_PUNCT_RE, _WHITESPACE_RE):
        matches = list(pattern.finditer(window))
        if matches:
            return lo + matches[-1].end()
    return None


def split_long_piece(text: str, max_chars: int) -> list[str]:
    """Pure mechanical splitter -- never rewrites text, only cuts it.
    Prefers sentence/soft-punctuation/whitespace boundaries near `max_chars`;
    falls back to a hard cut (needed for CJK text without word spacing)."""
    if max_chars <= 0:
        max_chars = 1
    pieces = []
    remaining = text.strip()
    while len(remaining) > max_chars:
        cut = _find_break_point(remaining, max_chars)
        if cut is None or cut <= 0:
            cut = max_chars
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return [p for p in pieces if p]


def _apply_cps_split(pieces_with_speaker: list[tuple[int, str]], total_duration: float,
                      cps_threshold: float, max_cue_duration_sec: float | None = None,
                      max_chars_per_cue: int | None = None) -> list[tuple[int, str]]:
    total_chars = sum(len(t) for _, t in pieces_with_speaker) or 1
    out: list[tuple[int, str]] = []
    for speaker_idx, text in pieces_with_speaker:
        alloc_dur = total_duration * (len(text) / total_chars)
        if alloc_dur <= 0:
            out.append((speaker_idx, text))
            continue
        cps = len(text) / alloc_dur

        # A piece can be well within the CPS (reading speed) limit yet still
        # sit on screen far too long if it's simply a long, slow-paced block
        # (e.g. a merged multi-speaker chunk with no [[SPEAKER]] markers --
        # exactly the case this max_cue_duration_sec branch exists for).
        # Without this, max_cue_duration_sec only clamped an *already-split*
        # piece's displayed duration in _allocate_timestamps and could never
        # trigger a split by itself, so a low-CPS/long-duration cue like that
        # silently stayed a single, very long cue.
        max_chars_options = []
        if cps_threshold and cps > cps_threshold:
            max_chars_options.append(max(1, math.floor(cps_threshold * alloc_dur)))
        if max_cue_duration_sec and alloc_dur > max_cue_duration_sec:
            local_rate = len(text) / alloc_dur  # this piece's own chars/sec
            max_chars_options.append(max(1, math.floor(local_rate * max_cue_duration_sec)))
        # CPS/duration alone can still leave a cue with more characters on
        # screen at once than fits comfortably (a short, dense, fast burst of
        # text reads "fine" by chars/sec but visually covers the screen).
        # This caps raw character count per cue directly, independent of
        # timing.
        if max_chars_per_cue and len(text) > max_chars_per_cue:
            max_chars_options.append(max_chars_per_cue)

        if not max_chars_options:
            out.append((speaker_idx, text))
            continue

        for sub in split_long_piece(text, min(max_chars_options)):
            out.append((speaker_idx, sub))
    return out


def _allocate_timestamps(final_pieces: list[tuple[int, str]], start_sec: float, end_sec: float,
                          min_cue_duration_sec: float | None, max_cue_duration_sec: float | None,
                          gap_sec: float) -> list[dict]:
    total_duration = max(0.0, end_sec - start_sec)
    total_chars = sum(len(t) for _, t in final_pieces) or 1
    n = len(final_pieces)

    raw_durations = [total_duration * (len(t) / total_chars) for _, t in final_pieces]
    if max_cue_duration_sec:
        raw_durations = [min(d, max_cue_duration_sec) for d in raw_durations]
    if min_cue_duration_sec:
        raw_durations = [max(d, min_cue_duration_sec) for d in raw_durations]
    # postprocessing.md SS11.4: "시간 배분은 근사치다" -- clamping to
    # min/max can make durations no longer sum to total_duration, so rescale
    # back to it. This is an approximation (bounds may drift slightly for
    # extreme outliers), consistent with the doc's stated limitation.
    duration_sum = sum(raw_durations)
    scale = total_duration / duration_sum if duration_sum > 0 else 1.0
    durations = [d * scale for d in raw_durations]

    cues = []
    cursor = start_sec
    for i, ((speaker_idx, text), dur) in enumerate(zip(final_pieces, durations)):
        piece_start = cursor
        piece_end = cursor + dur
        is_last = i == n - 1
        display_end = piece_end
        if not is_last and gap_sec > 0:
            display_end = max(piece_start + 0.01, piece_end - gap_sec)
        cues.append({
            "start_sec": piece_start,
            "end_sec": end_sec if is_last else display_end,
            "text": text,
            "speaker_idx": speaker_idx,
        })
        cursor = piece_end
    return cues


def split_by_speaker_cues(text: str, start_sec: float, end_sec: float,
                           gap_sec: float = DEFAULT_GAP_SEC, show_speaker_label: bool = False) -> list[dict]:
    """Step 1 entry point (postprocessing.md SS11.1 1차 분할, automatic Phase
    C). Splits only on [[SPEAKER]] markers -- no CPS-based length splitting,
    that is a separate manual step (`split_long_cue`) run later against a
    translated SRT. Returns a list of {start_sec, end_sec, text} dicts in
    chronological order (speaker_idx is cue-local per postprocessing.md
    SS10.1, dropped from the output unless show_speaker_label is on)."""
    text = text.strip()
    if not text:
        return []

    speaker_pieces = split_by_speaker(text)
    if not speaker_pieces:
        return []
    labeled = [(i + 1, p) for i, p in enumerate(speaker_pieces)]

    cues = _allocate_timestamps(labeled, start_sec, end_sec, None, None, gap_sec)
    for cue in cues:
        if show_speaker_label:
            cue["text"] = f"화자{cue['speaker_idx']}: {cue['text']}"
        del cue["speaker_idx"]
    return cues


def split_long_cue(text: str, start_sec: float, end_sec: float, cps_threshold: float,
                    min_cue_duration_sec: float | None = None, max_cue_duration_sec: float | None = None,
                    max_chars_per_cue: int | None = None, gap_sec: float = DEFAULT_GAP_SEC) -> list[dict]:
    """Step 2 entry point (postprocessing.md SS11.1 2차 분할, manual SRT
    post-processing tool). Takes one already-timed SRT cue (no [[SPEAKER]]
    markers expected -- the source has typically been through external
    translation by this point) and mechanically cuts it further wherever its
    CPS exceeds `cps_threshold`, its duration exceeds `max_cue_duration_sec`
    (catches long, slow-paced blocks that read fine CPS-wise but still sit on
    screen too long as one cue -- e.g. a merged multi-speaker chunk with no
    [[SPEAKER]] markers), OR its raw character count exceeds
    `max_chars_per_cue` (catches short, dense bursts that read fine by
    chars/sec but still visually cover the screen -- CPS/duration alone don't
    bound how much text sits on screen at once). Returns a list of
    {start_sec, end_sec, text} dicts in chronological order."""
    text = text.strip()
    if not text:
        return []

    total_duration = max(0.0, end_sec - start_sec)
    pieces = [(0, text)]
    final_pieces = (
        _apply_cps_split(pieces, total_duration, cps_threshold, max_cue_duration_sec, max_chars_per_cue)
        if (cps_threshold or max_cue_duration_sec or max_chars_per_cue) else pieces
    )
    if not final_pieces:
        return []

    cues = _allocate_timestamps(final_pieces, start_sec, end_sec, min_cue_duration_sec, max_cue_duration_sec, gap_sec)
    for cue in cues:
        del cue["speaker_idx"]
    return cues
