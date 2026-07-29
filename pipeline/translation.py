"""번역(Translation) 탭 지원 로직 (design.md SS5C, v3.4).

Pure marker-protocol + batching + merge logic (no Qt), used by
gui/translation_tab.py the same way pipeline/text_correction.py's
correct_one()/correct_all() are used by gui/worker.py -- this module never
touches the llama-server lifecycle itself (that's the caller's job, via
server_manager.ensure_llama_server()).

SS5C.3 `[nnnn]|` serial marker protocol: each SRT cue becomes one
`[nnnn]|text` block (nnnn = 1-based cue index, zero-padded to 4 digits). The
marker only prefixes a cue's first line; a multi-line cue's remaining lines
follow with their original newlines, up to the next marker.

SS5C.5 parsing is deliberately newline-independent (external tools like
DeepL are documented to sometimes insert spaces inside a marker's digits, or
drop the blank line between blocks): it locates markers by regex match
position, not line boundaries, and takes each cue's content as "everything
between this marker and the next one".
"""

import re

import requests

MARKER_RE = re.compile(r"\[\s*(?:\d\s*){4,}\]\s*\|")
_DIGITS_RE = re.compile(r"\d")


def _numbered(cues: list[dict]) -> list[tuple[int, dict]]:
    return list(enumerate(cues, 1))


def render_markers(numbered_cues: list[tuple[int, dict]]) -> str:
    return "\n".join(f"[{serial:04d}]|{cue['text']}" for serial, cue in numbered_cues)


def build_markers(cues: list[dict]) -> str:
    """SS5C.3 -- the marker-tagged plain text that becomes the 번역 탭's
    좌측(원문) textbox content."""
    return render_markers(_numbered(cues))


def render_translated_map(translated_map: dict[int, str]) -> str:
    """Inverse-ish of parse_markers(): renders a {serial: text} map back into
    `[nnnn]|text` marker text, ordered by serial -- used to fill the 번역 탭's
    우측(번역문) textbox after Translate finishes (SS5C.4)."""
    return render_markers([(serial, {"text": text}) for serial, text in sorted(translated_map.items())])


def parse_markers(text: str) -> dict[int, str]:
    """SS5C.5 -- robust, newline-independent marker parsing. Matches on
    marker *positions*, not line breaks, so it survives both documented
    external-tool mangling patterns: digits split by inserted spaces inside
    the brackets (`[00 07]|`), and a lost blank line between blocks.
    Returns {serial: trimmed cue text}; internal newlines within a cue are
    preserved, only leading/trailing whitespace is stripped."""
    matches = list(MARKER_RE.finditer(text))
    result: dict[int, str] = {}
    for idx, m in enumerate(matches):
        serial = int("".join(_DIGITS_RE.findall(m.group(0))))
        content_start = m.end()
        content_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        result[serial] = text[content_start:content_end].strip()
    return result


def make_batches(cues: list[dict], max_chars: int) -> list[list[tuple[int, dict]]]:
    """SS5C.6 -- accumulates (serial, cue) pairs in order until adding the
    next would exceed `max_chars`, never splitting a cue's marker+text
    across batches. A single cue that alone exceeds `max_chars` is still
    sent, alone, in its own batch (documented exception -- postprocessing.md
    SS6 "긴 파일 처리" precedent)."""
    batches: list[list[tuple[int, dict]]] = []
    current: list[tuple[int, dict]] = []
    current_len = 0
    for serial, cue in _numbered(cues):
        piece_len = len(f"[{serial:04d}]|{cue['text']}") + 1  # +1 for the joining "\n"
        if current and current_len + piece_len > max_chars:
            batches.append(current)
            current, current_len = [], 0
        current.append((serial, cue))
        current_len += piece_len
    if current:
        batches.append(current)
    return batches


def merge_translation(cues: list[dict], translated_map: dict[int, str]) -> tuple[list[dict], list[int]]:
    """SS5C.7 -- replaces each cue's text with translated_map[serial] when
    present (serial = 1-based original-order index), leaves any cue with no
    matching serial untouched rather than dropping it (silent-failure-
    forbidden requirement), and returns the list of missing serials so the
    caller can log/report them."""
    merged: list[dict] = []
    missing: list[int] = []
    for serial, cue in _numbered(cues):
        if serial in translated_map:
            merged.append({**cue, "text": translated_map[serial]})
        else:
            merged.append(dict(cue))
            missing.append(serial)
    return merged, missing


def build_translation_prompt(template: str, target_language: str, vocabulary: list[str]) -> str:
    """SS5C.6/SS8.7 template variables: {{target_language}} and {{vocabulary}}
    (the latter reused from config-stt.json's text_enhancement.custom_vocabulary,
    same block format as gui/worker.py's build_prompt() {{vocabulary}})."""
    vocabulary_block = ""
    if vocabulary:
        terms = "\n".join(f"- {v}" for v in vocabulary if v.strip())
        if terms:
            vocabulary_block = f"Vocabulary (proper nouns/terms that may appear):\n{terms}\n\n"
    return template.replace("{{target_language}}", target_language).replace("{{vocabulary}}", vocabulary_block)


def call_translation_llm(url: str, system_prompt: str, batch_markers: str, timeout: float = 120.0) -> str:
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": batch_markers},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def run_batches(cues: list[dict], prompt_template: str, max_chars: int, url: str,
                 target_language: str = "", vocabulary: list[str] | None = None,
                 log=print, should_stop=lambda: False,
                 on_progress=lambda done, total: None) -> dict[int, str]:
    """Shared batch-processing loop for both Preprocess and Translate
    (design.md SS5C.6 -- the two buttons differ only in which prompt
    template and which pane's cues they run over, per SS5C.4). Returns the
    accumulated {serial: text} map across all batches (partial if stopped
    early), ready for the caller to fill a textbox or run merge_translation()."""
    system_prompt = build_translation_prompt(prompt_template, target_language, vocabulary or [])
    batches = make_batches(cues, max_chars)
    total = len(batches)
    result: dict[int, str] = {}
    for done, batch in enumerate(batches, 1):
        if should_stop():
            break
        response = call_translation_llm(url, system_prompt, render_markers(batch))
        result.update(parse_markers(response))
        log(f"[translation] batch {done}/{total} ({len(batch)} cue(s)) done")
        on_progress(done, total)
    return result
