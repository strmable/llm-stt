"""Phase C step 1 (postprocessing.md SS6/SS8): full-context LLM correction.

Sends the whole raw transcript as a fixed prefix + one chunk's raw text as
the variable suffix, once per transcribed chunk, to a text-instruct
llama-server (a *different* server/model than the ASR one -- design.md
SS5B.3 "STT 모델과 텍스트 교정 모델을 동시에 상주시키지 않는다"). The model
returns the corrected text for that one chunk, optionally with `[[SPEAKER]]`
markers where it detects a speaker change inside the chunk (postprocessing.md
SS10). Cue splitting on those markers is a separate, deterministic step
(cue_splitter.py) -- this module never touches timestamps.

Usage:
    python pipeline/text_correction.py temp/3c6527d5
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

from common import load_config
from server_manager import adapt_text_correction_server_config, ensure_llama_server

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SPEAKER_MARKER = "[[SPEAKER]]"

# postprocessing.md SS8.1 (system prompt) + SS8.2 (few-shot examples), verbatim.
SYSTEM_PROMPT = """You are an Automatic Speech Recognition (ASR) post-processing assistant.

Your task is to recover the speaker's intended utterance from an ASR transcript.
You are given the FULL raw ASR transcript inside <FULL_TRANSCRIPT> tags for
reference only, and the current segment to correct inside <CURRENT> tags.

The current segment may contain recognition errors such as:
- homophone substitutions
- incorrect proper nouns (transliteration / character errors)
- duplicated words or phrases
- obvious character mistakes
- minor punctuation or spacing errors

Rules:
1. Correct only clear ASR recognition errors in the <CURRENT> segment.
2. Use <FULL_TRANSCRIPT> only to disambiguate homophones and proper nouns and to
   keep terminology/proper-noun spelling consistent across the whole document.
3. Preserve the speaker's intended message, style, and tone.
4. Do NOT rewrite, summarize, paraphrase, or improve the sentence.
5. Do NOT add information that was not spoken.
6. If multiple valid interpretations exist and context cannot decide, keep the original text unchanged.
7. NEVER modify or output the transcript context. It is for understanding only.
8. Correct ONLY the current segment.
9. If (and only if) the current segment clearly contains an exchange between
   different speakers (e.g. a question immediately followed by its answer),
   insert the marker [[SPEAKER]] at each point where the speaker changes.
   Do NOT number or name speakers. Do NOT insert the marker when a single
   speaker is talking continuously. When in doubt, do NOT insert it.
10. Output ONLY the corrected current segment wrapped in <OUTPUT> tags. No
    explanations, no comments, no markdown, no quotation marks, no numbering.

The transcript may be in Japanese, Chinese, Korean, or a mixture of these.
Apply each language's normal writing conventions only when needed to fix an obvious ASR error.

Be conservative. When uncertain, leave the original text unchanged and insert no markers.

Examples:

Example 1 (Japanese -- proper noun corrected via document context, katakana transliteration error):
<CURRENT>
新しいベルサージのバッグを買いました。
</CURRENT>
<OUTPUT>
新しいヴェルサーチのバッグを買いました。
</OUTPUT>

Example 2 (Chinese -- homophone-like proper noun corrected via context, hanzi substitution error):
<CURRENT>
第一站是克隆,然后去巴黎。
</CURRENT>
<OUTPUT>
第一站是科隆,然后去巴黎。
</OUTPUT>

Example 3 (Korean -- grammatically implausible word replaced with the word that actually fits):
<CURRENT>
아무리 빨라도 3개월은 조기 걸린다고 하더라고요.
</CURRENT>
<OUTPUT>
아무리 빨라도 3개월은 족히 걸린다고 하더라고요.
</OUTPUT>

Example 4 (Korean -- real word confused with another real word, resolved by sentence-level meaning):
<CURRENT>
저희가 경쟁사를 시장에서 보란했음을 강조했어요.
</CURRENT>
<OUTPUT>
저희가 경쟁사를 시장에서 몰아냈음을 강조했어요.
</OUTPUT>

Example 5 (Korean -- two equally valid real words, context insufficient -> leave unchanged):
<CURRENT>
네, 그런데 요즘 식욕이 무척 늘어서 걱정이에요.
</CURRENT>
<OUTPUT>
네, 그런데 요즘 식욕이 무척 늘어서 걱정이에요.
</OUTPUT>

Example 6 (Korean -- a question and its answer merged into one segment -> insert speaker-change marker):
<CURRENT>
이거 얼마예요 오천원입니다
</CURRENT>
<OUTPUT>
이거 얼마예요?[[SPEAKER]]오천원입니다.
</OUTPUT>

Example 7 (Korean -- single speaker talking continuously, no speaker change -> do NOT insert marker):
<CURRENT>
그래서 제가 어제 시장에 갔는데 사람이 정말 많더라고요 결국 아무것도 못 샀어요
</CURRENT>
<OUTPUT>
그래서 제가 어제 시장에 갔는데 사람이 정말 많더라고요. 결국 아무것도 못 샀어요.
</OUTPUT>

Do not copy these examples' content. They illustrate the correction and segmentation style only."""

OUTPUT_RE = re.compile(r"<OUTPUT>(.*?)</OUTPUT>", re.DOTALL)


def resolve_manifest_path(job_arg: Path) -> Path:
    if job_arg.is_dir():
        return job_arg / "manifest.json"
    return job_arg


def raw_text_for(job_dir: Path, chunk: dict) -> str:
    txt_path = (job_dir / chunk["file"]).with_suffix(".txt")
    if not txt_path.exists():
        return ""
    return " ".join(txt_path.read_text(encoding="utf-8").split())


def fixed_path_for(job_dir: Path, chunk: dict) -> Path:
    return (job_dir / chunk["file"]).with_suffix(".fixed.txt")


def correctable_chunks(manifest: dict, job_dir: Path) -> list[dict]:
    """Transcribed chunks with non-empty raw text -- the units full-context
    correction runs over. Failed/empty chunks pass through untouched (there
    is nothing to correct, and cue_splitter.py never sees them)."""
    out = []
    for chunk in manifest["chunks"]:
        if chunk["status"] != "transcribed":
            continue
        if raw_text_for(job_dir, chunk):
            out.append(chunk)
    return out


def split_into_segments(chunks: list[dict], job_dir: Path, max_segment_chars: int,
                         segment_split_count: int) -> list[list[dict]]:
    """postprocessing.md SS6 "긴 파일 처리": documents under max_segment_chars
    get a single full-context group; longer ones are cut into
    `segment_split_count` contiguous groups (chunk order preserved), each
    using only its own group's raw text as the fixed <FULL_TRANSCRIPT>
    context -- chronological order means nearby chunks (where cross-reference
    matters most) stay grouped together."""
    lengths = [len(raw_text_for(job_dir, c)) for c in chunks]
    total = sum(lengths)
    if total <= max_segment_chars or len(chunks) <= 1:
        return [chunks]

    target = total / segment_split_count
    groups: list[list[dict]] = []
    current: list[dict] = []
    current_len = 0
    for chunk, length in zip(chunks, lengths):
        if current and current_len >= target and len(groups) < segment_split_count - 1:
            groups.append(current)
            current, current_len = [], 0
        current.append(chunk)
        current_len += length
    if current:
        groups.append(current)
    return groups


def build_full_transcript_block(group: list[dict], job_dir: Path) -> str:
    lines = [f"{c['id']} {raw_text_for(job_dir, c)}" for c in group]
    return "<FULL_TRANSCRIPT>\n" + "\n".join(lines) + "\n</FULL_TRANSCRIPT>"


def build_user_message(full_transcript_block: str, chunk_id: int, raw_text: str) -> str:
    return f"{full_transcript_block}\n\n<CURRENT>\n{chunk_id} {raw_text}\n</CURRENT>"


def parse_output(raw: str, fallback: str) -> str:
    m = OUTPUT_RE.search(raw)
    if not m:
        return fallback
    text = m.group(1).strip()
    return text if text else fallback


def correct_one(url: str, full_transcript_block: str, chunk_id: int, raw_text: str,
                 sampling: dict, timeout: float) -> dict:
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(full_transcript_block, chunk_id, raw_text)},
        ],
        "temperature": sampling.get("temperature", 0.25),
        "top_p": sampling.get("top_p", 0.8),
        "top_k": sampling.get("top_k", 20),
        "presence_penalty": sampling.get("presence_penalty", 1.0),
        "repetition_penalty": sampling.get("repetition_penalty", 1.0),
        "max_tokens": sampling.get("max_tokens", 512),
        "stop": ["</OUTPUT>"],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.time()
    resp = requests.post(url, json=payload, timeout=timeout)
    elapsed = time.time() - t0
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    text = parse_output(content, fallback=raw_text)
    return {"elapsed": elapsed, "raw": content, "text": text}


def correct_all(manifest: dict, job_dir: Path, tc_cfg: dict, url: str,
                 log=print, should_stop=lambda: False,
                 on_progress=lambda done, total: None) -> None:
    """Writes each corrected chunk's text to chunk_NNNN.fixed.txt next to its
    raw chunk_NNNN.txt (design.md SS5B.3/postprocessing.md SS5: keep both so
    a bad correction can be compared/rolled back). Resume-safe: a chunk whose
    .fixed.txt already exists is skipped."""
    chunks = correctable_chunks(manifest, job_dir)
    todo = [c for c in chunks if not fixed_path_for(job_dir, c).exists()]
    total = len(chunks)
    done = total - len(todo)
    on_progress(done, total)
    if not todo:
        return

    fc_cfg = tc_cfg.get("full_context", {})
    sampling = tc_cfg.get("sampling", {})
    groups = split_into_segments(chunks, job_dir, fc_cfg.get("max_segment_chars", 60000),
                                  fc_cfg.get("segment_split_count", 3))
    if len(groups) > 1:
        log(f"[text_correction] document split into {len(groups)} full-context segment(s) "
            f"(postprocessing.md SS6 긴 파일 처리)")

    todo_ids = {c["id"] for c in todo}
    for group in groups:
        group_todo = [c for c in group if c["id"] in todo_ids]
        if not group_todo:
            continue
        full_transcript_block = build_full_transcript_block(group, job_dir)
        for chunk in group_todo:
            if should_stop():
                return
            raw_text = raw_text_for(job_dir, chunk)
            result = correct_one(url, full_transcript_block, chunk["id"], raw_text, sampling, timeout=120.0)
            fixed_path_for(job_dir, chunk).write_text(result["text"], encoding="utf-8")
            marker_note = " [+SPEAKER]" if SPEAKER_MARKER in result["text"] else ""
            log(f"[text_correction] chunk {chunk['id']:04d}  {result['elapsed']:5.2f}s{marker_note}  "
                f"{result['text'][:70]!r}")
            done += 1
            on_progress(done, total)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("job", type=Path, help="job directory (temp/{job_id}) or its manifest.json")
    args = ap.parse_args()

    manifest_path = resolve_manifest_path(args.job)
    if not manifest_path.exists():
        sys.exit(f"manifest.json not found: {manifest_path} (run Phase B / transcribe_chunks.py first)")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    job_dir = manifest_path.parent

    config = load_config()
    tc_cfg = config.get("text_enhancement", {}).get("text_correction", {})
    if not tc_cfg.get("enabled", False):
        sys.exit("text_enhancement.text_correction.enabled is false in config.json -- nothing to do")

    server_url = tc_cfg.get("server", {}).get(
        "url", "http://localhost:8081/v1/chat/completions")
    server_base = server_url.split("/v1/")[0]

    tc_config_for_server = adapt_text_correction_server_config(tc_cfg.get("server", {}))
    with ensure_llama_server(server_base, tc_config_for_server, log_path=job_dir / "llama-server-tc.log"):
        correct_all(manifest, job_dir, tc_cfg, server_url)

    n_fixed = sum(1 for c in correctable_chunks(manifest, job_dir) if fixed_path_for(job_dir, c).exists())
    print(f"\n[summary] {n_fixed} chunk(s) corrected (see chunks/*.fixed.txt)")


if __name__ == "__main__":
    main()
