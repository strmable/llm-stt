"""Stage 3 (phase_a_roadmap.md): transcribe Stage 2c's chunks via llama-server (Qwen3-ASR).

Walks a job's manifest.json + chunk WAVs, sends each un-transcribed chunk to
llama-server's OpenAI-compatible endpoint (design.md SS10.1, same request
shape already validated in tools/test_transcribe.py / tools/eval_language_hint.py),
and writes each result to chunk_NNNN.txt next to its WAV (design.md SS13 --
so a single bad chunk can be re-heard/re-run without touching the rest).

Deliberately NOT implemented yet (out of scope for this stage; see design.md
SS17/SS5B.2): Context Carryover and Custom Vocabulary prompt injection. Both
require prompt wording that hasn't been validated against Qwen3-ASR's fixed
"language {Lang}<asr_text>{text}" output convention -- the only prompts
confirmed to work with it so far are a bare "Transcribe this audio." or a
bare "language: {code}" hint (TESTING.md SS4.1), so that's what's used here.

Usage:
    python pipeline/transcribe_chunks.py temp/3c6527d5
    python pipeline/transcribe_chunks.py temp/3c6527d5/manifest.json --language ko
"""

import argparse
import base64
import json
import re
import sys
import time
from pathlib import Path

import requests

from common import load_config
from server_manager import ensure_llama_server

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_SERVER = "http://localhost:8080"
DEFAULT_MODEL = "qwen3-asr"

# Flags any 2-20 char substring immediately repeated 5+ times in a row --
# a cheap heuristic for the "infinite repetition" failure mode design.md
# SS5A.6/TESTING.md call out, not a guarantee (real check is still the
# human "정성" pass over chunk_NNNN.txt next to chunk_NNNN.wav).
REPEAT_RE = re.compile(r"(.{2,20}?)\1{4,}", re.DOTALL)


def resolve_manifest_path(job_arg: Path) -> Path:
    if job_arg.is_dir():
        return job_arg / "manifest.json"
    return job_arg


def build_prompt(language: str) -> str:
    if language == "auto":
        return "Transcribe this audio."
    return f"language: {language}"


def parse_response(content: str) -> tuple[str, str]:
    """Qwen3-ASR's fixed response shape: "language {Lang}<asr_text>{text}"."""
    if "<asr_text>" in content:
        head, text = content.split("<asr_text>", 1)
        lang = head.replace("language", "").strip()
        return lang, text.strip()
    return "?", content.strip()


def detect_repetition(text: str) -> str | None:
    m = REPEAT_RE.search(text)
    return m.group(1) if m else None


def transcribe_one(server: str, model: str, prompt: str, wav_path: Path, temperature: float,
                    timeout: float) -> dict:
    audio_b64 = base64.b64encode(wav_path.read_bytes()).decode("ascii")
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
            ],
        }],
        "temperature": temperature, "top_p": 0.95, "top_k": 64, "max_tokens": 4096,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.time()
    resp = requests.post(f"{server.rstrip('/')}/v1/chat/completions", json=payload, timeout=timeout)
    elapsed = time.time() - t0
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return {"elapsed": elapsed, "raw": content}


def transcribe_all(todo: list[dict], job_dir: Path, server: str, model: str, prompt: str,
                    temperature: float, timeout: float) -> tuple[int, int, int]:
    n_ok = n_failed = n_flagged = 0
    for entry in todo:
        wav_path = job_dir / entry["file"]
        txt_path = wav_path.with_suffix(".txt")

        result = None
        last_error = None
        for attempt in range(2):  # design.md SS21: 1 retry on timeout/transient error, then fail
            try:
                result = transcribe_one(server, model, prompt, wav_path, temperature, timeout)
                break
            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt == 0:
                    print(f"  chunk {entry['id']:04d}: request failed ({e}), retrying once...")
                    time.sleep(1.0)

        if result is None:
            print(f"[ERROR] chunk {entry['id']:04d} failed after retry: {last_error}")
            entry["status"] = "failed"
            n_failed += 1
            continue

        lang, text = parse_response(result["raw"])
        txt_path.write_text(text, encoding="utf-8")

        repeat = detect_repetition(text)
        flag = f"  [WARNING: possible infinite repetition of {repeat!r}]" if repeat else ""
        if repeat:
            n_flagged += 1

        print(f"  chunk {entry['id']:04d}  {result['elapsed']:5.2f}s  lang={lang:8s}  "
              f"raw={result['raw'][:70]!r}{flag}")

        entry["status"] = "transcribed"
        entry["language_detected"] = lang
        entry["transcribe_elapsed_sec"] = round(result["elapsed"], 3)
        if repeat:
            entry["flags"] = ["possible_infinite_repetition"]
        n_ok += 1
    return n_ok, n_failed, n_flagged


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("job", type=Path, help="job directory (temp/{job_id}) or its manifest.json")
    ap.add_argument("--server", default=DEFAULT_SERVER, help="llama-server base URL")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="model field sent in the request body")
    ap.add_argument("--language", default="auto",
                     help="'auto' (default, no hint) or a language code e.g. 'ko' (design.md SS5A.8)")
    ap.add_argument("--temperature", type=float, default=1.0, help="design.md SS6.1 default")
    ap.add_argument("--timeout", type=float, default=120.0, help="per-request timeout in seconds")
    ap.add_argument("--force", action="store_true",
                     help="re-transcribe chunks already marked 'transcribed' (default: skip them, resume-style)")
    args = ap.parse_args()

    manifest_path = resolve_manifest_path(args.job)
    if not manifest_path.exists():
        sys.exit(f"manifest.json not found: {manifest_path} (run Stage 2c chunk_export.py first)")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    job_dir = manifest_path.parent
    prompt = build_prompt(args.language)
    print(f"[transcribe] server={args.server} model={args.model} language={args.language} "
          f"prompt={prompt!r}")

    chunks = manifest["chunks"]
    todo = [c for c in chunks if args.force or c["status"] != "transcribed"]
    print(f"[transcribe] {len(todo)}/{len(chunks)} chunk(s) to process "
          f"({'--force: re-transcribing all' if args.force else 'skipping already-transcribed'})")

    config = load_config()
    # design.md SS6.3: server lifecycle is scoped to "Phase B" -- started (if
    # managed) right before this loop, torn down right after, not held for
    # the whole run_transcript.py invocation.
    with ensure_llama_server(args.server, config, log_path=job_dir / "llama-server.log"):
        n_ok, n_failed, n_flagged = transcribe_all(
            todo, job_dir, args.server, args.model, prompt, args.temperature, args.timeout,
        )

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[summary] transcribed={n_ok}  failed={n_failed}  flagged_for_review={n_flagged}  "
          f"(manifest updated: {manifest_path})")
    if n_flagged:
        print("[summary] review flagged chunks' .txt next to their .wav for hallucination/repetition "
              "(roadmap Stage 3 정성 검증)")


if __name__ == "__main__":
    main()
