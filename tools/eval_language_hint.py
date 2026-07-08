"""Compare STT accuracy with vs. without an explicit language hint.

For each sample in samples/manifest.jsonl (see tools/fetch_samples.py),
sends two requests to the local llama-server: one with a plain prompt
("auto", relies on the model's own language detection) and one with a
"language: {code}" hint (design.md SS5A.8 / SS8 "출력 언어 설정"). Reports
per-sample and average CER against the ground-truth transcription, plus
language-misdetection counts.

Usage:
    python tools/eval_language_hint.py
    python tools/eval_language_hint.py --samples-dir samples --temperature 0
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LANG_NAMES = {"ko": "Korean", "ja": "Japanese", "zh": "Chinese"}


def cer(ref: str, hyp: str) -> float:
    ref = ref.replace(" ", "")
    hyp = hyp.replace(" ", "")
    n, m = len(ref), len(hyp)
    if n == 0:
        return 0.0 if m == 0 else 1.0
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            tmp = dp[j]
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = tmp
    return dp[m] / n


def ask(server: str, prompt: str, wav_path: Path, temperature: float) -> str:
    audio_b64 = base64.b64encode(wav_path.read_bytes()).decode("ascii")
    payload = {
        "model": "qwen3-asr",
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
    r = requests.post(f"{server}/v1/chat/completions", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def parse_response(content: str):
    # Qwen3-ASR's fixed response shape: "language {Lang}<asr_text>{text}"
    if "<asr_text>" in content:
        head, text = content.split("<asr_text>", 1)
        lang = head.replace("language", "").strip()
        return lang, text.strip()
    return "?", content.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples-dir", type=Path, default=Path("samples"),
                     help="directory with manifest.jsonl + wav files (see tools/fetch_samples.py)")
    ap.add_argument("--server", default="http://localhost:8080", help="llama-server base URL")
    ap.add_argument("--temperature", type=float, default=0.0,
                     help="0 for deterministic output so the auto/hinted comparison isn't "
                          "confounded by sampling noise (design.md default is 1.0 for normal use)")
    args = ap.parse_args()

    manifest_path = args.samples_dir / "manifest.jsonl"
    manifest = [json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines()]
    results = []

    for entry in manifest:
        lang = entry["lang"]
        wav_path = args.samples_dir / entry["file"]
        ref = entry["transcription"]

        print(f"=== {entry['file']} (true lang={lang}) ===")

        auto_resp = ask(args.server, "Transcribe this audio.", wav_path, args.temperature)
        auto_lang, auto_text = parse_response(auto_resp)
        auto_cer = cer(ref, auto_text)
        print(f"  [auto]   detected={auto_lang!r:10} cer={auto_cer:.3f}  text={auto_text[:60]}")

        hinted_resp = ask(args.server, f"language: {lang}", wav_path, args.temperature)
        hinted_lang, hinted_text = parse_response(hinted_resp)
        hinted_cer = cer(ref, hinted_text)
        print(f"  [hinted] detected={hinted_lang!r:10} cer={hinted_cer:.3f}  text={hinted_text[:60]}")

        results.append({
            "file": entry["file"], "lang": lang, "ref": ref,
            "auto_lang": auto_lang, "auto_text": auto_text, "auto_cer": auto_cer,
            "hinted_lang": hinted_lang, "hinted_text": hinted_text, "hinted_cer": hinted_cer,
        })
        print()

    out_path = args.samples_dir / "language_hint_comparison.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    n = len(results)
    avg_auto = sum(r["auto_cer"] for r in results) / n
    avg_hinted = sum(r["hinted_cer"] for r in results) / n
    lang_mismatches_auto = sum(1 for r in results if LANG_NAMES[r["lang"]].lower() != r["auto_lang"].lower())
    lang_mismatches_hinted = sum(1 for r in results if LANG_NAMES[r["lang"]].lower() != r["hinted_lang"].lower())

    print("=== Summary ===")
    print(f"Average CER  auto={avg_auto:.3f}  hinted={avg_hinted:.3f}")
    print(f"Language misdetections  auto={lang_mismatches_auto}/{n}  hinted={lang_mismatches_hinted}/{n}")


if __name__ == "__main__":
    main()
