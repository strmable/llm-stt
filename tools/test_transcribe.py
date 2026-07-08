"""Manual smoke test for local llama-server STT backends (design.md SS10.1).

Usage:
    python tools/test_transcribe.py path/to/audio.(wav|mp3|m4a|...)
    python tools/test_transcribe.py path/to/audio.wav --server http://localhost:8080 --model qwen3-asr
"""

import argparse
import base64
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

DEFAULT_PROMPT = "Transcribe this audio."


def to_16k_mono_wav(src: Path) -> Path:
    tmp = Path(tempfile.mktemp(suffix=".wav"))
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-ac", "1", "-ar", "16000",
        str(tmp),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"ffmpeg conversion failed for {src}")
    return tmp


def transcribe(server: str, model: str, prompt: str, wav_path: Path) -> dict:
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
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "max_tokens": 4096,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    url = server.rstrip("/") + "/v1/chat/completions"
    t0 = time.time()
    resp = requests.post(url, json=payload, timeout=120)
    elapsed = time.time() - t0
    resp.raise_for_status()
    data = resp.json()
    return {"elapsed": elapsed, "response": data}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audio", type=Path, help="Input audio file (any ffmpeg-readable format)")
    ap.add_argument("--server", default="http://localhost:8080", help="llama-server base URL")
    ap.add_argument("--model", default="qwen3-asr", help="model field sent in the request body")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="text prompt sent alongside the audio")
    ap.add_argument("--keep-wav", action="store_true", help="don't delete the converted temp WAV")
    args = ap.parse_args()

    if not args.audio.exists():
        sys.exit(f"File not found: {args.audio}")

    print(f"[1/3] Converting to 16kHz mono WAV: {args.audio}")
    wav_path = to_16k_mono_wav(args.audio)
    print(f"      -> {wav_path} ({wav_path.stat().st_size / 1024:.1f} KB)")

    print(f"[2/3] Sending request to {args.server} (model={args.model})")
    try:
        result = transcribe(args.server, args.model, args.prompt, wav_path)
    finally:
        if not args.keep_wav:
            wav_path.unlink(missing_ok=True)

    print(f"[3/3] Done in {result['elapsed']:.2f}s\n")
    try:
        text = result["response"]["choices"][0]["message"]["content"]
        print("--- Transcription ---")
        print(text)
    except (KeyError, IndexError):
        print("--- Raw response (unexpected shape) ---")
        print(result["response"])


if __name__ == "__main__":
    main()
