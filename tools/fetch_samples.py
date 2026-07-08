"""Fetch a handful of labeled CJK speech samples for manual STT testing.

Pulls individual rows (audio + ground-truth transcription) from the
google/fleurs dataset via HF's datasets-server API, without downloading
the full dataset. Audio files are written locally (gitignored, not
committed) alongside a manifest.jsonl with the reference transcriptions
needed for CER comparison (design.md SS5A.4).

Usage:
    python tools/fetch_samples.py --lang ko ja zh --count 5
    python tools/fetch_samples.py --lang ko --count 10 --out samples/ko
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests

# Windows consoles are often cp949/cp1252 etc. and will raise UnicodeEncodeError
# on Korean/Japanese/Chinese transcriptions otherwise.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LANG_ALIASES = {
    "ko": "ko_kr",
    "ja": "ja_jp",
    "zh": "cmn_hans_cn",
}

ROWS_API = "https://datasets-server.huggingface.co/rows"


def fetch_rows(config: str, split: str, count: int, offset: int = 0, retries: int = 3) -> list[dict]:
    params = {
        "dataset": "google/fleurs",
        "config": config,
        "split": split,
        "offset": offset,
        "length": count,
    }
    last_err = None
    for attempt in range(retries):
        try:
            # First request for a rarely-queried config/split can be slow while
            # datasets-server builds its cache server-side; allow a long timeout
            # and retry transient 5xx errors with backoff.
            resp = requests.get(ROWS_API, params=params, timeout=120)
            resp.raise_for_status()
            return resp.json()["rows"]
        except (requests.HTTPError, requests.Timeout) as e:
            last_err = e
            if attempt < retries - 1:
                wait = 10 * (attempt + 1)
                print(f"  retrying in {wait}s after error: {e}", file=sys.stderr)
                time.sleep(wait)
    raise last_err


def download_sample(row: dict, out_dir: Path, lang: str, idx: int) -> dict:
    audio_url = row["row"]["audio"][0]["src"]
    audio_bytes = requests.get(audio_url, timeout=60).content

    wav_path = out_dir / f"{lang}_{idx:03d}.wav"
    wav_path.write_bytes(audio_bytes)

    return {
        "file": wav_path.name,
        "lang": lang,
        "transcription": row["row"]["transcription"],
        "raw_transcription": row["row"]["raw_transcription"],
        "num_samples": row["row"]["num_samples"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", nargs="+", default=["ko", "ja", "zh"],
                     choices=sorted(LANG_ALIASES), help="languages to fetch")
    ap.add_argument("--count", type=int, default=5, help="samples per language")
    ap.add_argument("--offset", type=int, default=0, help="row offset in the split")
    ap.add_argument("--split", default="validation", choices=["validation", "test", "train"],
                     help="HF datasets-server has a 300MB single-row-group scan limit; "
                          "ja_jp/cmn_hans_cn 'test' parquet shards exceed it, so 'validation' "
                          "(the smallest shard for all three languages) is the default")
    ap.add_argument("--out", type=Path, default=Path("samples"), help="output directory")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.jsonl"

    # Keep entries for languages not being (re-)fetched in this run, so running
    # the script once per language doesn't clobber earlier results.
    entries = []
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as f:
            entries = [json.loads(line) for line in f if line.strip()]
        entries = [e for e in entries if e["lang"] not in args.lang]

    for lang in args.lang:
        config = LANG_ALIASES[lang]
        print(f"[{lang}] fetching {args.count} rows from google/fleurs ({config}, {args.split})...")
        try:
            rows = fetch_rows(config, args.split, args.count, args.offset)
        except requests.HTTPError as e:
            print(f"  failed: {e}", file=sys.stderr)
            continue

        for i, row in enumerate(rows):
            entry = download_sample(row, args.out, lang, i)
            entries.append(entry)
            print(f"  -> {entry['file']}: {entry['transcription'][:40]}...")

    with manifest_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(entries)} samples + manifest to {args.out}/")


if __name__ == "__main__":
    main()
