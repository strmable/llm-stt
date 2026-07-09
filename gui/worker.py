"""Worker thread driving Phase A/B (design.md SS11/SS19) for the GUI.

Reuses the same functions the independently-runnable pipeline/*.py stage
scripts use (see phase_a_roadmap.md) instead of shelling out to them, so
progress/cancellation/SRT updates can be reported chunk-by-chunk via Qt
signals (design.md SS19: Worker Thread does the work, GUI Thread only
touches widgets in response to signals).

Two things this worker does beyond what the CLI stage scripts currently do
(they intentionally defer these -- see transcribe_chunks.py's docstring):
  - Context Carryover (design.md SS17) and Custom Vocabulary (SS5B.2) via
    config.json's prompt.template + {{context}}/{{language_hint}}/{{vocabulary}}.
  - Failed chunks render as a "[TRANSCRIPTION FAILED]" SRT placeholder
    (design.md SS21) instead of being silently dropped.
"""

import base64
import datetime
import json
import shutil
import sys
import time
from pathlib import Path

import requests
import soundfile as sf
from PySide6.QtCore import QThread, Signal

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from build_srt import srt_timestamp  # noqa: E402
from chunk_export import export_chunks, preserve_prior_transcriptions, validate_manifest  # noqa: E402
from common import job_dir as get_job_dir, read_source_info, write_source_info  # noqa: E402
from server_manager import ensure_llama_server  # noqa: E402
from transcribe_chunks import detect_repetition, parse_response  # noqa: E402
from vad_merge import merge_pipeline  # noqa: E402
from vad_raw_test import SAMPLE_RATE, ensure_extracted_wav, raw_segments, run_vad  # noqa: E402

MAX_CONSECUTIVE_FAILURES = 5  # design.md SS21
FAILED_PLACEHOLDER = "[TRANSCRIPTION FAILED]"


def build_prompt(template: str, context: str, language: str, vocabulary: list[str]) -> str:
    """design.md SS8 Template Variables."""
    language_hint = f"language: {language}\n\n" if language and language != "auto" else ""
    vocabulary_block = ""
    if vocabulary:
        terms = "\n".join(f"- {v}" for v in vocabulary if v.strip())
        if terms:
            vocabulary_block = f"Vocabulary (proper nouns/terms that may appear):\n{terms}\n\n"
    return (template.replace("{{language_hint}}", language_hint)
                     .replace("{{vocabulary}}", vocabulary_block)
                     .replace("{{context}}", context))


def transcribe_one_local(url: str, model: str, prompt: str, wav_path: Path, llm_cfg: dict,
                          disable_thinking: bool, timeout: float) -> dict:
    """design.md SS10.1 -- OpenAI-compatible input_audio content type.

    `url` is the full endpoint (config.json local_api.url, e.g.
    "http://host:port/v1/chat/completions") -- unlike transcribe_chunks.py's
    --server (a bare base URL it appends "/v1/chat/completions" to itself).
    """
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
        "temperature": llm_cfg.get("temperature", 1.0),
        "top_p": llm_cfg.get("top_p", 0.95),
        "top_k": llm_cfg.get("top_k", 64),
        "max_tokens": llm_cfg.get("max_tokens", 4096),
    }
    if disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    t0 = time.time()
    resp = requests.post(url, json=payload, timeout=timeout)
    elapsed = time.time() - t0
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return {"elapsed": elapsed, "raw": content}


def transcribe_one_gemini(api_key: str, model: str, prompt: str, wav_path: Path, llm_cfg: dict,
                           timeout: float) -> dict:
    """design.md SS10.2 -- generateContent inlineData."""
    audio_b64 = base64.b64encode(wav_path.read_bytes()).decode("ascii")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": "audio/wav", "data": audio_b64}},
        ]}],
        "generationConfig": {
            "temperature": llm_cfg.get("temperature", 1.0),
            "topP": llm_cfg.get("top_p", 0.95),
            "topK": llm_cfg.get("top_k", 64),
            "maxOutputTokens": llm_cfg.get("max_tokens", 4096),
        },
    }
    t0 = time.time()
    resp = requests.post(url, json=payload, timeout=timeout)
    elapsed = time.time() - t0
    resp.raise_for_status()
    text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    return {"elapsed": elapsed, "raw": text}


def render_srt_with_placeholders(manifest: dict, job_dir_path: Path) -> str:
    """Like build_srt.py's collect_entries+render_srt, but failed chunks get a
    "[TRANSCRIPTION FAILED]" placeholder entry instead of being dropped
    (design.md SS21), so users see gaps as they happen rather than a
    silently-shorter SRT."""
    blocks = []
    i = 0
    prev_end = -1.0
    for chunk in manifest["chunks"]:
        if chunk["status"] == "failed":
            text = FAILED_PLACEHOLDER
        elif chunk["status"] == "transcribed":
            txt_path = (job_dir_path / chunk["file"]).with_suffix(".txt")
            text = txt_path.read_text(encoding="utf-8").strip() if txt_path.exists() else ""
            text = " ".join(text.split())
            if not text:
                continue
        else:
            continue  # pending/vad_extracted -- not reached yet

        start_sec, end_sec = chunk["start_sec"], chunk["end_sec"]
        if start_sec < prev_end - 1e-6:
            continue  # shouldn't happen (chunks are already time-ordered), guard anyway
        prev_end = end_sec
        i += 1
        blocks.append(f"{i}\n{srt_timestamp(start_sec)} --> {srt_timestamp(end_sec)}\n{text}\n")
    return "\n".join(blocks)


class TranscriptionWorker(QThread):
    phaseChanged = Signal(str)
    progressChanged = Signal(int, int)  # (done, total) chunks, Phase B only
    srtUpdated = Signal(str)  # full current SRT text (design.md SS7 SRT Output TextBox)
    logMessage = Signal(str)
    jobFinished = Signal(str)  # output srt path
    jobFailed = Signal(str)
    jobStopped = Signal()

    def __init__(self, source: Path, config: dict, resume: bool, parent=None):
        super().__init__(parent)
        self.source = source
        self.config = config
        self.resume = resume
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        try:
            self._run()
        except Exception as e:  # noqa: BLE001 -- surfaced to the GUI, not swallowed
            self.jobFailed.emit(str(e))

    # -- Phase A -----------------------------------------------------------

    def _phase_a(self, job_dir_path: Path) -> dict:
        self.phaseChanged.emit("Phase A: 오디오 추출 중...")
        self.logMessage.emit(f"[INFO] Loading file: {self.source}")
        wav_path = ensure_extracted_wav(self.source)
        assert wav_path.parent == job_dir_path
        # ensure_extracted_wav() (vad_raw_test.py) calls extract_audio.extract()
        # directly rather than its main(), which is the only thing that writes
        # source_info.json -- write it here so read_source_info() below (and
        # any resumed run) always finds it, matching extract_audio.py's own
        # standalone behavior.
        write_source_info(job_dir_path, self.source)

        vad_cfg = self.config.get("vad", {})
        samples, sr = sf.read(str(wav_path), dtype="float32")
        assert sr == SAMPLE_RATE, f"expected {SAMPLE_RATE}Hz wav, got {sr}"
        total_duration = len(samples) / SAMPLE_RATE

        self.phaseChanged.emit("Phase A: VAD 분석 중...")
        self.logMessage.emit("[INFO] Starting VAD")
        probs = run_vad(samples)
        raw = raw_segments(probs, vad_cfg.get("threshold", 0.5))
        max_absorb_gap = vad_cfg.get("max_absorb_gap", 3.0)
        stages = merge_pipeline(
            raw, vad_cfg.get("min_silence", 0.7), vad_cfg.get("min_speech", 1.0),
            vad_cfg.get("max_chunk", 30.0), None if max_absorb_gap < 0 else max_absorb_gap,
        )

        self.phaseChanged.emit("Phase A: Chunk 생성 중...")
        chunk_entries = export_chunks(samples, stages["final"], job_dir_path / "chunks")
        validate_manifest(chunk_entries)
        n_restored = preserve_prior_transcriptions(job_dir_path, chunk_entries)
        if n_restored:
            self.logMessage.emit(f"[INFO] Resume: restored {n_restored} previously-transcribed chunk(s)")

        source_info = read_source_info(job_dir_path)
        source_file = source_info["source_file"]
        output_srt = str(Path(source_file).parent / f"{Path(source_file).stem}.srt")

        manifest = {
            "job_id": job_dir_path.name,
            "source_file": source_file,
            "source_mtime": source_info["source_mtime"],
            "source_size": source_info["source_size"],
            "provider": self.config.get("provider", "local_api"),
            "created_at": datetime.datetime.now().isoformat(),
            "output_srt": output_srt,
            "audio_sample_rate": SAMPLE_RATE,
            "source_duration_sec": round(total_duration, 3),
            "vad_params": {
                "threshold": vad_cfg.get("threshold", 0.5), "min_silence": vad_cfg.get("min_silence", 0.7),
                "min_speech": vad_cfg.get("min_speech", 1.0), "max_absorb_gap": max_absorb_gap,
                "max_chunk": vad_cfg.get("max_chunk", 30.0),
            },
            "chunks": chunk_entries,
        }
        manifest_path = job_dir_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        self.logMessage.emit(f"[INFO] Phase A complete ({len(chunk_entries)} chunks)")
        return manifest

    # -- Phase B -----------------------------------------------------------

    def _phase_b(self, job_dir_path: Path, manifest: dict) -> None:
        provider = self.config.get("provider", "local_api")
        language = self.config.get("language", "auto")
        llm_cfg = self.config.get("llm", {})
        prompt_template = self.config.get("prompt", {}).get("template", "{{context}}")
        vocabulary = self.config.get("text_enhancement", {}).get("custom_vocabulary", [])

        chunks = manifest["chunks"]
        todo = [c for c in chunks if c["status"] != "transcribed"]
        total = len(chunks)
        done = total - len(todo)
        self.progressChanged.emit(done, total)

        context = ""
        # Resume: recover context from the last successfully-transcribed chunk.
        for c in reversed(chunks):
            if c["status"] == "transcribed":
                txt_path = (job_dir_path / c["file"]).with_suffix(".txt")
                if txt_path.exists():
                    context = txt_path.read_text(encoding="utf-8").strip()
                break

        manifest_path = job_dir_path / "manifest.json"

        def flush():
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            srt_text = render_srt_with_placeholders(manifest, job_dir_path)
            Path(manifest["output_srt"]).write_text(srt_text, encoding="utf-8")
            self.srtUpdated.emit(srt_text)

        if not todo:
            flush()
            return

        n_consecutive_failures = 0

        def process_todo():
            nonlocal context, n_consecutive_failures, done
            for entry in todo:
                if self._stop_requested:
                    self.logMessage.emit("[INFO] Cancel requested, waiting current chunk")
                    return
                self.phaseChanged.emit(f"Phase B: Transcribing chunk {entry['id']}/{total}")
                wav_path = job_dir_path / entry["file"]
                prompt = build_prompt(prompt_template, context, language, vocabulary)

                result, last_error = None, None
                for attempt in range(2):  # design.md SS21: 1 retry then fail
                    try:
                        if provider == "gemini":
                            gemini_cfg = self.config.get("gemini", {})
                            result = transcribe_one_gemini(
                                gemini_cfg.get("api_key", ""), gemini_cfg.get("model", ""),
                                prompt, wav_path, llm_cfg, timeout=120.0,
                            )
                        else:
                            local_cfg = self.config.get("local_api", {})
                            result = transcribe_one_local(
                                server_url, local_cfg.get("model", "qwen3-asr"), prompt, wav_path,
                                llm_cfg, local_cfg.get("disable_thinking", True), timeout=120.0,
                            )
                        break
                    except requests.exceptions.RequestException as e:
                        last_error = e
                        if attempt == 0:
                            self.logMessage.emit(f"[WARNING] chunk {entry['id']} request failed, retrying: {e}")
                            time.sleep(1.0)

                if result is None:
                    self.logMessage.emit(f"[ERROR] chunk {entry['id']} failed after retry: {last_error}")
                    entry["status"] = "failed"
                    n_consecutive_failures += 1
                    done += 1
                    self.progressChanged.emit(done, total)
                    flush()
                    if n_consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        raise RuntimeError(
                            f"연속 {MAX_CONSECUTIVE_FAILURES}회 실패 -- 서버 상태를 확인하세요 (design.md SS21)"
                        )
                    continue

                n_consecutive_failures = 0
                if provider == "gemini":
                    text = result["raw"].strip()
                    lang = language
                else:
                    lang, text = parse_response(result["raw"])
                txt_path = wav_path.with_suffix(".txt")
                txt_path.write_text(text, encoding="utf-8")

                repeat = detect_repetition(text)
                if repeat:
                    self.logMessage.emit(f"[WARNING] chunk {entry['id']}: possible infinite repetition of {repeat!r}")
                    entry["flags"] = ["possible_infinite_repetition"]

                entry["status"] = "transcribed"
                entry["language_detected"] = lang
                entry["transcribe_elapsed_sec"] = round(result["elapsed"], 3)
                context = text  # SS17: only the most recent chunk's text is kept

                self.logMessage.emit(f"[INFO] Chunk #{entry['id']} completed ({result['elapsed']:.2f}s)")
                done += 1
                self.progressChanged.emit(done, total)
                flush()

        if provider == "gemini":
            process_todo()
        else:
            local_cfg = self.config.get("local_api", {})
            server_url = local_cfg.get("url", "http://localhost:8080/v1/chat/completions")
            server_base = server_url.split("/v1/")[0]
            if local_cfg.get("launch_mode", "external") == "managed":
                self.phaseChanged.emit("STT 서버 준비 중...")
            with ensure_llama_server(server_base, self.config, log_path=job_dir_path / "llama-server.log"):
                process_todo()

        flush()

    # -- Orchestration -------------------------------------------------------

    def _run(self):
        job_dir_path = get_job_dir(self.source)
        if not self.resume and (job_dir_path / "manifest.json").exists():
            shutil.rmtree(job_dir_path)
            job_dir_path.mkdir(parents=True)

        manifest = self._phase_a(job_dir_path)
        if self._stop_requested:
            self.jobStopped.emit()
            return

        self._phase_b(job_dir_path, manifest)
        if self._stop_requested:
            self.jobStopped.emit()
            return

        output_srt = manifest["output_srt"]
        n_failed = sum(1 for c in manifest["chunks"] if c["status"] == "failed")
        self.logMessage.emit(f"[INFO] SRT updated: {output_srt}")

        remove_on_success = self.config.get("cleanup", {}).get("remove_temp_on_success", True)
        if n_failed == 0 and remove_on_success:
            shutil.rmtree(job_dir_path)
            self.logMessage.emit(f"[INFO] Cleanup: {job_dir_path} removed")

        self.jobFinished.emit(output_srt)
