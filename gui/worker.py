"""Worker thread driving Phase A/B/C (design.md SS11/SS19, postprocessing.md)
for the GUI.

Reuses the same functions the independently-runnable pipeline/*.py stage
scripts use (see phase_a_roadmap.md) instead of shelling out to them, so
progress/cancellation/SRT updates can be reported chunk-by-chunk via Qt
signals (design.md SS19: Worker Thread does the work, GUI Thread only
touches widgets in response to signals).

Phase C (postprocessing.md) is opt-in via config.json
text_enhancement.text_correction.enabled (default OFF) and runs after Phase B
fully completes -- never concurrently with the STT server (design.md SS5B.3).

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
import re
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
from cue_splitter import DEFAULT_GAP_SEC, split_by_speaker_cues  # noqa: E402
from server_manager import adapt_text_correction_server_config, ensure_llama_server  # noqa: E402
from text_correction import correct_all  # noqa: E402
from transcribe_chunks import detect_repetition, parse_response  # noqa: E402
from vad_merge import merge_pipeline  # noqa: E402
from vad_raw_test import SAMPLE_RATE, ensure_extracted_wav, raw_segments, run_vad  # noqa: E402

MAX_CONSECUTIVE_FAILURES = 5  # design.md SS21
FAILED_PLACEHOLDER = "[TRANSCRIPTION FAILED]"

_TRAILING_PUNCT_RE = re.compile(r"[\s.,!?~…、。！？]+$")


def _normalize_for_dedup(text: str) -> str:
    """Loose equality for the repeated-chunk hallucination check below --
    collapses whitespace and ignores trailing punctuation so "그렇습니다."
    vs "그렇습니다" still count as the same echoed sentence."""
    return _TRAILING_PUNCT_RE.sub("", " ".join(text.split())).casefold()


def _format_eta(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


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

    def _load_existing_manifest(self, manifest_path: Path) -> dict:
        """Resume path when temp/{job_id}/manifest.json already exists: reuse
        it as-is instead of re-running audio extraction/VAD/chunk export.
        job_id already encodes (source path, mtime, size) (design.md SS14.1),
        so a manifest found under it is guaranteed to belong to this exact
        source file, and chunk WAVs/txt files persist on disk between runs
        (design principle 1) unless the job finished successfully or the user
        hit "완전 취소" -- neither of which leaves a resumable manifest
        behind. This also sidesteps a correctness trap the old
        always-re-run-Phase-A path had: re-running VAD with the *current*
        config instead of the manifest's pinned vad_params would silently
        produce different chunk boundaries if VAD settings changed between
        runs, which preserve_prior_transcriptions() matches by exact
        (start_sec, end_sec) -- a mismatch there would discard already-done
        transcription progress instead of just wasting time re-computing VAD."""
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        n_total = len(manifest["chunks"])
        n_done = sum(1 for c in manifest["chunks"] if c["status"] == "transcribed")
        self.phaseChanged.emit("Phase A: 이전 작업 재사용 (VAD 생략)")
        self.logMessage.emit(
            f"[INFO] Resume: reusing existing Phase A result ({n_total} chunks, "
            f"{n_done} already transcribed) -- VAD/오디오 추출 생략"
        )
        return manifest

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

        # Seed the running average from chunks already transcribed (resume-safe),
        # so the ETA is accurate from the first chunk of this run, not just this session's.
        elapsed_total = sum(
            c["transcribe_elapsed_sec"] for c in chunks
            if c["status"] == "transcribed" and "transcribe_elapsed_sec" in c
        )
        elapsed_count = sum(
            1 for c in chunks if c["status"] == "transcribed" and "transcribe_elapsed_sec" in c
        )

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
            nonlocal context, n_consecutive_failures, done, elapsed_total, elapsed_count
            for entry in todo:
                if self._stop_requested:
                    self.logMessage.emit("[INFO] Cancel requested, waiting current chunk")
                    return
                status_msg = f"Transcribing chunk {entry['id']}/{total}"
                if elapsed_count > 0:
                    avg = elapsed_total / elapsed_count
                    eta_sec = avg * (total - done)
                    status_msg += f" - {_format_eta(eta_sec)} remaining..."
                self.phaseChanged.emit(status_msg)
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

                # Faint/near-silent audio sometimes makes the model just echo
                # the previous chunk's text back verbatim instead of admitting
                # it heard nothing -- an observed hallucination pattern, not a
                # real repeated sentence (real repeats are rare across a VAD
                # chunk boundary and 30s max-chunk gap). Opt-in since a chunk
                # legitimately repeating the prior one is possible.
                dedup_enabled = self.config.get("text_enhancement", {}).get("dedup_repeated_chunks", False)
                is_duplicate = (
                    dedup_enabled and text.strip() and context.strip()
                    and _normalize_for_dedup(text) == _normalize_for_dedup(context)
                )

                if is_duplicate:
                    self.logMessage.emit(
                        f"[WARNING] chunk {entry['id']}: text identical to previous chunk, "
                        f"suppressed as suspected hallucination: {text[:60]!r}"
                    )
                    entry.setdefault("flags", []).append("possible_duplicate_hallucination")
                    txt_path.write_text("", encoding="utf-8")
                    # context deliberately left pointing at the last genuinely
                    # new text, not this echo, so a run of several faint
                    # chunks in a row doesn't compare against its own echo
                else:
                    repeat = detect_repetition(text)
                    strip_enabled = self.config.get("text_enhancement", {}).get(
                        "strip_infinite_repetition", False)
                    if repeat:
                        entry.setdefault("flags", []).append("possible_infinite_repetition")
                    if repeat and strip_enabled:
                        self.logMessage.emit(
                            f"[WARNING] chunk {entry['id']}: possible infinite repetition of {repeat!r}, "
                            f"suppressed as suspected hallucination: {text[:60]!r}"
                        )
                        txt_path.write_text("", encoding="utf-8")
                        # context deliberately left pointing at the last genuine
                        # text, same rationale as the duplicate-chunk branch above
                    else:
                        if repeat:
                            self.logMessage.emit(
                                f"[WARNING] chunk {entry['id']}: possible infinite repetition of {repeat!r}")
                        txt_path.write_text(text, encoding="utf-8")
                        context = text  # SS17: only the most recent chunk's text is kept

                entry["status"] = "transcribed"
                entry["language_detected"] = lang
                entry["transcribe_elapsed_sec"] = round(result["elapsed"], 3)
                elapsed_total += result["elapsed"]
                elapsed_count += 1

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

    # -- Phase C (postprocessing.md) ----------------------------------------

    def _render_final_srt(self, manifest: dict, job_dir_path: Path, cue_cfg: dict) -> str:
        """Like render_srt_with_placeholders, but each transcribed chunk's
        corrected text (chunk_NNNN.fixed.txt, falling back to the raw
        chunk_NNNN.txt for any chunk text_correction skipped) is run through
        the deterministic speaker-marker cue splitter (postprocessing.md
        SS11.1 1차 분할 only -- no CPS-based length splitting here) instead of
        becoming exactly one cue -- so a merged Q&A becomes multiple,
        appropriately-timed cues. CPS-based length splitting is a separate,
        manual step run later against a translated SRT (pipeline/
        srt_postprocess.py, gui/srt_postprocess_dialog.py)."""
        blocks = []
        i = 0
        for chunk in manifest["chunks"]:
            if chunk["status"] == "failed":
                cues = [{"start_sec": chunk["start_sec"], "end_sec": chunk["end_sec"], "text": FAILED_PLACEHOLDER}]
            elif chunk["status"] == "transcribed":
                fixed_path = (job_dir_path / chunk["file"]).with_suffix(".fixed.txt")
                raw_path = (job_dir_path / chunk["file"]).with_suffix(".txt")
                if fixed_path.exists():
                    text = fixed_path.read_text(encoding="utf-8").strip()
                elif raw_path.exists():
                    text = " ".join(raw_path.read_text(encoding="utf-8").split())
                else:
                    text = ""
                if not text:
                    continue
                cues = split_by_speaker_cues(
                    text, chunk["start_sec"], chunk["end_sec"],
                    gap_sec=cue_cfg.get("gap_sec", DEFAULT_GAP_SEC),
                    show_speaker_label=cue_cfg.get("show_speaker_label", False),
                )
            else:
                continue  # pending/vad_extracted -- not reached yet

            for cue in cues:
                if not cue["text"].strip():
                    continue
                i += 1
                blocks.append(
                    f"{i}\n{srt_timestamp(cue['start_sec'])} --> {srt_timestamp(cue['end_sec'])}\n{cue['text']}\n"
                )
        return "\n".join(blocks)

    def _phase_c(self, job_dir_path: Path, manifest: dict) -> None:
        """Phase B ends with exactly one cue per VAD chunk (design.md SS21
        placeholder rules only, no text correction). If Text Correction is
        enabled (postprocessing.md, opt-in/default OFF), Phase C: (1) runs
        full-context LLM correction per chunk on a *separate* text-instruct
        server (never concurrently with the STT server, design.md SS5B.3),
        which may also insert [[SPEAKER]] markers where a merged chunk holds
        more than one speaker; (2) preserves Phase B's single-cue-per-chunk
        SRT as {name}.raw.srt for comparison/rollback; (3) deterministically
        re-splits every chunk into cues on speaker markers only (approximate
        timing, SS11.1 1차 분할) and overwrites the final SRT with that.
        CPS-based length splitting (SS11.1 2차 분할) is intentionally NOT run
        here -- it happens manually, later, against a translated SRT (see
        pipeline/srt_postprocess.py), so external translation still sees
        whole sentences instead of pre-cut fragments."""
        tc_cfg = self.config.get("text_enhancement", {}).get("text_correction", {})
        if not tc_cfg.get("enabled", False):
            return

        output_srt = Path(manifest["output_srt"])
        if output_srt.exists():
            raw_srt_path = output_srt.parent / f"{output_srt.stem}.raw.srt"
            raw_srt_path.write_text(output_srt.read_text(encoding="utf-8"), encoding="utf-8")
            self.logMessage.emit(f"[INFO] Phase B SRT preserved as {raw_srt_path} before correction")

        server_cfg = tc_cfg.get("server", {})
        server_url = server_cfg.get("url", "http://localhost:8081/v1/chat/completions")
        server_base = server_url.split("/v1/")[0]
        server_manager_cfg = adapt_text_correction_server_config(server_cfg)

        if server_cfg.get("launch_mode", "external") == "managed":
            self.phaseChanged.emit("Phase C: 후처리 서버 준비 중...")

        def on_progress(done: int, total: int):
            self.progressChanged.emit(done, total)
            self.phaseChanged.emit(f"Phase C: 텍스트 교정 중... ({done}/{total})")

        with ensure_llama_server(server_base, server_manager_cfg, log_path=job_dir_path / "llama-server-tc.log"):
            correct_all(
                manifest, job_dir_path, tc_cfg, server_url,
                log=self.logMessage.emit,
                should_stop=lambda: self._stop_requested,
                on_progress=on_progress,
            )

        if self._stop_requested:
            self.logMessage.emit("[INFO] Cancel requested during Phase C, keeping raw.srt result")
            return

        self.phaseChanged.emit("Phase C: 화자 분할 중...")
        srt_text = self._render_final_srt(manifest, job_dir_path, tc_cfg.get("cue_splitter", {}))
        output_srt.write_text(srt_text, encoding="utf-8")
        self.srtUpdated.emit(srt_text)
        self.logMessage.emit(f"[INFO] Phase C complete, final SRT: {output_srt}")

    # -- Orchestration -------------------------------------------------------

    def _run(self):
        job_dir_path = get_job_dir(self.source)
        manifest_path = job_dir_path / "manifest.json"
        if not self.resume and manifest_path.exists():
            shutil.rmtree(job_dir_path)
            job_dir_path.mkdir(parents=True)

        if self.resume and manifest_path.exists():
            manifest = self._load_existing_manifest(manifest_path)
        else:
            manifest = self._phase_a(job_dir_path)
        if self._stop_requested:
            self.jobStopped.emit()
            return

        self._phase_b(job_dir_path, manifest)
        if self._stop_requested:
            self.jobStopped.emit()
            return

        self._phase_c(job_dir_path, manifest)
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
