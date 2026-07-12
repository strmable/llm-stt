"""Settings modal dialog (design.md SS8)."""

import io
import json
import struct
import sys
import wave
from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
    QPushButton, QRadioButton, QSpinBox, QDoubleSpinBox, QTabWidget, QVBoxLayout,
    QWidget,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from common import CONFIG_PATH  # noqa: E402


def _silence_wav_bytes(seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    """1s of 16kHz mono silence -- design.md SS6.4 Test Connection payload."""
    n_frames = int(seconds * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_frames}h", *([0] * n_frames)))
    return buf.getvalue()


class SettingsDialog(QDialog):
    NOTE_TEXT = (
        "검증 환경: llama.cpp llama-server(2026-06-05 이후 빌드) + Gemma 4 12B GGUF + gemma4uv mmproj, "
        "또는 ggml-org/Qwen3-ASR GGUF.\n"
        "LM Studio / Ollama 등 기타 서버는 동작을 보증하지 않습니다.\n"
        "llama.cpp의 오디오 입력은 \"experimental\"로 표기되어 있어 인식 품질이 저하될 수 있습니다.\n"
        "Managed 모드는 Job당 1회 모델 로딩 시간이 추가되나 일반적으로 비중이 낮으며, "
        "짧은 파일을 반복 처리하는 경우 External 모드를 권장합니다."
    )

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("설정")
        self.setMinimumWidth(560)
        self.config = json.loads(json.dumps(config))  # deep copy -- Cancel must discard edits
        self._build_ui()
        self._load_from_config()

    # -- UI construction ------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        tabs = QTabWidget()
        root.addWidget(tabs)

        tabs.addTab(self._build_vad_tab(), "VAD")
        tabs.addTab(self._build_provider_tab(), "Provider")
        tabs.addTab(self._build_vocabulary_tab(), "용어집")
        tabs.addTab(self._build_postprocessing_tab(), "후처리")
        tabs.addTab(self._build_prompt_tab(), "Prompt")
        tabs.addTab(self._build_params_tab(), "모델 파라미터")

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _build_provider_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        provider_box = QGroupBox("Provider")
        provider_layout = QVBoxLayout(provider_box)
        self.rb_local = QRadioButton("Local llama-server (OpenAI-Compatible)")
        self.rb_gemini = QRadioButton("Google Gemini API")
        self.rb_local.setChecked(True)
        provider_layout.addWidget(self.rb_local)
        provider_layout.addWidget(self.rb_gemini)
        layout.addWidget(provider_box)

        # -- Local llama-server settings --
        self.local_box = QGroupBox("Local llama-server 설정")
        form = QFormLayout(self.local_box)
        self.local_url = QLineEdit()
        form.addRow("API URL", self.local_url)
        self.local_model = QLineEdit()
        form.addRow("Model Name", self.local_model)

        launch_row = QHBoxLayout()
        self.rb_external = QRadioButton("External")
        self.rb_managed = QRadioButton("Managed")
        self.rb_external.setChecked(True)
        launch_row.addWidget(self.rb_external)
        launch_row.addWidget(self.rb_managed)
        form.addRow("Launch Mode", launch_row)

        self.server_binary = self._path_row(form, "llama-server 실행 파일 경로", is_dir=False,
                                             file_filter="Executable (*.exe);;All files (*)")
        self.model_path = self._path_row(form, "모델 파일 경로 (model_path)", is_dir=False,
                                          file_filter="GGUF (*.gguf);;All files (*)")
        self.mmproj_path = self._path_row(form, "mmproj 파일 경로", is_dir=False,
                                           file_filter="GGUF (*.gguf);;All files (*)")
        self.hf_repo = QLineEdit()
        form.addRow("hf_repo (model_path 비어있을 때 -hf로 자동 다운로드)", self.hf_repo)

        self.managed_port = QSpinBox()
        self.managed_port.setRange(1, 65535)
        form.addRow("포트 (Managed)", self.managed_port)
        self.managed_extra_args = QLineEdit()
        form.addRow("추가 인자 (Managed, 선택)", self.managed_extra_args)
        self.managed_timeout = QSpinBox()
        self.managed_timeout.setRange(1, 3600)
        self.managed_timeout.setSuffix(" s")
        form.addRow("기동 타임아웃 (Managed)", self.managed_timeout)

        self.disable_thinking = QCheckBox("Thinking 출력 억제 (disable_thinking)")
        form.addRow(self.disable_thinking)

        test_row = QHBoxLayout()
        self.btn_test_local = QPushButton("Test Connection")
        self.btn_test_local.clicked.connect(self._test_local_connection)
        test_row.addStretch()
        test_row.addWidget(self.btn_test_local)
        form.addRow(test_row)

        for rb in (self.rb_external, self.rb_managed):
            rb.toggled.connect(self._update_managed_enabled)
        layout.addWidget(self.local_box)

        # -- Gemini settings --
        self.gemini_box = QGroupBox("Google Gemini API 설정")
        gform = QFormLayout(self.gemini_box)
        self.gemini_key = QLineEdit()
        self.gemini_key.setEchoMode(QLineEdit.Password)
        gform.addRow("API Key", self.gemini_key)
        self.gemini_model = QLineEdit()
        gform.addRow("Model Name", self.gemini_model)
        gtest_row = QHBoxLayout()
        self.btn_test_gemini = QPushButton("Test Connection")
        self.btn_test_gemini.clicked.connect(self._test_gemini_connection)
        gtest_row.addStretch()
        gtest_row.addWidget(self.btn_test_gemini)
        gform.addRow(gtest_row)
        layout.addWidget(self.gemini_box)

        lang_box = QGroupBox("출력 언어 설정")
        lform = QVBoxLayout(lang_box)
        self.rb_lang_auto = QRadioButton("Auto (모델이 자동 감지)")
        self.rb_lang_auto.setChecked(True)
        lform.addWidget(self.rb_lang_auto)
        forced_row = QHBoxLayout()
        self.rb_lang_forced = QRadioButton("강제 지정")
        self.lang_code = QComboBox()
        self.lang_code.setEditable(True)
        self.lang_code.addItems(["ko", "ja", "zh", "en"])
        forced_row.addWidget(self.rb_lang_forced)
        forced_row.addWidget(self.lang_code)
        forced_row.addStretch()
        lform.addLayout(forced_row)
        layout.addWidget(lang_box)
        self.rb_lang_auto.toggled.connect(lambda: self.lang_code.setEnabled(self.rb_lang_forced.isChecked()))
        self.rb_lang_forced.toggled.connect(lambda: self.lang_code.setEnabled(self.rb_lang_forced.isChecked()))

        note = QLabel(self.NOTE_TEXT)
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(note)
        layout.addStretch()

        self.rb_local.toggled.connect(self._update_provider_enabled)
        return w

    def _path_row(self, form: QFormLayout, label: str, is_dir: bool, file_filter: str) -> QLineEdit:
        edit = QLineEdit()
        browse = QPushButton("찾아보기...")

        def on_browse():
            if is_dir:
                path = QFileDialog.getExistingDirectory(self, label)
            else:
                path, _ = QFileDialog.getOpenFileName(self, label, filter=file_filter)
            if path:
                edit.setText(path)

        browse.clicked.connect(on_browse)
        row = QHBoxLayout()
        row.addWidget(edit)
        row.addWidget(browse)
        container = QWidget()
        container.setLayout(row)
        form.addRow(label, container)
        return edit

    def _build_vocabulary_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        vocab_box = QGroupBox("Custom Vocabulary (줄바꿈으로 구분)")
        vform = QVBoxLayout(vocab_box)
        self.vocabulary = QPlainTextEdit()
        self.vocabulary.setPlaceholderText("등장 가능한 고유명사/전문용어를 한 줄에 하나씩 입력")
        vform.addWidget(self.vocabulary)
        layout.addWidget(vocab_box)
        layout.addStretch()
        return w

    def _build_postprocessing_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        self.text_correction = QCheckBox("Text Correction (STT 결과를 문맥 기반으로 재교정) -- 차기 버전")
        self.text_correction.setEnabled(False)
        layout.addWidget(self.text_correction)

        hallucination_box = QGroupBox("할루시네이션 필터")
        hform = QVBoxLayout(hallucination_box)
        self.dedup_repeated_chunks = QCheckBox(
            "직전 chunk와 텍스트가 동일하면 제거 (오디오가 흐릿할 때 이전 문장을 그대로 반복 출력하는 "
            "할루시네이션 억제, 기본 OFF)"
        )
        hform.addWidget(self.dedup_repeated_chunks)
        self.strip_infinite_repetition = QCheckBox(
            "동일 패턴이 5회 이상 연속 반복되면 해당 chunk 제거 (무한 반복 할루시네이션 억제, 기본 OFF)"
        )
        hform.addWidget(self.strip_infinite_repetition)
        layout.addWidget(hallucination_box)
        layout.addStretch()
        return w

    def _build_prompt_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        var_label = QLabel(
            "Template Variables: {{context}} 직전 chunk 인식 결과  |  "
            "{{language_hint}} 언어 힌트  |  {{vocabulary}} 용어집"
        )
        var_label.setWordWrap(True)
        var_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(var_label)

        self.prompt_edit = QPlainTextEdit()
        layout.addWidget(self.prompt_edit)

        btn_row = QHBoxLayout()
        btn_load = QPushButton("Load")
        btn_save = QPushButton("Save")
        btn_save_as = QPushButton("Save As")
        btn_load.clicked.connect(self._prompt_load)
        btn_save.clicked.connect(self._prompt_save)
        btn_save_as.clicked.connect(self._prompt_save_as)
        btn_row.addWidget(btn_load)
        btn_row.addWidget(btn_save)
        btn_row.addWidget(btn_save_as)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._last_prompt_path: Path | None = None
        return w

    def _build_params_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        params_box = QGroupBox("모델 파라미터")
        form = QFormLayout(params_box)
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.0, 2.0)
        self.temperature.setSingleStep(0.05)
        form.addRow("Temperature", self.temperature)
        self.top_p = QDoubleSpinBox()
        self.top_p.setRange(0.0, 1.0)
        self.top_p.setSingleStep(0.01)
        form.addRow("Top-P", self.top_p)
        self.top_k = QSpinBox()
        self.top_k.setRange(0, 1000)
        form.addRow("Top-K", self.top_k)
        self.max_tokens = QSpinBox()
        self.max_tokens.setRange(1, 32768)
        form.addRow("Max Tokens", self.max_tokens)
        layout.addWidget(params_box)

        self.cleanup_checkbox = QCheckBox("완료 후 임시 파일 정리 (기본 ON)")
        self.cleanup_checkbox.setChecked(True)
        layout.addWidget(self.cleanup_checkbox)
        layout.addStretch()
        return w

    def _build_vad_tab(self) -> QWidget:
        """design.md SS12.2/12.3 -- VAD 후처리/Chunk 경계 파라미터. CLI 스테이지
        스크립트(vad_merge.py 등)의 동일 옵션과 config.json "vad" 섹션을 공유한다."""
        w = QWidget()
        layout = QVBoxLayout(w)

        vad_box = QGroupBox("VAD / Chunk 분할 파라미터")
        form = QFormLayout(vad_box)

        self.vad_threshold = QDoubleSpinBox()
        self.vad_threshold.setRange(0.0, 1.0)
        self.vad_threshold.setSingleStep(0.05)
        self.vad_threshold.setDecimals(2)
        form.addRow("Threshold (VAD 확률 임계값)", self.vad_threshold)

        self.vad_min_silence = QDoubleSpinBox()
        self.vad_min_silence.setRange(0.0, 10.0)
        self.vad_min_silence.setSingleStep(0.1)
        form.addRow("Min Silence (초과 시에만 실제 분할, 이하는 병합)", self.vad_min_silence)

        self.vad_min_speech = QDoubleSpinBox()
        self.vad_min_speech.setRange(0.0, 10.0)
        self.vad_min_speech.setSingleStep(0.1)
        form.addRow("Min Speech (미만이면 이웃 구간에 흡수)", self.vad_min_speech)

        self.vad_max_absorb_gap = QDoubleSpinBox()
        self.vad_max_absorb_gap.setRange(-1.0, 60.0)
        self.vad_max_absorb_gap.setSingleStep(0.5)
        self.vad_max_absorb_gap.setSpecialValueText("무제한 (음수)")
        form.addRow("Max Absorb Gap (흡수 가능 거리 상한, 음수=무제한)", self.vad_max_absorb_gap)

        self.vad_max_chunk = QDoubleSpinBox()
        self.vad_max_chunk.setRange(1.0, 120.0)
        self.vad_max_chunk.setSingleStep(1.0)
        form.addRow("Max Chunk (하드 리밋, 모델 오디오 입력 상한)", self.vad_max_chunk)

        layout.addWidget(vad_box)

        note = QLabel(
            "값 변경은 다음 작업(새로 시작)부터 적용됩니다. 이미 Phase A가 끝난 job을 이어하기(Resume)하면 "
            "그 job은 최초 실행 시의 파라미터를 그대로 사용합니다 (manifest.json에 고정 기록됨)."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(note)
        layout.addStretch()
        return w

    # -- Load/save config <-> widgets ------------------------------------

    def _load_from_config(self):
        cfg = self.config
        self.rb_local.setChecked(cfg.get("provider", "local_api") != "gemini")
        self.rb_gemini.setChecked(cfg.get("provider", "local_api") == "gemini")

        local_api = cfg.get("local_api", {})
        self.local_url.setText(local_api.get("url", ""))
        self.local_model.setText(local_api.get("model", ""))
        self.rb_managed.setChecked(local_api.get("launch_mode", "external") == "managed")
        self.rb_external.setChecked(local_api.get("launch_mode", "external") != "managed")
        self.server_binary.setText(local_api.get("server_binary", ""))
        self.model_path.setText(local_api.get("model_path", ""))
        self.mmproj_path.setText(local_api.get("mmproj_path", ""))
        self.hf_repo.setText(local_api.get("hf_repo", ""))
        managed = local_api.get("managed", {})
        self.managed_port.setValue(managed.get("port", 8080))
        self.managed_extra_args.setText(managed.get("extra_args", ""))
        self.managed_timeout.setValue(managed.get("startup_timeout_sec", 120))
        self.disable_thinking.setChecked(local_api.get("disable_thinking", True))

        gemini = cfg.get("gemini", {})
        self.gemini_key.setText(gemini.get("api_key", ""))
        self.gemini_model.setText(gemini.get("model", "gemini-3.1-flash-lite"))

        language = cfg.get("language", "auto")
        self.rb_lang_auto.setChecked(language == "auto")
        self.rb_lang_forced.setChecked(language != "auto")
        if language != "auto":
            self.lang_code.setCurrentText(language)
        self.lang_code.setEnabled(language != "auto")

        vocab = cfg.get("text_enhancement", {}).get("custom_vocabulary", [])
        self.vocabulary.setPlainText("\n".join(vocab))
        self.dedup_repeated_chunks.setChecked(
            cfg.get("text_enhancement", {}).get("dedup_repeated_chunks", False))
        self.strip_infinite_repetition.setChecked(
            cfg.get("text_enhancement", {}).get("strip_infinite_repetition", False))

        self.prompt_edit.setPlainText(cfg.get("prompt", {}).get("template", ""))

        llm = cfg.get("llm", {})
        self.temperature.setValue(llm.get("temperature", 1.0))
        self.top_p.setValue(llm.get("top_p", 0.95))
        self.top_k.setValue(llm.get("top_k", 64))
        self.max_tokens.setValue(llm.get("max_tokens", 4096))

        self.cleanup_checkbox.setChecked(cfg.get("cleanup", {}).get("remove_temp_on_success", True))

        vad = cfg.get("vad", {})
        self.vad_threshold.setValue(vad.get("threshold", 0.5))
        self.vad_min_silence.setValue(vad.get("min_silence", 0.7))
        self.vad_min_speech.setValue(vad.get("min_speech", 1.0))
        self.vad_max_absorb_gap.setValue(vad.get("max_absorb_gap", 3.0))
        self.vad_max_chunk.setValue(vad.get("max_chunk", 30.0))

        self._update_provider_enabled()
        self._update_managed_enabled()

    def to_config(self) -> dict:
        cfg = json.loads(json.dumps(self.config))  # keep untouched keys (logging, etc.)
        cfg["provider"] = "gemini" if self.rb_gemini.isChecked() else "local_api"
        cfg["language"] = self.lang_code.currentText().strip() if self.rb_lang_forced.isChecked() else "auto"

        cfg.setdefault("local_api", {})
        cfg["local_api"].update({
            "url": self.local_url.text().strip(),
            "model": self.local_model.text().strip(),
            "disable_thinking": self.disable_thinking.isChecked(),
            "launch_mode": "managed" if self.rb_managed.isChecked() else "external",
            "server_binary": self.server_binary.text().strip(),
            "model_path": self.model_path.text().strip(),
            "mmproj_path": self.mmproj_path.text().strip(),
            "hf_repo": self.hf_repo.text().strip(),
            "managed": {
                "port": self.managed_port.value(),
                "extra_args": self.managed_extra_args.text().strip(),
                "startup_timeout_sec": self.managed_timeout.value(),
            },
        })

        cfg.setdefault("gemini", {})
        cfg["gemini"].update({
            "api_key": self.gemini_key.text().strip(),
            "model": self.gemini_model.text().strip(),
        })

        cfg.setdefault("llm", {})
        cfg["llm"].update({
            "temperature": self.temperature.value(),
            "top_p": self.top_p.value(),
            "top_k": self.top_k.value(),
            "max_tokens": self.max_tokens.value(),
        })

        cfg.setdefault("prompt", {})
        cfg["prompt"]["template"] = self.prompt_edit.toPlainText()

        vocab_lines = [line.strip() for line in self.vocabulary.toPlainText().splitlines() if line.strip()]
        cfg.setdefault("text_enhancement", {"custom_vocabulary": [], "text_correction": {
            "enabled": False, "provider": "local_api", "window_chunks": 5}})
        cfg["text_enhancement"]["custom_vocabulary"] = vocab_lines
        cfg["text_enhancement"]["dedup_repeated_chunks"] = self.dedup_repeated_chunks.isChecked()
        cfg["text_enhancement"]["strip_infinite_repetition"] = self.strip_infinite_repetition.isChecked()

        cfg.setdefault("cleanup", {})
        cfg["cleanup"]["remove_temp_on_success"] = self.cleanup_checkbox.isChecked()

        cfg.setdefault("vad", {})
        cfg["vad"].update({
            "threshold": self.vad_threshold.value(),
            "min_silence": self.vad_min_silence.value(),
            "min_speech": self.vad_min_speech.value(),
            "max_absorb_gap": self.vad_max_absorb_gap.value(),
            "max_chunk": self.vad_max_chunk.value(),
        })
        return cfg

    def accept(self):
        cfg = self.to_config()
        CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        self.config = cfg
        super().accept()

    # -- behavior ---------------------------------------------------------

    def _update_provider_enabled(self):
        self.local_box.setEnabled(self.rb_local.isChecked())
        self.gemini_box.setEnabled(self.rb_gemini.isChecked())

    def _update_managed_enabled(self):
        managed = self.rb_managed.isChecked()
        self.managed_port.setEnabled(managed)
        self.managed_extra_args.setEnabled(managed)
        self.managed_timeout.setEnabled(managed)

    def _prompt_load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Prompt", filter="Text files (*.txt);;All files (*)")
        if path:
            self.prompt_edit.setPlainText(Path(path).read_text(encoding="utf-8"))
            self._last_prompt_path = Path(path)

    def _prompt_save(self):
        if self._last_prompt_path is None:
            self._prompt_save_as()
            return
        self._last_prompt_path.write_text(self.prompt_edit.toPlainText(), encoding="utf-8")

    def _prompt_save_as(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Prompt As", filter="Text files (*.txt);;All files (*)")
        if path:
            Path(path).write_text(self.prompt_edit.toPlainText(), encoding="utf-8")
            self._last_prompt_path = Path(path)

    def _test_local_connection(self):
        import requests
        cfg = self.to_config()
        local_api = cfg["local_api"]
        if local_api["launch_mode"] == "managed":
            missing = [k for k in ("server_binary",) if not local_api[k]]
            if not local_api["model_path"] and not local_api["hf_repo"]:
                missing.append("model_path 또는 hf_repo")
            if missing:
                QMessageBox.warning(self, "Test Connection", f"Managed 모드 필수 항목 누락: {', '.join(missing)}")
                return
        try:
            wav_b64 = __import__("base64").b64encode(_silence_wav_bytes()).decode("ascii")
            payload = {
                "model": local_api["model"],
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe this audio."},
                        {"type": "input_audio", "input_audio": {"data": wav_b64, "format": "wav"}},
                    ],
                }],
                "max_tokens": 32,
            }
            resp = requests.post(local_api["url"], json=payload, timeout=10)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            QMessageBox.information(self, "Test Connection", f"OK\n\n{content[:200]}")
        except Exception as e:  # noqa: BLE001 -- shown to the user, not logged
            QMessageBox.critical(self, "Test Connection", f"실패: {e}")

    def _test_gemini_connection(self):
        import base64
        import requests
        cfg = self.to_config()
        gemini = cfg["gemini"]
        if not gemini["api_key"]:
            QMessageBox.warning(self, "Test Connection", "API Key가 비어있습니다.")
            return
        try:
            wav_b64 = base64.b64encode(_silence_wav_bytes()).decode("ascii")
            url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                   f"{gemini['model']}:generateContent?key={gemini['api_key']}")
            payload = {"contents": [{"parts": [
                {"text": "Transcribe this audio."},
                {"inline_data": {"mime_type": "audio/wav", "data": wav_b64}},
            ]}]}
            resp = requests.post(url, json=payload, timeout=10)
            resp.raise_for_status()
            candidates = resp.json().get("candidates")
            if not candidates:
                raise RuntimeError("응답에 candidates 없음")
            QMessageBox.information(self, "Test Connection", "OK")
        except Exception as e:  # noqa: BLE001 -- shown to the user, not logged
            QMessageBox.critical(self, "Test Connection", f"실패: {e}")
