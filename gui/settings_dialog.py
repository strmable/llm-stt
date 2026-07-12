"""Settings modal dialog (design.md SS8)."""

import io
import json
import struct
import sys
import wave
from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
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

        hallucination_box = QGroupBox("할루시네이션 필터 (Phase B, STT 청크 단위)")
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

        self.text_correction_enabled = QCheckBox(
            "Text Correction 활성화 (postprocessing.md §12.2) -- 설정 저장만 지원, 파이프라인 실행 연동은 차기 버전"
        )
        layout.addWidget(self.text_correction_enabled)

        self.tc_box = QGroupBox("Text Correction 설정")
        tc_layout = QVBoxLayout(self.tc_box)

        # -- server (§12.2 "server") --
        tc_server_box = QGroupBox("서버 (llama-server)")
        sform = QFormLayout(tc_server_box)
        self.tc_url = QLineEdit()
        sform.addRow("API URL", self.tc_url)

        tc_launch_row = QHBoxLayout()
        self.tc_rb_external = QRadioButton("External")
        self.tc_rb_managed = QRadioButton("Managed")
        self.tc_rb_external.setChecked(True)
        tc_launch_row.addWidget(self.tc_rb_external)
        tc_launch_row.addWidget(self.tc_rb_managed)
        sform.addRow("Launch Mode", tc_launch_row)

        self.tc_server_binary = self._path_row(sform, "llama-server 실행 파일 경로", is_dir=False,
                                                file_filter="Executable (*.exe);;All files (*)")
        self.tc_model_path = self._path_row(sform, "모델 파일 경로 (model_path)", is_dir=False,
                                             file_filter="GGUF (*.gguf);;All files (*)")

        self.tc_port = QSpinBox()
        self.tc_port.setRange(1, 65535)
        sform.addRow("포트 (Managed)", self.tc_port)
        self.tc_extra_args = QLineEdit()
        sform.addRow("추가 인자 (Managed, §3 검증 커맨드 기본 반영)", self.tc_extra_args)
        self.tc_startup_timeout = QSpinBox()
        self.tc_startup_timeout.setRange(1, 3600)
        self.tc_startup_timeout.setSuffix(" s")
        sform.addRow("기동 타임아웃 (Managed)", self.tc_startup_timeout)

        tc_test_row = QHBoxLayout()
        self.btn_test_tc = QPushButton("Test Connection")
        self.btn_test_tc.clicked.connect(self._test_tc_connection)
        tc_test_row.addStretch()
        tc_test_row.addWidget(self.btn_test_tc)
        sform.addRow(tc_test_row)

        for rb in (self.tc_rb_external, self.tc_rb_managed):
            rb.toggled.connect(self._update_tc_managed_enabled)
        tc_layout.addWidget(tc_server_box)

        # -- sampling (§12.2 "sampling") --
        sampling_box = QGroupBox("샘플링 파라미터")
        spform = QFormLayout(sampling_box)
        self.tc_temperature = QDoubleSpinBox()
        self.tc_temperature.setRange(0.0, 2.0)
        self.tc_temperature.setSingleStep(0.05)
        spform.addRow("Temperature", self.tc_temperature)
        self.tc_top_p = QDoubleSpinBox()
        self.tc_top_p.setRange(0.0, 1.0)
        self.tc_top_p.setSingleStep(0.01)
        spform.addRow("Top-P", self.tc_top_p)
        self.tc_top_k = QSpinBox()
        self.tc_top_k.setRange(0, 1000)
        spform.addRow("Top-K", self.tc_top_k)
        self.tc_presence_penalty = QDoubleSpinBox()
        self.tc_presence_penalty.setRange(0.0, 2.0)
        self.tc_presence_penalty.setSingleStep(0.1)
        spform.addRow("Presence Penalty", self.tc_presence_penalty)
        self.tc_repetition_penalty = QDoubleSpinBox()
        self.tc_repetition_penalty.setRange(0.0, 2.0)
        self.tc_repetition_penalty.setSingleStep(0.05)
        spform.addRow("Repetition Penalty", self.tc_repetition_penalty)
        self.tc_max_tokens = QSpinBox()
        self.tc_max_tokens.setRange(1, 32768)
        spform.addRow("Max Tokens", self.tc_max_tokens)
        tc_layout.addWidget(sampling_box)

        # -- full_context (§12.2 "full_context", §6) --
        fc_box = QGroupBox("Full-context 교정 (§6)")
        fcform = QFormLayout(fc_box)
        self.tc_max_segment_chars = QSpinBox()
        self.tc_max_segment_chars.setRange(1000, 500000)
        self.tc_max_segment_chars.setSingleStep(1000)
        fcform.addRow("Max Segment Chars (초과 시 대구간 분할)", self.tc_max_segment_chars)
        self.tc_segment_split_count = QSpinBox()
        self.tc_segment_split_count.setRange(1, 10)
        fcform.addRow("Segment Split Count", self.tc_segment_split_count)
        self.tc_glossary_assist = QCheckBox("Glossary 보조 (표기 흔들림 관측 시에만 ON, 기본 OFF)")
        fcform.addRow(self.tc_glossary_assist)
        tc_layout.addWidget(fc_box)

        # -- speaker_detection (§12.2 "speaker_detection", §10) --
        sd_box = QGroupBox("화자 전환 탐지 (§10)")
        sdform = QFormLayout(sd_box)
        self.tc_speaker_enabled = QCheckBox("화자 전환 탐지 활성화 (기본 ON)")
        sdform.addRow(self.tc_speaker_enabled)
        self.tc_trigger_length_chars = QSpinBox()
        self.tc_trigger_length_chars.setRange(1, 10000)
        sdform.addRow("Trigger Length Chars", self.tc_trigger_length_chars)
        tc_layout.addWidget(sd_box)

        # -- cue_splitter (§12.2 "cue_splitter", §11) --
        cs_box = QGroupBox("Cue Splitter (§11)")
        csform = QFormLayout(cs_box)
        self.tc_cps_threshold = QSpinBox()
        self.tc_cps_threshold.setRange(1, 100)
        csform.addRow("CPS Threshold (초당 문자수)", self.tc_cps_threshold)
        self.tc_max_cue_duration_sec = QDoubleSpinBox()
        self.tc_max_cue_duration_sec.setRange(0.0, 60.0)
        self.tc_max_cue_duration_sec.setSingleStep(0.5)
        self.tc_max_cue_duration_sec.setSpecialValueText("미설정 (null, 무제한)")
        csform.addRow("Max Cue Duration (sec)", self.tc_max_cue_duration_sec)
        self.tc_min_cue_duration_sec = QDoubleSpinBox()
        self.tc_min_cue_duration_sec.setRange(0.0, 10.0)
        self.tc_min_cue_duration_sec.setSingleStep(0.1)
        self.tc_min_cue_duration_sec.setSpecialValueText("미설정 (null, 무제한)")
        csform.addRow("Min Cue Duration (sec)", self.tc_min_cue_duration_sec)
        self.tc_show_speaker_label = QCheckBox("화자 라벨 화면 노출 (기본 OFF)")
        csform.addRow(self.tc_show_speaker_label)
        tc_layout.addWidget(cs_box)

        layout.addWidget(self.tc_box)
        self.text_correction_enabled.toggled.connect(self.tc_box.setEnabled)
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

        text_enhancement = cfg.get("text_enhancement", {})
        vocab = text_enhancement.get("custom_vocabulary", [])
        self.vocabulary.setPlainText("\n".join(vocab))
        self.dedup_repeated_chunks.setChecked(text_enhancement.get("dedup_repeated_chunks", False))
        self.strip_infinite_repetition.setChecked(text_enhancement.get("strip_infinite_repetition", False))

        tc = text_enhancement.get("text_correction", {})
        self.text_correction_enabled.setChecked(tc.get("enabled", False))

        tc_server = tc.get("server", {})
        self.tc_url.setText(tc_server.get("url", "http://localhost:8081/v1/chat/completions"))
        self.tc_rb_managed.setChecked(tc_server.get("launch_mode", "external") == "managed")
        self.tc_rb_external.setChecked(tc_server.get("launch_mode", "external") != "managed")
        self.tc_server_binary.setText(tc_server.get("server_binary", ""))
        self.tc_model_path.setText(tc_server.get("model_path", ""))
        self.tc_port.setValue(tc_server.get("port", 8081))
        self.tc_extra_args.setText(tc_server.get(
            "extra_args",
            "--ctx-size 32768 --parallel 1 -fa on --cache-type-k q8_0 --cache-type-v q8_0 "
            "--reasoning-budget 0 --jinja"))
        self.tc_startup_timeout.setValue(tc_server.get("startup_timeout_sec", 120))

        tc_sampling = tc.get("sampling", {})
        self.tc_temperature.setValue(tc_sampling.get("temperature", 0.25))
        self.tc_top_p.setValue(tc_sampling.get("top_p", 0.8))
        self.tc_top_k.setValue(tc_sampling.get("top_k", 20))
        self.tc_presence_penalty.setValue(tc_sampling.get("presence_penalty", 1.0))
        self.tc_repetition_penalty.setValue(tc_sampling.get("repetition_penalty", 1.0))
        self.tc_max_tokens.setValue(tc_sampling.get("max_tokens", 512))

        tc_full_context = tc.get("full_context", {})
        self.tc_max_segment_chars.setValue(tc_full_context.get("max_segment_chars", 60000))
        self.tc_segment_split_count.setValue(tc_full_context.get("segment_split_count", 3))
        self.tc_glossary_assist.setChecked(tc_full_context.get("glossary_assist", False))

        tc_speaker = tc.get("speaker_detection", {})
        self.tc_speaker_enabled.setChecked(tc_speaker.get("enabled", True))
        self.tc_trigger_length_chars.setValue(tc_speaker.get("trigger_length_chars", 100))

        tc_cue = tc.get("cue_splitter", {})
        self.tc_cps_threshold.setValue(tc_cue.get("cps_threshold", 15))
        self.tc_max_cue_duration_sec.setValue(tc_cue.get("max_cue_duration_sec") or 0.0)
        self.tc_min_cue_duration_sec.setValue(tc_cue.get("min_cue_duration_sec") or 0.0)
        self.tc_show_speaker_label.setChecked(tc_cue.get("show_speaker_label", False))

        self.tc_box.setEnabled(self.text_correction_enabled.isChecked())

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
        self._update_tc_managed_enabled()

    def to_config(self) -> dict:
        cfg = json.loads(json.dumps(self.config))  # keep untouched keys (logging, etc.)
        cfg["provider"] = "gemini" if self.rb_gemini.isChecked() else "local_api"
        # "language" is owned by the main window's output-language dropdown now
        # (moved below the progress bar) -- deliberately left untouched here so
        # opening/accepting Settings can't clobber it with a stale value.

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
        cfg.setdefault("text_enhancement", {})
        cfg["text_enhancement"]["custom_vocabulary"] = vocab_lines
        cfg["text_enhancement"]["dedup_repeated_chunks"] = self.dedup_repeated_chunks.isChecked()
        cfg["text_enhancement"]["strip_infinite_repetition"] = self.strip_infinite_repetition.isChecked()
        cfg["text_enhancement"]["text_correction"] = {
            "enabled": self.text_correction_enabled.isChecked(),
            "server": {
                "url": self.tc_url.text().strip(),
                "launch_mode": "managed" if self.tc_rb_managed.isChecked() else "external",
                "server_binary": self.tc_server_binary.text().strip(),
                "model_path": self.tc_model_path.text().strip(),
                "port": self.tc_port.value(),
                "extra_args": self.tc_extra_args.text().strip(),
                "startup_timeout_sec": self.tc_startup_timeout.value(),
            },
            "sampling": {
                "temperature": self.tc_temperature.value(),
                "top_p": self.tc_top_p.value(),
                "top_k": self.tc_top_k.value(),
                "presence_penalty": self.tc_presence_penalty.value(),
                "repetition_penalty": self.tc_repetition_penalty.value(),
                "max_tokens": self.tc_max_tokens.value(),
            },
            "full_context": {
                "max_segment_chars": self.tc_max_segment_chars.value(),
                "segment_split_count": self.tc_segment_split_count.value(),
                "glossary_assist": self.tc_glossary_assist.isChecked(),
            },
            "speaker_detection": {
                "enabled": self.tc_speaker_enabled.isChecked(),
                "trigger_length_chars": self.tc_trigger_length_chars.value(),
            },
            "cue_splitter": {
                "cps_threshold": self.tc_cps_threshold.value(),
                "max_cue_duration_sec": self.tc_max_cue_duration_sec.value() or None,
                "min_cue_duration_sec": self.tc_min_cue_duration_sec.value() or None,
                "show_speaker_label": self.tc_show_speaker_label.isChecked(),
            },
        }

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

    def _update_tc_managed_enabled(self):
        managed = self.tc_rb_managed.isChecked()
        self.tc_port.setEnabled(managed)
        self.tc_extra_args.setEnabled(managed)
        self.tc_startup_timeout.setEnabled(managed)

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

    def _test_tc_connection(self):
        import requests
        cfg = self.to_config()
        tc_server = cfg["text_enhancement"]["text_correction"]["server"]
        if tc_server["launch_mode"] == "managed":
            missing = [k for k in ("server_binary", "model_path") if not tc_server[k]]
            if missing:
                QMessageBox.warning(self, "Test Connection", f"Managed 모드 필수 항목 누락: {', '.join(missing)}")
                return
        try:
            payload = {
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 8,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            resp = requests.post(tc_server["url"], json=payload, timeout=10)
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
