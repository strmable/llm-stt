"""Standalone SRT post-processing modal (postprocessing.md SS11 2차 분할,
design.md SS7/SS8.4).

Runs the CPS-based length-splitting step of the Cue Splitter against an
externally-translated SRT file, independently of the main Transcript job --
this is deliberately a separate manual tool from automatic Phase C (which
only does speaker-marker splitting, gui/worker.py's `_phase_c()`), because
splitting by length before translation would hand the translator sentence
fragments instead of whole sentences.
"""

import json
import sys
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QSpinBox, QVBoxLayout,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from common import CONFIG_PATH  # noqa: E402
from srt_postprocess import DEFAULT_CUE_CFG, postprocess_srt_text  # noqa: E402


class SrtPostprocessWorker(QThread):
    logMessage = Signal(str)
    progressChanged = Signal(int, int)
    finished_ok = Signal(str, str)  # (output_path, backup_path)
    failed = Signal(str)

    def __init__(self, input_path: Path, cue_cfg: dict, parent=None):
        super().__init__(parent)
        self.input_path = input_path
        self.cue_cfg = cue_cfg

    def run(self):
        try:
            self._run()
        except Exception as e:  # noqa: BLE001 -- surfaced to the dialog, not swallowed
            self.failed.emit(str(e))

    def _run(self):
        srt_text = self.input_path.read_text(encoding="utf-8")
        result = postprocess_srt_text(
            srt_text, self.cue_cfg,
            log=self.logMessage.emit,
            on_progress=lambda done, total: self.progressChanged.emit(done, total),
        )
        if not result:
            self.failed.emit("변환된 cue가 없습니다. 입력 SRT 형식을 확인하세요.")
            return

        backup_path = self.input_path.with_suffix(self.input_path.suffix + ".bak")
        backup_path.write_bytes(self.input_path.read_bytes())
        self.input_path.write_text(result, encoding="utf-8")
        self.finished_ok.emit(str(self.input_path), str(backup_path))


class SrtPostprocessDialog(QDialog):
    """design.md SS7 -- "후처리" button next to Settings, opens this modal.
    Flow: SRT 파일 선택 -> 옵션(CPS/길이/gap) -> 실행 -> 로그 출력."""

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SRT 후처리 (길이 분할)")
        self.resize(560, 480)
        self.setAcceptDrops(True)
        self.config = config
        self.input_path: Path | None = None
        self.worker: SrtPostprocessWorker | None = None

        self._build_ui()
        self._load_from_config()

    # -- UI -----------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)

        intro = QLabel(
            "외부 도구로 번역까지 마친 SRT 파일을 선택해, 한 화면에 너무 오래 표시되는 "
            "대사를 CPS(초당 문자수) 기준으로 여러 cue로 나눕니다. 원본은 같은 위치에 "
            "\"{파일명}.srt.bak\"으로 백업된 뒤, 같은 파일명으로 결과가 저장됩니다."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(intro)

        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("SRT 파일 :"))
        self.file_edit = QLineEdit()
        self.file_edit.setReadOnly(True)
        self.file_edit.setPlaceholderText("(번역 완료된 .srt 파일을 선택하거나 여기로 드래그하세요)")
        file_row.addWidget(self.file_edit, stretch=1)
        self.btn_select = QPushButton("파일 선택")
        self.btn_select.clicked.connect(self._select_file)
        file_row.addWidget(self.btn_select)
        layout.addLayout(file_row)

        opt_box = QGroupBox("옵션")
        opt_form = QFormLayout(opt_box)
        self.cps_threshold = QSpinBox()
        self.cps_threshold.setRange(1, 100)
        opt_form.addRow("CPS Threshold (초당 문자수)", self.cps_threshold)
        self.max_cue_duration_sec = QDoubleSpinBox()
        self.max_cue_duration_sec.setRange(0.0, 60.0)
        self.max_cue_duration_sec.setSingleStep(0.5)
        self.max_cue_duration_sec.setSpecialValueText("미설정 (null, 무제한)")
        opt_form.addRow("Max Cue Duration (sec)", self.max_cue_duration_sec)
        self.min_cue_duration_sec = QDoubleSpinBox()
        self.min_cue_duration_sec.setRange(0.0, 10.0)
        self.min_cue_duration_sec.setSingleStep(0.1)
        self.min_cue_duration_sec.setSpecialValueText("미설정 (null, 무제한)")
        opt_form.addRow("Min Cue Duration (sec)", self.min_cue_duration_sec)
        self.max_chars_per_cue = QSpinBox()
        self.max_chars_per_cue.setRange(0, 300)
        self.max_chars_per_cue.setSpecialValueText("미설정 (0, 무제한)")
        opt_form.addRow("Max Characters per Cue (화면당 최대 글자수)", self.max_chars_per_cue)
        self.gap_sec = QDoubleSpinBox()
        self.gap_sec.setRange(0.0, 2.0)
        self.gap_sec.setSingleStep(0.01)
        self.gap_sec.setDecimals(2)
        opt_form.addRow("분할된 cue 간 대기시간 (sec)", self.gap_sec)
        layout.addWidget(opt_box)

        self.btn_run = QPushButton("실행")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self._on_run_clicked)
        layout.addWidget(self.btn_run)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("실행 로그가 여기에 표시됩니다.")
        layout.addWidget(self.log_view, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_close = QPushButton("닫기")
        self.btn_close.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_close)
        layout.addLayout(btn_row)

    def _load_from_config(self):
        cue_cfg = {**DEFAULT_CUE_CFG, **self.config.get("srt_postprocess", {})}
        self.cps_threshold.setValue(cue_cfg.get("cps_threshold", 15))
        self.max_cue_duration_sec.setValue(cue_cfg.get("max_cue_duration_sec") or 0.0)
        self.min_cue_duration_sec.setValue(cue_cfg.get("min_cue_duration_sec") or 0.0)
        self.max_chars_per_cue.setValue(cue_cfg.get("max_chars_per_cue") or 0)
        self.gap_sec.setValue(cue_cfg.get("gap_sec", 0.08))

    def _current_cue_cfg(self) -> dict:
        return {
            "cps_threshold": self.cps_threshold.value(),
            "max_cue_duration_sec": self.max_cue_duration_sec.value() or None,
            "min_cue_duration_sec": self.min_cue_duration_sec.value() or None,
            "max_chars_per_cue": self.max_chars_per_cue.value() or None,
            "gap_sec": self.gap_sec.value(),
        }

    def _save_to_config(self):
        cfg = json.loads(json.dumps(self.config))
        cfg["srt_postprocess"] = self._current_cue_cfg()
        CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        self.config = cfg

    # -- File selection / drag & drop ------------------------------------

    def _select_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "SRT 파일 선택", filter="SRT files (*.srt);;All files (*)")
        if not path:
            return
        self._set_source(Path(path))

    def _set_source(self, path: Path):
        self.input_path = path
        self.file_edit.setText(str(self.input_path))
        self.btn_run.setEnabled(True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        if self.worker is not None and self.worker.isRunning():
            return
        urls = event.mimeData().urls()
        if not urls:
            return
        path = Path(urls[0].toLocalFile())
        if path.suffix.lower() != ".srt":
            QMessageBox.warning(self, "지원하지 않는 형식", f"{path.suffix} 형식은 지원하지 않습니다. .srt 파일을 선택하세요.")
            return
        self._set_source(path)

    # -- Run ------------------------------------------------------------------

    def _set_controls_enabled(self, enabled: bool):
        self.btn_select.setEnabled(enabled)
        self.btn_run.setEnabled(enabled and self.input_path is not None)
        self.cps_threshold.setEnabled(enabled)
        self.max_cue_duration_sec.setEnabled(enabled)
        self.min_cue_duration_sec.setEnabled(enabled)
        self.max_chars_per_cue.setEnabled(enabled)
        self.gap_sec.setEnabled(enabled)

    def _append_log(self, message: str):
        self.log_view.appendPlainText(message)

    def _on_run_clicked(self):
        assert self.input_path is not None
        self._save_to_config()
        self.log_view.clear()
        self.progress_bar.setValue(0)
        self._set_controls_enabled(False)

        self.worker = SrtPostprocessWorker(self.input_path, self._current_cue_cfg())
        self.worker.logMessage.connect(self._append_log)
        self.worker.progressChanged.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_progress(self, done: int, total: int):
        pct = int(100 * done / total) if total else 0
        self.progress_bar.setValue(pct)
        self.progress_bar.setFormat(f"{done}/{total} (%p%)")

    def _on_finished(self, output_path: str, backup_path: str):
        self._append_log(f"[완료] {output_path} (백업: {backup_path})")
        self._cleanup_worker()

    def _on_failed(self, message: str):
        self._append_log(f"[오류] {message}")
        QMessageBox.critical(self, "후처리 실패", message)
        self._cleanup_worker()

    def _cleanup_worker(self):
        if self.worker is not None:
            self.worker.wait()
        self.worker = None
        self._set_controls_enabled(True)

    # -- behavior ---------------------------------------------------------

    def reject(self):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "실행 중", "후처리가 끝난 뒤 닫을 수 있습니다.")
            return
        super().reject()
