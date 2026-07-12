"""Main window (design.md SS7)."""

import json
import shutil
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from common import CONFIG_PATH, TEMP_ROOT, compute_job_id, job_dir as get_job_dir, load_config  # noqa: E402

from .settings_dialog import SettingsDialog
from .worker import TranscriptionWorker

SUPPORTED_EXTENSIONS = {
    ".wav", ".mp3", ".aac", ".m4a", ".flac", ".ogg", ".wma", ".opus",  # design.md SS4 audio
    ".mp4", ".mkv", ".webm", ".mov", ".avi", ".mpg", ".mpeg", ".wmv", ".ts", ".m2ts", ".3gp",  # SS4 video
}  # extract_audio.py just calls ffmpeg -i, which reads far more than SS4's original
   # example list -- these are the other containers actually asked for/commonly seen.


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Media Transcriber")
        self.resize(760, 560)
        self.setAcceptDrops(True)

        self.config = load_config()
        self.source_path: Path | None = None
        self.worker: TranscriptionWorker | None = None
        self._drop_after_stop = False  # SS14.4 요청: Stop과 별개로 temp/{job_id} 완전 삭제

        self._build_ui()
        self._update_controls()

    # -- UI ---------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("File :"))
        self.file_label = QLabel("(파일을 선택하거나 여기로 드래그하세요)")
        self.file_label.setStyleSheet("color: gray;")
        file_row.addWidget(self.file_label, stretch=1)
        btn_select = QPushButton("Select File")
        btn_select.clicked.connect(self._select_file)
        file_row.addWidget(btn_select)
        root.addLayout(file_row)

        self.srt_view = QPlainTextEdit()
        self.srt_view.setReadOnly(True)
        self.srt_view.setPlaceholderText("SRT 결과가 여기에 실시간으로 표시됩니다.")
        root.addWidget(self.srt_view, stretch=1)

        self.phase_label = QLabel("")
        root.addWidget(self.phase_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        root.addWidget(self.progress_bar)

        lang_row = QHBoxLayout()
        lang_row.addWidget(QLabel("출력 언어 :"))
        self.language_combo = QComboBox()
        self.language_combo.setEditable(True)
        self.language_combo.setInsertPolicy(QComboBox.NoInsert)
        self.language_combo.addItems(["auto", "ko", "ja", "zh", "en"])
        self.language_combo.setCurrentText(self.config.get("language", "auto"))
        self.language_combo.activated.connect(self._on_language_changed)
        self.language_combo.lineEdit().editingFinished.connect(self._on_language_changed)
        lang_row.addWidget(self.language_combo)
        lang_row.addStretch()
        root.addLayout(lang_row)

        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("Transcript")
        self.btn_start.clicked.connect(self._on_start_stop_clicked)
        btn_row.addWidget(self.btn_start)
        self.btn_cancel = QPushButton("완전 취소 (임시파일 삭제)")
        self.btn_cancel.clicked.connect(self._on_cancel_clicked)
        btn_row.addWidget(self.btn_cancel)
        self.btn_copy = QPushButton("Copy")
        self.btn_copy.clicked.connect(self._copy_srt)
        btn_row.addWidget(self.btn_copy)
        self.btn_settings = QPushButton("Settings")
        self.btn_settings.clicked.connect(self._open_settings)
        btn_row.addWidget(self.btn_settings)
        btn_row.addStretch()
        root.addLayout(btn_row)

    # -- File selection / drag & drop --------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if not urls:
            return
        path = Path(urls[0].toLocalFile())
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            QMessageBox.warning(self, "지원하지 않는 형식", f"{path.suffix} 형식은 지원하지 않습니다.")
            return
        self._set_source(path)

    def _select_file(self):
        exts = " ".join(f"*{e}" for e in sorted(SUPPORTED_EXTENSIONS))
        path, _ = QFileDialog.getOpenFileName(self, "파일 선택", filter=f"Media files ({exts});;All files (*)")
        if path:
            self._set_source(Path(path))

    def _set_source(self, path: Path):
        self.source_path = path
        self.file_label.setText(str(path))
        self.file_label.setStyleSheet("")
        self._update_controls()

    # -- Controls -----------------------------------------------------------

    def _update_controls(self):
        running = self.worker is not None
        self.btn_start.setEnabled(self.source_path is not None)
        self.btn_settings.setEnabled(not running)
        if running:
            self.btn_start.setText("Stop")
        else:
            self.btn_start.setText("Transcript")
        self.btn_cancel.setEnabled(self.source_path is not None and (running or self._job_dir_exists()))

    def _job_dir_exists(self) -> bool:
        """True if a resumable temp/{job_id}/manifest.json exists for the
        currently selected file -- gates the "완전 취소" button when idle."""
        if self.source_path is None or not self.source_path.exists():
            return False
        job_id = compute_job_id(self.source_path)
        return (TEMP_ROOT / job_id / "manifest.json").exists()

    def _drop_job_dir(self):
        """Delete temp/{job_id} outright (design.md SS14.4's normal cleanup is
        success-only; this is the explicit user-requested "완전 취소" -- forfeits
        Resume for this file, unlike a plain Stop which always preserves it)."""
        if self.source_path is None:
            return
        job_id = compute_job_id(self.source_path)
        path = TEMP_ROOT / job_id
        if path.exists():
            shutil.rmtree(path)

    def _on_start_stop_clicked(self):
        if self.worker is not None:
            self._request_stop()
        else:
            self._start_job()

    def _start_job(self):
        assert self.source_path is not None
        job_dir_path = get_job_dir(self.source_path)
        resume = False
        if (job_dir_path / "manifest.json").exists():
            box = QMessageBox(self)
            box.setWindowTitle("이전 작업 발견")
            box.setText("이전 작업을 발견했습니다. 이어서 진행하시겠습니까?")
            btn_resume = box.addButton("이어하기", QMessageBox.AcceptRole)
            box.addButton("새로 시작", QMessageBox.DestructiveRole)
            btn_cancel = box.addButton("취소", QMessageBox.RejectRole)
            box.exec()
            if box.clickedButton() is btn_cancel:
                return
            resume = box.clickedButton() is btn_resume

        self.config = load_config()  # pick up any Settings changes since launch
        self.srt_view.clear()
        self.progress_bar.setValue(0)
        self.phase_label.setText("시작 중...")

        self.worker = TranscriptionWorker(self.source_path, self.config, resume)
        self.worker.phaseChanged.connect(self.phase_label.setText)
        self.worker.progressChanged.connect(self._on_progress)
        self.worker.srtUpdated.connect(self._on_srt_updated)
        self.worker.logMessage.connect(print)
        self.worker.jobFinished.connect(self._on_job_finished)
        self.worker.jobFailed.connect(self._on_job_failed)
        self.worker.jobStopped.connect(self._on_job_stopped)
        self.worker.start()
        self._update_controls()

    def _request_stop(self):
        if self.worker is not None:
            self.worker.request_stop()
            self.phase_label.setText("중단 요청됨 -- 현재 chunk 완료 후 정지합니다...")
            self.btn_start.setEnabled(False)

    def _on_cancel_clicked(self):
        if self.worker is not None:
            reply = QMessageBox.question(
                self, "완전 취소",
                "작업을 중단하고 임시 파일을 삭제합니다. 진행 상황은 복구할 수 없으며 다음 실행은 "
                "처음부터 다시 시작합니다. 계속하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
            self._drop_after_stop = True
            self._request_stop()
            self.btn_cancel.setEnabled(False)
            return

        reply = QMessageBox.question(
            self, "완전 취소",
            "이 파일의 임시 작업 데이터(temp/)를 삭제합니다. 이어하기(Resume)가 더 이상 불가능하며 "
            "다음 실행은 처음부터 다시 시작합니다. 계속하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._drop_job_dir()
        self.phase_label.setText("취소됨 (임시 파일 삭제됨)")
        self._update_controls()

    def _cleanup_worker(self):
        if self.worker is not None:
            self.worker.wait()
        self.worker = None
        self._update_controls()

    # -- Worker signal handlers -----------------------------------------------

    def _on_progress(self, done: int, total: int):
        pct = int(100 * done / total) if total else 0
        self.progress_bar.setValue(pct)
        self.progress_bar.setFormat(f"{done}/{total} (%p%)")

    def _on_srt_updated(self, text: str):
        self.srt_view.setPlainText(text)
        cursor = self.srt_view.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.srt_view.setTextCursor(cursor)

    def _on_job_finished(self, output_srt: str):
        self.phase_label.setText(f"완료: {output_srt}")
        self._cleanup_worker()

    def _on_job_failed(self, message: str):
        self.phase_label.setText("오류 발생")
        QMessageBox.critical(self, "작업 실패", message)
        self._cleanup_worker()

    def _on_job_stopped(self):
        if self._drop_after_stop:
            self._drop_after_stop = False
            self._drop_job_dir()
            self.phase_label.setText("취소됨 (임시 파일 삭제됨)")
        else:
            self.phase_label.setText("중단됨 (재시작 시 이어서 진행 가능)")
        self._cleanup_worker()

    # -- Misc buttons ---------------------------------------------------------

    def _copy_srt(self):
        QApplication.clipboard().setText(self.srt_view.toPlainText())

    def _on_language_changed(self, *_args):
        language = self.language_combo.currentText().strip() or "auto"
        if self.config.get("language", "auto") == language:
            return
        self.config = load_config()
        self.config["language"] = language
        CONFIG_PATH.write_text(json.dumps(self.config, ensure_ascii=False, indent=2), encoding="utf-8")

    def _open_settings(self):
        dialog = SettingsDialog(self.config, self)
        if dialog.exec():
            self.config = dialog.config

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            reply = QMessageBox.question(
                self, "작업 중",
                "작업이 진행 중입니다. 중단하고 종료하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return
            self.worker.request_stop()
            self.worker.wait()
        event.accept()
