"""Main window (design.md SS7)."""

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from common import compute_job_id, job_dir as get_job_dir, load_config  # noqa: E402

from .settings_dialog import SettingsDialog
from .worker import TranscriptionWorker

SUPPORTED_EXTENSIONS = {
    ".wav", ".mp3", ".aac", ".m4a", ".flac", ".ogg",  # design.md SS4 audio
    ".mp4", ".mkv", ".webm", ".mov", ".avi",  # design.md SS4 video
}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Media Transcriber")
        self.resize(760, 560)
        self.setAcceptDrops(True)

        self.config = load_config()
        self.source_path: Path | None = None
        self.worker: TranscriptionWorker | None = None

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

        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("Transcript")
        self.btn_start.clicked.connect(self._on_start_stop_clicked)
        btn_row.addWidget(self.btn_start)
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
        self.phase_label.setText("중단됨 (재시작 시 이어서 진행 가능)")
        self._cleanup_worker()

    # -- Misc buttons ---------------------------------------------------------

    def _copy_srt(self):
        QApplication.clipboard().setText(self.srt_view.toPlainText())

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
