"""Main window (design.md SS7, v3.4: STT/번역/후처리 3-탭 구조 + STT 배치 큐)."""

import json
import shutil
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QTabWidget, QVBoxLayout, QWidget,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from common import CONFIG_STT_PATH, TEMP_ROOT, compute_job_id, job_dir as get_job_dir, load_stt_config  # noqa: E402

from .batch_queue import BatchQueue, BatchQueueItem
from .settings_dialog import SettingsDialog
from .srt_postprocess_tab import SrtPostprocessTab
from .translation_tab import TranslationTab
from .worker import TranscriptionWorker

SUPPORTED_EXTENSIONS = {
    ".wav", ".mp3", ".aac", ".m4a", ".flac", ".ogg", ".wma", ".opus",  # design.md SS4 audio
    ".mp4", ".mkv", ".webm", ".mov", ".avi", ".mpg", ".mpeg", ".wmv", ".ts", ".m2ts", ".3gp",  # SS4 video
}  # extract_audio.py just calls ffmpeg -i, which reads far more than SS4's original
   # example list -- these are the other containers actually asked for/commonly seen.

_QUEUE_STATUS_LABELS = {"waiting": "대기중", "running": "진행중", "done": "완료", "failed": "실패"}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Media Transcriber")
        self.resize(820, 640)
        self.setAcceptDrops(True)

        self.config = load_stt_config()
        self.source_path: Path | None = None
        self.worker: TranscriptionWorker | None = None
        self._drop_after_stop = False  # SS14.4 요청: Stop과 별개로 temp/{job_id} 완전 삭제
        self.queue = BatchQueue()  # SS7.1 STT 배치 큐
        self._current_item: BatchQueueItem | None = None  # queue item behind the running job, if any

        self._build_ui()
        self._update_controls()

    # -- UI ---------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs, stretch=1)

        self.tabs.addTab(self._build_stt_tab(), "STT")

        self.translation_tab = TranslationTab()
        self.translation_tab.mergedFile.connect(self._on_translation_merged)
        self.translation_tab.settingsRequested.connect(self._open_settings)
        self.tabs.addTab(self.translation_tab, "번역")

        self.postprocess_tab = SrtPostprocessTab()
        self.postprocess_tab.settingsRequested.connect(self._open_settings)
        self.tabs.addTab(self.postprocess_tab, "후처리")

        # Connected only after all tabs exist -- QTabWidget emits
        # currentChanged as soon as the first tab is added (index -1 -> 0),
        # which would fire _on_tab_changed before translation_tab/
        # postprocess_tab are assigned otherwise.
        self.tabs.currentChanged.connect(self._on_tab_changed)

    def _build_stt_tab(self) -> QWidget:
        w = QWidget()
        root = QVBoxLayout(w)

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

        # SS7.1 배치 큐 -- 항상 표시, 대기중 항목만 개별 제거 가능.
        root.addWidget(QLabel("대기열"))
        self.queue_list = QListWidget()
        self.queue_list.setMaximumHeight(90)
        root.addWidget(self.queue_list)
        self.btn_remove_from_queue = QPushButton("Remove Selected")
        self.btn_remove_from_queue.clicked.connect(self._on_remove_queue_item_clicked)
        root.addWidget(self.btn_remove_from_queue)

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
        self.btn_cancel = QPushButton("Full Cancel (Delete Temp Files)")
        self.btn_cancel.clicked.connect(self._on_cancel_clicked)
        btn_row.addWidget(self.btn_cancel)
        self.btn_copy = QPushButton("Copy")
        self.btn_copy.clicked.connect(self._copy_srt)
        btn_row.addWidget(self.btn_copy)
        btn_row.addStretch()
        self.btn_settings = QPushButton("Settings")
        self.btn_settings.clicked.connect(self._open_settings)
        btn_row.addWidget(self.btn_settings)
        root.addLayout(btn_row)

        return w

    # -- Tab switching (design.md SS7.2/SS7.3 자동 로드 체인) -------------------

    def _on_tab_changed(self, index: int):
        widget = self.tabs.widget(index)
        if widget is self.translation_tab:
            self.translation_tab.activate()
        elif widget is self.postprocess_tab:
            self.postprocess_tab.activate()

    def _on_translation_merged(self, path_str: str):
        self.postprocess_tab.notify_upstream_srt(Path(path_str))

    # -- File selection / drag & drop / batch queue (design.md SS7.1) -------

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if not urls:
            return
        paths = [Path(u.toLocalFile()) for u in urls]
        supported = [p for p in paths if p.suffix.lower() in SUPPORTED_EXTENSIONS]
        unsupported = [p for p in paths if p.suffix.lower() not in SUPPORTED_EXTENSIONS]
        if unsupported:
            names = ", ".join(p.name for p in unsupported)
            QMessageBox.warning(self, "지원하지 않는 형식", f"다음 파일은 지원하지 않는 형식이라 제외됩니다: {names}")
        if supported:
            self._handle_new_files(supported)

    def _select_file(self):
        exts = " ".join(f"*{e}" for e in sorted(SUPPORTED_EXTENSIONS))
        paths, _ = QFileDialog.getOpenFileNames(self, "파일 선택", filter=f"Media files ({exts});;All files (*)")
        if paths:
            self._handle_new_files([Path(p) for p in paths])

    def _handle_new_files(self, paths: list[Path]):
        """SS7.1: idle + single file behaves like pre-v3.4 (plain selection,
        no queue); everything else (idle + multiple, or any drop while
        running) appends to the visible queue without auto-starting."""
        running = self.worker is not None
        action = self.queue.handle_drop(paths, running=running)
        if action == "single":
            self._set_source(paths[0])
        self._refresh_queue_list()
        self._update_controls()

    def _set_source(self, path: Path):
        self.source_path = path
        self.file_label.setText(str(path))
        self.file_label.setStyleSheet("")
        self._update_controls()

    def _on_remove_queue_item_clicked(self):
        list_item = self.queue_list.currentItem()
        if list_item is None:
            return
        item = list_item.data(Qt.UserRole)
        self.queue.remove(item)
        self._refresh_queue_list()
        self._update_controls()

    def _refresh_queue_list(self):
        self.queue_list.clear()
        for item in self.queue.items:
            label = f"{_QUEUE_STATUS_LABELS.get(item.status, item.status)}   {item.path.name}"
            list_item = QListWidgetItem(label)
            list_item.setData(Qt.UserRole, item)
            self.queue_list.addItem(list_item)

    # -- Controls -----------------------------------------------------------

    def _update_controls(self):
        running = self.worker is not None
        can_start = self.source_path is not None or self.queue.next_waiting() is not None
        self.btn_start.setEnabled(running or can_start)
        self.btn_settings.setEnabled(not running)
        self.btn_start.setText("Stop" if running else "Transcript")
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

    # -- Job start/stop (design.md SS7.1 manual vs auto-advance) -------------

    def _on_start_stop_clicked(self):
        if self.worker is not None:
            self._request_stop()
        else:
            self._start_next_manual()

    def _start_next_manual(self):
        """User clicked [Transcript]: prefer the plain single-file selection
        (pre-v3.4 behavior), else start the next queued item manually --
        either way, a manually-started job still shows the Resume 3-button
        dialog if a previous temp exists (design.md SS7.1)."""
        path = self.source_path
        if path is None:
            item = self.queue.next_waiting()
            if item is None:
                return
            path = item.path
        self._begin_job(path, manual=True)

    def _resolve_resume(self, path: Path, manual: bool) -> tuple[bool, bool]:
        """Returns (proceed, resume). A manually-started job with an existing
        temp shows the 3-button dialog (may cancel -> proceed=False); an
        auto-advanced queue item with an existing temp always resumes
        silently (design.md SS7.1 "무인 배치 처리가 목적")."""
        job_dir_path = get_job_dir(path)
        has_existing = (job_dir_path / "manifest.json").exists()
        if not has_existing:
            return True, False
        if not manual:
            return True, True
        box = QMessageBox(self)
        box.setWindowTitle("이전 작업 발견")
        box.setText("이전 작업을 발견했습니다. 이어서 진행하시겠습니까?")
        btn_resume = box.addButton("Resume", QMessageBox.AcceptRole)
        box.addButton("Start Over", QMessageBox.DestructiveRole)
        btn_cancel = box.addButton("Cancel", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is btn_cancel:
            return False, False
        return True, box.clickedButton() is btn_resume

    def _begin_job(self, path: Path, manual: bool):
        proceed, resume = self._resolve_resume(path, manual)
        if not proceed:
            return

        item = next((it for it in self.queue.items if it.path == path and it.status == "waiting"), None)
        if item is not None:
            self.queue.start(item)
        self._current_item = item
        self.queue.clear_stopped()  # a fresh manual/auto start un-halts the queue

        self._set_source(path)
        self.config = load_stt_config()  # pick up any Settings changes since launch
        self.srt_view.clear()
        self.progress_bar.setValue(0)
        self.phase_label.setText("시작 중...")

        self.worker = TranscriptionWorker(path, self.config, resume)
        self.worker.phaseChanged.connect(self.phase_label.setText)
        self.worker.progressChanged.connect(self._on_progress)
        self.worker.srtUpdated.connect(self._on_srt_updated)
        self.worker.logMessage.connect(print)
        self.worker.jobFinished.connect(self._on_job_finished)
        self.worker.jobFailed.connect(self._on_job_failed)
        self.worker.jobStopped.connect(self._on_job_stopped)
        self.worker.start()
        self._refresh_queue_list()
        self._update_controls()

    def _request_stop(self):
        if self.worker is not None:
            self.worker.request_stop()
            self.queue.stop()  # SS7.1: Stop halts auto-advance too, queue stays intact
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

    def _finish_current_item_after_stop(self, cancelled: bool):
        if self._current_item is None:
            return
        if cancelled:
            # 완전 취소: temp already deleted, nothing left to resume -- drop
            # it from the queue entirely rather than leaving a dead entry.
            self.queue.items = [it for it in self.queue.items if it is not self._current_item]
        else:
            # plain Stop: temp is preserved, put it back to "waiting" so a
            # later manual [Transcript] click (or a future auto-advance) can
            # pick it up again.
            self._current_item.status = "waiting"

    def _maybe_auto_advance(self):
        self._refresh_queue_list()
        self._update_controls()
        if self.worker is not None or not self.queue.should_auto_advance():
            return
        next_item = self.queue.next_waiting()
        self._begin_job(next_item.path, manual=False)

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
        if self._current_item is not None:
            self.queue.finish(self._current_item, success=True)
        self._current_item = None
        self.translation_tab.notify_upstream_srt(Path(output_srt))
        self._cleanup_worker()
        self._maybe_auto_advance()

    def _on_job_failed(self, message: str):
        self.phase_label.setText("오류 발생")
        if self._current_item is not None:
            self.queue.finish(self._current_item, success=False)
        self._current_item = None
        QMessageBox.critical(self, "작업 실패", message)
        self._cleanup_worker()
        self._maybe_auto_advance()

    def _on_job_stopped(self):
        if self._drop_after_stop:
            self._drop_after_stop = False
            self._drop_job_dir()
            self.phase_label.setText("취소됨 (임시 파일 삭제됨)")
            self._finish_current_item_after_stop(cancelled=True)
        else:
            self.phase_label.setText("중단됨 (재시작 시 이어서 진행 가능)")
            self._finish_current_item_after_stop(cancelled=False)
        self._current_item = None
        self._cleanup_worker()
        self._refresh_queue_list()
        # no auto-advance here -- self.queue.stop() (set in _request_stop)
        # already halts it; user must press [Transcript] again (SS7.1).

    # -- Misc buttons ---------------------------------------------------------

    def _copy_srt(self):
        QApplication.clipboard().setText(self.srt_view.toPlainText())

    def _on_language_changed(self, *_args):
        language = self.language_combo.currentText().strip() or "auto"
        if self.config.get("language", "auto") == language:
            return
        self.config = load_stt_config()
        self.config["language"] = language
        CONFIG_STT_PATH.write_text(json.dumps(self.config, ensure_ascii=False, indent=2), encoding="utf-8")

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
