"""번역 탭 (design.md SS5C, SS7.2, v3.4).

Marker-based translation workflow: converts an SRT to `[nnnn]|` marker-tagged
plain text (pipeline/translation.py's build_markers), offers local-LLM
Preprocess/Translate (batched, same server-lifecycle pattern as Phase C's
text_correction.py/gui/worker.py) plus Copy/Paste round-trip to external
tools (DeepL etc.), and Merges the right pane back into the original SRT.
"""

import sys
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QSplitter, QVBoxLayout, QWidget,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from build_srt import parse_srt, render_srt  # noqa: E402
from common import load_stt_config, load_translate_config  # noqa: E402
from server_manager import adapt_flat_server_config, ensure_llama_server  # noqa: E402
import translation as translation_lib  # noqa: E402

DEFAULT_TRANSLATION_CFG = {
    "target_language": "en",
    "context_max_chars": 20000,
    "preprocess_prompt": "",
    "translate_prompt": "",
    "server": {
        "url": "http://localhost:8082/v1/chat/completions",
        "launch_mode": "external",
        "server_binary": "",
        "model_path": "",
        "port": 8082,
        "extra_args": "",
        "startup_timeout_sec": 120,
    },
}


class TranslationBatchWorker(QThread):
    logMessage = Signal(str)
    progressChanged = Signal(int, int)
    finished_ok = Signal(dict)  # {serial: text}
    failed = Signal(str)

    def __init__(self, cues: list[dict], prompt_template: str, max_chars: int, server_cfg: dict,
                 target_language: str, vocabulary: list[str], parent=None):
        super().__init__(parent)
        self.cues = cues
        self.prompt_template = prompt_template
        self.max_chars = max_chars
        self.server_cfg = server_cfg
        self.target_language = target_language
        self.vocabulary = vocabulary
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        try:
            self._run()
        except Exception as e:  # noqa: BLE001 -- surfaced to the tab, not swallowed
            self.failed.emit(str(e))

    def _run(self):
        url = self.server_cfg.get("url", DEFAULT_TRANSLATION_CFG["server"]["url"])
        server_base = url.split("/v1/")[0]
        server_manager_cfg = adapt_flat_server_config(self.server_cfg, default_port=8082)
        if self.server_cfg.get("launch_mode", "external") == "managed":
            self.logMessage.emit("[translation] 번역 서버 준비 중...")
        with ensure_llama_server(server_base, server_manager_cfg, log_path=None):
            result = translation_lib.run_batches(
                self.cues, self.prompt_template, self.max_chars, url,
                target_language=self.target_language, vocabulary=self.vocabulary,
                log=self.logMessage.emit, should_stop=lambda: self._stop_requested,
                on_progress=lambda done, total: self.progressChanged.emit(done, total),
            )
        self.finished_ok.emit(result)


class TranslationTab(QWidget):
    """design.md SS7.2/SS5C.4 -- 원문/번역문 two-pane 편집기 +
    Preprocess/Copy/Translate/Paste/Merge. Emits `mergedFile` after a
    successful Merge so MainWindow can hand the result to the 후처리 탭
    (SS7.3's "직전 탭 산출물 자동 로드" chain)."""

    mergedFile = Signal(str)
    settingsRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.input_path: Path | None = None
        self.cues: list[dict] = []
        self.worker: TranslationBatchWorker | None = None
        self._pending_auto_path: Path | None = None
        self._user_loaded = False

        self._build_ui()

    # -- UI -----------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)

        file_row = QHBoxLayout()
        self.btn_open = QPushButton("Open File")
        self.btn_open.clicked.connect(self._select_file)
        file_row.addWidget(self.btn_open)
        self.file_label = QLabel("(Select an SRT file or drag and drop one here)")
        self.file_label.setStyleSheet("color: gray;")
        file_row.addWidget(self.file_label, stretch=1)
        layout.addLayout(file_row)

        splitter = QSplitter()
        self.left_edit = QPlainTextEdit()
        self.left_edit.setPlaceholderText("Source text (with [nnnn]| markers)")
        splitter.addWidget(self.left_edit)
        self.right_edit = QPlainTextEdit()
        self.right_edit.setPlaceholderText("Translated text")
        splitter.addWidget(self.right_edit)
        layout.addWidget(splitter, stretch=1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(self.status_label)

        # STT 탭과 동일하게 progress bar 아래에 버튼 줄을 둔다 (v3.4 후속 정정).
        btn_row = QHBoxLayout()
        self.btn_preprocess = QPushButton("Preprocess")
        self.btn_preprocess.clicked.connect(self._on_preprocess_clicked)
        btn_row.addWidget(self.btn_preprocess)
        self.btn_copy = QPushButton("Copy")
        self.btn_copy.clicked.connect(self._on_copy_clicked)
        btn_row.addWidget(self.btn_copy)
        self.btn_translate = QPushButton("Translate")
        self.btn_translate.clicked.connect(self._on_translate_clicked)
        btn_row.addWidget(self.btn_translate)
        self.btn_paste = QPushButton("Paste")
        self.btn_paste.clicked.connect(self._on_paste_clicked)
        btn_row.addWidget(self.btn_paste)
        self.btn_merge = QPushButton("Merge")
        self.btn_merge.clicked.connect(self._on_merge_clicked)
        btn_row.addWidget(self.btn_merge)
        btn_row.addStretch()
        self.btn_settings = QPushButton("Settings")
        self.btn_settings.clicked.connect(self.settingsRequested.emit)
        btn_row.addWidget(self.btn_settings)
        layout.addLayout(btn_row)

        self._update_buttons_enabled(not self.is_running())

    # -- File loading / auto-load chain (design.md SS7.2) --------------------

    def is_running(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def notify_upstream_srt(self, path: Path):
        """Called by MainWindow when the STT tab finishes a job -- remembers
        the path but only loads it once this tab is actually switched to
        (SS7.2), and never overrides a file the user picked themselves."""
        self._pending_auto_path = Path(path)

    def activate(self):
        """Called by MainWindow when this tab becomes the current tab."""
        if self._pending_auto_path is not None and not self._user_loaded and not self.is_running():
            self.load_file(self._pending_auto_path)
        self._pending_auto_path = None

    def load_file(self, path: Path):
        if self.is_running():
            return
        path = Path(path)
        if not path.exists():
            return
        srt_text = path.read_text(encoding="utf-8")
        self.cues = parse_srt(srt_text, preserve_newlines=True)
        self.input_path = path
        self.file_label.setText(str(path))
        self.file_label.setStyleSheet("")
        self.left_edit.setPlainText(translation_lib.build_markers(self.cues))
        self.right_edit.clear()
        self.status_label.setText(f"{len(self.cues)}개 cue 로드됨")

    def _select_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "SRT 파일 선택", filter="SRT files (*.srt);;All files (*)")
        if not path:
            return
        self._user_loaded = True
        self.load_file(Path(path))

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        if self.is_running():
            return
        urls = event.mimeData().urls()
        if not urls:
            return
        path = Path(urls[0].toLocalFile())
        if path.suffix.lower() != ".srt":
            QMessageBox.warning(self, "지원하지 않는 형식", f"{path.suffix} 형식은 지원하지 않습니다. .srt 파일을 선택하세요.")
            return
        self._user_loaded = True
        self.load_file(path)

    # -- Config ---------------------------------------------------------------

    def _translation_cfg(self) -> dict:
        cfg = load_translate_config().get("translation", {})
        return {**DEFAULT_TRANSLATION_CFG, **cfg, "server": {**DEFAULT_TRANSLATION_CFG["server"], **cfg.get("server", {})}}

    def _vocabulary(self) -> list[str]:
        return load_stt_config().get("text_enhancement", {}).get("custom_vocabulary", [])

    def _current_left_cues(self) -> list[dict]:
        """Re-parses the left pane's current text back onto self.cues'
        timings -- Preprocess/Translate always run over whatever's currently
        shown (a prior Preprocess result, or manual edits), not the
        originally-loaded text (design.md SS5C.6)."""
        parsed = translation_lib.parse_markers(self.left_edit.toPlainText())
        return [{**cue, "text": parsed.get(serial, cue["text"])} for serial, cue in enumerate(self.cues, 1)]

    # -- Buttons (design.md SS5C.4) --------------------------------------------

    def _update_buttons_enabled(self, enabled: bool):
        for btn in (self.btn_open, self.btn_preprocess, self.btn_copy,
                    self.btn_translate, self.btn_paste, self.btn_merge):
            btn.setEnabled(enabled)

    def _on_copy_clicked(self):
        QApplication.clipboard().setText(self.left_edit.toPlainText())

    def _on_paste_clicked(self):
        self.right_edit.setPlainText(QApplication.clipboard().text())

    def _on_preprocess_clicked(self):
        self._run_batch_job(
            prompt_key="preprocess_prompt",
            target="left",
        )

    def _on_translate_clicked(self):
        self._run_batch_job(
            prompt_key="translate_prompt",
            target="right",
        )

    def _run_batch_job(self, prompt_key: str, target: str):
        if self.input_path is None or not self.cues:
            QMessageBox.warning(self, "번역", "먼저 SRT 파일을 로드하세요.")
            return
        cfg = self._translation_cfg()
        prompt_template = cfg.get(prompt_key, "")
        working_cues = self._current_left_cues()

        self.progress_bar.setValue(0)
        self.status_label.setText("처리 중...")
        self._update_buttons_enabled(False)

        self.worker = TranslationBatchWorker(
            working_cues, prompt_template, cfg.get("context_max_chars", 20000), cfg.get("server", {}),
            cfg.get("target_language", "en"), self._vocabulary(),
        )
        self.worker.logMessage.connect(print)
        self.worker.progressChanged.connect(self._on_progress)
        self.worker.finished_ok.connect(lambda result: self._on_batch_finished(result, working_cues, target))
        self.worker.failed.connect(self._on_batch_failed)
        self.worker.start()

    def _on_progress(self, done: int, total: int):
        pct = int(100 * done / total) if total else 0
        self.progress_bar.setValue(pct)
        self.status_label.setText(f"배치 {done}/{total} 처리 중...")

    def _on_batch_finished(self, result: dict, working_cues: list[dict], target: str):
        if target == "left":
            merged, missing = translation_lib.merge_translation(working_cues, result)
            self.cues = merged
            self.left_edit.setPlainText(translation_lib.build_markers(self.cues))
            self.status_label.setText(f"Preprocess 완료 ({len(merged) - len(missing)}/{len(merged)} cue)")
        else:
            self.right_edit.setPlainText(translation_lib.render_translated_map(result))
            self.status_label.setText(f"Translate 완료 ({len(result)}/{len(working_cues)} cue)")
        self._cleanup_worker()

    def _on_batch_failed(self, message: str):
        self.status_label.setText("오류 발생")
        QMessageBox.critical(self, "번역 실패", message)
        self._cleanup_worker()

    def _cleanup_worker(self):
        if self.worker is not None:
            self.worker.wait()
        self.worker = None
        self._update_buttons_enabled(True)

    def _on_merge_clicked(self):
        if self.input_path is None or not self.cues:
            QMessageBox.warning(self, "번역", "먼저 SRT 파일을 로드하세요.")
            return
        translated_map = translation_lib.parse_markers(self.right_edit.toPlainText())
        merged, missing = translation_lib.merge_translation(self.cues, translated_map)

        backup_path = self.input_path.with_suffix(self.input_path.suffix + ".bak")
        backup_path.write_bytes(self.input_path.read_bytes())
        self.input_path.write_text(render_srt(merged), encoding="utf-8")

        n = len(merged)
        matched = n - len(missing)
        msg = f"원본 cue 수 {n}개 / 번역 매칭 {matched}개 (누락 {len(missing)}개)"
        if missing:
            msg += f" -- 누락 시리얼: {missing}"
        self.status_label.setText(msg)
        print(f"[translation] Merge: {msg} -> {self.input_path} (백업: {backup_path})")

        self.mergedFile.emit(str(self.input_path))
