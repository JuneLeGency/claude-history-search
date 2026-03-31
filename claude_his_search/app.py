"""Claude Code 对话历史搜索 GUI 主程序"""

import sys
import json
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QLineEdit, QPushButton, QListWidget, QListWidgetItem,
    QTextBrowser, QLabel, QProgressBar, QComboBox, QDialog,
    QFormLayout, QDialogButtonBox, QMessageBox, QTabWidget,
    QSpinBox, QToolBar, QStatusBar,
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QAction

from .config import AppConfig, detect_backend, BACKEND_MLX, BACKEND_TORCH
from .history_parser import scan_all_sessions, Session, Message
from .embedding_engine import EmbeddingEngine


# ── Worker Threads ──────────────────────────────────────────────


class ScanWorker(QThread):
    finished = Signal(list)
    progress = Signal(str)

    def run(self):
        self.progress.emit("正在扫描会话文件...")
        sessions = scan_all_sessions()
        self.progress.emit(f"扫描完成，共 {len(sessions)} 个会话")
        self.finished.emit(sessions)


class IndexWorker(QThread):
    finished = Signal(int)  # 新增条数
    progress = Signal(int, int, str)
    error = Signal(str)

    def __init__(self, engine: EmbeddingEngine, sessions: list[Session], force_rebuild: bool = False):
        super().__init__()
        self.engine = engine
        self.sessions = sessions
        self.force_rebuild = force_rebuild

    def run(self):
        try:
            old_size = self.engine.index_size
            self.engine.build_index(
                self.sessions,
                progress_callback=self._on_progress,
                force_rebuild=self.force_rebuild,
            )
            new_count = self.engine.index_size - old_size
            self.finished.emit(max(new_count, 0))
        except Exception as e:
            self.error.emit(str(e))

    def _on_progress(self, current, total, msg):
        self.progress.emit(current, total, msg)


class SemanticSearchWorker(QThread):
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, engine: EmbeddingEngine, query: str):
        super().__init__()
        self.engine = engine
        self.query = query

    def run(self):
        try:
            results = self.engine.search(self.query)
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class ModelLoadWorker(QThread):
    """后台预加载模型"""
    finished = Signal()
    error = Signal(str)

    def __init__(self, engine: EmbeddingEngine):
        super().__init__()
        self.engine = engine

    def run(self):
        try:
            self.engine._ensure_backend()
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


# ── Config Dialog ───────────────────────────────────────────────


class ConfigDialog(QDialog):
    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("设置")
        self.setMinimumWidth(520)

        layout = QFormLayout(self)

        # 后端选择
        detected = detect_backend()
        self.backend_combo = QComboBox()
        self.backend_combo.addItem("自动检测", "auto")
        self.backend_combo.addItem("MLX (Apple Silicon 推荐)", "mlx")
        self.backend_combo.addItem("PyTorch (sentence-transformers)", "torch")
        idx = self.backend_combo.findData(config.backend)
        if idx >= 0:
            self.backend_combo.setCurrentIndex(idx)
        layout.addRow("推理后端:", self.backend_combo)

        backend_info = QLabel(f"当前检测到: {detected or '无 (请安装 mlx 或 torch)'}")
        backend_info.setStyleSheet("color: #a6e3a1; font-size: 12px;" if detected else "color: #f38ba8; font-size: 12px;")
        layout.addRow("", backend_info)

        # 模型源
        self.source_combo = QComboBox()
        self.source_combo.addItem("ModelScope (国内推荐)", "modelscope")
        self.source_combo.addItem("HuggingFace", "huggingface")
        idx = self.source_combo.findData(config.model_source)
        if idx >= 0:
            self.source_combo.setCurrentIndex(idx)
        layout.addRow("模型源:", self.source_combo)

        # 模型名
        self.model_edit = QLineEdit(config.embedding_model)
        self.model_edit.setPlaceholderText("Qwen/Qwen3-Embedding-0.6B")
        layout.addRow("Embedding 模型:", self.model_edit)

        info = QLabel("首次使用会自动下载模型 (~1.2GB)")
        info.setStyleSheet("color: #6c7086; font-size: 12px;")
        layout.addRow("", info)

        # 维度
        self.dim_spin = QSpinBox()
        self.dim_spin.setRange(32, 1024)
        self.dim_spin.setValue(config.embedding_dimension)
        layout.addRow("Embedding 维度:", self.dim_spin)

        dim_info = QLabel("维度越高精度越好但索引越大，推荐 512")
        dim_info.setStyleSheet("color: #6c7086; font-size: 12px;")
        layout.addRow("", dim_info)

        # 每页消息数
        self.page_size_spin = QSpinBox()
        self.page_size_spin.setRange(10, 500)
        self.page_size_spin.setValue(config.page_size)
        layout.addRow("每页消息数:", self.page_size_spin)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def accept(self):
        self.config.backend = self.backend_combo.currentData()
        self.config.model_source = self.source_combo.currentData()
        self.config.embedding_model = self.model_edit.text().strip()
        self.config.embedding_dimension = self.dim_spin.value()
        self.config.page_size = self.page_size_spin.value()
        self.config.save()
        super().accept()


# ── Main Window ─────────────────────────────────────────────────


CHAT_CSS = """
body { font-family: -apple-system, 'Segoe UI', sans-serif; font-size: 14px; margin: 8px; background: #1e1e2e; color: #cdd6f4; }
.msg { margin: 8px 0; padding: 10px 14px; border-radius: 10px; max-width: 90%; line-height: 1.5; }
.user { background: #313244; border-left: 3px solid #89b4fa; }
.assistant { background: #1e1e2e; border-left: 3px solid #a6e3a1; }
.role { font-size: 11px; font-weight: 600; margin-bottom: 4px; }
.role-user { color: #89b4fa; }
.role-assistant { color: #a6e3a1; }
.ts { font-size: 10px; color: #6c7086; margin-top: 4px; }
.highlight { background: #f9e2af33; border: 1px solid #f9e2af; }
pre { background: #181825; padding: 8px; border-radius: 6px; overflow-x: auto; font-size: 13px; }
code { font-family: 'Menlo', 'Fira Code', monospace; font-size: 13px; }
a { color: #89b4fa; }
"""


def _msg_to_html(msg: Message, anchor_id: str = "", highlight: bool = False) -> str:
    import html as html_mod
    import re

    hl = " highlight" if highlight else ""
    role_cls = "role-user" if msg.role == "user" else "role-assistant"
    side_cls = "user" if msg.role == "user" else "assistant"
    role_label = "You" if msg.role == "user" else "Assistant"
    ts = msg.timestamp[:19].replace("T", " ") if msg.timestamp else ""

    text = html_mod.escape(msg.content)
    text = re.sub(r'```(\w*)\n(.*?)```', r'<pre><code>\2</code></pre>', text, flags=re.DOTALL)
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    text = text.replace("\n", "<br>")

    anchor = f'<a name="{anchor_id}"></a>' if anchor_id else ""
    return (
        f'{anchor}<div class="msg {side_cls}{hl}" id="{anchor_id}">'
        f'<div class="role {role_cls}">{role_label}</div>'
        f'{text}'
        f'<div class="ts">{ts}</div>'
        f'</div>'
    )


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = AppConfig.load()
        self.engine = EmbeddingEngine(self.config)
        self.sessions: list[Session] = []
        self.filtered_sessions: list[Session] = []
        self.current_session: Session | None = None
        self._current_page = 0
        self._workers = []
        self._model_loaded = False
        self._indexing = False  # 索引构建中防重入

        self.setWindowTitle("Claude Code 对话历史搜索")
        self.resize(1400, 900)
        self._build_ui()
        self._start_scan()

    def _build_ui(self):
        # Toolbar
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        settings_action = QAction("设置", self)
        settings_action.triggered.connect(self._open_settings)
        toolbar.addAction(settings_action)

        build_index_action = QAction("更新索引", self)
        build_index_action.triggered.connect(self._build_index)
        toolbar.addAction(build_index_action)

        rebuild_index_action = QAction("重建索引", self)
        rebuild_index_action.triggered.connect(self._rebuild_index)
        toolbar.addAction(rebuild_index_action)

        clear_index_action = QAction("清除索引", self)
        clear_index_action.triggered.connect(self._clear_index)
        toolbar.addAction(clear_index_action)

        refresh_action = QAction("刷新会话", self)
        refresh_action.triggered.connect(self._start_scan)
        toolbar.addAction(refresh_action)

        preload_action = QAction("预加载模型", self)
        preload_action.triggered.connect(self._preload_model)
        toolbar.addAction(preload_action)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(300)
        self.progress_bar.hide()
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.model_status_label = QLabel("")
        self.status_bar.addPermanentWidget(self.model_status_label)

        # Main layout
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # Search tabs
        search_tabs = QTabWidget()
        search_tabs.setMaximumHeight(90)

        # Tab 1: 关键词搜索
        kw_widget = QWidget()
        kw_layout = QHBoxLayout(kw_widget)
        kw_layout.setContentsMargins(8, 8, 8, 8)
        self.keyword_input = QLineEdit()
        self.keyword_input.setPlaceholderText("关键词搜索会话内容...")
        self.keyword_input.returnPressed.connect(self._keyword_search)
        kw_layout.addWidget(self.keyword_input)
        kw_btn = QPushButton("搜索")
        kw_btn.clicked.connect(self._keyword_search)
        kw_layout.addWidget(kw_btn)
        clear_btn = QPushButton("清除")
        clear_btn.clicked.connect(self._clear_search)
        kw_layout.addWidget(clear_btn)
        search_tabs.addTab(kw_widget, "关键词搜索")

        # Tab 2: 语义搜索
        sem_widget = QWidget()
        sem_layout = QHBoxLayout(sem_widget)
        sem_layout.setContentsMargins(8, 8, 8, 8)
        self.semantic_input = QLineEdit()
        self.semantic_input.setPlaceholderText("用自然语言描述你要找的内容 (需先构建索引)...")
        self.semantic_input.returnPressed.connect(self._semantic_search)
        sem_layout.addWidget(self.semantic_input)
        sem_btn = QPushButton("语义搜索")
        sem_btn.clicked.connect(self._semantic_search)
        sem_layout.addWidget(sem_btn)
        self.index_label = QLabel(f"索引: {self.engine.index_size} 条")
        sem_layout.addWidget(self.index_label)
        search_tabs.addTab(sem_widget, "语义搜索 (本地 Embedding)")

        main_layout.addWidget(search_tabs)

        # Splitter: session list | chat view
        splitter = QSplitter(Qt.Horizontal)

        # Left panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.session_count_label = QLabel("会话列表")
        left_layout.addWidget(self.session_count_label)
        self.session_list = QListWidget()
        self.session_list.currentRowChanged.connect(self._on_session_selected)
        left_layout.addWidget(self.session_list)
        splitter.addWidget(left_panel)

        # Right panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        header_layout = QHBoxLayout()
        self.chat_title = QLabel("选择一个会话")
        self.chat_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        header_layout.addWidget(self.chat_title)
        header_layout.addStretch()
        self.copy_resume_btn = QPushButton("复制 Resume 命令")
        self.copy_resume_btn.clicked.connect(self._copy_resume)
        self.copy_resume_btn.setEnabled(False)
        header_layout.addWidget(self.copy_resume_btn)
        right_layout.addLayout(header_layout)

        self.chat_browser = QTextBrowser()
        self.chat_browser.setOpenExternalLinks(False)
        self.chat_browser.setFont(QFont("Menlo", 13))
        right_layout.addWidget(self.chat_browser)

        page_layout = QHBoxLayout()
        self.prev_btn = QPushButton("上一页")
        self.prev_btn.clicked.connect(self._prev_page)
        self.prev_btn.setEnabled(False)
        page_layout.addWidget(self.prev_btn)
        self.page_label = QLabel("")
        page_layout.addWidget(self.page_label)
        self.next_btn = QPushButton("下一页")
        self.next_btn.clicked.connect(self._next_page)
        self.next_btn.setEnabled(False)
        page_layout.addWidget(self.next_btn)
        page_layout.addStretch()
        right_layout.addLayout(page_layout)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        main_layout.addWidget(splitter)

        # Style (Catppuccin Mocha)
        self.setStyleSheet("""
            QMainWindow { background: #1e1e2e; }
            QWidget { background: #1e1e2e; color: #cdd6f4; }
            QLineEdit { background: #313244; border: 1px solid #45475a; border-radius: 6px;
                         padding: 6px 10px; color: #cdd6f4; font-size: 14px; }
            QLineEdit:focus { border-color: #89b4fa; }
            QPushButton { background: #313244; border: 1px solid #45475a; border-radius: 6px;
                          padding: 6px 16px; color: #cdd6f4; font-size: 13px; }
            QPushButton:hover { background: #45475a; }
            QPushButton:pressed { background: #585b70; }
            QPushButton:disabled { color: #6c7086; }
            QListWidget { background: #181825; border: 1px solid #313244; border-radius: 6px;
                          font-size: 12px; color: #cdd6f4; }
            QListWidget::item { padding: 6px 8px; border-bottom: 1px solid #313244; }
            QListWidget::item:selected { background: #313244; color: #89b4fa; }
            QTextBrowser { background: #1e1e2e; border: 1px solid #313244; border-radius: 6px; }
            QTabWidget::pane { border: 1px solid #313244; border-radius: 6px; }
            QTabBar::tab { background: #181825; color: #6c7086; padding: 6px 16px;
                           border-top-left-radius: 6px; border-top-right-radius: 6px; }
            QTabBar::tab:selected { background: #313244; color: #cdd6f4; }
            QToolBar { background: #181825; border-bottom: 1px solid #313244; spacing: 8px; padding: 4px; }
            QStatusBar { background: #181825; color: #6c7086; }
            QProgressBar { background: #313244; border-radius: 4px; text-align: center; color: #cdd6f4; }
            QProgressBar::chunk { background: #89b4fa; border-radius: 4px; }
            QLabel { color: #cdd6f4; }
        """)

    # ── Scan ──

    def _start_scan(self):
        worker = ScanWorker()
        worker.progress.connect(self.status_bar.showMessage)
        worker.finished.connect(self._on_scan_done)
        self._workers.append(worker)
        worker.start()

    def _on_scan_done(self, sessions: list[Session]):
        self.sessions = sessions
        self.filtered_sessions = sessions
        self._refresh_session_list()
        self.status_bar.showMessage(f"共 {len(sessions)} 个会话", 5000)

    def _refresh_session_list(self):
        self.session_list.clear()
        for s in self.filtered_sessions:
            item = QListWidgetItem(s.display_name)
            item.setData(Qt.UserRole, s.session_id)
            self.session_list.addItem(item)
        self.session_count_label.setText(f"会话列表 ({len(self.filtered_sessions)})")

    # ── Session Selection ──

    def _on_session_selected(self, row: int):
        if row < 0 or row >= len(self.filtered_sessions):
            return
        session = self.filtered_sessions[row]
        self.current_session = session
        self._current_page = 0
        self.copy_resume_btn.setEnabled(True)
        self.chat_title.setText(session.display_name)
        self._render_chat()

    def _render_chat(self, highlight_uuid: str = ""):
        if not self.current_session:
            return
        msgs = self.current_session.messages
        page_size = self.config.page_size
        total_pages = max(1, (len(msgs) + page_size - 1) // page_size)
        start = self._current_page * page_size
        end = min(start + page_size, len(msgs))
        page_msgs = msgs[start:end]

        html_parts = [f"<html><head><style>{CHAT_CSS}</style></head><body>"]
        for msg in page_msgs:
            hl = (msg.uuid == highlight_uuid)
            html_parts.append(_msg_to_html(msg, anchor_id=msg.uuid, highlight=hl))
        html_parts.append("</body></html>")

        self.chat_browser.setHtml("\n".join(html_parts))
        if highlight_uuid:
            self.chat_browser.scrollToAnchor(highlight_uuid)

        self.page_label.setText(f"第 {self._current_page + 1}/{total_pages} 页 (共 {len(msgs)} 条)")
        self.prev_btn.setEnabled(self._current_page > 0)
        self.next_btn.setEnabled(self._current_page < total_pages - 1)

    def _prev_page(self):
        if self._current_page > 0:
            self._current_page -= 1
            self._render_chat()

    def _next_page(self):
        if self.current_session:
            page_size = self.config.page_size
            total_pages = max(1, (len(self.current_session.messages) + page_size - 1) // page_size)
            if self._current_page < total_pages - 1:
                self._current_page += 1
                self._render_chat()

    # ── Keyword Search ──

    def _keyword_search(self):
        keyword = self.keyword_input.text().strip()
        if not keyword:
            self._clear_search()
            return
        kw_lower = keyword.lower()
        matched = []
        for s in self.sessions:
            for msg in s.messages:
                if kw_lower in msg.content.lower():
                    matched.append(s)
                    break
        self.filtered_sessions = matched
        self._refresh_session_list()
        self.status_bar.showMessage(f"关键词搜索: 找到 {len(matched)} 个匹配会话", 5000)

    def _clear_search(self):
        self.keyword_input.clear()
        self.semantic_input.clear()
        self.filtered_sessions = self.sessions
        self._refresh_session_list()

    # ── Semantic Search ──

    def _semantic_search(self):
        query = self.semantic_input.text().strip()
        if not query:
            return
        if self.engine.index_size == 0:
            QMessageBox.warning(self, "无索引", "请先点击「构建索引」生成嵌入向量索引")
            return

        self.status_bar.showMessage("语义搜索中...")
        worker = SemanticSearchWorker(self.engine, query)
        worker.finished.connect(self._on_semantic_results)
        worker.error.connect(lambda e: QMessageBox.critical(self, "搜索失败", e))
        self._workers.append(worker)
        worker.start()

    def _on_semantic_results(self, results: list[dict]):
        self.status_bar.showMessage(f"语义搜索完成，找到 {len(results)} 条相关结果", 5000)
        if not results:
            return

        session_scores: dict[str, float] = {}
        session_highlight: dict[str, str] = {}
        for r in results:
            sid = r["session_id"]
            if sid not in session_scores or r["score"] > session_scores[sid]:
                session_scores[sid] = r["score"]
                session_highlight[sid] = r["message_uuid"]

        matched = [s for s in self.sessions if s.session_id in session_scores]
        matched.sort(key=lambda s: session_scores[s.session_id], reverse=True)
        self.filtered_sessions = matched
        self._refresh_session_list()

        if matched:
            self.session_list.setCurrentRow(0)
            best_uuid = session_highlight.get(matched[0].session_id, "")
            if best_uuid:
                self._jump_to_message(matched[0], best_uuid)

    def _jump_to_message(self, session: Session, message_uuid: str):
        self.current_session = session
        page_size = self.config.page_size
        for i, msg in enumerate(session.messages):
            if msg.uuid == message_uuid:
                self._current_page = i // page_size
                break
        self.copy_resume_btn.setEnabled(True)
        self.chat_title.setText(session.display_name)
        self._render_chat(highlight_uuid=message_uuid)

    # ── Build Index ──

    def _build_index(self, force_rebuild=False):
        if self._indexing:
            self.status_bar.showMessage("索引正在构建中，请等待完成...", 3000)
            return
        if not self.sessions:
            QMessageBox.warning(self, "无会话", "未找到任何会话历史")
            return

        self._indexing = True
        self.progress_bar.show()
        self.progress_bar.setRange(0, 100)
        mode = "全量重建" if force_rebuild else "增量更新"
        self.status_bar.showMessage(f"正在{mode}索引 (首次需下载模型 ~1.2GB)...")

        worker = IndexWorker(self.engine, self.sessions, force_rebuild=force_rebuild)
        worker.progress.connect(self._on_index_progress)
        worker.finished.connect(self._on_index_done)
        worker.error.connect(self._on_index_error)
        self._workers.append(worker)
        worker.start()

    def _rebuild_index(self):
        self._build_index(force_rebuild=True)

    def _on_index_progress(self, current: int, total: int, msg: str):
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(current)
        else:
            self.progress_bar.setRange(0, 0)
        self.status_bar.showMessage(msg)

    def _on_index_done(self, new_count: int):
        self._indexing = False
        self.progress_bar.hide()
        self._model_loaded = True
        if self.engine.backend_name:
            self.model_status_label.setText(f"后端: {self.engine.backend_name} | 模型: 已加载")
        self.index_label.setText(f"索引: {self.engine.index_size} 条")
        if new_count > 0:
            self.status_bar.showMessage(f"索引更新完成! 新增 {new_count} 条，共 {self.engine.index_size} 条", 5000)
        else:
            self.status_bar.showMessage(f"索引已是最新，共 {self.engine.index_size} 条", 5000)

    def _on_index_error(self, err: str):
        self._indexing = False
        self.progress_bar.hide()
        QMessageBox.critical(self, "索引构建失败", err)

    def _clear_index(self):
        if self._indexing:
            self.status_bar.showMessage("索引正在构建中，无法清除", 3000)
            return
        self.engine.clear_index()
        self.index_label.setText("索引: 0 条")
        self.status_bar.showMessage("索引已清除", 3000)

    # ── Preload Model ──

    def _preload_model(self):
        if self._model_loaded:
            self.status_bar.showMessage("模型已加载", 3000)
            return
        if self._indexing:
            self.status_bar.showMessage("索引正在构建中，模型会自动加载", 3000)
            return
        self.status_bar.showMessage("正在加载模型 (首次需下载 ~1.2GB)...")
        worker = ModelLoadWorker(self.engine)
        worker.finished.connect(self._on_model_loaded)
        worker.error.connect(lambda e: QMessageBox.critical(self, "模型加载失败", e))
        self._workers.append(worker)
        worker.start()

    def _on_model_loaded(self):
        self._model_loaded = True
        self.model_status_label.setText(f"后端: {self.engine.backend_name} | 模型: 已加载")
        self.status_bar.showMessage("模型加载完成!", 3000)

    # ── Copy Resume ──

    def _copy_resume(self):
        if not self.current_session:
            return
        cmd = f"claude --resume {self.current_session.session_id}"
        clipboard = QApplication.clipboard()
        clipboard.setText(cmd)
        self.status_bar.showMessage(f"已复制: {cmd}", 3000)

    # ── Settings ──

    def _open_settings(self):
        dialog = ConfigDialog(self.config, self)
        if dialog.exec() == QDialog.Accepted:
            self.config = AppConfig.load()
            self.engine = EmbeddingEngine(self.config)
            self._model_loaded = False
            self.model_status_label.setText("")
            self.index_label.setText(f"索引: {self.engine.index_size} 条")
            self.status_bar.showMessage("设置已保存", 3000)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Claude History Search")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
