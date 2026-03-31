#include "mainwindow.h"
#include <QApplication>
#include <QClipboard>
#include <QHBoxLayout>
#include <QMenuBar>
#include <QMessageBox>
#include <QSplitter>
#include <QStatusBar>
#include <QVBoxLayout>
#include <QtConcurrent>
#include <QFutureWatcher>
#include <QTimer>
#include <QStyleFactory>

// ── macOS 风格: 深色 / 浅色 双主题 ──

static QString buildTheme(bool dark) {
    // 配色变量
    QString bg1   = dark ? "#1c1c1e" : "#ffffff";  // 主背景
    QString bg2   = dark ? "#2c2c2e" : "#f2f2f7";  // 二级背景
    QString bg3   = dark ? "#3a3a3c" : "#e5e5ea";  // 三级/hover
    QString fg    = dark ? "#e5e5e7" : "#1c1c1e";  // 主文字
    QString fg2   = dark ? "#98989d" : "#6c6c70";  // 二级文字
    QString fg3   = dark ? "#636366" : "#aeaeb2";  // 淡文字
    QString bdr   = dark ? "#3a3a3c" : "#d1d1d6";  // 边框
    QString sel   = dark ? "#3a3a3c" : "#d1d1d6";  // 选中
    QString sbh   = dark ? "#48484a" : "#c7c7cc";  // 滚动条

    return QString(R"(
* { font-family: -apple-system, 'PingFang SC', 'Helvetica Neue', sans-serif; }
QMainWindow { background: %1; }
QWidget { background: %1; color: %4; }
QLineEdit {
    background: %2; border: 1px solid %7; border-radius: 8px;
    padding: 8px 12px; color: %4; font-size: 13px;
    selection-background-color: #0a84ff;
}
QLineEdit:focus { border-color: #0a84ff; }
QPushButton {
    background: %3; border: none; border-radius: 8px;
    padding: 8px 18px; color: %4; font-size: 12px; font-weight: 500;
}
QPushButton:hover { background: %7; }
QPushButton:disabled { color: %6; background: %2; }
QPushButton#accent { background: #0a84ff; color: white; font-weight: 600; }
QPushButton#accent:hover { background: #409cff; }
QListWidget {
    background: %1; border: none; outline: none; font-size: 12px; color: %4;
}
QListWidget::item { padding: 8px 10px; border-radius: 6px; margin: 1px 4px; }
QListWidget::item:selected { background: %8; }
QListWidget::item:hover:!selected { background: %2; }
QStatusBar { background: %1; border-top: 1px solid %2; color: %6; font-size: 11px; }
QStatusBar QLabel { color: %6; font-size: 11px; background: transparent; }
QProgressBar {
    background: %3; border-radius: 4px; border: none;
    text-align: center; color: %4; font-size: 9px; min-height: 6px; max-height: 6px;
}
QProgressBar::chunk { background: #0a84ff; border-radius: 4px; }
QLabel { background: transparent; }
QSplitter::handle { background: %2; width: 1px; }
QScrollBar:vertical { background: transparent; width: 8px; }
QScrollBar::handle:vertical { background: %9; border-radius: 4px; min-height: 32px; }
QScrollBar::handle:vertical:hover { background: %6; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { height: 0; background: transparent; }
    )").arg(bg1, bg2, bg3, fg, fg2, fg3, bdr, sel, sbh);
}

// ── 消息气泡 (QPainter) — macOS 风格: 灰底, 细线, 微妙色彩 ──
// 在 chatwidget.h 的 MessageBubble 里处理, 这里只管布局

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    m_engine = new EmbeddingEngine(this);
    setWindowTitle("Claude History");
    resize(1400, 900);
    setMinimumSize(900, 600);

    buildUI();
    qApp->setStyleSheet(buildTheme(m_darkMode));
    startScan();

    // Engine signals
    connect(m_engine, &EmbeddingEngine::indexProgress, this, [this](int c, int t, const QString &m) {
        m_progressBar->show(); m_progressBar->setRange(0, t); m_progressBar->setValue(c); m_statusLabel->setText(m);
    });
    connect(m_engine, &EmbeddingEngine::indexDone, this, [this](int n, int t) {
        m_indexing = false; m_progressBar->hide();
        m_indexLabel->setText(QString(" 索引 %1 ").arg(t));
        m_statusLabel->setText(n > 0 ? QString("新增 %1，共 %2").arg(n).arg(t) : QString("已最新 · %1 条").arg(t));
    });
    connect(m_engine, &EmbeddingEngine::indexError, this, [this](const QString &e) {
        m_indexing = false; m_progressBar->hide(); QMessageBox::warning(this, tr("索引失败"), e);
    });
    connect(m_engine, &EmbeddingEngine::searchDone, this, [this](const QVector<SearchResult> &results) {
        if (results.isEmpty()) { m_statusLabel->setText(tr("未找到结果")); return; }
        QMap<QString, QPair<float, QString>> best;
        for (auto &r : results) {
            auto it = best.find(r.sessionId);
            if (it == best.end() || r.score > it->first) best[r.sessionId] = {r.score, r.messageUuid};
        }
        QVector<QPair<int, float>> scored;
        for (int i = 0; i < m_sessions.size(); i++) {
            auto it = best.find(m_sessions[i].sessionId);
            if (it != best.end()) scored.append({i, it->first});
        }
        std::sort(scored.begin(), scored.end(), [](auto &a, auto &b) { return a.second > b.second; });
        m_filteredIndices.clear();
        for (auto &p : scored) m_filteredIndices.append(p.first);
        refreshList();
        if (!scored.isEmpty()) {
            m_sessionList->setCurrentRow(0);
            jumpToMessage(scored[0].first, best[m_sessions[scored[0].first].sessionId].second);
        }
        m_statusLabel->setText(QString("找到 %1 个").arg(scored.size()));
    });
}

void MainWindow::buildUI() {
    // ── macOS 原生菜单栏 ──
    auto *mb = menuBar();

    auto *fileMenu = mb->addMenu(tr("文件"));
    fileMenu->addAction(tr("刷新会话"), QKeySequence("Ctrl+R"), this, &MainWindow::refreshSessions);
    fileMenu->addSeparator();
    fileMenu->addAction(tr("设置..."), QKeySequence("Ctrl+,"), this, &MainWindow::openSettings);

    auto *viewMenu = mb->addMenu(tr("显示"));
    viewMenu->addAction(tr("切换深色/浅色模式"), QKeySequence("Ctrl+T"), this, [this]() {
        m_darkMode = !m_darkMode;
        qApp->setStyleSheet(buildTheme(m_darkMode));
        // 更新 chatwidget 和气泡的背景色
        QString chatBg = m_darkMode ? "#1c1c1e" : "#ffffff";
        m_chatWidget->setStyleSheet(QString("QScrollArea{background:%1; border:none;}").arg(chatBg));
        if (m_selectedSession >= 0) renderChat();
    });

    auto *indexMenu = mb->addMenu(tr("索引"));
    indexMenu->addAction(tr("更新索引"), QKeySequence("Ctrl+U"), this, [this](){ buildIndex(false); });
    indexMenu->addAction(tr("重建索引"), QKeySequence("Ctrl+Shift+U"), this, [this](){ buildIndex(true); });
    indexMenu->addAction(tr("清除索引"), this, &MainWindow::clearIndex);

    auto *searchMenu = mb->addMenu(tr("搜索"));
    searchMenu->addAction(tr("关键词搜索"), QKeySequence("Ctrl+F"), this, [this](){
        m_keywordInput->setFocus(); m_keywordInput->selectAll();
    });
    searchMenu->addAction(tr("语义搜索"), QKeySequence("Ctrl+Shift+F"), this, &MainWindow::semanticSearch);
    searchMenu->addAction(tr("清除搜索"), QKeySequence("Escape"), this, &MainWindow::clearSearch);
    searchMenu->addSeparator();
    searchMenu->addAction(tr("下一个结果"), QKeySequence("Ctrl+G"), this, &MainWindow::nextMatch);
    searchMenu->addAction(tr("上一个结果"), QKeySequence("Ctrl+Shift+G"), this, &MainWindow::prevMatch);

    auto *navMenu = mb->addMenu(tr("导航"));
    navMenu->addAction(tr("上一页"), QKeySequence("Ctrl+["), this, [this](){
        if (m_currentPage > 0) { m_currentPage--; renderChat(); }
    });
    navMenu->addAction(tr("下一页"), QKeySequence("Ctrl+]"), this, [this](){
        if (m_selectedSession >= 0) {
            int tp = (m_sessions[m_selectedSession].messages.size() + m_pageSize - 1) / m_pageSize;
            if (m_currentPage + 1 < tp) { m_currentPage++; renderChat(); }
        }
    });
    navMenu->addSeparator();
    navMenu->addAction(tr("复制 Resume 命令"), QKeySequence("Ctrl+Shift+C"), this, &MainWindow::copyResume);

    // ── 状态栏 ──
    m_statusLabel = new QLabel(tr("扫描中..."));
    statusBar()->addWidget(m_statusLabel, 1);
    m_progressBar = new QProgressBar;
    m_progressBar->setMaximumWidth(200); m_progressBar->hide();
    statusBar()->addPermanentWidget(m_progressBar);
    m_indexLabel = new QLabel(QString(" %1 indexed ").arg(m_engine->indexSize()));
    m_indexLabel->setStyleSheet("background:#2c2c2e; border-radius:4px; padding:1px 8px; color:#636366; font-size:10px;");
    statusBar()->addPermanentWidget(m_indexLabel);

    // ── 主布局 ──
    auto *central = new QWidget;
    setCentralWidget(central);
    auto *mainVBox = new QVBoxLayout(central);
    mainVBox->setContentsMargins(0, 0, 0, 0);
    mainVBox->setSpacing(0);

    // 搜索行
    auto *searchRow = new QHBoxLayout;
    searchRow->setContentsMargins(10, 8, 10, 8);
    searchRow->setSpacing(6);

    m_keywordInput = new QLineEdit;
    m_keywordInput->setPlaceholderText(tr("搜索会话内容..."));
    connect(m_keywordInput, &QLineEdit::returnPressed, this, &MainWindow::keywordSearch);
    searchRow->addWidget(m_keywordInput, 1);

    auto *searchBtn = new QPushButton(tr("搜索"));
    searchBtn->setObjectName("accent");
    connect(searchBtn, &QPushButton::clicked, this, &MainWindow::keywordSearch);
    searchRow->addWidget(searchBtn);

    auto *semBtn = new QPushButton(tr("语义搜索"));
    connect(semBtn, &QPushButton::clicked, this, &MainWindow::semanticSearch);
    searchRow->addWidget(semBtn);

    auto *clrBtn = new QPushButton(tr("清除"));
    connect(clrBtn, &QPushButton::clicked, this, &MainWindow::clearSearch);
    searchRow->addWidget(clrBtn);

    // 结果导航
    m_prevMatchBtn = new QPushButton("▲");
    m_prevMatchBtn->setFixedWidth(32);
    m_prevMatchBtn->setToolTip(tr("上一个结果 (Cmd+Shift+G)"));
    m_prevMatchBtn->setEnabled(false);
    connect(m_prevMatchBtn, &QPushButton::clicked, this, &MainWindow::prevMatch);
    searchRow->addWidget(m_prevMatchBtn);

    m_matchLabel = new QLabel;
    m_matchLabel->setStyleSheet("color:#636366; font-size:11px; min-width:60px;");
    m_matchLabel->setAlignment(Qt::AlignCenter);
    searchRow->addWidget(m_matchLabel);

    m_nextMatchBtn = new QPushButton("▼");
    m_nextMatchBtn->setFixedWidth(32);
    m_nextMatchBtn->setToolTip(tr("下一个结果 (Cmd+G)"));
    m_nextMatchBtn->setEnabled(false);
    connect(m_nextMatchBtn, &QPushButton::clicked, this, &MainWindow::nextMatch);
    searchRow->addWidget(m_nextMatchBtn);

    m_semanticInput = m_keywordInput;
    m_searchTabs = nullptr;
    mainVBox->addLayout(searchRow);

    // 分割线
    auto *sep = new QWidget;
    sep->setFixedHeight(1);
    sep->setStyleSheet("background:#2c2c2e;");
    mainVBox->addWidget(sep);

    // ── Splitter ──
    auto *splitter = new QSplitter(Qt::Horizontal);
    splitter->setChildrenCollapsible(false);
    splitter->setHandleWidth(1);

    // 左: Sessions
    auto *leftW = new QWidget;
    auto *leftL = new QVBoxLayout(leftW);
    leftL->setContentsMargins(8, 8, 0, 8);
    leftL->setSpacing(4);

    m_sessionCountLabel = new QLabel("SESSIONS");
    m_sessionCountLabel->setStyleSheet("color:#636366; font-size:10px; font-weight:600; letter-spacing:1px; padding-left:8px;");
    m_sessionCountLabel->setFixedHeight(18);
    leftL->addWidget(m_sessionCountLabel);

    m_sessionList = new QListWidget;
    connect(m_sessionList, &QListWidget::currentRowChanged, this, &MainWindow::onSessionSelected);
    leftL->addWidget(m_sessionList);
    splitter->addWidget(leftW);

    // 右: Chat
    auto *rightW = new QWidget;
    auto *rightL = new QVBoxLayout(rightW);
    rightL->setContentsMargins(0, 8, 8, 8);
    rightL->setSpacing(4);

    // 标题行
    auto *titleRow = new QHBoxLayout;
    titleRow->setContentsMargins(8, 0, 0, 0);
    m_chatTitle = new QLabel(tr("选择一个会话"));
    m_chatTitle->setStyleSheet("color:#98989d; font-size:12px; font-weight:500;");
    m_chatTitle->setWordWrap(true);
    titleRow->addWidget(m_chatTitle, 1);
    m_copyResumeBtn = new QPushButton(tr("复制 Resume"));
    m_copyResumeBtn->setEnabled(false);
    m_copyResumeBtn->setStyleSheet("QPushButton{font-size:11px; padding:4px 12px;}");
    connect(m_copyResumeBtn, &QPushButton::clicked, this, &MainWindow::copyResume);
    titleRow->addWidget(m_copyResumeBtn);
    rightL->addLayout(titleRow);

    // 聊天区
    m_chatWidget = new ChatWidget;
    rightL->addWidget(m_chatWidget, 1);

    // 分页行
    auto *pageRow = new QHBoxLayout;
    pageRow->setContentsMargins(8, 4, 0, 0);
    m_prevBtn = new QPushButton("←"); m_prevBtn->setEnabled(false);
    m_prevBtn->setStyleSheet("font-size:11px; padding:4px 12px;");
    connect(m_prevBtn, &QPushButton::clicked, this, [this](){
        if(m_currentPage > 0) { m_currentPage--; renderChat(); }
    });
    pageRow->addWidget(m_prevBtn);
    m_pageLabel = new QLabel;
    m_pageLabel->setStyleSheet("color:#636366; font-size:11px;");
    pageRow->addWidget(m_pageLabel);
    m_nextBtn = new QPushButton("→"); m_nextBtn->setEnabled(false);
    m_nextBtn->setStyleSheet("font-size:11px; padding:4px 12px;");
    connect(m_nextBtn, &QPushButton::clicked, this, [this](){
        if(m_selectedSession >= 0) {
            int tp = (m_sessions[m_selectedSession].messages.size() + m_pageSize - 1) / m_pageSize;
            if (m_currentPage + 1 < tp) { m_currentPage++; renderChat(); }
        }
    });
    pageRow->addWidget(m_nextBtn);
    pageRow->addStretch();
    rightL->addLayout(pageRow);

    splitter->addWidget(rightW);
    splitter->setStretchFactor(0, 1);
    splitter->setStretchFactor(1, 3);
    splitter->setSizes({320, 1080});

    mainVBox->addWidget(splitter, 1);
}

// ── 其余方法不变 ──

void MainWindow::applyTheme() { qApp->setStyleSheet(buildTheme(m_darkMode)); }

void MainWindow::startScan() {
    m_statusLabel->setText(tr("扫描中..."));
    auto *w = new QFutureWatcher<QVector<Session>>(this);
    connect(w, &QFutureWatcher<QVector<Session>>::finished, this, [this, w](){
        m_sessions = w->result(); w->deleteLater(); onScanDone();
    });
    w->setFuture(QtConcurrent::run([]() { return scanAllSessions(); }));
}

void MainWindow::onScanDone() {
    m_filteredIndices.clear();
    for (int i = 0; i < m_sessions.size(); i++) m_filteredIndices.append(i);
    refreshList();
    m_statusLabel->setText(QString("%1 个会话").arg(m_sessions.size()));
}

void MainWindow::refreshSessionList() { refreshList(); }

void MainWindow::refreshList() {
    m_sessionList->clear();
    for (int idx : m_filteredIndices)
        m_sessionList->addItem(m_sessions[idx].displayName());
    m_sessionCountLabel->setText(QString("SESSIONS (%1)").arg(m_filteredIndices.size()));
}

void MainWindow::onSessionSelected(int row) {
    if (row < 0 || row >= m_filteredIndices.size()) return;
    int idx = m_filteredIndices[row];
    m_copyResumeBtn->setEnabled(true);
    m_chatTitle->setText(m_sessions[idx].displayName());
    m_chatTitle->setStyleSheet("color:#e5e5e7; font-size:12px; font-weight:500;");

    if (m_matchedUuids.contains(idx)) {
        QString uuid = m_matchedUuids[idx];
        if (!uuid.isEmpty()) {
            jumpToMessage(idx, uuid);
        } else {
            // 关键词在 summary/title 中 → 显示提示卡片
            m_selectedSession = idx;
            m_currentPage = 0;
            renderChat();
            m_chatWidget->showSummaryMatch(
                m_sessions[idx].summary, m_sessions[idx].customTitle);
            m_statusLabel->setText(
                QString::fromUtf8("关键词在会话摘要中，可翻页查找具体消息"));
        }
    } else {
        m_selectedSession = idx;
        m_currentPage = 0;
        renderChat();
    }
}

void MainWindow::renderChat(const QString &hlUuid) {
    if (m_selectedSession < 0) return;
    const auto &s = m_sessions[m_selectedSession];
    int total = s.messages.size();
    int pages = qMax(1, (total + m_pageSize - 1) / m_pageSize);
    int start = m_currentPage * m_pageSize;
    int end = qMin(start + m_pageSize, total);

    QVector<Message> pageMsgs;
    for (int i = start; i < end; i++) pageMsgs.append(s.messages[i]);
    m_chatWidget->setMessages(pageMsgs, hlUuid);
    if (!hlUuid.isEmpty())
        QTimer::singleShot(100, m_chatWidget, &ChatWidget::scrollToHighlight);

    m_pageLabel->setText(QString(" %1 / %2 · %3 msgs ").arg(m_currentPage + 1).arg(pages).arg(total));
    m_prevBtn->setEnabled(m_currentPage > 0);
    m_nextBtn->setEnabled(m_currentPage + 1 < pages);
}

void MainWindow::keywordSearch() {
    QString q = m_keywordInput->text().trimmed();
    if (q.isEmpty()) { clearSearch(); return; }
    QStringList words = normalizeForSearch(q).split(' ', Qt::SkipEmptyParts);
    if (words.isEmpty()) { clearSearch(); return; }

    m_filteredIndices.clear();
    m_matchedUuids.clear();
    m_allMatches.clear();
    m_currentMatch = -1;

    QStringList rawWords = q.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
    m_chatWidget->setKeywords(rawWords);
    m_chatWidget->setNormalizedKeywords(words);

    auto msgMatches = [&](const Message &msg) -> bool {
        for (auto &rw : rawWords) {
            if (msg.content.contains(rw, Qt::CaseInsensitive) ||
                msg.searchText.contains(rw, Qt::CaseInsensitive))
                return true;
        }
        QString nc = normalizeForSearch(msg.content);
        QString ns = normalizeForSearch(msg.searchText);
        for (auto &w : words) {
            if (nc.contains(w) || ns.contains(w)) return true;
        }
        return false;
    };

    for (int i = 0; i < m_sessions.size(); i++) {
        if (!std::all_of(words.begin(), words.end(), [&](const QString &w) {
            return m_sessions[i].searchTextLower.contains(w);
        })) continue;

        m_filteredIndices.append(i);
        bool firstFound = false;

        // 收集所有匹配消息
        for (const auto &msg : m_sessions[i].messages) {
            if (msgMatches(msg)) {
                m_allMatches.append({i, msg.uuid});
                if (!firstFound) {
                    m_matchedUuids[i] = msg.uuid;
                    firstFound = true;
                }
            }
        }
        if (!firstFound) {
            m_matchedUuids[i] = QString(); // 摘要命中
        }
    }

    refreshList();
    // 更新导航状态
    bool hasMatches = !m_allMatches.isEmpty();
    m_prevMatchBtn->setEnabled(hasMatches);
    m_nextMatchBtn->setEnabled(hasMatches);

    if (hasMatches) {
        m_currentMatch = 0;
        m_matchLabel->setText(QString("1/%1").arg(m_allMatches.size()));
        goToMatch(0);
    } else {
        m_matchLabel->setText("");
        m_statusLabel->setText(QString("找到 %1 个会话").arg(m_filteredIndices.size()));
        if (!m_filteredIndices.isEmpty()) {
            m_sessionList->setCurrentRow(0);
        }
    }
}

void MainWindow::semanticSearch() {
    QString q = m_keywordInput->text().trimmed();
    if (q.isEmpty()) return;
    if (!m_engine->hasIndex()) { QMessageBox::information(this, tr("无索引"), tr("请先通过菜单「索引 → 更新索引」构建")); return; }
    m_statusLabel->setText(tr("语义搜索中..."));
    m_engine->search(q);
}

void MainWindow::clearSearch() {
    m_keywordInput->clear();
    m_selectedSession = -1;
    m_matchedUuids.clear();
    m_allMatches.clear();
    m_currentMatch = -1;
    m_chatWidget->setKeywords({});
    m_chatWidget->setNormalizedKeywords({});
    m_prevMatchBtn->setEnabled(false);
    m_nextMatchBtn->setEnabled(false);
    m_matchLabel->setText("");
    onScanDone();
}

void MainWindow::buildIndex(bool f) {
    if (m_indexing) return;
    m_indexing = true;
    m_progressBar->show(); m_progressBar->setRange(0, 0);
    m_engine->buildIndex(m_sessions, f);
}

void MainWindow::clearIndex() {
    if (m_indexing) return;
    m_engine->clearIndex();
    m_indexLabel->setText(" 0 indexed ");
    m_statusLabel->setText(tr("索引已清除"));
}

void MainWindow::copyResume() {
    if (m_selectedSession < 0) return;
    auto cmd = "claude --resume " + m_sessions[m_selectedSession].sessionId;
    QApplication::clipboard()->setText(cmd);
    m_statusLabel->setText("✓ 已复制: " + cmd);
}

void MainWindow::nextMatch() {
    if (m_allMatches.isEmpty()) return;
    m_currentMatch = (m_currentMatch + 1) % m_allMatches.size();
    goToMatch(m_currentMatch);
}

void MainWindow::prevMatch() {
    if (m_allMatches.isEmpty()) return;
    m_currentMatch = (m_currentMatch - 1 + m_allMatches.size()) % m_allMatches.size();
    goToMatch(m_currentMatch);
}

void MainWindow::goToMatch(int idx) {
    if (idx < 0 || idx >= m_allMatches.size()) return;
    auto [sessionIdx, uuid] = m_allMatches[idx];
    m_currentMatch = idx;
    m_matchLabel->setText(QString("%1/%2").arg(idx + 1).arg(m_allMatches.size()));

    // 选中对应会话
    for (int row = 0; row < m_filteredIndices.size(); row++) {
        if (m_filteredIndices[row] == sessionIdx) {
            m_sessionList->blockSignals(true);
            m_sessionList->setCurrentRow(row);
            m_sessionList->blockSignals(false);
            break;
        }
    }

    m_chatTitle->setText(m_sessions[sessionIdx].displayName());
    m_chatTitle->setStyleSheet("color:#e5e5e7; font-size:12px; font-weight:500;");
    m_copyResumeBtn->setEnabled(true);
    jumpToMessage(sessionIdx, uuid);
    m_statusLabel->setText(QString("结果 %1/%2").arg(idx + 1).arg(m_allMatches.size()));
}

void MainWindow::openSettings() { m_statusLabel->setText(tr("设置开发中...")); }
void MainWindow::refreshSessions() { startScan(); }

void MainWindow::jumpToMessage(int idx, const QString &uuid) {
    m_selectedSession = idx;
    for (int i = 0; i < m_sessions[idx].messages.size(); i++) {
        if (m_sessions[idx].messages[i].uuid == uuid) { m_currentPage = i / m_pageSize; break; }
    }
    m_copyResumeBtn->setEnabled(true);
    m_chatTitle->setText(m_sessions[idx].displayName());
    m_chatTitle->setStyleSheet("color:#e5e5e7; font-size:12px; font-weight:500;");
    renderChat(uuid);
}
