#pragma once
#include "engine.h"
#include "parser.h"
#include "types.h"
#include <QMainWindow>
#include <QListWidget>
#include "chatwidget.h"
#include <QLineEdit>
#include <QPushButton>
#include <QLabel>
#include <QProgressBar>
#include <QTabWidget>
#include <QSplitter>
#include <QToolButton>
#include <QMap>

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = nullptr);

private slots:
    void onScanDone();
    void onSessionSelected(int row);
    void keywordSearch();
    void semanticSearch();
    void clearSearch();
    void buildIndex(bool force = false);
    void clearIndex();
    void copyResume();
    void openSettings();
    void refreshSessions();
    void nextMatch();
    void prevMatch();
    void goToMatch(int idx);

private:
    void buildUI();
    void applyTheme();
    void startScan();
    void refreshSessionList();
    void refreshList();
    void renderChat(const QString &highlightUuid = {});
    void jumpToMessage(int sessionIdx, const QString &uuid);

    // 数据
    QVector<Session> m_sessions;
    QVector<int> m_filteredIndices;
    QMap<int, QString> m_matchedUuids; // sessionIdx -> first matched message uuid
    // 结果导航: 所有匹配消息的有序列表 (sessionIdx, uuid)
    QVector<QPair<int, QString>> m_allMatches;
    int m_currentMatch = -1;
    int m_selectedSession = -1;
    int m_currentPage = 0;
    int m_pageSize = 50;
    bool m_indexing = false;
    bool m_darkMode = true;

    // 引擎
    EmbeddingEngine *m_engine;

    // 控件
    QListWidget *m_sessionList;
    ChatWidget *m_chatWidget;
    QLineEdit *m_keywordInput;
    QLineEdit *m_semanticInput;
    QLabel *m_sessionCountLabel;
    QLabel *m_chatTitle;
    QLabel *m_pageLabel;
    QLabel *m_statusLabel;
    QLabel *m_indexLabel;
    QProgressBar *m_progressBar;
    QPushButton *m_prevBtn;
    QPushButton *m_nextBtn;
    QPushButton *m_copyResumeBtn;
    QPushButton *m_prevMatchBtn;
    QPushButton *m_nextMatchBtn;
    QLabel *m_matchLabel;
    QTabWidget *m_searchTabs;
};
