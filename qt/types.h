#pragma once
#include <QString>
#include <QVector>
#include <QSet>

struct Message {
    QString uuid;
    QString role;       // "user" / "assistant"
    QString content;    // 显示文本
    QString searchText; // 搜索文本 (含 tool_result)
    QString timestamp;
    int lineNumber = 0;
    QString model;
};

struct Session {
    QString sessionId;
    QString projectPath;
    QString filePath;
    QVector<Message> messages;
    QString firstTimestamp;
    QString lastTimestamp;
    QString summary;
    QString customTitle;
    QString cwd;
    QString model;
    QString searchTextLower; // 预计算的搜索文本

    QString displayName() const {
        QString ts = firstTimestamp.left(19).replace('T', ' ');
        QString proj = projectPath.section('/', -1);
        if (proj.isEmpty()) proj = "unknown";
        QString title = customTitle.isEmpty() ? summary.left(60) : customTitle;
        if (title.isEmpty()) title = sessionId.left(8);
        return QString("[%1] %2 - %3").arg(ts, proj, title);
    }
};

struct SearchResult {
    QString sessionId;
    QString messageUuid;
    float score = 0;
};

struct IndexMeta {
    QString sessionId;
    QString messageUuid;
    QString textPreview;
    QString projectPath;
    int lineNumber = 0;
};
