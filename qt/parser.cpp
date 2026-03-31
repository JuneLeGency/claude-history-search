#include "parser.h"
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QStandardPaths>
#include <QtConcurrent>
#include <algorithm>

static QString claudeProjectsDir() {
    return QDir::homePath() + "/.claude/projects";
}

// ── CJK 标点 → 空格 (参考 claude-history) ──

static bool isCjkPunctuation(QChar c) {
    ushort u = c.unicode();
    return u == 0x3000 || u == 0x3001 || u == 0x3002 ||
           (u >= 0x3008 && u <= 0x3011) || (u >= 0x3014 && u <= 0x3017) ||
           u == 0xFF01 || u == 0xFF08 || u == 0xFF09 || u == 0xFF0C ||
           u == 0xFF1A || u == 0xFF1B || u == 0xFF1F ||
           u == 0x201C || u == 0x201D || u == 0x2018 || u == 0x2019 ||
           u == 0x2014 || u == 0x2026 || u == 0x00B7;
}

QString normalizeForSearch(const QString &text) {
    QString out;
    out.reserve(text.size());
    for (QChar ch : text) {
        if (ch == '_' || ch == '-' || ch == '/' || isCjkPunctuation(ch)) {
            out.append(' ');
        } else {
            out.append(ch.toLower());
        }
    }
    return out;
}

// ── 文本提取 ──

static QString extractTextFromBlocks(const QJsonArray &blocks) {
    QStringList parts;
    for (const auto &b : blocks) {
        QJsonObject obj = b.toObject();
        if (obj["type"].toString() == "text") {
            parts << obj["text"].toString();
        }
    }
    return parts.join(" ");
}

static QString extractToolResultText(const QJsonValue &content) {
    if (content.isString()) {
        QString s = content.toString().trimmed();
        return s.isEmpty() ? QString() : s;
    }
    if (content.isArray()) {
        QStringList parts;
        for (const auto &item : content.toArray()) {
            if (item.isString()) {
                parts << item.toString();
            } else if (item.isObject()) {
                QJsonObject obj = item.toObject();
                QString type = obj["type"].toString();
                if (type.isEmpty() || type == "text") {
                    QString t = obj["text"].toString();
                    if (!t.isEmpty()) parts << t;
                }
            }
        }
        QString joined = parts.join(" ");
        return joined.trimmed().isEmpty() ? QString() : joined;
    }
    return {};
}

static QString extractSearchTextFromBlocks(const QJsonArray &blocks) {
    QStringList parts;
    for (const auto &b : blocks) {
        QJsonObject obj = b.toObject();
        QString type = obj["type"].toString();
        if (type == "text") {
            parts << obj["text"].toString();
        } else if (type == "tool_result" && obj.contains("content")) {
            QString t = extractToolResultText(obj["content"]);
            if (!t.isEmpty()) {
                // 截断到 16KB
                if (t.size() > 16384) t = t.left(12288) + " " + t.right(4096);
                parts << t;
            }
        }
    }
    return parts.join(" ");
}

static QString extractUserContent(const QJsonValue &content, bool forSearch) {
    if (content.isString()) return content.toString();
    if (content.isArray()) {
        return forSearch ? extractSearchTextFromBlocks(content.toArray())
                         : extractTextFromBlocks(content.toArray());
    }
    return {};
}

// ── 解析 JSONL ──

std::optional<Session> parseSessionFile(const QString &path) {
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) return std::nullopt;

    QFileInfo fi(path);
    QString projectDir = fi.dir().dirName();
    QString projectPath = projectDir.replace('-', '/');

    Session session;
    session.sessionId = fi.baseName();
    session.projectPath = projectPath;
    session.filePath = path;

    QStringList allSearchParts;
    int lineNo = 0;

    while (!file.atEnd()) {
        lineNo++;
        QByteArray lineData = file.readLine();
        if (lineData.trimmed().isEmpty()) continue;

        QJsonParseError err;
        QJsonDocument doc = QJsonDocument::fromJson(lineData, &err);
        if (err.error != QJsonParseError::NoError) continue;
        QJsonObject entry = doc.object();

        QString type = entry["type"].toString();

        if (type == "summary" && session.summary.isEmpty()) {
            session.summary = entry["summary"].toString();
            continue;
        }
        if (type == "custom-title") {
            session.customTitle = entry["customTitle"].toString();
            continue;
        }

        if (type != "user" && type != "assistant") continue;

        QJsonObject msg = entry["message"].toObject();
        if (msg.isEmpty()) continue;

        QString ts = entry["timestamp"].toString();
        if (session.firstTimestamp.isEmpty() && !ts.isEmpty()) session.firstTimestamp = ts;
        if (!ts.isEmpty()) session.lastTimestamp = ts;

        if (type == "user" && session.cwd.isEmpty()) {
            session.cwd = entry["cwd"].toString();
        }

        QJsonValue contentVal = msg["content"];
        QString displayText = extractUserContent(contentVal, false).trimmed();
        QString searchText = extractUserContent(contentVal, true).trimmed();

        if (displayText.isEmpty() && searchText.isEmpty()) continue;

        if (session.summary.isEmpty() && type == "user" && !displayText.isEmpty()) {
            session.summary = displayText.left(200);
        }

        if (session.model.isEmpty() && msg.contains("model")) {
            session.model = msg["model"].toString();
        }

        allSearchParts << searchText;

        Message m;
        m.uuid = entry["uuid"].toString();
        m.role = msg["role"].toString(type);
        m.content = displayText;
        m.searchText = searchText;
        m.timestamp = ts;
        m.lineNumber = lineNo;
        m.model = msg["model"].toString();
        session.messages.append(m);
    }

    if (session.messages.isEmpty()) return std::nullopt;

    // 预计算搜索文本
    QString fullSearch;
    if (!session.customTitle.isEmpty()) fullSearch += session.customTitle + " ";
    fullSearch += session.summary + " ";
    fullSearch += allSearchParts.join(" ");
    session.searchTextLower = normalizeForSearch(fullSearch);

    return session;
}

// ── 扫描所有会话 ──

QVector<Session> scanAllSessions(std::function<void(int, int)> progress) {
    QDir projectsRoot(claudeProjectsDir());
    if (!projectsRoot.exists()) return {};

    // 收集所有 JSONL 路径
    QStringList allPaths;
    for (const auto &projEntry : projectsRoot.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot)) {
        QDir projDir(projEntry.absoluteFilePath());
        for (const auto &fi : projDir.entryInfoList({"*.jsonl"}, QDir::Files)) {
            if (!fi.fileName().startsWith("agent-") &&
                !fi.absoluteFilePath().contains("subagents")) {
                allPaths << fi.absoluteFilePath();
            }
        }
    }

    // 并行解析
    QAtomicInt done(0);
    int total = allPaths.size();

    QVector<Session> sessions = QtConcurrent::blockingMapped(allPaths,
        [&done, total, &progress](const QString &path) -> Session {
            auto result = parseSessionFile(path);
            int d = done.fetchAndAddRelaxed(1) + 1;
            if (progress && d % 50 == 0) progress(d, total);
            return result.value_or(Session{});
        }
    );

    // 过滤空会话 + 排序
    sessions.erase(
        std::remove_if(sessions.begin(), sessions.end(),
                        [](const Session &s) { return s.messages.isEmpty(); }),
        sessions.end()
    );
    std::sort(sessions.begin(), sessions.end(),
              [](const Session &a, const Session &b) {
                  return a.lastTimestamp > b.lastTimestamp;
              });

    if (progress) progress(total, total);
    return sessions;
}
