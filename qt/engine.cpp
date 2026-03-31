#include "engine.h"
#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QStandardPaths>

static QString indexMetaPath() {
    return QDir::homePath() + "/.claude_his_search/index/meta.json";
}

EmbeddingEngine::EmbeddingEngine(QObject *parent) : QObject(parent) {
    // 读取已有索引大小
    QFile f(indexMetaPath());
    if (f.open(QIODevice::ReadOnly)) {
        QJsonArray arr = QJsonDocument::fromJson(f.readAll()).array();
        m_indexSize = arr.size();
    }
}

void EmbeddingEngine::buildIndex(const QVector<Session> &sessions, bool forceRebuild) {
    // 调用 Python embedding 模块
    QStringList args;
    args << "-c"
         << QString("from claude_his_search.embedding_engine import EmbeddingEngine; "
                     "from claude_his_search.config import AppConfig; "
                     "from claude_his_search.history_parser import scan_all_sessions; "
                     "import json, sys; "
                     "config = AppConfig.load(); "
                     "engine = EmbeddingEngine(config); "
                     "sessions = scan_all_sessions(); "
                     "old = engine.index_size; "
                     "engine.build_index(sessions, force_rebuild=%1, "
                     "progress_callback=lambda c,t,m: print(json.dumps({'p':c,'t':t,'m':m}), flush=True)); "
                     "print(json.dumps({'done': True, 'new': engine.index_size - old, 'total': engine.index_size}), flush=True)")
                .arg(forceRebuild ? "True" : "False");

    if (m_process) {
        m_process->kill();
        m_process->deleteLater();
    }

    m_process = new QProcess(this);
    m_pendingAction = "index";

    connect(m_process, &QProcess::readyReadStandardOutput, this, [this]() {
        while (m_process->canReadLine()) {
            QByteArray line = m_process->readLine().trimmed();
            QJsonObject obj = QJsonDocument::fromJson(line).object();
            if (obj.contains("done")) {
                m_indexSize = obj["total"].toInt();
                emit indexDone(obj["new"].toInt(), m_indexSize);
            } else if (obj.contains("p")) {
                emit indexProgress(obj["p"].toInt(), obj["t"].toInt(), obj["m"].toString());
            }
        }
    });
    connect(m_process, &QProcess::readyReadStandardError, this, [this]() {
        QString err = m_process->readAllStandardError();
        if (!err.trimmed().isEmpty() && m_pendingAction == "index") {
            // 只在进程结束后报错，中间的 stderr 可能是 warning
        }
    });
    connect(m_process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, [this](int code, QProcess::ExitStatus) {
        if (code != 0 && m_pendingAction == "index") {
            emit indexError(m_process->readAllStandardError());
        }
        m_pendingAction.clear();
    });

    // 使用 uv run 确保在 venv 中运行
    QString projectDir = QCoreApplication::applicationDirPath() + "/..";
    // 找到项目根目录 (含 pyproject.toml)
    QDir dir(projectDir);
    while (!dir.exists("pyproject.toml") && dir.cdUp()) {}

    m_process->setWorkingDirectory(dir.absolutePath());
    m_process->start("uv", QStringList() << "run" << "python" << args);
}

void EmbeddingEngine::search(const QString &query) {
    QStringList args;
    args << "-c"
         << QString("from claude_his_search.embedding_engine import EmbeddingEngine; "
                     "from claude_his_search.config import AppConfig; "
                     "import json, sys; "
                     "config = AppConfig.load(); "
                     "engine = EmbeddingEngine(config); "
                     "results = engine.search('%1'); "
                     "print(json.dumps(results), flush=True)")
                .arg(QString(query).replace("'", "\\'"));

    if (m_process) {
        m_process->kill();
        m_process->deleteLater();
    }

    m_process = new QProcess(this);
    m_pendingAction = "search";

    connect(m_process, &QProcess::readyReadStandardOutput, this, [this]() {
        QByteArray data = m_process->readAllStandardOutput().trimmed();
        // 取最后一行 (可能有 warning 输出在前面)
        QList<QByteArray> lines = data.split('\n');
        QByteArray lastLine = lines.last().trimmed();
        QJsonArray arr = QJsonDocument::fromJson(lastLine).array();
        QVector<SearchResult> results;
        for (const auto &item : arr) {
            QJsonObject obj = item.toObject();
            SearchResult r;
            r.sessionId = obj["session_id"].toString();
            r.messageUuid = obj["message_uuid"].toString();
            r.score = obj["score"].toDouble();
            results.append(r);
        }
        emit searchDone(results);
    });
    connect(m_process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, [this](int code, QProcess::ExitStatus) {
        if (code != 0 && m_pendingAction == "search") {
            emit searchError(m_process->readAllStandardError());
        }
        m_pendingAction.clear();
    });

    QDir dir(QCoreApplication::applicationDirPath() + "/..");
    while (!dir.exists("pyproject.toml") && dir.cdUp()) {}
    m_process->setWorkingDirectory(dir.absolutePath());
    m_process->start("uv", QStringList() << "run" << "python" << args);
}

void EmbeddingEngine::clearIndex() {
    QFile::remove(QDir::homePath() + "/.claude_his_search/index/embeddings.npz");
    QFile::remove(indexMetaPath());
    m_indexSize = 0;
}
