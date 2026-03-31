#pragma once
#include "types.h"
#include <QObject>
#include <QProcess>
#include <QVector>

/// Embedding 引擎 — 通过 Python MLX 子进程执行推理
/// Qt C++ 负责 UI + 解析, Python 负责模型推理
class EmbeddingEngine : public QObject {
    Q_OBJECT
public:
    explicit EmbeddingEngine(QObject *parent = nullptr);

    int indexSize() const { return m_indexSize; }
    bool hasIndex() const { return m_indexSize > 0; }

    /// 异步构建索引
    void buildIndex(const QVector<Session> &sessions, bool forceRebuild = false);
    /// 异步语义搜索
    void search(const QString &query);
    /// 清除索引
    void clearIndex();

signals:
    void indexProgress(int current, int total, const QString &msg);
    void indexDone(int newCount, int total);
    void indexError(const QString &error);
    void searchDone(const QVector<SearchResult> &results);
    void searchError(const QString &error);

private:
    void runPythonCommand(const QString &cmd, const QByteArray &input = {});
    int m_indexSize = 0;
    QProcess *m_process = nullptr;
    QString m_pendingAction;
};
