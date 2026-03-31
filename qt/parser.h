#pragma once
#include "types.h"
#include <QVector>
#include <functional>

// CJK 搜索文本标准化
QString normalizeForSearch(const QString &text);

// 解析单个 JSONL 会话文件
std::optional<Session> parseSessionFile(const QString &path);

// 并行扫描所有会话
QVector<Session> scanAllSessions(std::function<void(int, int)> progress = nullptr);
