#pragma once
#include "types.h"
#include <QWidget>
#include <QScrollArea>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QPainter>
#include <QPainterPath>
#include <QRegularExpression>

namespace Colors {
    const QColor bg      (0x1c, 0x1c, 0x1e);
    const QColor surface (0x2c, 0x2c, 0x2e);
    const QColor elevated(0x3a, 0x3a, 0x3c);
    const QColor border  (0x48, 0x48, 0x4a);
    const QColor text    (0xe5, 0xe5, 0xe7);
    const QColor subtext (0x98, 0x98, 0x9d);
    const QColor dim     (0x63, 0x63, 0x66);
    const QColor accent  (0x0a, 0x84, 0xff);
    const QColor green   (0x30, 0xd1, 0x58);
    const QColor yellow  (0xff, 0xd6, 0x0a);
}

// ── 文本处理工具 ──

static const int COLLAPSE_THRESHOLD = 300; // 超过此字符数折叠
static const int SNIPPET_CONTEXT = 80;     // 关键词前后保留字符数

// 高亮关键词
static QString highlightText(const QString &text, const QStringList &keywords) {
    if (keywords.isEmpty()) return text.toHtmlEscaped().replace('\n', "<br>");

    QStringList escaped;
    for (auto &kw : keywords) escaped << QRegularExpression::escape(kw);
    QRegularExpression re("(" + escaped.join('|') + ")", QRegularExpression::CaseInsensitiveOption);

    QString html;
    int last = 0;
    auto it = re.globalMatch(text);
    while (it.hasNext()) {
        auto m = it.next();
        html += text.mid(last, m.capturedStart() - last).toHtmlEscaped();
        html += "<span style='background-color:#f9a825; color:#1c1c1e; padding:0 2px;'>"
              + m.captured().toHtmlEscaped() + "</span>";
        last = m.capturedEnd();
    }
    html += text.mid(last).toHtmlEscaped();
    html.replace('\n', "<br>");
    return html;
}

// 提取关键词周围的片段 (折叠模式用)
static QString extractSnippets(const QString &text, const QStringList &keywords) {
    if (keywords.isEmpty()) {
        // 无关键词：截取开头
        if (text.length() <= COLLAPSE_THRESHOLD) return text;
        int end = text.lastIndexOf(' ', COLLAPSE_THRESHOLD);
        if (end < 100) end = COLLAPSE_THRESHOLD;
        return text.left(end) + "...";
    }

    // 找到所有匹配位置
    QStringList escaped;
    for (auto &kw : keywords) escaped << QRegularExpression::escape(kw);
    QRegularExpression re("(" + escaped.join('|') + ")", QRegularExpression::CaseInsensitiveOption);

    struct Range { int start; int end; };
    QVector<Range> ranges;
    auto it = re.globalMatch(text);
    while (it.hasNext()) {
        auto m = it.next();
        int s = qMax(0, m.capturedStart() - SNIPPET_CONTEXT);
        int e = qMin(text.length(), m.capturedEnd() + SNIPPET_CONTEXT);
        ranges.append({s, e});
    }

    if (ranges.isEmpty()) {
        // 没找到匹配(可能在 searchText 里匹配但不在 content 里)
        if (text.length() <= COLLAPSE_THRESHOLD) return text;
        return text.left(COLLAPSE_THRESHOLD) + "...";
    }

    // 合并重叠区间
    std::sort(ranges.begin(), ranges.end(), [](auto &a, auto &b){ return a.start < b.start; });
    QVector<Range> merged;
    merged.append(ranges[0]);
    for (int i = 1; i < ranges.size(); i++) {
        if (ranges[i].start <= merged.last().end + 20) {
            merged.last().end = qMax(merged.last().end, ranges[i].end);
        } else {
            merged.append(ranges[i]);
        }
    }

    // 拼接片段
    QString result;
    for (int i = 0; i < merged.size(); i++) {
        if (i == 0 && merged[i].start > 0) result += "...";
        else if (i > 0) result += " ... ";
        result += text.mid(merged[i].start, merged[i].end - merged[i].start);
    }
    if (merged.last().end < text.length()) result += "...";
    return result;
}

// ── 消息气泡 ──

class MessageBubble : public QWidget {
    Q_OBJECT
public:
    MessageBubble(const Message &msg, bool highlight, const QStringList &keywords,
                  bool dark = true, QWidget *parent = nullptr)
        : QWidget(parent), m_isUser(msg.role == "user"), m_highlight(highlight),
          m_dark(dark), m_keywords(keywords)
    {
        // 优先显示 content，content 为空时用 searchText (tool_result)
        QString displayContent = msg.content.trimmed();
        bool isToolResult = false;
        if (displayContent.isEmpty() && !msg.searchText.trimmed().isEmpty()) {
            // 有关键词 → 提取关键词上下文; 无关键词 → 截取前 500 字
            if (!keywords.isEmpty()) {
                displayContent = extractSnippets(msg.searchText.trimmed(), keywords);
            } else {
                displayContent = msg.searchText.trimmed();
                if (displayContent.length() > 500)
                    displayContent = displayContent.left(500) + "...";
            }
            isToolResult = true;
        }
        m_empty = displayContent.isEmpty();
        m_fullContent = displayContent;
        if (m_empty) { setFixedHeight(0); return; }

        m_needsCollapse = (displayContent.length() > COLLAPSE_THRESHOLD);
        m_collapsed = m_needsCollapse;

        auto *outerLayout = new QVBoxLayout(this);
        outerLayout->setContentsMargins(20, 10, 16, 10);
        outerLayout->setSpacing(4);

        QString fg = dark ? "#e5e5e7" : "#1c1c1e";
        QString fg3 = dark ? "#48484a" : "#aeaeb2";
        QString linkColor = dark ? "#0a84ff" : "#007aff";

        // 角色 + 时间 (同一行)
        auto *topRow = new QHBoxLayout;
        topRow->setSpacing(8);
        QString roleText = m_isUser ? "You" : "Assistant";
        if (isToolResult) roleText += " · Tool Result";
        auto *roleLabel = new QLabel(roleText);
        QColor roleColor = m_isUser ? Colors::accent : Colors::green;
        roleLabel->setStyleSheet(QString("color:%1; font-size:11px; font-weight:600; background:transparent;").arg(roleColor.name()));
        topRow->addWidget(roleLabel);

        QString meta = msg.timestamp.left(19).replace('T', ' ');
        if (!msg.model.isEmpty()) meta += QString::fromUtf8("  \xc2\xb7  ") + msg.model.section('-', 0, 2);
        auto *tsLabel = new QLabel(meta);
        tsLabel->setStyleSheet(QString("color:%1; font-size:10px; background:transparent;").arg(fg3));
        topRow->addWidget(tsLabel);
        topRow->addStretch();

        // 字符数 (长消息才显示)
        if (m_needsCollapse) {
            auto *lenLabel = new QLabel(QString("%1 字").arg(msg.content.length()));
            lenLabel->setStyleSheet(QString("color:%1; font-size:10px; background:transparent;").arg(fg3));
            topRow->addWidget(lenLabel);
        }
        outerLayout->addLayout(topRow);

        // 正文
        m_contentLabel = new QLabel;
        m_contentLabel->setWordWrap(true);
        m_contentLabel->setTextFormat(Qt::RichText);
        m_contentLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
        m_contentLabel->setStyleSheet(QString("color:%1; font-size:13px; background:transparent;").arg(fg));
        outerLayout->addWidget(m_contentLabel);

        // 展开/收起按钮
        if (m_needsCollapse) {
            m_toggleBtn = new QPushButton;
            m_toggleBtn->setCursor(Qt::PointingHandCursor);
            m_toggleBtn->setStyleSheet(QString(
                "QPushButton { color:%1; font-size:11px; background:transparent; "
                "border:none; padding:2px 0; text-align:left; font-weight:500; }"
                "QPushButton:hover { color:%2; }"
            ).arg(linkColor, dark ? "#409cff" : "#0056b3"));
            connect(m_toggleBtn, &QPushButton::clicked, this, &MessageBubble::toggleCollapse);
            outerLayout->addWidget(m_toggleBtn);
        }

        updateContent();
    }

protected:
    void paintEvent(QPaintEvent *) override {
        if (m_empty) return;
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        QRectF r = QRectF(rect()).adjusted(8, 2, -8, -2);

        QColor bgColor = m_highlight
            ? (m_dark ? QColor(0x3a, 0x3a, 0x3c) : QColor(0xfe, 0xf9, 0xe7))
            : (m_dark ? Colors::surface : QColor(0xf2, 0xf2, 0xf7));
        QColor bdrColor = m_highlight
            ? Colors::yellow
            : (m_dark ? QColor(0x3a, 0x3a, 0x3c) : QColor(0xd1, 0xd1, 0xd6));

        QPainterPath bg;
        bg.addRoundedRect(r, 10, 10);
        p.fillPath(bg, bgColor);
        p.setPen(QPen(bdrColor, m_highlight ? 1.5 : 0.5));
        p.drawPath(bg);

        QColor bar = m_highlight ? Colors::yellow : (m_isUser ? Colors::accent : Colors::green);
        p.setPen(Qt::NoPen);
        p.setBrush(bar);
        p.drawRoundedRect(QRectF(r.left() + 1, r.top() + 8, 3, r.height() - 16), 1.5, 1.5);
    }

private slots:
    void toggleCollapse() {
        m_collapsed = !m_collapsed;
        updateContent();
    }

private:
    void updateContent() {
        if (m_collapsed && m_needsCollapse) {
            QString snippet = extractSnippets(m_fullContent, m_keywords);
            m_contentLabel->setText(highlightText(snippet, m_keywords));
            if (m_toggleBtn) m_toggleBtn->setText(QString::fromUtf8("▼ 展开全文"));
        } else {
            m_contentLabel->setText(highlightText(m_fullContent, m_keywords));
            if (m_toggleBtn) m_toggleBtn->setText(QString::fromUtf8("▲ 收起"));
        }
    }

    bool m_isUser, m_highlight, m_empty, m_dark;
    bool m_needsCollapse = false;
    bool m_collapsed = false;
    QString m_fullContent;
    QStringList m_keywords;
    QLabel *m_contentLabel = nullptr;
    QPushButton *m_toggleBtn = nullptr;
};

// ── 聊天消息列表 ──

class ChatWidget : public QScrollArea {
    Q_OBJECT
public:
    explicit ChatWidget(QWidget *parent = nullptr) : QScrollArea(parent) {
        setWidgetResizable(true);
        setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        setFrameShape(QFrame::NoFrame);
        setStyleSheet("QScrollArea{background:#1c1c1e; border:none;}");

        m_container = new QWidget;
        m_container->setStyleSheet("background:#1c1c1e;");
        m_layout = new QVBoxLayout(m_container);
        m_layout->setContentsMargins(4, 8, 4, 8);
        m_layout->setSpacing(2);
        m_layout->addStretch();
        setWidget(m_container);

        showEmpty(QString::fromUtf8("选择一个会话"));
    }

    void setKeywords(const QStringList &kw) { m_keywords = kw; }
    void setDarkMode(bool dark) { m_dark = dark; }

    void showEmpty(const QString &text) {
        clear();
        auto *l = new QLabel(text);
        l->setAlignment(Qt::AlignCenter);
        l->setStyleSheet("color:#48484a; font-size:14px; padding:60px; background:transparent;");
        m_layout->insertWidget(0, l);
    }

    void setNormalizedKeywords(const QStringList &nkw) { m_normalizedKeywords = nkw; }

    /// 在消息列表顶部显示摘要匹配提示
    void showSummaryMatch(const QString &summary, const QString &title) {
        QString text;
        if (!title.isEmpty()) text += "<b>标题:</b> " + highlightText(title, m_keywords) + "<br>";
        if (!summary.isEmpty()) text += "<b>摘要:</b> " + highlightText(summary, m_keywords);
        if (text.isEmpty()) return;

        auto *banner = new QWidget(m_container);
        auto *bl = new QVBoxLayout(banner);
        bl->setContentsMargins(16, 10, 16, 10);
        auto *icon = new QLabel(QString::fromUtf8("🔍 关键词匹配在会话摘要中:"));
        icon->setStyleSheet("color:#f9a825; font-size:11px; font-weight:600; background:transparent;");
        bl->addWidget(icon);
        auto *content = new QLabel(text);
        content->setWordWrap(true);
        content->setTextFormat(Qt::RichText);
        content->setStyleSheet(QString("color:%1; font-size:13px; background:transparent;")
            .arg(m_dark ? "#e5e5e7" : "#1c1c1e"));
        bl->addWidget(content);
        banner->setStyleSheet(QString("background:%1; border:1px solid #f9a825; border-radius:10px;")
            .arg(m_dark ? "#2a2a1e" : "#fef9e7"));
        m_layout->insertWidget(0, banner);
    }

    void setMessages(const QVector<Message> &msgs, const QString &hlUuid = {}) {
        clear();
        for (auto &msg : msgs) {
            // content 为空但 searchText 有内容(tool_result) → 用 searchText 摘要显示
            bool contentEmpty = msg.content.trimmed().isEmpty();
            bool searchEmpty = msg.searchText.trimmed().isEmpty();
            if (contentEmpty && searchEmpty) continue;

            bool hl = false;
            if (!m_keywords.isEmpty()) {
                // 1) 原始关键词直接匹配 (中文等)
                for (auto &kw : m_keywords) {
                    if (msg.content.contains(kw, Qt::CaseInsensitive) ||
                        msg.searchText.contains(kw, Qt::CaseInsensitive)) {
                        hl = true; break;
                    }
                }
                // 2) normalized 匹配 (处理 -_/ 等分隔符)
                if (!hl && !m_normalizedKeywords.isEmpty()) {
                    auto normalize = [](const QString &s) {
                        QString out;
                        out.reserve(s.size());
                        for (QChar ch : s) {
                            ushort u = ch.unicode();
                            if (ch == '_' || ch == '-' || ch == '/' ||
                                u == 0x3000 || u == 0x3001 || u == 0x3002 ||
                                (u >= 0x3008 && u <= 0x3011) || (u >= 0xFF01 && u <= 0xFF1F)) {
                                out.append(' ');
                            } else {
                                out.append(ch.toLower());
                            }
                        }
                        return out;
                    };
                    QString normContent = normalize(msg.content);
                    QString normSearch = normalize(msg.searchText);
                    for (auto &nkw : m_normalizedKeywords) {
                        if (normContent.contains(nkw) || normSearch.contains(nkw)) {
                            hl = true; break;
                        }
                    }
                }
            }
            if (!hlUuid.isEmpty() && msg.uuid == hlUuid) hl = true;

            auto *b = new MessageBubble(msg, hl, m_keywords, m_dark, m_container);
            m_layout->insertWidget(m_layout->count() - 1, b);
            if (!hlUuid.isEmpty() && msg.uuid == hlUuid) m_hlWidget = b;
        }
    }

    void scrollToHighlight() {
        if (m_hlWidget) ensureWidgetVisible(m_hlWidget, 0, 80);
    }

    void clear() {
        m_hlWidget = nullptr;
        while (m_layout->count() > 1) {
            auto *item = m_layout->takeAt(0);
            if (item->widget()) delete item->widget();
            delete item;
        }
    }

private:
    QWidget *m_container = nullptr;
    QVBoxLayout *m_layout = nullptr;
    QWidget *m_hlWidget = nullptr;
    QStringList m_keywords;
    QStringList m_normalizedKeywords;
    bool m_dark = true;
};
