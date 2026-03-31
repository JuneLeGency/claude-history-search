#pragma once
#include <QStyledItemDelegate>
#include <QPainter>
#include <QApplication>

class SessionDelegate : public QStyledItemDelegate {
    Q_OBJECT
public:
    bool m_dark = true;

    using QStyledItemDelegate::QStyledItemDelegate;

    QSize sizeHint(const QStyleOptionViewItem &, const QModelIndex &) const override {
        return QSize(0, 56); // 增高
    }

    void paint(QPainter *p, const QStyleOptionViewItem &opt, const QModelIndex &idx) const override {
        p->setRenderHint(QPainter::Antialiasing);
        QRect r = opt.rect.adjusted(4, 2, -4, -2);

        // 选中/hover 背景
        bool selected = opt.state & QStyle::State_Selected;
        bool hover = opt.state & QStyle::State_MouseOver;
        if (selected) {
            QPainterPath bg;
            bg.addRoundedRect(QRectF(r), 8, 8);
            p->fillPath(bg, QColor(m_dark ? "#3a3a3c" : "#d1d1d6"));
        } else if (hover) {
            QPainterPath bg;
            bg.addRoundedRect(QRectF(r), 8, 8);
            p->fillPath(bg, QColor(m_dark ? "#2c2c2e" : "#e5e5ea"));
        }

        // 解析 display text: "[2026-03-30 14:09:27] project - summary"
        QString text = idx.data(Qt::DisplayRole).toString();
        QString time, project, summary;

        int bracket = text.indexOf(']');
        if (bracket > 0) {
            time = text.mid(1, bracket - 1); // "2026-03-30 14:09:27"
            QString rest = text.mid(bracket + 2); // "project - summary"
            int dash = rest.indexOf(" - ");
            if (dash > 0) {
                project = rest.left(dash);
                summary = rest.mid(dash + 3);
            } else {
                summary = rest;
            }
        } else {
            summary = text;
        }

        int x = r.left() + 10;
        int y = r.top() + 4;
        int w = r.width() - 20;

        QFont f = p->font();

        // 第一行: 目录名(强调色) — 摘要(普通色)
        if (!project.isEmpty()) {
            // 目录名: 蓝/青色
            f.setPointSize(12);
            f.setWeight(QFont::DemiBold);
            p->setFont(f);
            QColor projColor = m_dark ? QColor("#64d2ff") : QColor("#0a84ff"); // cyan / blue
            if (selected) projColor = m_dark ? QColor("#ffffff") : QColor("#0a84ff");
            p->setPen(projColor);
            QString projText = p->fontMetrics().elidedText(project, Qt::ElideRight, w * 0.4);
            p->drawText(QRect(x, y, w, 20), Qt::AlignLeft | Qt::AlignVCenter, projText);
            int projW = p->fontMetrics().horizontalAdvance(projText);

            // " — "
            f.setWeight(QFont::Normal);
            p->setFont(f);
            p->setPen(QColor(m_dark ? "#48484a" : "#c7c7cc"));
            p->drawText(QRect(x + projW, y, 30, 20), Qt::AlignLeft | Qt::AlignVCenter, " — ");

            // 摘要: 普通白/黑
            p->setPen(QColor(m_dark ? "#a1a1a6" : "#3c3c43"));
            int sumX = x + projW + 30;
            QString sumText = p->fontMetrics().elidedText(summary, Qt::ElideRight, w - (sumX - x));
            p->drawText(QRect(sumX, y, w - (sumX - x), 20), Qt::AlignLeft | Qt::AlignVCenter, sumText);
        } else {
            f.setPointSize(12);
            f.setWeight(QFont::Medium);
            p->setFont(f);
            p->setPen(QColor(m_dark ? "#e5e5e7" : "#1c1c1e"));
            p->drawText(QRect(x, y, w, 20), Qt::AlignLeft | Qt::AlignVCenter,
                         p->fontMetrics().elidedText(summary, Qt::ElideRight, w));
        }

        // 第二行: 时间
        y += 24;
        f.setPointSize(10);
        f.setWeight(QFont::Normal);
        p->setFont(f);
        p->setPen(QColor(m_dark ? "#636366" : "#aeaeb2"));
        p->drawText(QRect(x, y, w, 18), Qt::AlignLeft | Qt::AlignVCenter, time);
    }
};
