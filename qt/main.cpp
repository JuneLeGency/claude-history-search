#include <QApplication>
#include "mainwindow.h"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    app.setApplicationName("Claude History Search");
    app.setOrganizationName("claude-his-search");

    MainWindow window;
    window.show();
    return app.exec();
}
