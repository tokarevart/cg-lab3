#include "mainwindow.h"

#include <QApplication>

int main(int argc, char *argv[]) {
    QApplication a(argc, argv);
    QApplication::setStyle("windowsvista");

    MainWindow w;
    w.show();
    return a.exec();
}
