#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QGraphicsScene>
#include <QMainWindow>
#include <array>

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow {
    Q_OBJECT

  public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

  private slots:
    void on_gen_btn_clicked();

    void on_gen_builtin_btn_clicked();

    void on_thous_gen_builtin_btn_clicked();

    void on_thous_gen_btn_clicked();

    void on_gen_gl_btn_clicked();

    void on_thous_gen_gl_btn_clicked();

  private:
    Ui::MainWindow *ui;
    std::array<QGraphicsScene *, 4> scenes;
};
#endif // MAINWINDOW_H
