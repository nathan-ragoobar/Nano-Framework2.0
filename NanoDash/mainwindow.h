// mainwindow.h
#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QtCharts>
#include <QLineSeries>
#include <QDateTimeAxis>
#include <QValueAxis>

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void loadDataFile();

private:
    QChartView *chartView;
    QPushButton *loadButton;
    void setupUI();
    void createChart();
};

#endif // MAINWINDOW_H
