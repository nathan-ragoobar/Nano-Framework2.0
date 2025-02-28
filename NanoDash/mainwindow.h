#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QtCharts>
#include <QLineSeries>
#include <QChartView>
#include <QValueAxis>
#include <QDateTime>
#include <QFileSystemWatcher>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

QT_USE_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void updateCharts();
    void fileChanged(const QString &path);

private:
    Ui::MainWindow *ui;
    QFileSystemWatcher *fileWatcher;
    QVector<QChart*> charts;
    
    struct TrainingData {
        QDateTime timestamp;
        double accuracy;
        double perplexity;
        double tokensPerSecond;
        double learningRate;
        double normRate;
    };
    QVector<TrainingData> trainingData;
    
    void loadData();
    void setupCharts();
    void updateSingleChart(QChart* chart, int metricIndex);
};
#endif // MAINWINDOW_H