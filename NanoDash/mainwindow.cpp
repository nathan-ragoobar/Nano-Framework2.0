#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include <QFile>
#include <QTextStream>
#include <QRegularExpression>
#include <QDebug>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , fileWatcher(new QFileSystemWatcher(this))
{
    ui->setupUi(this);
    
    setupCharts();
    
    // Connect file watcher
    connect(fileWatcher, &QFileSystemWatcher::fileChanged,
            this, &MainWindow::fileChanged);
            
    // Start watching the file
    fileWatcher->addPath("training_data.txt");
    
    loadData();
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::setupCharts()
{
    // Create charts for each metric
    QVector<QChartView*> chartViews = {
        ui->accuracyChart,
        ui->perplexityChart,
        ui->tokensChart,
        ui->learningRateChart,
        ui->normRateChart
    };
    
    QStringList titles = {
        "Training Accuracy",
        "Perplexity",
        "Tokens per Second",
        "Learning Rate",
        "Normalization Rate"
    };
    
    for (int i = 0; i < chartViews.size(); ++i) {
        QChart* chart = new QChart();
        charts.append(chart);
        
        chartViews[i]->setChart(chart);
        chartViews[i]->setRenderHint(QPainter::Antialiasing);
        
        chart->setTitle(titles[i]);
    }
}

void MainWindow::loadData()
{
    QFile file("training_data.txt");
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qDebug() << "Failed to open file";
        return;
    }

    trainingData.clear();
    QTextStream in(&file);
    while (!in.atEnd()) {
        QString line = in.readLine();
        QStringList fields = line.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
        
        if (fields.size() >= 6) {
            TrainingData data;
            data.timestamp = QDateTime::fromString(fields[0], Qt::ISODate);
            
            // Check each field for placeholder '*'
            data.accuracy = fields[1] == "*" ? std::numeric_limits<double>::quiet_NaN() 
                                           : fields[1].toDouble();
            data.perplexity = fields[2] == "*" ? std::numeric_limits<double>::quiet_NaN() 
                                              : fields[2].toDouble();
            data.tokensPerSecond = fields[3] == "*" ? std::numeric_limits<double>::quiet_NaN() 
                                                   : fields[3].toDouble();
            data.learningRate = fields[4] == "*" ? std::numeric_limits<double>::quiet_NaN() 
                                                : fields[4].toDouble();
            data.normRate = fields[5] == "*" ? std::numeric_limits<double>::quiet_NaN() 
                                           : fields[5].toDouble();
            
            trainingData.append(data);
        }
    }
    
    file.close();
    updateCharts();
}

void MainWindow::updateSingleChart(QChart* chart, int metricIndex)
{
    if (trainingData.isEmpty())
        return;

    chart->removeAllSeries();
    QLineSeries *series = new QLineSeries();
    
    qint64 minTime = trainingData.first().timestamp.toSecsSinceEpoch();
    
    std::sort(trainingData.begin(), trainingData.end(), 
              [](const TrainingData& a, const TrainingData& b) {
                  return a.timestamp < b.timestamp;
              });
    
    double lastValidValue = std::numeric_limits<double>::quiet_NaN();
    QVector<QPointF> points;
    
    for (const TrainingData& data : trainingData) {
        double x = data.timestamp.toSecsSinceEpoch() - minTime;
        double y;
        
        switch(metricIndex) {
            case 0: y = data.accuracy; break;
            case 1: y = data.perplexity; break;
            case 2: y = data.tokensPerSecond; break;
            case 3: y = data.learningRate; break;
            case 4: y = data.normRate; break;
            default: y = std::numeric_limits<double>::quiet_NaN();
        }
        
        if (!std::isnan(y)) {
            points.append(QPointF(x, y));
            lastValidValue = y;
        } else if (!std::isnan(lastValidValue)) {
            points.append(QPointF(x, lastValidValue));
        }
    }
    
    series->replace(points);
    chart->addSeries(series);
    
    // Remove old axes
    for(auto* axis : chart->axes()) {
        chart->removeAxis(axis);
        delete axis;
    }
    
    chart->createDefaultAxes();
    
    if (QAbstractAxis *axisX = chart->axes(Qt::Horizontal).first()) {
        axisX->setTitleText("Time (seconds)");
    }
}

void MainWindow::updateCharts()
{
    // Update all charts
    for (int i = 0; i < charts.size(); ++i) {
        updateSingleChart(charts[i], i);
    }
}

void MainWindow::fileChanged(const QString &path)
{
    // Rewatch the file
    if (!fileWatcher->files().contains(path)) {
        fileWatcher->addPath(path);
    }
    
    loadData();
}