// mainwindow.cpp
#include "mainwindow.h"
#include <QVBoxLayout>
#include <QPushButton>
#include <QFileDialog>
#include <QDateTime>
#include <QFile>
#include <QTextStream>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    setupUI();
    createChart();
}

MainWindow::~MainWindow()
{
}

void MainWindow::setupUI()
{
    // Create central widget and layout
    QWidget *centralWidget = new QWidget(this);
    QVBoxLayout *layout = new QVBoxLayout(centralWidget);
    
    // Create chart view
    chartView = new QChartView(this);
    chartView->setMinimumSize(800, 600);
    layout->addWidget(chartView);
    
    // Create load button
    loadButton = new QPushButton("Load Data File", this);
    connect(loadButton, &QPushButton::clicked, this, &MainWindow::loadDataFile);
    layout->addWidget(loadButton);
    
    setCentralWidget(centralWidget);
    setWindowTitle("Training Accuracy Visualizer");
}

void MainWindow::createChart()
{
    QChart *chart = new QChart();
    chart->setTitle("Training Accuracy Over Time");
    
    // Create axes
    QDateTimeAxis *axisX = new QDateTimeAxis;
    axisX->setTitleText("Time");
    axisX->setFormat("yyyy-MM-dd HH:mm:ss");
    
    QValueAxis *axisY = new QValueAxis;
    axisY->setTitleText("Accuracy");
    axisY->setRange(0, 1);
    
    chart->addAxis(axisX, Qt::AlignBottom);
    chart->addAxis(axisY, Qt::AlignLeft);
    
    chartView->setChart(chart);
    chartView->setRenderHint(QPainter::Antialiasing);
}

void MainWindow::loadDataFile()
{
    QString fileName = QFileDialog::getOpenFileName(this,
        tr("Open Data File"), "",
        tr("Text Files (*.txt);;All Files (*)"));
        
    if (fileName.isEmpty())
        return;
        
    QFile file(fileName);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        return;

    // Create a new series for the data
    QLineSeries *series = new QLineSeries();
    
    QTextStream in(&file);
    while (!in.atEnd()) {
        QString line = in.readLine();
        QStringList fields = line.split(" ");
        
        if (fields.size() >= 2) {
            qint64 timestamp = fields[0].toLongLong();
            double accuracy = fields[1].toDouble();
            
            // Convert Unix timestamp to QDateTime
            QDateTime datetime = QDateTime::fromSecsSinceEpoch(timestamp);
            series->append(datetime.toMSecsSinceEpoch(), accuracy);
        }
    }
    
    // Update chart with new series
    QChart *chart = chartView->chart();
    chart->removeAllSeries();
    chart->addSeries(series);
    
    // Attach axes to the series
    series->attachAxis(chart->axes(Qt::Horizontal).first());
    series->attachAxis(chart->axes(Qt::Vertical).first());
    
    // Adjust axes ranges
    chart->axes(Qt::Horizontal).first()->setRange(
        QDateTime::fromMSecsSinceEpoch(series->at(0).x()),
        QDateTime::fromMSecsSinceEpoch(series->at(series->count()-1).x())
    );
    
    double minY = 0;
    double maxY = 1;
    for (int i = 0; i < series->count(); ++i) {
        double y = series->at(i).y();
        minY = qMin(minY, y);
        maxY = qMax(maxY, y);
    }
    chart->axes(Qt::Vertical).first()->setRange(minY * 0.9, maxY * 1.1);
}
