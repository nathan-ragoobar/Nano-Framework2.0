// TrainingVisualizer.cpp
#include "TrainingVisualizer.hpp"
#include <QApplication>
#include <QMainWindow>
#include <QtCharts>
#include <QVBoxLayout>

std::unique_ptr<QApplication> TrainingVisualizer::app;

struct TrainingVisualizer::Private {
    std::unique_ptr<QMainWindow> window;
    QChartView* chartView;
    QLineSeries* series;
    
    Private() : window(std::make_unique<QMainWindow>()),
                chartView(nullptr),
                series(nullptr) {
        setupUI();
    }

    void setupUI() {
        window->setWindowTitle("Training Progress");
        window->resize(800, 600);

        // Create central widget and layout
        auto centralWidget = new QWidget(window.get());
        auto layout = new QVBoxLayout(centralWidget);

        // Create chart
        auto chart = new QChart();
        series = new QLineSeries(chart);
        chart->addSeries(series);
        chart->setTitle("Training Accuracy Over Time");

        // Create axes
        auto axisX = new QDateTimeAxis;
        axisX->setTitleText("Time");
        axisX->setFormat("hh:mm:ss");
        
        auto axisY = new QValueAxis;
        axisY->setTitleText("Accuracy");
        axisY->setRange(0, 1);

        chart->addAxis(axisX, Qt::AlignBottom);
        chart->addAxis(axisY, Qt::AlignLeft);
        
        series->attachAxis(axisX);
        series->attachAxis(axisY);

        // Create chart view
        chartView = new QChartView(chart);
        chartView->setRenderHint(QPainter::Antialiasing);
        layout->addWidget(chartView);

        window->setCentralWidget(centralWidget);
    }
};

bool TrainingVisualizer::initialize(int& argc, char** argv) {
    if (!app) {
        app = std::make_unique<QApplication>(argc, argv);
        return true;
    }
    return false;
}

TrainingVisualizer::TrainingVisualizer() : d(std::make_unique<Private>()) {}

TrainingVisualizer::~TrainingVisualizer() = default;

void TrainingVisualizer::addDataPoint(double timestamp, double accuracy) {
    if (d->series) {
        QDateTime datetime = QDateTime::fromSecsSinceEpoch(timestamp);
        d->series->append(datetime.toMSecsSinceEpoch(), accuracy);
        
        // Update axes ranges
        auto chart = d->chartView->chart();
        if (d->series->count() > 0) {
            chart->axes(Qt::Horizontal).first()->setRange(
                QDateTime::fromMSecsSinceEpoch(d->series->at(0).x()),
                QDateTime::fromMSecsSinceEpoch(d->series->at(d->series->count()-1).x())
            );
            
            // Find min/max accuracy for Y axis
            double minY = accuracy;
            double maxY = accuracy;
            for (int i = 0; i < d->series->count(); ++i) {
                double y = d->series->at(i).y();
                minY = std::min(minY, y);
                maxY = std::max(maxY, y);
            }
            
            // Add 10% padding to Y axis
            double padding = (maxY - minY) * 0.1;
            chart->axes(Qt::Vertical).first()->setRange(
                std::max(0.0, minY - padding),
                std::min(1.0, maxY + padding)
            );
        }
    }
}

void TrainingVisualizer::show() {
    if (d->window) {
        d->window->show();
    }
}

void TrainingVisualizer::processEvents() {
    if (app) {
        app->processEvents();
    }
}
