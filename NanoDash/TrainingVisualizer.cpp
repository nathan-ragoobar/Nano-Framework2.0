// TrainingVisualizer.cpp
#include "TrainingVisualizer.hpp"
#include <QApplication>
#include <QMainWindow>
#include <QtCharts>
#include <QVBoxLayout>
#include <QFileSystemWatcher>

std::unique_ptr<QApplication> TrainingVisualizer::app;

struct TrainingVisualizer::Private {
    struct TabData {
        QChartView* chartView;
        QLineSeries* series;
        TabData() : chartView(nullptr), series(nullptr) {}
    };

    std::unique_ptr<QMainWindow> window;
    QTabWidget* tabWidget;
    QMap<QString, TabData> tabData;
    QChartView* chartView;  // Add these member variables
    QLineSeries* series;    // for backward compatibility
    QFileSystemWatcher fileWatcher;
    QString lastFileName;
    
    Private() : window(std::make_unique<QMainWindow>()),
                tabWidget(nullptr),
                chartView(nullptr),
                series(nullptr) {
        setupUI();
    }

    void setupUI() {
        window->setWindowTitle("Training Dashboard");
        window->resize(800, 600);
    
        // Create toolbar and load button
        auto toolbar = new QToolBar();
        auto loadButton = new QPushButton("Load Data");
        toolbar->addWidget(loadButton);
        window->addToolBar(Qt::TopToolBarArea, toolbar);
    
        // Connect load button
        QObject::connect(loadButton, &QPushButton::clicked, [this]() {
            QString fileName = QFileDialog::getOpenFileName(window.get(),
                "Load Training Data", "",
                "CSV Files (*.csv);;All Files (*)");
            
            if (!fileName.isEmpty()) {
                loadDataFromFile(fileName);
            }
        });
    
        // Create tab widget
        tabWidget = new QTabWidget(window.get());
    
        // Create all tabs with charts
        createAccuracyTab();
        //createLossTab();
        createTrainingLossTab();
        createValidationLossTab();
        createPerplexityTab();
        createTokensTab();
        createLearningRateTab();
        
        
    
        window->setCentralWidget(tabWidget);
    }
    
    void createAccuracyTab() {
        auto widget = new QWidget();
        auto layout = new QVBoxLayout(widget);
    
        auto chart = new QChart();
        series = new QLineSeries(chart);
        chart->addSeries(series);
        chart->setTitle("Training Accuracy Over Time");

        // Enable hover events on the chart
        chart->setAcceptHoverEvents(true);
    
        // Create axes
        auto axisX = new QValueAxis;
        axisX->setTitleText("Steps");  // Updated label
        axisX->setLabelFormat("%i");   // Integer format for steps
        
        auto axisY = new QValueAxis;
        axisY->setTitleText("Accuracy");
        axisY->setRange(0, 1);
    
        chart->addAxis(axisX, Qt::AlignBottom);
        chart->addAxis(axisY, Qt::AlignLeft);
        series->attachAxis(axisX);
        series->attachAxis(axisY);
    
        chartView = new QChartView(chart);
        chartView->setRenderHint(QPainter::Antialiasing);

        // Enable rubber-band zooming and panning
        chartView->setRubberBand(QChartView::RectangleRubberBand);
        chartView->setDragMode(QGraphicsView::ScrollHandDrag);

        // Optionally show data points
        series->setPointsVisible(true);

        // Create a local pointer to chartView to capture
        QChartView* localChartView = chartView;

        // Connect hover signal to show tooltips
        QObject::connect(series, &QLineSeries::hovered, chartView,
                        [localChartView](const QPointF &point, bool state) {
            if (state) {
                QToolTip::showText(
                    QCursor::pos(),
                    QString("Step: %1\nValue: %2").arg(point.x()).arg(point.y()),
                    localChartView
                );
            } else {
                QToolTip::hideText();
            }
        });

        layout->addWidget(chartView);
        tabWidget->addTab(widget, "Accuracy");

        TabData data;
        data.chartView = chartView;
        data.series = series;
        tabData["Accuracy"] = data;
    }
    
    // Add similar methods for other tabs
    void createPerplexityTab() {
        auto widget = new QWidget();
        auto layout = new QVBoxLayout(widget);
    
        auto chart = new QChart();
        series = new QLineSeries(chart);
        chart->addSeries(series);
        chart->setTitle("Perplexity Over Time");

        // Enable hover events on the chart
        chart->setAcceptHoverEvents(true);
    
        // Create axes
        auto axisX = new QValueAxis;
        axisX->setTitleText("Steps");  // Updated label
        axisX->setLabelFormat("%i");   // Integer format for steps
            
        auto axisY = new QValueAxis;
        axisY->setTitleText("Perplexity");
        axisY->setRange(0, 1);
    
        chart->addAxis(axisX, Qt::AlignBottom);
        chart->addAxis(axisY, Qt::AlignLeft);
        series->attachAxis(axisX);
        series->attachAxis(axisY);
    
        chartView = new QChartView(chart);
        chartView->setRenderHint(QPainter::Antialiasing);

        // Enable rubber-band zooming and panning
        chartView->setRubberBand(QChartView::RectangleRubberBand);
        chartView->setDragMode(QGraphicsView::ScrollHandDrag);

        // Optionally show data points
        series->setPointsVisible(true);

        // Create a local pointer to chartView to capture
        QChartView* localChartView = chartView;

        // Connect hover signal to show tooltips
        QObject::connect(series, &QLineSeries::hovered, chartView,
                        [localChartView](const QPointF &point, bool state) {
            if (state) {
                QToolTip::showText(
                    QCursor::pos(),
                    QString("Step: %1\nValue: %2").arg(point.x()).arg(point.y()),
                    localChartView
                );
            } else {
                QToolTip::hideText();
            }
        });

        layout->addWidget(chartView);
        tabWidget->addTab(widget, "Perplexity");

        TabData data;
        data.chartView = chartView;
        data.series = series;
        tabData["Perplexity"] = data;
    }


    void createTokensTab() {
        auto widget = new QWidget();
        auto layout = new QVBoxLayout(widget);
    
        auto chart = new QChart();
        series = new QLineSeries(chart);
        chart->addSeries(series);
        chart->setTitle("Tokens Per Second Over Time");

        // Enable hover events on the chart
        chart->setAcceptHoverEvents(true);
    
        // Create axes
        auto axisX = new QValueAxis;
        axisX->setTitleText("Steps");  // Updated label
        axisX->setLabelFormat("%i");   // Integer format for steps
        
        auto axisY = new QValueAxis;
        axisY->setTitleText("Tokens/Second");
        axisY->setRange(0, 1);
    
        chart->addAxis(axisX, Qt::AlignBottom);
        chart->addAxis(axisY, Qt::AlignLeft);
        series->attachAxis(axisX);
        series->attachAxis(axisY);
    
        chartView = new QChartView(chart);
        chartView->setRenderHint(QPainter::Antialiasing);

        // Enable rubber-band zooming and panning
        chartView->setRubberBand(QChartView::RectangleRubberBand);
        chartView->setDragMode(QGraphicsView::ScrollHandDrag);

        // Optionally show data points
        series->setPointsVisible(true);

        // Create a local pointer to chartView to capture
        QChartView* localChartView = chartView;

        // Connect hover signal to show tooltips
        QObject::connect(series, &QLineSeries::hovered, chartView,
                        [localChartView](const QPointF &point, bool state) {
            if (state) {
                QToolTip::showText(
                    QCursor::pos(),
                    QString("Step: %1\nValue: %2").arg(point.x()).arg(point.y()),
                    localChartView
                );
            } else {
                QToolTip::hideText();
            }
        });

        layout->addWidget(chartView);
        tabWidget->addTab(widget, "Tokens");

        TabData data;
        data.chartView = chartView;
        data.series = series;
        tabData["Tokens"] = data;
    }

    void createLearningRateTab() {
        auto widget = new QWidget();
        auto layout = new QVBoxLayout(widget);
    
        auto chart = new QChart();
        series = new QLineSeries(chart);
        chart->addSeries(series);
        chart->setTitle("Learning Rate Over Time");
    
        // Enable hover events on the chart
        chart->setAcceptHoverEvents(true);

        // Create axes
        auto axisX = new QValueAxis;
        axisX->setTitleText("Steps");  // Updated label
        axisX->setLabelFormat("%i");   // Integer format for steps
        
        auto axisY = new QValueAxis;
        axisY->setTitleText("Learning Rate");
        axisY->setRange(0, 1);
    
        chart->addAxis(axisX, Qt::AlignBottom);
        chart->addAxis(axisY, Qt::AlignLeft);
        series->attachAxis(axisX);
        series->attachAxis(axisY);
    
        chartView = new QChartView(chart);
        chartView->setRenderHint(QPainter::Antialiasing);

        // Enable rubber-band zooming and panning
        chartView->setRubberBand(QChartView::RectangleRubberBand);
        chartView->setDragMode(QGraphicsView::ScrollHandDrag);

        // Optionally show data points
        series->setPointsVisible(true);

        // Create a local pointer to chartView to capture
        QChartView* localChartView = chartView;

        // Connect hover signal to show tooltips
        QObject::connect(series, &QLineSeries::hovered, chartView,
                        [localChartView](const QPointF &point, bool state) {
            if (state) {
                QToolTip::showText(
                    QCursor::pos(),
                    QString("Step: %1\nValue: %2").arg(point.x()).arg(point.y()),
                    localChartView
                );
            } else {
                QToolTip::hideText();
            }
        });

        layout->addWidget(chartView);
        tabWidget->addTab(widget, "Learning Rate");

        TabData data;
        data.chartView = chartView;
        data.series = series;
        tabData["Learning Rate"] = data;
    }
/*
    void createLossTab() {
        auto widget = new QWidget();
        auto layout = new QVBoxLayout(widget);
    
        auto chart = new QChart();
        series = new QLineSeries(chart);
        chart->addSeries(series);
        chart->setTitle("Loss Over Time");
    
        // Create axes
        auto axisX = new QDateTimeAxis;
        axisX->setTitleText("Time");
        axisX->setFormat("hh:mm:ss");
        
        auto axisY = new QValueAxis;
        axisY->setTitleText("Loss");
        axisY->setRange(0, 1);
    
        chart->addAxis(axisX, Qt::AlignBottom);
        chart->addAxis(axisY, Qt::AlignLeft);
        series->attachAxis(axisX);
        series->attachAxis(axisY);
    
        chartView = new QChartView(chart);
        chartView->setRenderHint(QPainter::Antialiasing);
        layout->addWidget(chartView);
    
        tabWidget->addTab(widget, "Loss");

        // Store the tab data
        TabData data;
        data.chartView = chartView;
        data.series = series;
        tabData["Loss"] = data;
    }
*/
    void createTrainingLossTab() {
        auto widget = new QWidget();
        auto layout = new QVBoxLayout(widget);

        auto chart = new QChart();
        series = new QLineSeries(chart);
        chart->addSeries(series);
        chart->setTitle("Training Loss Over Time");

        // Enable hover events on the chart
        chart->setAcceptHoverEvents(true);

        auto axisX = new QValueAxis;
        axisX->setTitleText("Steps");  // Updated label
        axisX->setLabelFormat("%i");   // Integer format for steps

        auto axisY = new QValueAxis;
        axisY->setTitleText("Training Loss");
        axisY->setRange(0, 1);

        chart->addAxis(axisX, Qt::AlignBottom);
        chart->addAxis(axisY, Qt::AlignLeft);
        series->attachAxis(axisX);
        series->attachAxis(axisY);

        chartView = new QChartView(chart);
        chartView->setRenderHint(QPainter::Antialiasing);

        // Enable rubber-band zooming and panning
        chartView->setRubberBand(QChartView::RectangleRubberBand);
        chartView->setDragMode(QGraphicsView::ScrollHandDrag);

        // Optionally show data points
        series->setPointsVisible(true);

        // Create a local pointer to chartView to capture
        QChartView* localChartView = chartView;

        // Connect hover signal to show tooltips
        QObject::connect(series, &QLineSeries::hovered, chartView,
                        [localChartView](const QPointF &point, bool state) {
            if (state) {
                QToolTip::showText(
                    QCursor::pos(),
                    QString("Step: %1\nValue: %2").arg(point.x()).arg(point.y()),
                    localChartView
                );
            } else {
                QToolTip::hideText();
            }
        });

        layout->addWidget(chartView);
        tabWidget->addTab(widget, "Training Loss");

        TabData data;
        data.chartView = chartView;
        data.series = series;
        tabData["Training Loss"] = data;
    }

    void createValidationLossTab() {
        auto widget = new QWidget();
        auto layout = new QVBoxLayout(widget);

        auto chart = new QChart();
        series = new QLineSeries(chart);
        chart->addSeries(series);
        chart->setTitle("Validation Loss Over Time");
        
        // Enable hover events on the chart
        chart->setAcceptHoverEvents(true);

        // Create axes
        auto axisX = new QValueAxis;
        axisX->setTitleText("Steps");  // Updated label
        axisX->setLabelFormat("%i");   // Integer format for steps

        auto axisY = new QValueAxis;
        axisY->setTitleText("Validation Loss");
        axisY->setRange(0, 1);

        chart->addAxis(axisX, Qt::AlignBottom);
        chart->addAxis(axisY, Qt::AlignLeft);
        series->attachAxis(axisX);
        series->attachAxis(axisY);

        chartView = new QChartView(chart);
        chartView->setRenderHint(QPainter::Antialiasing);

        // Enable rubber-band zooming and panning
        chartView->setRubberBand(QChartView::RectangleRubberBand);
        chartView->setDragMode(QGraphicsView::ScrollHandDrag);

        // Optionally show data points
        series->setPointsVisible(true);

        // Create a local pointer to chartView to capture
        QChartView* localChartView = chartView;

        // Connect hover signal to show tooltips
        QObject::connect(series, &QLineSeries::hovered, chartView,
                        [localChartView](const QPointF &point, bool state) {
            if (state) {
                QToolTip::showText(
                    QCursor::pos(),
                    QString("Step: %1\nValue: %2").arg(point.x()).arg(point.y()),
                    localChartView
                );
            } else {
                QToolTip::hideText();
            }
        });

        layout->addWidget(chartView);
        tabWidget->addTab(widget, "Validation Loss");

        TabData data;
        data.chartView = chartView;
        data.series = series;
        tabData["Validation Loss"] = data;
    }

    void loadDataFromFile(const QString& fileName) {
        // If a file is already being watched, remove it:
        if (!lastFileName.isEmpty()) {
            fileWatcher.removePath(lastFileName);
        }

        // Start watching the new file
        fileWatcher.addPath(fileName);
        lastFileName = fileName;

        // When the file changes, reload data
        QObject::connect(&fileWatcher, &QFileSystemWatcher::fileChanged, [this](const QString &path){
            // Reload data if file is modified
            loadDataFromFile(path);
        });
        
        QFile file(fileName);
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
            qDebug() << "Failed to open file";
            return;
        }
    
        // Create a mapping from tab names to CSV column names
        QMap<QString, QString> tabToHeader {
            {"Accuracy", "accuracy"},
            {"Training Loss", "train_loss"},
            {"Validation Loss", "val_loss"},
            {"Perplexity", "perplexity"},
            {"Tokens", "tokens_per_second"},
            {"Learning Rate", "learning_rate"},
            {"Normalization Rate", "normalization_rate"}  // Adjust if needed
        };
    
        // Prepare a map of tabName -> column index
        QMap<QString, int> tabColumnIndex;
        // Prepare a map of tabName -> new QLineSeries
        QMap<QString, QLineSeries*> newSeriesMap;
    
        QTextStream in(&file);
        QString headerLine = in.readLine();
        QStringList headers = headerLine.split(',');

        // Find the steps column
        int stepsIndex = headers.indexOf("Step");  // Add this
        if (stepsIndex < 0) {
            qDebug() << "No 'step' column found in CSV";
            file.close();
            return;
        }
    
        // Identify each tab’s CSV column index and create its series
        for (auto it = tabData.begin(); it != tabData.end(); ++it) {
            QString tabName = it.key();
            if (!tabToHeader.contains(tabName)){
                continue; // Skip if we don’t have a header mapping for this tab
            }
            int colIndex = headers.indexOf(tabToHeader[tabName]);
            if (colIndex < 0){
                continue; // Skip if column not found in the CSV
            }
            tabColumnIndex[tabName] = colIndex;
            newSeriesMap[tabName] = new QLineSeries();
        }
    
        // Read each line, parse timestamp & add data points for every tab that has a valid column
        int currentStep = 0;  // Fallback if no step column
        while (!in.atEnd()) {
            QString line = in.readLine();
            QStringList fields = line.split(',');
            if (fields.size() < 2) continue;
    
            // Get step value
            bool ok;
            double step = fields[stepsIndex].toDouble(&ok);
            if (!ok) {
                step = currentStep++;  // Use incremental step if parsing fails
            }
    
            // For each tab with a known column, parse and append data
            for (auto it = tabColumnIndex.begin(); it != tabColumnIndex.end(); ++it) {
                QString tabName = it.key();
                int colIndex = it.value();
                if (colIndex >= fields.size()) continue;
    
                QString valueStr = fields[colIndex];
                if (valueStr == "NA") continue;
    
                bool ok;
                double value = valueStr.toDouble(&ok);
                if (!ok) continue;
    
                newSeriesMap[tabName]->append(step, value);  // Use step instead of timestamp
            }
        }
        file.close();
    
        // Update each tab's chart
        for (auto it = newSeriesMap.begin(); it != newSeriesMap.end(); ++it) {
            QString tabName = it.key();
            QLineSeries* series = it.value();
            if (!series || series->count() == 0) {
                delete series;
                continue;
            }
    
            auto& currentTabData = tabData[tabName];
            QChart* chart = currentTabData.chartView->chart();
            chart->removeAllSeries();
            chart->addSeries(series);
            series->attachAxis(chart->axes(Qt::Horizontal).first());
            series->attachAxis(chart->axes(Qt::Vertical).first());
    
            // Update X axis range
            auto xAxis = qobject_cast<QValueAxis*>(chart->axes(Qt::Horizontal).first());
            if (xAxis) {
                double minX = series->at(0).x();
                double maxX = series->at(series->count() - 1).x();
                double padding = (maxX - minX) * 0.1;
                xAxis->setRange(minX - padding, maxX + padding);
            }
    
            // Update Y axis range with padding
            double minY = std::numeric_limits<double>::max();
            double maxY = std::numeric_limits<double>::lowest();
            for (int i = 0; i < series->count(); ++i) {
                double y = series->at(i).y();
                minY = qMin(minY, y);
                maxY = qMax(maxY, y);
            }
            double padding = (maxY - minY) * 0.1;
            auto yAxis = qobject_cast<QValueAxis*>(chart->axes(Qt::Vertical).first());
            if (yAxis) {
                yAxis->setRange(minY - padding, maxY + padding);
            }
        }
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


int TrainingVisualizer::exec() {
    if (app) {
        return app->exec();
    }
    return 1;
}
