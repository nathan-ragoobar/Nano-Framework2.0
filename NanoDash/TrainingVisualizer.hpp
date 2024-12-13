#ifndef NANO_DASH_TRAINING_VISUALIZER_HPP_
#define NANO_DASH_TRAINING_VISUALIZER_HPP_


#include <memory>
#include <string>

// Forward declarations to avoid Qt dependencies in header
class QApplication;
class QMainWindow;
class QChartView;
class QPushButton;

class TrainingVisualizer {
public:
    // Initialize with argc and argv needed for Qt
    static bool initialize(int& argc, char** argv);
    
    // Create a visualizer instance
    TrainingVisualizer();
    ~TrainingVisualizer();

    // Methods to update the visualization
    void addDataPoint(double timestamp, double accuracy);
    void show();
    void processEvents();  // Allow UI to update during training

    // Prevent copying
    TrainingVisualizer(const TrainingVisualizer&) = delete;
    TrainingVisualizer& operator=(const TrainingVisualizer&) = delete;

private:
    static std::unique_ptr<QApplication> app;
    struct Private;
    std::unique_ptr<Private> d;
};

#endif  // NANO_DASH_TRAINING_VISUALIZER_HPP_
