#include "TrainingVisualizer.hpp"

int main(int argc, char** argv) {
    // Initialize Qt application
    TrainingVisualizer::initialize(argc, argv);
    
    // Create visualizer
    TrainingVisualizer viz;
    viz.show();
/*
    // In your training loop:
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // ... training code ...
        
        double accuracy = calculate_accuracy();
        viz.addDataPoint(std::time(nullptr), accuracy);
        viz.processEvents();  // Update UI
    }
    */
   // Start the Qt event loop
   return TrainingVisualizer::exec();

    //return 0;
}