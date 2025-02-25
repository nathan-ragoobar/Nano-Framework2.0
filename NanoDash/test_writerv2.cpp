#include "writer.hpp"
#include <iostream>
#include <random>
#include <thread>

int main() {
    // Define metrics we want to track
    std::vector<std::string> metrics = {
        "loss", 
        "accuracy", 
        "learning_rate"
    };

    // Create metric writer
    MetricWriter writer("training_metrics", metrics);

    // Random number generation
    std::random_device rd;
    std::mt19937 gen(rd());

    // Initial values
    double loss = 2.5;
    double accuracy = 0.1;
    double learning_rate = 0.001;

    // Simulate 100 training steps
    for (int step = 0; step < 100; step++) {
        // Simulate training patterns
        loss *= std::uniform_real_distribution<>(0.95, 0.98)(gen);
        accuracy = std::min(0.99, accuracy + std::uniform_real_distribution<>(0.01, 0.03)(gen));
        learning_rate *= 0.99;  // Learning rate decay

        // Log metrics
        writer.addScalar("loss", loss);
        writer.addScalar("accuracy", accuracy);
        writer.addScalar("learning_rate", learning_rate);

        // Print progress
        if (step % 10 == 0) {
            std::cout << "Step " << step 
                      << " - Loss: " << loss 
                      << " - Accuracy: " << accuracy 
                      << " - LR: " << learning_rate << std::endl;
        }

        // Small delay to simulate training time
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "Data generation complete. Check training_metrics.csv" << std::endl;
    return 0;
}