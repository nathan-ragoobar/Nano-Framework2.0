#include "writer.hpp"
#include <iostream>
#include <random>
#include <thread>
#include <cmath>

int main() {
    // Define metrics we want to track
    std::vector<std::string> metrics = {
        "accuracy",
        "learning_rate",
        "train_loss",
        "val_loss"
    };

    // Create metric writer
    MetricWriter writer("training_metrics", metrics);

    // Random number generation
    std::random_device rd;
    std::mt19937 gen(rd());

    // Training parameters
    const int num_epochs = 10;
    const int steps_per_epoch = 10;
    const double initial_lr = 0.01;

    // Initial values
    double train_loss = 5.0;
    double val_loss = 5.5;    // Validation starts higher
    double accuracy = 0.1;
    double learning_rate = initial_lr;

    // Training loop - epochs and steps
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        std::cout << "\nEpoch " << epoch + 1 << "/" << num_epochs << std::endl;

        // Training phase
        for (int step = 0; step < steps_per_epoch; step++) {
            int global_step = epoch * steps_per_epoch + step;

            // Simulate training patterns with noise
            double noise = std::uniform_real_distribution<>(-0.1, 0.1)(gen);
            train_loss = 5.0 * std::exp(-0.2 * epoch) + noise;
            accuracy = 0.5 * (1.0 + std::tanh(0.5 * epoch + noise));
            learning_rate = initial_lr * std::exp(-0.1 * epoch);

            // Log training metrics
            writer.addTrainingLoss(train_loss, global_step);
            writer.addScalar("accuracy", accuracy, global_step);
            writer.addScalar("learning_rate", learning_rate, global_step);

            // Print progress
            std::cout << "Step " << global_step 
                      << " - Train Loss: " << train_loss 
                      << " - Accuracy: " << accuracy 
                      << " - LR: " << learning_rate << std::endl;
        }

        // Validation phase at end of epoch
        double val_noise = std::uniform_real_distribution<>(-0.05, 0.05)(gen);
        val_loss = 5.0 * std::exp(-0.15 * epoch) + val_noise;  // Slower improvement than training
        
        // Log validation metrics
        writer.addValidationLoss(val_loss, (epoch + 1) * steps_per_epoch - 1);
        std::cout << "Validation Loss: " << val_loss << std::endl;

        // Small delay to simulate computation time
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "\nTraining complete. Data saved to training_metrics.csv" << std::endl;
    return 0;
}