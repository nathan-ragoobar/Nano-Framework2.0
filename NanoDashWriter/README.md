# MetricWriter Documentation

A thread-safe C++ library for logging training metrics to CSV files. Useful for tracking machine learning experiments and visualizing training progress.

## Features

- Thread-safe metric logging
- Automatic CSV file creation and management
- Support for training and validation metrics
- Timestamped entries
- Step-based tracking
- Automatic header generation

## Usage

### Basic Setup

```cpp
#include "writer.hpp"

// Define metrics to track
std::vector<std::string> metrics = {
    "accuracy",
    "learning_rate",
    "train_loss",
    "val_loss"
};

// Create writer instance
MetricWriter writer("experiment_metrics", metrics);
```

### Logging Metrics

```cpp
// Log training loss
writer.addTrainingLoss(0.5);  // Auto-increment step
writer.addTrainingLoss(0.4, 10);  // Explicit step

// Log validation loss
writer.addValidationLoss(0.6);
writer.addValidationLoss(0.55, 10);

// Log other metrics
writer.addScalar("accuracy", 0.85);
writer.addScalar("learning_rate", 0.001);
```

### Training Loop Example

```cpp
const int num_epochs = 10;
const int steps_per_epoch = 100;

for (int epoch = 0; epoch < num_epochs; epoch++) {
    // Training phase
    for (int step = 0; step < steps_per_epoch; step++) {
        int global_step = epoch * steps_per_epoch + step;
        
        // Your training code here
        double train_loss = /* ... */;
        double accuracy = /* ... */;
        
        // Log metrics
        writer.addTrainingLoss(train_loss, global_step);
        writer.addScalar("accuracy", accuracy, global_step);
    }
    
    // Validation phase
    double val_loss = /* ... */;
    writer.addValidationLoss(val_loss, (epoch + 1) * steps_per_epoch);
}
```

## Output Format

The metrics are saved in CSV format with the following columns:
- Timestamp (ISO format)
- Step
- Metrics (as specified in construction)

Example output:
```csv
Timestamp,Step,train_loss,val_loss,accuracy,learning_rate
2025-02-25T10:15:30,1,0.500000,NA,0.850000,0.001000
2025-02-25T10:15:31,2,0.450000,NA,0.860000,0.001000
2025-02-25T10:15:32,10,0.400000,0.550000,0.870000,0.001000
```

## Building

```bash
# In your project directory
mkdir build && cd build
cmake ..
make
```

## Requirements

- C++17 or later
- CMake 3.14 or later
- pthread support