#include <iostream>
#include <random>
#include <vector>
#include "../nn/fastfeedforward.hpp"
#include "../optimizer/optim.hpp"

// A simple synthetic training task: XOR problem with more dimensions
// This is a classic non-linear problem that requires hidden layers to solve
void create_xor_dataset(int num_samples, int input_dim, int output_dim,
                        std::vector<std::vector<float>>& inputs,
                        std::vector<std::vector<float>>& targets) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  
  inputs.resize(num_samples);
  targets.resize(num_samples);
  
  for (int i = 0; i < num_samples; ++i) {
    // Create input vector
    inputs[i].resize(input_dim);
    for (int j = 0; j < input_dim; ++j) {
      inputs[i][j] = dist(gen);
    }
    
    // Compute XOR on first 2 dimensions, then extend pattern
    bool xor_result = (inputs[i][0] > 0.0f) != (inputs[i][1] > 0.0f);
    
    // Create target vector
    targets[i].resize(output_dim);
    for (int j = 0; j < output_dim; ++j) {
      // First output is XOR of inputs, others are variations
      if (j == 0) {
        targets[i][j] = xor_result ? 1.0f : -1.0f;
      } else {
        // Add some pattern variation for multi-dimensional output
        targets[i][j] = xor_result ? (j % 2 == 0 ? 1.0f : -1.0f) : (j % 2 == 0 ? -1.0f : 1.0f);
      }
    }
  }
}

// Calculate MSE loss
float calculate_loss(const std::vector<std::vector<float>>& outputs,
                    const std::vector<std::vector<float>>& targets) {
  float total_loss = 0.0f;
  int num_samples = outputs.size();
  int output_dim = outputs[0].size();
  
  for (int i = 0; i < num_samples; ++i) {
    for (int j = 0; j < output_dim; ++j) {
      float diff = outputs[i][j] - targets[i][j];
      total_loss += diff * diff;
    }
  }
  
  return total_loss / (num_samples * output_dim);
}

int main() {
  // Parameters
  const int input_dim = 4;
  const int hidden_dim = 16;
  const int output_dim = 2;
  const int depth = 2;  // Use depth=2 (4 leaves) for more capacity
  const int num_train_samples = 1000;
  const int num_test_samples = 200;
  const int batch_size = 32;
  const int epochs = 100;
  const float learning_rate = 0.01f;
  
  std::cout << "Creating FastFeedforward network with:"
            << "\n - Input dimension: " << input_dim
            << "\n - Hidden dimension: " << hidden_dim
            << "\n - Output dimension: " << output_dim
            << "\n - Tree depth: " << depth
            << std::endl;
  
  // Create FFN model
  gpt::FastFeedforwardNetwork model(input_dim, hidden_dim, output_dim, depth);
  
  // Create Adam optimizer
  std::vector<nn::Parameter*> params;
  model.Parameters(&params);
  optim::AdamW optimizer(params, learning_rate); 
  
  // Create synthetic dataset
  std::vector<std::vector<float>> train_inputs, train_targets;
  std::vector<std::vector<float>> test_inputs, test_targets;
  
  std::cout << "Creating training dataset with " << num_train_samples << " samples..." << std::endl;
  create_xor_dataset(num_train_samples, input_dim, output_dim, train_inputs, train_targets);
  
  std::cout << "Creating test dataset with " << num_test_samples << " samples..." << std::endl;
  create_xor_dataset(num_test_samples, input_dim, output_dim, test_inputs, test_targets);
  
  // Buffer for batch processing
  nn::Parameter x_data(nn::DataTypeToEnum<float>::value, batch_size * input_dim);
  nn::Parameter y_data(nn::DataTypeToEnum<float>::value, batch_size * output_dim);
  nn::Parameter y_pred(nn::DataTypeToEnum<float>::value, batch_size * output_dim);
  nn::Parameter y_grad(nn::DataTypeToEnum<float>::value, batch_size * output_dim);
  nn::Parameter x_grad(nn::DataTypeToEnum<float>::value, batch_size * input_dim);
  
  // Training loop
  std::cout << "Starting training for " << epochs << " epochs..." << std::endl;
  for (int epoch = 0; epoch < epochs; ++epoch) {
    float epoch_loss = 0.0f;
    int num_batches = (num_train_samples + batch_size - 1) / batch_size;
    
    for (int batch = 0; batch < num_batches; ++batch) {
      int start_idx = batch * batch_size;
      int end_idx = std::min(start_idx + batch_size, num_train_samples);
      int current_batch_size = end_idx - start_idx;

      // Create properly sized parameter buffers for this batch
    nn::Parameter x_data(nn::DataTypeToEnum<float>::value, current_batch_size * input_dim);
    nn::Parameter y_data(nn::DataTypeToEnum<float>::value, current_batch_size * output_dim);
    nn::Parameter y_pred(nn::DataTypeToEnum<float>::value, current_batch_size * output_dim);
    nn::Parameter y_grad(nn::DataTypeToEnum<float>::value, current_batch_size * output_dim);
    nn::Parameter x_grad(nn::DataTypeToEnum<float>::value, current_batch_size * input_dim);
          
      // Prepare input and target tensors
      auto x_batch = x_data.matrix<float>(current_batch_size, input_dim);
      auto y_batch = y_data.matrix<float>(current_batch_size, output_dim);
      
      for (int i = 0; i < current_batch_size; ++i) {
        int sample_idx = start_idx + i;
        for (int j = 0; j < input_dim; ++j) {
          x_batch(i, j) = train_inputs[sample_idx][j];
        }
        for (int j = 0; j < output_dim; ++j) {
          y_batch(i, j) = train_targets[sample_idx][j];
        }
      }
      
      // Forward pass
      auto x_const = x_data.const_matrix<float>(current_batch_size, input_dim);
      auto y_output = y_pred.matrix<float>(current_batch_size, output_dim);
      model.Forward(x_const, y_output, true);  // Training mode
      
      // Compute loss and gradient
      auto y_target = y_data.const_matrix<float>(current_batch_size, output_dim);
      auto grad = y_grad.matrix<float>(current_batch_size, output_dim);
      auto x_grad_mat = x_grad.matrix<float>(current_batch_size, input_dim);
      
      float batch_loss = 0.0f;
      for (int i = 0; i < current_batch_size; ++i) {
        for (int j = 0; j < output_dim; ++j) {
          float diff = y_output(i, j) - y_target(i, j);
          batch_loss += diff * diff;
          grad(i, j) = 2.0f * diff / (current_batch_size * output_dim);  // MSE gradient
        }
      }
      batch_loss /= (current_batch_size * output_dim);
      epoch_loss += batch_loss;
      
      // Backward pass
      model.Backward(x_const, y_grad.const_matrix<float>(current_batch_size, output_dim), x_grad_mat);
      
      // Update parameters
      optimizer.Step(epoch * num_batches + batch + 1, learning_rate);
        optimizer.ZeroGrad();
    }
    
    epoch_loss /= num_batches;
    
    // Evaluate on test set every 10 epochs
    if (epoch % 10 == 0 || epoch == epochs - 1) {
      std::vector<std::vector<float>> test_outputs(num_test_samples, std::vector<float>(output_dim));
      
      // Process test data in batches
      for (int batch = 0; batch < (num_test_samples + batch_size - 1) / batch_size; ++batch) {
        int start_idx = batch * batch_size;
        int end_idx = std::min(start_idx + batch_size, num_test_samples);
        int current_batch_size = end_idx - start_idx;

        // Create properly sized buffers for this test batch
        nn::Parameter x_data(nn::DataTypeToEnum<float>::value, current_batch_size * input_dim);
        nn::Parameter y_pred(nn::DataTypeToEnum<float>::value, current_batch_size * output_dim);
        
        auto x_test = x_data.matrix<float>(current_batch_size, input_dim);
        auto y_out = y_pred.matrix<float>(current_batch_size, output_dim);
        
        for (int i = 0; i < current_batch_size; ++i) {
          int sample_idx = start_idx + i;
          for (int j = 0; j < input_dim; ++j) {
            x_test(i, j) = test_inputs[sample_idx][j];
          }
        }
        
        // Forward pass in evaluation mode
        model.Forward(x_data.const_matrix<float>(current_batch_size, input_dim), 
                     y_out, false);  // Eval mode
        
        // Copy outputs
        for (int i = 0; i < current_batch_size; ++i) {
          int sample_idx = start_idx + i;
          for (int j = 0; j < output_dim; ++j) {
            test_outputs[sample_idx][j] = y_out(i, j);
          }
        }
      }
      
      float test_loss = calculate_loss(test_outputs, test_targets);
      std::cout << "Epoch " << epoch << ", Train Loss: " << epoch_loss
                << ", Test Loss: " << test_loss << std::endl;
    } else {
      std::cout << "Epoch " << epoch << ", Train Loss: " << epoch_loss << std::endl;
    }
  }
  
  std::cout << "Training completed successfully!" << std::endl;
  return 0;
}