#include <gtest/gtest.h>
#include "../nn/fastfeedforward.hpp"
#include <random>
#include <cmath>

namespace gpt {
namespace testing {

class DecisionNodeTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Default setup for tests
    input_size_ = 4;
    batch_size_ = 2;
    
    // Create random input
    std::default_random_engine generator(42); // Fixed seed for reproducibility
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    
    input_data_ = std::make_unique<nn::Parameter>(nn::DataTypeToEnum<float>::value, 
                                                batch_size_ * input_size_);
    auto input = input_data_->matrix<float>(batch_size_, input_size_);
    
    for (int b = 0; b < batch_size_; ++b) {
      for (int i = 0; i < input_size_; ++i) {
        input(b, i) = distribution(generator);
      }
    }
    
    // Create output buffer
    output_data_ = std::make_unique<nn::Parameter>(nn::DataTypeToEnum<float>::value, 
                                                 batch_size_);
    
    // Create gradient buffers
    output_grad_ = std::make_unique<nn::Parameter>(nn::DataTypeToEnum<float>::value, 
                                                 batch_size_);
    input_grad_ = std::make_unique<nn::Parameter>(nn::DataTypeToEnum<float>::value, 
                                                batch_size_ * input_size_);
    
    // Random gradients
    auto grad = output_grad_->matrix<float>(batch_size_, 1);
    for (int b = 0; b < batch_size_; ++b) {
      grad(b, 0) = distribution(generator);
    }
  }
  
  DecisionNode CreateDecisionNode(int node_index = 0) {
    return DecisionNode(input_size_, node_index);
  }
  
  int input_size_;
  int batch_size_;
  std::unique_ptr<nn::Parameter> input_data_;
  std::unique_ptr<nn::Parameter> output_data_;
  std::unique_ptr<nn::Parameter> output_grad_;
  std::unique_ptr<nn::Parameter> input_grad_;
};

TEST_F(DecisionNodeTest, ForwardOutputsInRange) {
  // Test that the sigmoid output is always between 0 and 1
  auto node = CreateDecisionNode();
  
  auto input = input_data_->const_matrix<float>(batch_size_, input_size_);
  auto output = output_data_->matrix<float>(batch_size_, 1);
  
  node.Forward(input, output);
  
  for (int b = 0; b < batch_size_; ++b) {
    EXPECT_GE(output(b, 0), 0.0f) << "Output should be >= 0";
    EXPECT_LE(output(b, 0), 1.0f) << "Output should be <= 1";
  }
}

TEST_F(DecisionNodeTest, NodeIndexPreserved) {
  // Test that the node_index_ is correctly set
  const int test_index = 42;
  auto node = CreateDecisionNode(test_index);
  EXPECT_EQ(node.node_index_, test_index);
}

TEST_F(DecisionNodeTest, BackwardProducesGradient) {
  // Test that backward pass produces non-zero gradients
  auto node = CreateDecisionNode();
  
  auto input = input_data_->const_matrix<float>(batch_size_, input_size_);
  auto output = output_data_->matrix<float>(batch_size_, 1);
  auto output_grad = output_grad_->const_matrix<float>(batch_size_, 1);
  auto input_grad = input_grad_->matrix<float>(batch_size_, input_size_);
  
  // Initialize input gradients to zero
  for (int b = 0; b < batch_size_; ++b) {
    for (int i = 0; i < input_size_; ++i) {
      input_grad(b, i) = 0.0f;
    }
  }
  
  // Forward pass
  node.Forward(input, output);
  
  // Backward pass
  node.Backward(input, output_grad, input_grad);
  
  // Check that at least some gradients are non-zero
  bool has_nonzero_grad = false;
  for (int b = 0; b < batch_size_; ++b) {
    for (int i = 0; i < input_size_; ++i) {
      if (std::abs(input_grad(b, i)) > 1e-6) {
        has_nonzero_grad = true;
        break;
      }
    }
    if (has_nonzero_grad) break;
  }
  
  EXPECT_TRUE(has_nonzero_grad) << "Backward pass should produce non-zero gradients";
}

TEST_F(DecisionNodeTest, ForwardOutputConsistency) {
  // Test that the output is consistent across multiple Forward calls
  auto node = CreateDecisionNode();
  
  auto input = input_data_->const_matrix<float>(batch_size_, input_size_);
  auto output1 = output_data_->matrix<float>(batch_size_, 1);
  
  // First forward pass
  node.Forward(input, output1);
  
  // Create a second output buffer
  nn::Parameter output_data2(nn::DataTypeToEnum<float>::value, batch_size_);
  auto output2 = output_data2.matrix<float>(batch_size_, 1);
  
  // Second forward pass
  node.Forward(input, output2);
  
  // Compare outputs
  for (int b = 0; b < batch_size_; ++b) {
    EXPECT_NEAR(output1(b, 0), output2(b, 0), 1e-6) 
        << "Forward pass should be deterministic";
  }
}

TEST_F(DecisionNodeTest, ZeroGradInput) {
  // Test that zero gradient input produces zero gradient output
  auto node = CreateDecisionNode();
  
  auto input = input_data_->const_matrix<float>(batch_size_, input_size_);
  auto output = output_data_->matrix<float>(batch_size_, 1);
  auto input_grad = input_grad_->matrix<float>(batch_size_, input_size_);
  
  // Initialize input gradients to zero
  for (int b = 0; b < batch_size_; ++b) {
    for (int i = 0; i < input_size_; ++i) {
      input_grad(b, i) = 0.0f;
    }
  }
  
  // Forward pass
  node.Forward(input, output);
  
  // Create zero gradient input
  nn::Parameter zero_grad(nn::DataTypeToEnum<float>::value, batch_size_);
  auto zero_grad_matrix = zero_grad.matrix<float>(batch_size_, 1);
  for (int b = 0; b < batch_size_; ++b) {
    zero_grad_matrix(b, 0) = 0.0f;
  }
  
  // Backward pass with zero gradients
  node.Backward(input, zero_grad.const_matrix<float>(batch_size_, 1), input_grad);
  
  // Check that all output gradients are zero
  for (int b = 0; b < batch_size_; ++b) {
    for (int i = 0; i < input_size_; ++i) {
      EXPECT_NEAR(input_grad(b, i), 0.0f, 1e-6) 
          << "Zero gradient input should produce zero gradient output";
    }
  }
}

TEST_F(DecisionNodeTest, NumParameters) {
  // Test that NumParameters returns the correct count
  auto node = CreateDecisionNode();
  
  // A linear layer with input_size inputs and 1 output has input_size + 1 parameters
  // (weights + bias)
  size_t expected_num_params = input_size_ + 1;
  
  EXPECT_EQ(node.NumParameters(), expected_num_params);
}

TEST_F(DecisionNodeTest, NumActivations) {
  // Test that NumActivations returns the correct count
  auto node = CreateDecisionNode();
  
  // Forward pass to allocate activations
  auto input = input_data_->const_matrix<float>(batch_size_, input_size_);
  auto output = output_data_->matrix<float>(batch_size_, 1);
  node.Forward(input, output);
  
  // Expected activations:
  // 1. Linear layer activations
  // 2. decision_output_ (batch_size * 1)
  // 3. sigmoid_output_ (batch_size * 1)
  size_t expected_activations = node.decision_->NumActivations() + 
                               batch_size_ + 
                               batch_size_;
  
  EXPECT_EQ(node.NumActivations(), expected_activations);
}

TEST_F(DecisionNodeTest, ParametersCollection) {
  // Test that Parameters collects all parameters
  auto node = CreateDecisionNode();
  
  std::vector<nn::Parameter*> params;
  node.Parameters(&params);
  
  // Should collect all parameters from the linear layer
  EXPECT_EQ(params.size(), 2); // Weights and bias
}

TEST_F(DecisionNodeTest, GradientCheckSigmoid) {
  // Simple numerical gradient check
  auto node = CreateDecisionNode();
  
  // Use a small test input
  nn::Parameter small_input(nn::DataTypeToEnum<float>::value, 4);
  auto x = small_input.matrix<float>(1, 4);
  x(0, 0) = 0.1f; x(0, 1) = -0.2f; x(0, 2) = 0.3f; x(0, 3) = -0.4f;
  
  // Initialize parameters for gradient check
  auto* weight = node.decision_->weight();
  auto w = weight->matrix<float>(4, 1);
  w(0, 0) = 0.5f; w(1, 0) = -0.5f; w(2, 0) = 0.5f; w(3, 0) = -0.5f;
  
  auto* bias = node.decision_->bias();
  auto b = bias->matrix<float>(1, 1);
  b(0, 0) = 0.1f;
  
  // Forward pass
  nn::Parameter output(nn::DataTypeToEnum<float>::value, 1);
  auto y = output.matrix<float>(1, 1);
  node.Forward(small_input.const_matrix<float>(1, 4), y);
  
  // Create output gradient (1.0)
  nn::Parameter y_grad(nn::DataTypeToEnum<float>::value, 1);
  auto dy = y_grad.matrix<float>(1, 1);
  dy(0, 0) = 1.0f;
  
  // Backward pass
  nn::Parameter x_grad(nn::DataTypeToEnum<float>::value, 4);
  auto dx = x_grad.matrix<float>(1, 4);
  for (int i = 0; i < 4; ++i) dx(0, i) = 0.0f;
  
  node.Backward(small_input.const_matrix<float>(1, 4), y_grad.const_matrix<float>(1, 1), dx);
  
  // For each input dimension, calculate numerical gradient
  const float epsilon = 1e-4f;
  for (int i = 0; i < 4; ++i) {
    // Slightly increase input
    float orig_val = x(0, i);
    x(0, i) = orig_val + epsilon;
    
    nn::Parameter output_plus(nn::DataTypeToEnum<float>::value, 1);
    auto y_plus = output_plus.matrix<float>(1, 1);
    node.Forward(small_input.const_matrix<float>(1, 4), y_plus);
    
    // Slightly decrease input
    x(0, i) = orig_val - epsilon;
    
    nn::Parameter output_minus(nn::DataTypeToEnum<float>::value, 1);
    auto y_minus = output_minus.matrix<float>(1, 1);
    node.Forward(small_input.const_matrix<float>(1, 4), y_minus);
    
    // Restore original value
    x(0, i) = orig_val;
    
    // Calculate numerical gradient
    float numerical_grad = (y_plus(0, 0) - y_minus(0, 0)) / (2 * epsilon);
    
    // Compare with analytical gradient
    EXPECT_NEAR(dx(0, i), numerical_grad, 1e-2) 
        << "Gradient check failed for input dimension " << i;
  }
}

}  // namespace testing
}  // namespace gpt

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}