#include <gtest/gtest.h>
#include "./fastfeedforward.hpp"
#include <cmath>

class FastFeedforwardTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup code
    }
};

// Test LeafNetwork
TEST_F(FastFeedforwardTest, LeafNetworkForward) {
    LeafNetwork leaf(2, 3);  // 2 inputs, 3 outputs
    
    // Set specific weights and biases for deterministic testing
    leaf.weights = {{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}};
    leaf.biases = {0.1, 0.2, 0.3};
    
    std::vector<double> input = {1.0, 2.0};
    auto output = leaf.forward(input);
    
    // Expected values calculated manually
    EXPECT_NEAR(output[0], 0.1 * 1.0 + 0.2 * 2.0 + 0.1, 1e-6);
    EXPECT_NEAR(output[1], 0.3 * 1.0 + 0.4 * 2.0 + 0.2, 1e-6);
    EXPECT_NEAR(output[2], 0.5 * 1.0 + 0.6 * 2.0 + 0.3, 1e-6);
}

// Test DecisionNode
TEST_F(FastFeedforwardTest, DecisionNodeForward) {
    DecisionNode node(2);
    
    // Set specific weights and bias
    node.weights = {0.1, 0.2};
    node.bias = 0.1;
    
    std::vector<double> input = {1.0, 2.0};
    double output = node.forward(input);
    
    // Expected sigmoid(0.1 * 1.0 + 0.2 * 2.0 + 0.1)
    double expected = sigmoid(0.1 + 0.4 + 0.1);
    EXPECT_NEAR(output, expected, 1e-6);
}

// Test FastFeedforwardNetwork
TEST_F(FastFeedforwardTest, NetworkForward) {
    FastFeedforwardNetwork network(2, 2);  // 2 inputs, 2 outputs
    
    // Set deterministic weights
    network.decision.weights = {0.1, 0.2};
    network.decision.bias = 0.1;
    
    network.left_leaf.weights = {{0.1, 0.2}, {0.3, 0.4}};
    network.left_leaf.biases = {0.1, 0.2};
    
    network.right_leaf.weights = {{0.5, 0.6}, {0.7, 0.8}};
    network.right_leaf.biases = {0.3, 0.4};
    
    std::vector<double> input = {1.0, 2.0};
    auto output = network.forward(input);
    
    // Test output size
    EXPECT_EQ(output.size(), 2);
    
    // Test output values (calculated manually)
    double decision_output = sigmoid(0.1 * 1.0 + 0.2 * 2.0 + 0.1);
    
    std::vector<double> left_expected = {
        0.1 * 1.0 + 0.2 * 2.0 + 0.1,
        0.3 * 1.0 + 0.4 * 2.0 + 0.2
    };
    
    std::vector<double> right_expected = {
        0.5 * 1.0 + 0.6 * 2.0 + 0.3,
        0.7 * 1.0 + 0.8 * 2.0 + 0.4
    };
    
    for (size_t i = 0; i < output.size(); i++) {
        double expected = decision_output * right_expected[i] + 
                        (1 - decision_output) * left_expected[i];
        EXPECT_NEAR(output[i], expected, 1e-6);
    }
}

// Test backward pass
TEST_F(FastFeedforwardTest, NetworkBackward) {
    FastFeedforwardNetwork network(2, 2);
    std::vector<double> input = {1.0, 2.0};
    
    // Forward pass
    auto output = network.forward(input);
    
    // Create gradient
    std::vector<double> grad_output = {0.1, 0.2};
    double learning_rate = 0.01;
    
    // Store initial weights
    auto initial_decision_weights = network.decision.weights;
    auto initial_left_weights = network.left_leaf.weights;
    auto initial_right_weights = network.right_leaf.weights;
    
    // Backward pass
    network.backward(grad_output, learning_rate);
    
    // Verify weights changed
    EXPECT_NE(network.decision.weights[0], initial_decision_weights[0]);
    EXPECT_NE(network.left_leaf.weights[0][0], initial_left_weights[0][0]);
    EXPECT_NE(network.right_leaf.weights[0][0], initial_right_weights[0][0]);
}

// Test parameter count
TEST_F(FastFeedforwardTest, ParameterCount) {
    FastFeedforwardNetwork network(2, 3);  // 2 inputs, 3 outputs
    
    // Expected parameters:
    // Decision node: 2 weights + 1 bias = 3
    // Left leaf: (2 * 3) weights + 3 biases = 9
    // Right leaf: (2 * 3) weights + 3 biases = 9
    // Total: 21
    EXPECT_EQ(network.getParameterCount(), 21);
}