#ifndef PROFILE_TRACE_FN
#define PROFILE_TRACE_FN(name)
#endif

#include <gtest/gtest.h>
#include "./fastfeedforward.hpp"
#include "./Parameter.hpp"
#include <vector>
#include <cmath>

using namespace nn;
using namespace gpt;

class FastFeedforwardTest : public ::testing::Test {
protected:
    void SetUp() override {
    }
};

TEST_F(FastFeedforwardTest, ForwardPassFloat) {
    // Setup parameters
    FastFeedforward network(4);  // embed_dim = 4
    
    // Create input tensor
    Parameter x(DT_FLOAT, 8);  // batch_size * embed_dim = 2 * 4 = 8
    Parameter y(DT_FLOAT, 8);
    
    // Set input values
    auto x_span = x.span<float>();
    for(int i = 0; i < 8; i++) {
        x_span[i] = i * 0.1f;
    }
    
    // Forward pass
    network.Forward(x.const_matrix<float>(2, 4),  // (batch_size, embed_dim)
                   y.matrix<float>(2, 4));  
    
    // Verify output size
    EXPECT_EQ(y.size(), 8);  // batch_size * embed_dim
}

TEST_F(FastFeedforwardTest, BackwardPassFloat) {
    // Setup network
    FastFeedforward network(4);  // embed_dim = 4
    
    // Create tensors
    Parameter x(DT_FLOAT, 8);
    Parameter y(DT_FLOAT, 8);
    Parameter grad_y(DT_FLOAT, 8);
    Parameter grad_x(DT_FLOAT, 8);
    
    // Initialize input
    auto x_span = x.span<float>();
    for(int i = 0; i < 8; i++) {
        x_span[i] = i * 0.1f;
    }
    
   // In BackwardPassFloat test
    network.Forward(x.const_matrix<float>(2, 4), y.matrix<float>(2, 4));

    // Set gradient
    auto grad_y_span = grad_y.span<float>();
    for(int i = 0; i < 8; i++) {
        grad_y_span[i] = 0.1f;
    }
    
    // And for the backward pass
    network.Backward(x.const_matrix<float>(2, 4),
    grad_y.const_matrix<float>(2, 4),
    grad_x.matrix<float>(2, 4));
    
    // Verify grad_x dimensions
    EXPECT_EQ(grad_x.size(), 8);  // batch_size * embed_dim = 2 * 4

}

TEST_F(FastFeedforwardTest, ParameterCount) {
    FastFeedforward network(4);  // embed_dim = 4
    
    // Expected parameters:
    // Decision FC: 4*1 weights + 1 bias = 5
    // Left FC: 4*4 weights + 4 bias = 20
    // Right FC: 4*4 weights + 4 bias = 20
    // Total: 45
    EXPECT_EQ(network.NumParameters(), 45);
}

TEST_F(FastFeedforwardTest, ActivationCount) {
    FastFeedforward network(4);  // embed_dim = 4
    
    // Forward pass to allocate activations
    Parameter x(DT_FLOAT, 8);  // batch_size * embed_dim = 2 * 4 = 8
    Parameter y(DT_FLOAT, 8);
    
    // Forward pass with matrix reshape
    network.Forward(x.const_matrix<float>(2, 4),  // (batch_size, embed_dim)
                   y.matrix<float>(2, 4));
    
    // Verify activation count is non-zero
    EXPECT_GT(network.NumActivations(), 0);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}