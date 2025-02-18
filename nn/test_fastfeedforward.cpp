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

TEST_F(FastFeedforwardTest, Training) {
    // Setup network
    FastFeedforward network(4);  // embed_dim = 4
    
    // Create training data
    Parameter x(DT_FLOAT, 8);  // batch_size * embed_dim = 2 * 4
    Parameter y_target(DT_FLOAT, 8);
    Parameter y_pred(DT_FLOAT, 8);
    Parameter grad_y(DT_FLOAT, 8);
    Parameter grad_x(DT_FLOAT, 8);
    
    // Set input values
    auto x_span = x.span<float>();
    for(int i = 0; i < 8; i++) {
        x_span[i] = i * 0.1f;  // Simple increasing sequence
    }
    
    // Set target values (let's say we want to learn a simple transformation)
    auto y_target_span = y_target.span<float>();
    for(int i = 0; i < 8; i++) {
        y_target_span[i] = x_span[i] * 2.0f;  // Target is 2x the input
    }
    
    // Training loop
    const float learning_rate = 0.01f;
    const int num_epochs = 100;
    
    std::vector<nn::Parameter*> params;
    network.Parameters(&params);
    
    for(int epoch = 0; epoch < num_epochs; epoch++) {
        // Forward pass
        network.Forward(x.const_matrix<float>(2, 4),
                       y_pred.matrix<float>(2, 4));
        
        // Compute MSE loss gradient
        auto y_pred_span = y_pred.span<float>();
        auto grad_y_span = grad_y.span<float>();
        float loss = 0.0f;
        for(int i = 0; i < 8; i++) {
            float diff = y_pred_span[i] - y_target_span[i];
            loss += diff * diff;
            grad_y_span[i] = 2.0f * diff;  // MSE loss gradient
        }
        loss /= 8.0f;
        
        // Backward pass
        network.Backward(x.const_matrix<float>(2, 4),
                        grad_y.const_matrix<float>(2, 4),
                        grad_x.matrix<float>(2, 4));
        
        // Update parameters
        for(auto param : params) {
            auto param_data = param->span<float>();
            auto param_grad = param->span_grad<float>();
            for(int i = 0; i < param->size(); i++) {
                param_data[i] -= learning_rate * param_grad[i];
            }
        }
        
        // Print progress every 10 epochs
        if(epoch % 10 == 0) {
            printf("Epoch %d, Loss: %f\n", epoch, loss);
        }
    }
    
    // Verify final predictions are close to targets
    network.Forward(x.const_matrix<float>(2, 4),
                   y_pred.matrix<float>(2, 4));
                   
    auto final_pred_span = y_pred.span<float>();
    for(int i = 0; i < 8; i++) {
        EXPECT_NEAR(final_pred_span[i], y_target_span[i], 0.1f);
    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}