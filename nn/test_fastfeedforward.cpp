#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "nn.hpp"
#include "fastfeedforward.hpp"

class FastFeedforwardTest : public ::testing::Test {
protected:

    FastFeedforwardTest() : 
        x(nn::DataTypeToEnum<float>::value, 2 * 4),
        y(nn::DataTypeToEnum<float>::value, 2 * 4),
        grad_y(nn::DataTypeToEnum<float>::value, 2 * 4),
        grad_x(nn::DataTypeToEnum<float>::value, 2 * 4) {}
    
    void SetUp() override {
        // Setup network with embed_dim = 4
        network = std::make_unique<gpt::FastFeedforwardNetwork>(4, 4);
        
        // Fill x with some test data
        auto x_span = x.span<float>();
        for(int i = 0; i < 8; i++) {
            x_span[i] = i * 0.1f;
        }
    }

    std::unique_ptr<gpt::FastFeedforwardNetwork> network;
    nn::Parameter x;
    nn::Parameter y;
    nn::Parameter grad_y;
    nn::Parameter grad_x;
};

TEST_F(FastFeedforwardTest, ForwardPassDimensions) {
    // Test forward pass dimensions
    network->Forward(x.const_matrix<float>(2, 4), y.matrix<float>(2, 4));
    
    // Verify output dimensions
    EXPECT_EQ(y.size(), 8);  // batch_size * output_size = 2 * 4
}

TEST_F(FastFeedforwardTest, ForwardPassValues) {
    // Forward pass
    network->Forward(x.const_matrix<float>(2, 4), y.matrix<float>(2, 4));
    
    // Check that outputs are within reasonable range for initialized network
    // (Just a basic sanity check since exact values depend on weight initialization)
    auto y_span = y.span<float>();
    for(int i = 0; i < 8; i++) {
        // Values should be finite
        EXPECT_TRUE(std::isfinite(y_span[i])) << "Output at position " << i << " is not finite";
    }
}

TEST_F(FastFeedforwardTest, BackwardPassDimensions) {
    // First do forward pass
    network->Forward(x.const_matrix<float>(2, 4), y.matrix<float>(2, 4));
    
    // Set gradient
    auto grad_y_span = grad_y.span<float>();
    for(int i = 0; i < 8; i++) {
        grad_y_span[i] = 0.1f;
    }
    
    // Then backward pass
    network->Backward(x.const_matrix<float>(2, 4),
                                        grad_y.const_matrix<float>(2, 4),
                                        grad_x.matrix<float>(2, 4));
    
    // Verify grad_x dimensions
    EXPECT_EQ(grad_x.size(), 8);  // batch_size * embed_dim = 2 * 4
}

TEST_F(FastFeedforwardTest, BackwardPassFloat) {
    // Forward pass
    network->Forward(x.const_matrix<float>(2, 4), y.matrix<float>(2, 4));

    // Set gradient
    auto grad_y_span = grad_y.span<float>();
    for(int i = 0; i < 8; i++) {
        grad_y_span[i] = 0.1f;
    }
    
    // And for the backward pass
    network->Backward(x.const_matrix<float>(2, 4),
                                        grad_y.const_matrix<float>(2, 4),
                                        grad_x.matrix<float>(2, 4));
    
    // Verify grad_x has valid values
    auto grad_x_span = grad_x.span<float>();
    for(int i = 0; i < 8; i++) {
        EXPECT_TRUE(std::isfinite(grad_x_span[i])) 
                << "Gradient at position " << i << " is not finite";
    }
}

TEST_F(FastFeedforwardTest, ParameterCount) {
    gpt::FastFeedforwardNetwork network(4, 4);  // embed_dim = 4, output_size = 4
    
    // Expected parameters:
    // Decision FC: 4*1 weights + 1 bias = 5
    // Left FC: 4*4 weights + 4 bias = 20
    // Right FC: 4*4 weights + 4 bias = 20
    // Total: 45 parameters
    EXPECT_EQ(network.NumParameters(), 45);
}

TEST_F(FastFeedforwardTest, ActivationCount) {
    // Setup with specific test dimensions
    int embed_dim = 4;
    int batch_size = 2;
    int output_size = 4;
    
    gpt::FastFeedforwardNetwork network(embed_dim, output_size);
    
    // Do forward pass to allocate activations
    nn::Parameter x(nn::DT_FLOAT, batch_size * embed_dim);
    nn::Parameter y(nn::DT_FLOAT, batch_size * output_size);
    network.Forward(x.const_matrix<float>(batch_size, embed_dim), 
                                 y.matrix<float>(batch_size, output_size));
    
    // Expected activations:
    // Decision node activations: 2*batch_size + decision node internal
    // Left leaf activations: batch_size*output_size + leaf internal
    // Right leaf activations: batch_size*output_size + leaf internal
    // Choice tensor: batch_size
    // Left output tensor: batch_size*output_size
    // Right output tensor: batch_size*output_size
    
    // We can't easily test the exact number without knowing implementation details
    // of the Linear layer, but we can check it's positive
    EXPECT_GT(network.NumActivations(), 0);
}

TEST_F(FastFeedforwardTest, RouteInterpolation) {
    // Create a custom network with predetermined decision weights
    gpt::FastFeedforwardNetwork network(4, 4);
    
    // Get access to parameters
    std::vector<nn::Parameter*> params;
    network.Parameters(&params);
    
    // Set decision weights to all zeros and bias to 0
    // This should make sigmoid(0) = 0.5, causing equal routing
    auto decision_weights = params[0]->matrix<float>(4, 1);
    auto decision_bias = params[1]->flat<float>();
    
    for(int i = 0; i < 4; i++) {
        decision_weights(i, 0) = 0.0f;
    }
    decision_bias(0) = 0.0f;
    
    // Set left and right leaf to output different constant values
    // Left leaf: all 1.0
    auto left_weights = params[2]->matrix<float>(4, 4);
    auto left_bias = params[3]->flat<float>();
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            left_weights(i, j) = 0.0f;
        }
        left_bias(i) = 1.0f;
    }
    
    // Right leaf: all 2.0
    auto right_weights = params[4]->matrix<float>(4, 4);
    auto right_bias = params[5]->flat<float>();
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            right_weights(i, j) = 0.0f;
        }
        right_bias(i) = 2.0f;
    }
    
    // Test input
    nn::Parameter test_x(nn::DT_FLOAT, 1 * 4);
    nn::Parameter test_y(nn::DT_FLOAT, 1 * 4);
    auto test_x_span = test_x.span<float>();
    for(int i = 0; i < 4; i++) {
        test_x_span[i] = 0.0f;  // All zeros input
    }
    
    // Forward pass
    network.Forward(test_x.const_matrix<float>(1, 4), test_y.matrix<float>(1, 4));
    
    // With sigmoid(0) = 0.5, we should get outputs that are 50% blend of 1.0 and 2.0
    auto test_y_span = test_y.span<float>();
    for(int i = 0; i < 4; i++) {
        EXPECT_NEAR(test_y_span[i], 1.5f, 1e-5f) 
                << "Output should be ~1.5 (50% blend of 1.0 and 2.0)";
    }
}

TEST_F(FastFeedforwardTest, GradientConsistency) {
    // Create a small network for testing
    gpt::FastFeedforwardNetwork network(4, 4);
    
    // Prepare input and output
    nn::Parameter x(nn::DT_FLOAT, 2 * 4);
    nn::Parameter y(nn::DT_FLOAT, 2 * 4);
    nn::Parameter grad_y(nn::DT_FLOAT, 2 * 4);
    nn::Parameter grad_x(nn::DT_FLOAT, 2 * 4);
    
    // Fill with test data
    auto x_span = x.span<float>();
    auto grad_y_span = grad_y.span<float>();
    for(int i = 0; i < 8; i++) {
        x_span[i] = i * 0.1f;
        grad_y_span[i] = 1.0f;
    }
    
    // Forward and backward pass
    network.Forward(x.const_matrix<float>(2, 4), y.matrix<float>(2, 4));
    network.Backward(x.const_matrix<float>(2, 4),
                                    grad_y.const_matrix<float>(2, 4),
                                    grad_x.matrix<float>(2, 4));
    
    // Store the first gradients
    std::vector<float> first_grads(8);
    auto grad_x_span = grad_x.span<float>();
    for(int i = 0; i < 8; i++) {
        first_grads[i] = grad_x_span[i];
    }
    
    // Do another backward pass with the same gradients
    grad_x.ZeroData();
    network.Backward(x.const_matrix<float>(2, 4),
                                    grad_y.const_matrix<float>(2, 4),
                                    grad_x.matrix<float>(2, 4));
    
    // Compare - gradients should be consistent between calls
    grad_x_span = grad_x.span<float>();
    for(int i = 0; i < 8; i++) {
        EXPECT_NEAR(grad_x_span[i], first_grads[i], 1e-5f) 
                << "Gradient not consistent between backward calls at index " << i;
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}