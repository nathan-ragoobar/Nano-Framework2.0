#include "AttentionLayer.hpp"
#include <gtest/gtest.h>
#include "./../tensor/fixed_point.hpp" // Include the fixed_point header if needed

constexpr float EPSILON = 1e-4;

class CausalSelfAttentionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }
};

TEST_F(CausalSelfAttentionTest, Forward) {
    using Type = fixed_point_7pt8;
    int block_size = 4;
    int n_head = 2;
    int n_embed = 8;
    gpt::CausalSelfAttention attention_layer(block_size, n_head, n_embed);

    Eigen::Tensor<Type, 3> x_data(2, block_size, n_embed);
    Eigen::Tensor<Type, 3> y_data(2, block_size, n_embed);

    // Initialize input tensor x
    x_data.setRandom(); // Random initialization for testing

    // Create TensorMap for x and y
    TTypes<Type, 3>::ConstTensor x(x_data.data(), x_data.dimensions());
    TTypes<Type, 3>::Tensor y(y_data.data(), y_data.dimensions());

    // Call the Forward function
    attention_layer.Forward(x, y);

    // Verify the output (this is a placeholder check, replace with actual expected values)
    for (int i = 0; i < y.dimension(0); ++i) {
        for (int j = 0; j < y.dimension(1); ++j) {
            for (int k = 0; k < y.dimension(2); ++k) {
                EXPECT_NEAR(y(i, j, k).to_float(), 0.0f, EPSILON); // Replace 0.0f with actual expected value
            }
        }
    }
}

TEST_F(CausalSelfAttentionTest, Backward) {
    using Type = fixed_point_7pt8;
    int block_size = 4;
    int n_head = 2;
    int n_embed = 8;
    gpt::CausalSelfAttention attention_layer(block_size, n_head, n_embed);

    Eigen::Tensor<Type, 3> x_data(2, block_size, n_embed);
    Eigen::Tensor<Type, 3> y_grad_data(2, block_size, n_embed);
    Eigen::Tensor<Type, 3> x_grad_data(2, block_size, n_embed);

    // Initialize input tensor x and gradient tensor y_grad
    x_data.setRandom(); // Random initialization for testing
    y_grad_data.setRandom(); // Random initialization for testing

    // Initialize gradient tensor x_grad
    x_grad_data.setZero();

    // Create TensorMap for x, y_grad, and x_grad
    TTypes<Type, 3>::ConstTensor x(x_data.data(), x_data.dimensions());
    TTypes<Type, 3>::ConstTensor y_grad(y_grad_data.data(), y_grad_data.dimensions());
    TTypes<Type, 3>::Tensor x_grad(x_grad_data.data(), x_grad_data.dimensions());

    // Call the Backward function
    attention_layer.Backward(x, y_grad, x_grad);

    // Verify the output for x_grad (this is a placeholder check, replace with actual expected values)
    for (int i = 0; i < x_grad.dimension(0); ++i) {
        for (int j = 0; j < x_grad.dimension(1); ++j) {
            for (int k = 0; k < x_grad.dimension(2); ++k) {
                EXPECT_NEAR(x_grad(i, j, k).to_float(), 0.0f, EPSILON); // Replace 0.0f with actual expected value
            }
        }
    }
}

TEST_F(CausalSelfAttentionTest, NumParameters) {
    int block_size = 4;
    int n_head = 2;
    int n_embed = 8;
    gpt::CausalSelfAttention attention_layer(block_size, n_head, n_embed);

    size_t num_parameters = attention_layer.NumParameters();
    EXPECT_GT(num_parameters, 0); // Should be greater than 0
}

TEST_F(CausalSelfAttentionTest, NumActivations) {
    int block_size = 4;
    int n_head = 2;
    int n_embed = 8;
    gpt::CausalSelfAttention attention_layer(block_size, n_head, n_embed);

    size_t num_activations = attention_layer.NumActivations();
    EXPECT_GT(num_activations, 0); // Should be greater than 0
}

TEST_F(CausalSelfAttentionTest, Parameters) {
    int block_size = 4;
    int n_head = 2;
    int n_embed = 8;
    gpt::CausalSelfAttention attention_layer(block_size, n_head, n_embed);

    std::vector<nn::Parameter*> parameters;
    attention_layer.Parameters(&parameters);

    EXPECT_GT(parameters.size(), 0); // Should be greater than 0
}