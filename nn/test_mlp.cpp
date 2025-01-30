#include "./MLP.hpp"
#include <gtest/gtest.h>
#include "./../tensor/fixed_point.hpp" // Include the fixed_point header if needed

constexpr float EPSILON = 1e-4;

class MLPTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }
};

TEST_F(MLPTest, Forward) {
    using T = fixed_point_7pt8;
    int n_embed = 3;
    gpt::MLP mlp_layer(n_embed);

    Eigen::Tensor<T, 2> x_data(2, n_embed);
    Eigen::Tensor<T, 2> y_data(2, n_embed);

    // Initialize input tensor x
    x_data(0, 0) = T(1.0f); x_data(0, 1) = T(2.0f); x_data(0, 2) = T(3.0f);
    x_data(1, 0) = T(4.0f); x_data(1, 1) = T(5.0f); x_data(1, 2) = T(6.0f);

    // Create TensorMap for x and y
    TTypes<T>::ConstMatrix x(x_data.data(), x_data.dimensions());
    TTypes<T>::Matrix y(y_data.data(), y_data.dimensions());

    // Call the Forward function
    mlp_layer.Forward(x, y);

    // Verify the output (this is a placeholder check, replace with actual expected values)
    for (int i = 0; i < y.dimension(0); ++i) {
        for (int j = 0; j < y.dimension(1); ++j) {
            EXPECT_NEAR(y(i, j).to_float(), 0.0f, EPSILON); // Replace 0.0f with actual expected value
        }
    }
}


TEST_F(MLPTest, Backward) {
    using T = fixed_point_7pt8;
    int n_embed = 3;
    gpt::MLP mlp_layer(n_embed);

    Eigen::Tensor<T, 2> x_data(2, n_embed);
    Eigen::Tensor<T, 2> y_data(2, n_embed);
    Eigen::Tensor<T, 2> y_grad_data(2, n_embed);
    Eigen::Tensor<T, 2> x_grad_data(2, n_embed);

    // Initialize input tensor x
    x_data(0, 0) = T(1.0f); x_data(0, 1) = T(2.0f); x_data(0, 2) = T(3.0f);
    x_data(1, 0) = T(4.0f); x_data(1, 1) = T(5.0f); x_data(1, 2) = T(6.0f);

    // Create TensorMap for forward pass
    TTypes<T>::ConstMatrix x(x_data.data(), x_data.dimensions());
    TTypes<T>::Matrix y(y_data.data(), y_data.dimensions());

    // Do forward pass first to initialize activations
    mlp_layer.Forward(x, y);

    // Initialize gradient tensors
    y_grad_data(0, 0) = T(1.0f); y_grad_data(0, 1) = T(2.0f); y_grad_data(0, 2) = T(3.0f);
    y_grad_data(1, 0) = T(4.0f); y_grad_data(1, 1) = T(5.0f); y_grad_data(1, 2) = T(6.0f);
    x_grad_data.setZero();

    // Create TensorMap for backward pass
    TTypes<T>::ConstMatrix y_grad(y_grad_data.data(), y_grad_data.dimensions());
    TTypes<T>::Matrix x_grad(x_grad_data.data(), x_grad_data.dimensions());

    // Call the Backward function
    mlp_layer.Backward(x, y_grad, x_grad);

    // Verify gradients
    for (int i = 0; i < x_grad.dimension(0); ++i) {
        for (int j = 0; j < x_grad.dimension(1); ++j) {
            EXPECT_NEAR(x_grad(i, j).to_float(), 0.0f, EPSILON);
        }
    }
}

TEST_F(MLPTest, NumParameters) {
    int n_embed = 3;
    gpt::MLP mlp_layer(n_embed);

    size_t num_parameters = mlp_layer.NumParameters();
    EXPECT_GT(num_parameters, 0); // Should be greater than 0
}

TEST_F(MLPTest, NumActivations) {
    int n_embed = 3;
    gpt::MLP mlp_layer(n_embed);

    size_t num_activations = mlp_layer.NumActivations();
    EXPECT_GT(num_activations, 0); // Should be greater than 0
}

TEST_F(MLPTest, Parameters) {
    int n_embed = 3;
    gpt::MLP mlp_layer(n_embed);

    std::vector<nn::Parameter*> parameters;
    mlp_layer.Parameters(&parameters);

    EXPECT_GT(parameters.size(), 0); // Should be greater than 0
}