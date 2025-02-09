#include "./LayerNorm.hpp"
#include <gtest/gtest.h>
#include "./../tensor/fixed_point.hpp" // Include the fixed_point header if needed

constexpr float EPSILON = 1e-4;

class LayerNormTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }
};

TEST_F(LayerNormTest, Forward) {
    using T = float;
    int normalized_shape = 3;
    nn::LayerNorm layernorm_layer(normalized_shape);

    Eigen::Tensor<T, 2> x_data(2, normalized_shape);
    Eigen::Tensor<T, 2> y_data(2, normalized_shape);
    Eigen::Tensor<T, 1> mean_data(2);
    Eigen::Tensor<T, 1> rstd_data(2);

    // Initialize input tensor x
    x_data(0, 0) = T(1.0f); x_data(0, 1) = T(2.0f); x_data(0, 2) = T(3.0f);
    x_data(1, 0) = T(4.0f); x_data(1, 1) = T(5.0f); x_data(1, 2) = T(6.0f);

    // Create TensorMap for x, y, mean, and rstd
    TTypes<T>::ConstMatrix x(x_data.data(), x_data.dimensions());
    TTypes<T>::Matrix y(y_data.data(), y_data.dimensions());
    TTypes<T>::Flat mean(mean_data.data(), mean_data.dimensions());
    TTypes<T>::Flat rstd(rstd_data.data(), rstd_data.dimensions());

    // Call the Forward function
    layernorm_layer.Forward(x, y, mean, rstd);

    // Verify the output (this is a placeholder check, replace with actual expected values)
    for (int i = 0; i < y.dimension(0); ++i) {
        for (int j = 0; j < y.dimension(1); ++j) {
            EXPECT_NEAR(y(i, j), 0.0f, EPSILON); // Replace 0.0f with actual expected value
        }
    }
}

TEST_F(LayerNormTest, Backward) {
    using T = float;
    int normalized_shape = 3;
    nn::LayerNorm layernorm_layer(normalized_shape);

    Eigen::Tensor<T, 2> x_data(2, normalized_shape);
    Eigen::Tensor<T, 2> y_grad_data(2, normalized_shape);
    Eigen::Tensor<T, 2> x_grad_data(2, normalized_shape);
    Eigen::Tensor<T, 1> mean_data(2);
    Eigen::Tensor<T, 1> rstd_data(2);

    // Initialize input tensor x and gradient tensor y_grad
    x_data(0, 0) = T(1.0f); x_data(0, 1) = T(2.0f); x_data(0, 2) = T(3.0f);
    x_data(1, 0) = T(4.0f); x_data(1, 1) = T(5.0f); x_data(1, 2) = T(6.0f);

    y_grad_data(0, 0) = T(1.0f); y_grad_data(0, 1) = T(2.0f); y_grad_data(0, 2) = T(3.0f);
    y_grad_data(1, 0) = T(4.0f); y_grad_data(1, 1) = T(5.0f); y_grad_data(1, 2) = T(6.0f);

    // Initialize gradient tensor x_grad
    x_grad_data.setZero();

    // Create TensorMap for x, y_grad, x_grad, mean, and rstd
    TTypes<T>::ConstMatrix x(x_data.data(), x_data.dimensions());
    TTypes<T>::ConstMatrix y_grad(y_grad_data.data(), y_grad_data.dimensions());
    TTypes<T>::Matrix x_grad(x_grad_data.data(), x_grad_data.dimensions());
    TTypes<T>::ConstFlat mean(mean_data.data(), mean_data.dimensions());
    TTypes<T>::ConstFlat rstd(rstd_data.data(), rstd_data.dimensions());

    // Call the Backward function
    layernorm_layer.Backward(x, y_grad, mean, rstd, x_grad);

    // Verify the output for x_grad (this is a placeholder check, replace with actual expected values)
    for (int i = 0; i < x_grad.dimension(0); ++i) {
        for (int j = 0; j < x_grad.dimension(1); ++j) {
            EXPECT_NEAR(x_grad(i, j), 0.0f, EPSILON); // Replace 0.0f with actual expected value
        }
    }
}

TEST_F(LayerNormTest, NumParameters) {
    int normalized_shape = 3;
    nn::LayerNorm layernorm_layer(normalized_shape);

    size_t num_parameters = layernorm_layer.NumParameters();
    EXPECT_EQ(num_parameters, normalized_shape * 2); // Including weight and bias
}

TEST_F(LayerNormTest, NumActivations) {
    int normalized_shape = 3;
    nn::LayerNorm layernorm_layer(normalized_shape);

    size_t num_activations = layernorm_layer.NumActivations();
    EXPECT_GT(num_activations, 0); // Should be greater than 0
}

TEST_F(LayerNormTest, Parameters) {
    int normalized_shape = 3;
    nn::LayerNorm layernorm_layer(normalized_shape);

    std::vector<nn::Parameter*> parameters;
    layernorm_layer.Parameters(&parameters);

    EXPECT_EQ(parameters.size(), 2); // Including weight and bias
}