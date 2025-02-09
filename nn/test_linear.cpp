#include "./Linear.hpp"
#include <gtest/gtest.h>
#include "./../tensor/fixed_point.hpp" // Include the fixed_point header if needed
#include "./../tensor/fpm/fpm.hpp"

constexpr float EPSILON = 1e-4;

class LinearTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }
};

TEST_F(LinearTest, Forward) {
    using T = fpm::fixed_16_16;
    int in_features = 3;
    int out_features = 2;
    nn::Linear linear_layer(in_features, out_features);

    Eigen::Tensor<T, 2> x_data(2, in_features);
    Eigen::Tensor<T, 2> y_data(2, out_features);

    // Initialize input tensor x
    x_data(0, 0) = T(1.0f); x_data(0, 1) = T(2.0f); x_data(0, 2) = T(3.0f);
    x_data(1, 0) = T(4.0f); x_data(1, 1) = T(5.0f); x_data(1, 2) = T(6.0f);

    // Create TensorMap for x and y
    TTypes<T>::ConstMatrix x(x_data.data(), x_data.dimensions());
    TTypes<T>::Matrix y(y_data.data(), y_data.dimensions());

    // Call the Forward function
    linear_layer.Forward(x, y);

    // Verify the output (this is a placeholder check, replace with actual expected values)
    for (int i = 0; i < y.dimension(0); ++i) {
        for (int j = 0; j < y.dimension(1); ++j) {
            EXPECT_NEAR(float(y(i, j)), 0.0f, EPSILON); // Replace 0.0f with actual expected value
        }
    }
}

TEST_F(LinearTest, Backward) {
    using T = fpm::fixed_16_16;
    int in_features = 3;
    int out_features = 2;
    nn::Linear linear_layer(in_features, out_features);

    Eigen::Tensor<T, 2> x_data(2, in_features);
    Eigen::Tensor<T, 2> y_grad_data(2, out_features);
    Eigen::Tensor<T, 2> x_grad_data(2, in_features);

    // Initialize input tensor x and gradient tensor y_grad
    x_data(0, 0) = T(1.0f); x_data(0, 1) = T(2.0f); x_data(0, 2) = T(3.0f);
    x_data(1, 0) = T(4.0f); x_data(1, 1) = T(5.0f); x_data(1, 2) = T(6.0f);

    y_grad_data(0, 0) = T(1.0f); y_grad_data(0, 1) = T(2.0f);
    y_grad_data(1, 0) = T(3.0f); y_grad_data(1, 1) = T(4.0f);

    // Initialize gradient tensor x_grad
    x_grad_data.setZero();

    // Create TensorMap for x, y_grad, and x_grad
    TTypes<T>::ConstMatrix x(x_data.data(), x_data.dimensions());
    TTypes<T>::ConstMatrix y_grad(y_grad_data.data(), y_grad_data.dimensions());
    TTypes<T>::Matrix x_grad(x_grad_data.data(), x_grad_data.dimensions());

    // Call the Backward function
    linear_layer.Backward(x, y_grad, x_grad);

    // Verify the output for x_grad (this is a placeholder check, replace with actual expected values)
    for (int i = 0; i < x_grad.dimension(0); ++i) {
        for (int j = 0; j < x_grad.dimension(1); ++j) {
            EXPECT_NEAR(float(x_grad(i, j)), 0.0f, EPSILON); // Replace 0.0f with actual expected value
        }
    }
}

TEST_F(LinearTest, NumParameters) {
    int in_features = 3;
    int out_features = 2;
    nn::Linear linear_layer(in_features, out_features);

    size_t num_parameters = linear_layer.NumParameters();
    EXPECT_EQ(num_parameters, in_features * out_features + out_features); // Including bias
}

TEST_F(LinearTest, NumActivations) {
    int in_features = 3;
    int out_features = 2;
    nn::Linear linear_layer(in_features, out_features);

    size_t num_activations = linear_layer.NumActivations();
    EXPECT_EQ(num_activations, 0);
}

TEST_F(LinearTest, Parameters) {
    int in_features = 3;
    int out_features = 2;
    nn::Linear linear_layer(in_features, out_features);

    std::vector<nn::Parameter*> parameters;
    linear_layer.Parameters(&parameters);

    EXPECT_EQ(parameters.size(), 2); // Including bias
}
