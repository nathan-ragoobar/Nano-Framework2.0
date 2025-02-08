#include "Softmax.hpp"
#include <gtest/gtest.h>
#include "./../tensor/fixed_point.hpp" // Include the fixed_point header if needed
#include "./../tensor/fpm/fpm.hpp"

constexpr float EPSILON = 0.01f;

class SoftmaxTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }
};

TEST_F(SoftmaxTest, Forward) {
    using T = fpm::fixed_16_16;
    
    Eigen::Tensor<T, 2> x_data(2, 3);
    x_data(0, 0) = T(1.0f); x_data(0, 1) = T(2.0f); x_data(0, 2) = T(3.0f);
    x_data(1, 0) = T(1.0f); x_data(1, 1) = T(2.0f); x_data(1, 2) = T(3.0f);

    // Updated with exact calculated values
    Eigen::Tensor<T, 2> expected_y_data(2, 3);
    expected_y_data(0, 0) = T(0.090f);
    expected_y_data(0, 1) = T(0.245f);
    expected_y_data(0, 2) = T(0.665f);
    expected_y_data(1, 0) = T(0.090f);
    expected_y_data(1, 1) = T(0.245f);
    expected_y_data(1, 2) = T(0.665f);

    // Create output tensor
    Eigen::Tensor<T, 2> y_data(2, 3);
    TTypes<T>::ConstMatrix x(x_data.data(), x_data.dimensions());
    TTypes<T>::Matrix y(y_data.data(), y_data.dimensions());

    nn::Softmax::Forward(x, y);

    // Verify with debug output
    for (int i = 0; i < 2; ++i) {
        float row_sum = 0;
        for (int j = 0; j < 3; ++j) {
            row_sum += float(y(i, j));
            EXPECT_NEAR(float(y(i, j)), float(expected_y_data(i, j)), EPSILON);
        }
        EXPECT_NEAR(row_sum, 1.0f, EPSILON);
    }
}

// 3. Modified Backward test
TEST_F(SoftmaxTest, Backward) {
    using T = fpm::fixed_16_16;
    
    // Adjusted y values to match forward pass
    Eigen::Tensor<T, 2> y_data(2, 3);
    y_data(0, 0) = T(0.09f); y_data(0, 1) = T(0.24f); y_data(0, 2) = T(0.67f);
    y_data(1, 0) = T(0.09f); y_data(1, 1) = T(0.24f); y_data(1, 2) = T(0.67f);

    // Simple gradient values
    Eigen::Tensor<T, 2> y_grad_data(2, 3);
    y_grad_data(0, 0) = T(0.1f); y_grad_data(0, 1) = T(0.2f); y_grad_data(0, 2) = T(0.3f);
    y_grad_data(1, 0) = T(0.1f); y_grad_data(1, 1) = T(0.2f); y_grad_data(1, 2) = T(0.3f);

    // Adjusted expected gradients
    Eigen::Tensor<T, 2> expected_x_grad_data(2, 3);
    expected_x_grad_data(0, 0) = T(-0.014f);
    expected_x_grad_data(0, 1) = T(-0.014f);
    expected_x_grad_data(0, 2) = T(0.028f);
    expected_x_grad_data(1, 0) = T(-0.014f);
    expected_x_grad_data(1, 1) = T(-0.014f);
    expected_x_grad_data(1, 2) = T(0.028f);

    Eigen::Tensor<T, 2> x_grad_data(2, 3);
    TTypes<T>::ConstMatrix y(y_data.data(), y_data.dimensions());
    TTypes<T>::ConstMatrix y_grad(y_grad_data.data(), y_grad_data.dimensions());
    TTypes<T>::Matrix x_grad(x_grad_data.data(), x_grad_data.dimensions());

    nn::Softmax::Backward(y, y_grad, x_grad);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(float(x_grad(i, j)), float(expected_x_grad_data(i, j)), EPSILON);
        }
    }
}

/*
TEST_F(SoftmaxTest, Forward) {
    using T = fixed_point_7pt8;
    Eigen::Tensor<T, 2> x_data(2, 3);
    Eigen::Tensor<T, 2> y_data(2, 3);
    Eigen::Tensor<T, 2> expected_y_data(2, 3);

    // Initialize input tensor x
    x_data(0, 0) = T(1.0f);
    x_data(0, 1) = T(2.0f);
    x_data(0, 2) = T(3.0f);
    x_data(1, 0) = T(1.0f);
    x_data(1, 1) = T(2.0f);
    x_data(1, 2) = T(3.0f);

    // Initialize expected output tensor y
    expected_y_data(0, 0) = T(0.09003057f);
    expected_y_data(0, 1) = T(0.24472847f);
    expected_y_data(0, 2) = T(0.66524096f);
    expected_y_data(1, 0) = T(0.09003057f);
    expected_y_data(1, 1) = T(0.24472847f);
    expected_y_data(1, 2) = T(0.66524096f);

    // Create TensorMap for x and y
    TTypes<T>::ConstMatrix x(x_data.data(), x_data.dimensions());
    TTypes<T>::Matrix y(y_data.data(), y_data.dimensions());

    // Call the Forward function
    nn::Softmax::Forward(x, y);

    // Verify the output
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(y(i, j).to_float(), expected_y_data(i, j).to_float(), EPSILON);
        }
    }
}

TEST_F(SoftmaxTest, Backward) {
    using T = fixed_point_7pt8;
    Eigen::Tensor<T, 2> y_data(2, 3);
    Eigen::Tensor<T, 2> y_grad_data(2, 3);
    Eigen::Tensor<T, 2> x_grad_data(2, 3);

    // Initialize output tensor y
    y_data(0, 0) = T(0.09003057f);
    y_data(0, 1) = T(0.24472847f);
    y_data(0, 2) = T(0.66524096f);
    y_data(1, 0) = T(0.09003057f);
    y_data(1, 1) = T(0.24472847f);
    y_data(1, 2) = T(0.66524096f);

    // Initialize gradient tensor y_grad
    y_grad_data(0, 0) = T(0.1f);
    y_grad_data(0, 1) = T(0.2f);
    y_grad_data(0, 2) = T(0.3f);
    y_grad_data(1, 0) = T(0.1f);
    y_grad_data(1, 1) = T(0.2f);
    y_grad_data(1, 2) = T(0.3f);

    // Expected gradient tensor x_grad
    Eigen::Tensor<T, 2> expected_x_grad_data(2, 3);
    expected_x_grad_data(0, 0) = T(-0.00721254f);
    expected_x_grad_data(0, 1) = T(0.00489494f);
    expected_x_grad_data(0, 2) = T(0.00231760f);
    expected_x_grad_data(1, 0) = T(-0.00721254f);
    expected_x_grad_data(1, 1) = T(0.00489494f);
    expected_x_grad_data(1, 2) = T(0.00231760f);

    // Create TensorMap for y, y_grad, and x_grad
    TTypes<T>::ConstMatrix y(y_data.data(), y_data.dimensions());
    TTypes<T>::ConstMatrix y_grad(y_grad_data.data(), y_grad_data.dimensions());
    TTypes<T>::Matrix x_grad(x_grad_data.data(), x_grad_data.dimensions());

    // Call the Backward function
    nn::Softmax::Backward(y, y_grad, x_grad);

    // Verify the output
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(x_grad(i, j).to_float(), expected_x_grad_data(i, j).to_float(), EPSILON);
        }
    }
}
*/
class SoftmaxCrossEntropyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }
};

TEST_F(SoftmaxCrossEntropyTest, Forward) {
    using T = fpm::fixed_16_16;
    Eigen::Tensor<T, 2> logits_data(2, 3);
    Eigen::Tensor<T, 2> probs_data(2, 3);
    std::vector<int> targets = {1, 2};
    float loss;

    // Initialize input tensor logits
    logits_data(0, 0) = T(1.0f);
    logits_data(0, 1) = T(2.0f);
    logits_data(0, 2) = T(3.0f);
    logits_data(1, 0) = T(1.0f);
    logits_data(1, 1) = T(2.0f);
    logits_data(1, 2) = T(3.0f);

    // Create TensorMap for logits and probs
    TTypes<T>::ConstMatrix logits(logits_data.data(), logits_data.dimensions());
    TTypes<T>::Matrix probs(probs_data.data(), probs_data.dimensions());

    // Create SoftmaxCrossEntropy object
    nn::SoftmaxCrossEntropy softmax_cross_entropy(nn::SoftmaxCrossEntropy::MEAN);

    // Call the Forward function
    softmax_cross_entropy.Forward(logits, targets, probs, &loss);

    // Verify the output
    float expected_loss = -std::log(0.245f) - std::log(0.665f);
    expected_loss /= 2.0f; // MEAN reduction

    EXPECT_NEAR(loss, expected_loss, EPSILON);
}

TEST_F(SoftmaxCrossEntropyTest, Backward) {
    using T = fpm::fixed_16_16;
    Eigen::Tensor<T, 2> probs_data(2, 3);
    Eigen::Tensor<T, 2> logits_grad_data(2, 3);
    std::vector<int> targets = {1, 2};

    // Initialize input tensor probs
    probs_data(0, 0) = T(0.09003057f);
    probs_data(0, 1) = T(0.24472847f);
    probs_data(0, 2) = T(0.66524096f);
    probs_data(1, 0) = T(0.09003057f);
    probs_data(1, 1) = T(0.24472847f);
    probs_data(1, 2) = T(0.66524096f);

    // Initialize gradient tensor logits_grad
    logits_grad_data.setZero();

    // Create TensorMap for probs and logits_grad
    TTypes<T>::ConstMatrix probs(probs_data.data(), probs_data.dimensions());
    TTypes<T>::Matrix logits_grad(logits_grad_data.data(), logits_grad_data.dimensions());

    // Create SoftmaxCrossEntropy object
    nn::SoftmaxCrossEntropy softmax_cross_entropy(nn::SoftmaxCrossEntropy::MEAN);

    // Call the Backward function
    softmax_cross_entropy.Backward(probs, targets, logits_grad);

    // Verify the output
    Eigen::Tensor<T, 2> expected_logits_grad(2, 3);
    expected_logits_grad.setZero();
    expected_logits_grad(0, 1) = T((0.245f - 1.0f) / 2.0f); // MEAN reduction
    expected_logits_grad(1, 2) = T((0.665f - 1.0f) / 2.0f); // MEAN reduction

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(float(logits_grad(i, j)), float(expected_logits_grad(i, j)), EPSILON);
        }
    }
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}