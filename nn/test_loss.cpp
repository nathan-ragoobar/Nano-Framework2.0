#include "Loss.hpp"
#include <gtest/gtest.h>
#include "./../tensor/fixed_point.hpp" // Include the fixed_point header if needed

constexpr float EPSILON = 1e-4;

class CrossEntropyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }
};

TEST_F(CrossEntropyTest, Forward) {
    using T = fixed_point_7pt8;
    Eigen::Tensor<T, 2> probs_data(2, 3);
    std::vector<int> targets = {1, 2};
    float loss;

    // Initialize input tensor probs
    probs_data(0, 0) = T(0.1f);
    probs_data(0, 1) = T(0.7f);
    probs_data(0, 2) = T(0.2f);
    probs_data(1, 0) = T(0.3f);
    probs_data(1, 1) = T(0.4f);
    probs_data(1, 2) = T(0.3f);

    // Create TensorMap for probs
    TTypes<T>::ConstMatrix probs(probs_data.data(), probs_data.dimensions());

    // Create CrossEntropy object
    nn::CrossEntropy cross_entropy(nn::CrossEntropy::MEAN);

    // Call the Forward function
    cross_entropy.Forward(probs, targets, &loss);

    // Verify the output
    float expected_loss = -std::log(0.7f) - std::log(0.3f);
    expected_loss /= 2.0f; // MEAN reduction

    EXPECT_NEAR(loss, expected_loss, EPSILON);
}

TEST_F(CrossEntropyTest, Backward) {
    using T = fixed_point_7pt8;
    Eigen::Tensor<T, 2> probs_data(2, 3);
    std::vector<int> targets = {1, 2};
    Eigen::Tensor<T, 2> probs_grad_data(2, 3);

    // Initialize input tensor probs
    probs_data(0, 0) = T(0.1f);
    probs_data(0, 1) = T(0.7f);
    probs_data(0, 2) = T(0.2f);
    probs_data(1, 0) = T(0.3f);
    probs_data(1, 1) = T(0.4f);
    probs_data(1, 2) = T(0.3f);

    // Initialize gradient tensor probs_grad
    probs_grad_data.setZero();

    // Create TensorMap for probs and probs_grad
    TTypes<T>::ConstMatrix probs(probs_data.data(), probs_data.dimensions());
    TTypes<T>::Matrix probs_grad(probs_grad_data.data(), probs_grad_data.dimensions());

    // Create CrossEntropy object
    nn::CrossEntropy cross_entropy(nn::CrossEntropy::MEAN);

    // Call the Backward function
    cross_entropy.Backward(probs, targets, probs_grad);

    // Verify the output
    Eigen::Tensor<T, 2> expected_probs_grad(2, 3);
    expected_probs_grad.setZero();
    expected_probs_grad(0, 1) = T(-1.0f / 0.7f / 2.0f); // MEAN reduction
    expected_probs_grad(1, 2) = T(-1.0f / 0.3f / 2.0f); // MEAN reduction

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(probs_grad(i, j).to_float(), expected_probs_grad(i, j).to_float(), EPSILON);
        }
    }
}