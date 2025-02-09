#include "./MatMul.hpp"
#include <gtest/gtest.h>
#include "./../tensor/fixed_point.hpp" // Include the fixed_point header if needed
#include "./../tensor/fpm/fpm.hpp"

constexpr float EPSILON = 1e-4;

class MatMulTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }
};
/*
TEST_F(MatMulTest, ForwardDebug) {
    using T = fixed_point_7pt8;
    
    // Create smaller test case
    Eigen::Tensor<T, 2> x1_data(2, 2);
    Eigen::Tensor<T, 2> x2_data(2, 2);
    Eigen::Tensor<T, 2> y_data(2, 2);

    // Simple values to trace computation
    x1_data(0,0) = T(1.0f); x1_data(0,1) = T(2.0f);
    x1_data(1,0) = T(3.0f); x1_data(1,1) = T(4.0f);

    x2_data(0,0) = T(1.0f); x2_data(0,1) = T(2.0f);
    x2_data(1,0) = T(3.0f); x2_data(1,1) = T(4.0f);

    TTypes<T>::ConstMatrix x1(x1_data.data(), x1_data.dimensions());
    TTypes<T>::ConstMatrix x2(x2_data.data(), x2_data.dimensions());
    TTypes<T>::Matrix y(y_data.data(), y_data.dimensions());

    // Print inputs
    std::cout << "x1 values:\n";
    for(int i=0; i<2; i++) {
        for(int j=0; j<2; j++) {
            std::cout << x1(i,j).to_float() << " ";
        }
        std::cout << "\n";
    }

    nn::MatMul::Forward(x1, x2, y);

    // Print outputs
    std::cout << "Result values:\n";
    for(int i=0; i<2; i++) {
        for(int j=0; j<2; j++) {
            std::cout << y(i,j).to_float() << " ";
        }
        std::cout << "\n";
    }
}
*/

TEST_F(MatMulTest, ForwardFixedPoint) {
    using T = fpm::fixed_16_16;
    Eigen::Tensor<T, 2> x1_data(2, 3);
    Eigen::Tensor<T, 2> x2_data(3, 2);
    Eigen::Tensor<T, 2> y_data(2, 2);

    // Initialize input tensors x1 and x2
    x1_data(0, 0) = T(1.0f); x1_data(0, 1) = T(2.0f); x1_data(0, 2) = T(3.0f);
    x1_data(1, 0) = T(4.0f); x1_data(1, 1) = T(5.0f); x1_data(1, 2) = T(6.0f);

    x2_data(0, 0) = T(7.0f); x2_data(0, 1) = T(8.0f);
    x2_data(1, 0) = T(9.0f); x2_data(1, 1) = T(10.0f);
    x2_data(2, 0) = T(11.0f); x2_data(2, 1) = T(12.0f);

    // Create TensorMap for x1, x2, and y
    TTypes<T>::ConstMatrix x1(x1_data.data(), x1_data.dimensions());
    TTypes<T>::ConstMatrix x2(x2_data.data(), x2_data.dimensions());
    TTypes<T>::Matrix y(y_data.data(), y_data.dimensions());

    // Call the Forward function
    nn::MatMul::Forward(x1, x2, y);

    // Verify the output
    Eigen::Tensor<T, 2> expected_y(2, 2);
    expected_y(0, 0) = T(58.0f); expected_y(0, 1) = T(64.0f);
    expected_y(1, 0) = T(139.0f); expected_y(1, 1) = T(154.0f);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_NEAR(float(y(i, j)), float(expected_y(i, j)), EPSILON);
        }
    }
}


TEST_F(MatMulTest, ForwardFloat) {
    using T = float;  // Changed from fpm::fixed_16_16 to float
    Eigen::Tensor<T, 2> x1_data(2, 3);
    Eigen::Tensor<T, 2> x2_data(3, 2);
    Eigen::Tensor<T, 2> y_data(2, 2);

    // Initialize input tensors x1 and x2 (removed T() conversion)
    x1_data(0, 0) = 1.0f; x1_data(0, 1) = 2.0f; x1_data(0, 2) = 3.0f;
    x1_data(1, 0) = 4.0f; x1_data(1, 1) = 5.0f; x1_data(1, 2) = 6.0f;

    x2_data(0, 0) = 7.0f; x2_data(0, 1) = 8.0f;
    x2_data(1, 0) = 9.0f; x2_data(1, 1) = 10.0f;
    x2_data(2, 0) = 11.0f; x2_data(2, 1) = 12.0f;

    // Create TensorMap for x1, x2, and y
    TTypes<T>::ConstMatrix x1(x1_data.data(), x1_data.dimensions());
    TTypes<T>::ConstMatrix x2(x2_data.data(), x2_data.dimensions());
    TTypes<T>::Matrix y(y_data.data(), y_data.dimensions());

    // Call the Forward function
    nn::MatMul::Forward(x1, x2, y);

    // Verify the output (removed T() conversion)
    Eigen::Tensor<T, 2> expected_y(2, 2);
    expected_y(0, 0) = 58.0f; expected_y(0, 1) = 64.0f;
    expected_y(1, 0) = 139.0f; expected_y(1, 1) = 154.0f;

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            // Removed float conversion since we're already using float
            EXPECT_NEAR(y(i, j), expected_y(i, j), EPSILON);
        }
    }
}

TEST_F(MatMulTest, BackwardFixedPoint) {
    using T = fpm::fixed_16_16;
    Eigen::Tensor<T, 2> x1_data(2, 3);
    Eigen::Tensor<T, 2> x2_data(3, 2);
    Eigen::Tensor<T, 2> y_grad_data(2, 2);
    Eigen::Tensor<T, 2> x1_grad_data(2, 3);
    Eigen::Tensor<T, 2> x2_grad_data(3, 2);

    // Initialize input tensors x1, x2, and y_grad
    x1_data(0, 0) = T(1.0f); x1_data(0, 1) = T(2.0f); x1_data(0, 2) = T(3.0f);
    x1_data(1, 0) = T(4.0f); x1_data(1, 1) = T(5.0f); x1_data(1, 2) = T(6.0f);

    x2_data(0, 0) = T(7.0f); x2_data(0, 1) = T(8.0f);
    x2_data(1, 0) = T(9.0f); x2_data(1, 1) = T(10.0f);
    x2_data(2, 0) = T(11.0f); x2_data(2, 1) = T(12.0f);

    y_grad_data(0, 0) = T(1.0f); y_grad_data(0, 1) = T(2.0f);
    y_grad_data(1, 0) = T(3.0f); y_grad_data(1, 1) = T(4.0f);

    // Initialize gradient tensors x1_grad and x2_grad
    x1_grad_data.setZero();
    x2_grad_data.setZero();

    // Create TensorMap for x1, x2, y_grad, x1_grad, and x2_grad
    TTypes<T>::ConstMatrix x1(x1_data.data(), x1_data.dimensions());
    TTypes<T>::ConstMatrix x2(x2_data.data(), x2_data.dimensions());
    TTypes<T>::ConstMatrix y_grad(y_grad_data.data(), y_grad_data.dimensions());
    TTypes<T>::Matrix x1_grad(x1_grad_data.data(), x1_grad_data.dimensions());
    TTypes<T>::Matrix x2_grad(x2_grad_data.data(), x2_grad_data.dimensions());

    // Call the Backward function
    nn::MatMul::Backward(x1, x2, y_grad, x1_grad, x2_grad);

    // Verify the output for x1_grad
    Eigen::Tensor<T, 2> expected_x1_grad(2, 3);
    expected_x1_grad(0, 0) = T(23.0f); expected_x1_grad(0, 1) = T(29.0f); expected_x1_grad(0, 2) = T(35.0f);
    expected_x1_grad(1, 0) = T(31.0f); expected_x1_grad(1, 1) = T(39.0f); expected_x1_grad(1, 2) = T(47.0f);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(float(x1_grad(i, j)), float(expected_x1_grad(i, j)), EPSILON);
        }
    }

    // Verify the output for x2_grad
    Eigen::Tensor<T, 2> expected_x2_grad(3, 2);
    expected_x2_grad(0, 0) = T(14.0f); expected_x2_grad(0, 1) = T(20.0f);
    expected_x2_grad(1, 0) = T(32.0f); expected_x2_grad(1, 1) = T(44.0f);
    expected_x2_grad(2, 0) = T(50.0f); expected_x2_grad(2, 1) = T(68.0f);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_NEAR(float(x2_grad(i, j)), float(expected_x2_grad(i, j)), EPSILON);
        }
    }
}

TEST_F(MatMulTest, BackwardFloat) {
    using T = float;  // Changed from fpm::fixed_16_16 to float
    Eigen::Tensor<T, 2> x1_data(2, 3);
    Eigen::Tensor<T, 2> x2_data(3, 2);
    Eigen::Tensor<T, 2> y_grad_data(2, 2);
    Eigen::Tensor<T, 2> x1_grad_data(2, 3);
    Eigen::Tensor<T, 2> x2_grad_data(3, 2);

    // Initialize input tensors x1, x2, and y_grad (removed T() conversion)
    x1_data(0, 0) = 1.0f; x1_data(0, 1) = 2.0f; x1_data(0, 2) = 3.0f;
    x1_data(1, 0) = 4.0f; x1_data(1, 1) = 5.0f; x1_data(1, 2) = 6.0f;

    x2_data(0, 0) = 7.0f; x2_data(0, 1) = 8.0f;
    x2_data(1, 0) = 9.0f; x2_data(1, 1) = 10.0f;
    x2_data(2, 0) = 11.0f; x2_data(2, 1) = 12.0f;

    y_grad_data(0, 0) = 1.0f; y_grad_data(0, 1) = 2.0f;
    y_grad_data(1, 0) = 3.0f; y_grad_data(1, 1) = 4.0f;

    // Initialize gradient tensors x1_grad and x2_grad
    x1_grad_data.setZero();
    x2_grad_data.setZero();

    // Create TensorMap for x1, x2, y_grad, x1_grad, and x2_grad
    TTypes<T>::ConstMatrix x1(x1_data.data(), x1_data.dimensions());
    TTypes<T>::ConstMatrix x2(x2_data.data(), x2_data.dimensions());
    TTypes<T>::ConstMatrix y_grad(y_grad_data.data(), y_grad_data.dimensions());
    TTypes<T>::Matrix x1_grad(x1_grad_data.data(), x1_grad_data.dimensions());
    TTypes<T>::Matrix x2_grad(x2_grad_data.data(), x2_grad_data.dimensions());

    // Call the Backward function
    nn::MatMul::Backward(x1, x2, y_grad, x1_grad, x2_grad);

    // Verify the output for x1_grad (removed T() conversion)
    Eigen::Tensor<T, 2> expected_x1_grad(2, 3);
    expected_x1_grad(0, 0) = 23.0f; expected_x1_grad(0, 1) = 29.0f; expected_x1_grad(0, 2) = 35.0f;
    expected_x1_grad(1, 0) = 31.0f; expected_x1_grad(1, 1) = 39.0f; expected_x1_grad(1, 2) = 47.0f;

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(x1_grad(i, j), expected_x1_grad(i, j), EPSILON);
        }
    }

    // Verify the output for x2_grad (removed T() conversion)
    Eigen::Tensor<T, 2> expected_x2_grad(3, 2);
    expected_x2_grad(0, 0) = 14.0f; expected_x2_grad(0, 1) = 20.0f;
    expected_x2_grad(1, 0) = 32.0f; expected_x2_grad(1, 1) = 44.0f;
    expected_x2_grad(2, 0) = 50.0f; expected_x2_grad(2, 1) = 68.0f;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_NEAR(x2_grad(i, j), expected_x2_grad(i, j), EPSILON);
        }
    }
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}