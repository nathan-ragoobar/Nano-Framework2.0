#pragma once
#include <gtest/gtest.h>
#include "./fastfeedforward.hpp"
#include "./../tensor/tensor_util.hpp"


constexpr float EPSILON = 1e-4;

class AutogradTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }
};

TEST_F(AutogradTest, SigmoidForward) {
    using Type = float;
    
    // Initialize input tensor with known values
    Eigen::Tensor<Type, 2> x_data(2, 2);
    x_data.setValues({{-1.0f, 0.0f},
                      { 1.0f, 2.0f}});
    
    // Create TensorMap for input
    TTypes<Type, 2>::ConstTensor x(x_data.data(), x_data.dimensions());
    
    // Create output tensor
    Eigen::Tensor<Type, 2> y_data(2, 2);
    TTypes<Type, 2>::Tensor y(y_data.data(), y_data.dimensions());
    
    // Create and apply sigmoid function
    nn::SigmoidFunction<Type> sigmoid;
    y = sigmoid.forward(x);
    
    // Expected values (computed using 1/(1 + exp(-x)))
    Eigen::Tensor<Type, 2> expected(2, 2);
    expected.setValues({{0.26894142f, 0.5f},
                       {0.73105858f, 0.88079708f}});
    
    // Verify results
    for (int i = 0; i < y.dimension(0); ++i) {
        for (int j = 0; j < y.dimension(1); ++j) {
            EXPECT_NEAR(y(i, j), expected(i, j), EPSILON);
        }
    }
}

TEST_F(AutogradTest, SigmoidBackward) {
    using Type = float;
    
    // Initialize tensors
    Eigen::Tensor<Type, 2> x_data(2, 2);
    Eigen::Tensor<Type, 2> y_data(2, 2);
    Eigen::Tensor<Type, 2> grad_output_data(2, 2);
    Eigen::Tensor<Type, 2> grad_input_data(2, 2);
    
    x_data.setValues({{-1.0f, 0.0f},
                      { 1.0f, 2.0f}});
    grad_output_data.setConstant(1.0f);
    grad_input_data.setZero();
    
    // Create TensorMaps
    TTypes<Type, 2>::ConstTensor x(x_data.data(), x_data.dimensions());
    TTypes<Type, 2>::Tensor y(y_data.data(), y_data.dimensions());
    TTypes<Type, 2>::ConstTensor grad_output(grad_output_data.data(), grad_output_data.dimensions());
    TTypes<Type, 2>::Tensor grad_input(grad_input_data.data(), grad_input_data.dimensions());
    
    // Run forward and backward passes
    nn::SigmoidFunction<Type> sigmoid;
    y = sigmoid.forward(x);
    sigmoid.backward(grad_output, );
    
    // Expected gradients: sigmoid(x) * (1 - sigmoid(x))
    Eigen::Tensor<Type, 2> expected_grad(2, 2);
    expected_grad.setValues({{0.196611933f, 0.25f},
                            {0.196611933f, 0.104994f}});
    
    // Verify gradients
    for (int i = 0; i < grad_input.dimension(0); ++i) {
        for (int j = 0; j < grad_input.dimension(1); ++j) {
            EXPECT_NEAR(grad_input(i, j), expected_grad(i, j), EPSILON);
        }
    }
}

TEST_F(AutogradTest, AutogradContextSaving) {
    using Type = float;
    
    // Initialize tensors
    Eigen::Tensor<Type, 2> x_data(2, 2);
    x_data.setRandom();
    
    // Create context
    nn::AutogradContext<Type> ctx;
    
    // Save tensor
    TTypes<Type, 2>::ConstTensor x(x_data.data(), x_data.dimensions());
    ctx.save_for_backward(x);
    
    // Retrieve and verify
    auto retrieved = ctx.get_saved_tensor(0);
    
    for (int i = 0; i < x.dimension(0); ++i) {
        for (int j = 0; j < x.dimension(1); ++j) {
            EXPECT_EQ(retrieved(i, j), x(i, j));
        }
    }
}

TEST_F(AutogradTest, ChainedOperations) {
    using Type = float;
    
    // Test sigmoid(sigmoid(x))
    Eigen::Tensor<Type, 1> x_data(2);
    x_data.setValues({1.0f, 2.0f});
    
    TTypes<Type, 1>::ConstTensor x(x_data.data(), x_data.dimensions());
    Eigen::Tensor<Type, 1> h1_data(2);
    Eigen::Tensor<Type, 1> h2_data(2);
    TTypes<Type, 1>::Tensor h1(h1_data.data(), h1_data.dimensions());
    TTypes<Type, 1>::Tensor h2(h2_data.data(), h2_data.dimensions());
    
    // Forward passes
    nn::SigmoidFunction<Type> sigmoid1, sigmoid2;
    sigmoid1.Forward(x, h1);
    sigmoid2.Forward(h1, h2);
    
    // Backward passes
    Eigen::Tensor<Type, 1> grad_h2(2);
    Eigen::Tensor<Type, 1> grad_h1(2);
    Eigen::Tensor<Type, 1> grad_x(2);
    grad_h2.setConstant(1.0f);
    grad_h1.setZero();
    grad_x.setZero();
    
    TTypes<Type, 1>::ConstTensor grad_output(grad_h2.data(), grad_h2.dimensions());
    TTypes<Type, 1>::Tensor grad_h1_tensor(grad_h1.data(), grad_h1.dimensions());
    TTypes<Type, 1>::Tensor grad_x_tensor(grad_x.data(), grad_x.dimensions());
    
    sigmoid2.Backward(grad_output, grad_h1_tensor);
    sigmoid1.Backward(grad_h1_tensor, grad_x_tensor);
    
    // Verify gradients flow through chain
    EXPECT_GT(grad_x_tensor(0), 0.0f);
    EXPECT_GT(grad_x_tensor(1), 0.0f);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}