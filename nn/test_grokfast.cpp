#include "grokfast.hpp"
#include <gtest/gtest.h>
#include "./grokfast.hpp"
//#include "./../tensor/fixed_point.hpp"

constexpr float EPSILON = 1e-4;

class GradientFilterTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup test parameters
        using Type = float;
        param1_ = std::make_unique<nn::Parameter>(DT_FLOAT, 4); // 2x2 tensor
        param2_ = std::make_unique<nn::Parameter>(DT_FLOAT, 4); // 2x2 tensor
        
        // Initialize parameter data
        auto p1_data = param1_->matrix<Type>(2, 2);
        auto p2_data = param2_->matrix<Type>(2, 2);
        p1_data.setValues({{1.0f, 2.0f}, {3.0f, 4.0f}});
        p2_data.setValues({{5.0f, 6.0f}, {7.0f, 8.0f}});
        
        // Initialize gradients
        param1_->LazyAllocateGradient();
        param2_->LazyAllocateGradient();
        auto p1_grad = param1_->matrix_grad<Type>(2, 2);
        auto p2_grad = param2_->matrix_grad<Type>(2, 2);
        p1_grad.setValues({{0.1f, 0.2f}, {0.3f, 0.4f}});
        p2_grad.setValues({{0.5f, 0.6f}, {0.7f, 0.8f}});
        
        parameters_.push_back(param1_.get());
        parameters_.push_back(param2_.get());
    }

    std::unique_ptr<nn::Parameter> param1_;
    std::unique_ptr<nn::Parameter> param2_;
    std::vector<nn::Parameter*> parameters_;
};

TEST_F(GradientFilterTest, MAWithoutWarmup) {
    using Type = float;
    nn::GradientFilter<Type> filter(2, 1.0f, nn::FilterType::MEAN, false);
    
    // Store original gradient
    auto orig_grad = param1_->const_matrix_grad<Type>(2, 2);
    Eigen::Tensor<Type, 2> orig_grad_copy = orig_grad;
    
    // Apply moving average
    filter.ApplyMA(parameters_);
    
    // Verify gradients changed
    auto new_grad = param1_->const_matrix_grad<Type>(2, 2);
    bool gradients_changed = false;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            if (std::abs(new_grad(i,j).to_float() - orig_grad_copy(i,j).to_float()) > EPSILON) {
                gradients_changed = true;
                break;
            }
        }
    }
    EXPECT_TRUE(gradients_changed);
}

TEST_F(GradientFilterTest, MAWithWarmup) {
    using Type = float;
    nn::GradientFilter<Type> filter(2, 1.0f, nn::FilterType::MEAN, true);
    
    // Store original gradient
    auto orig_grad = param1_->const_matrix_grad<Type>(2, 2);
    Eigen::Tensor<Type, 2> orig_grad_copy = orig_grad;
    
    // First application (warmup)
    filter.ApplyMA(parameters_);
    
    // Verify gradients unchanged during warmup
    auto warmup_grad = param1_->const_matrix_grad<Type>(2, 2);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_NEAR(warmup_grad(i,j).to_float(), orig_grad_copy(i,j).to_float(), EPSILON);
        }
    }
    
    // Second application (after warmup)
    filter.ApplyMA(parameters_);
    
    // Verify gradients changed after warmup
    auto final_grad = param1_->const_matrix_grad<Type>(2, 2);
    bool gradients_changed = false;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            if (std::abs(final_grad(i,j).to_float() - orig_grad_copy(i,j).to_float()) > EPSILON) {
                gradients_changed = true;
                break;
            }
        }
    }
    EXPECT_TRUE(gradients_changed);
}

TEST_F(GradientFilterTest, EMABehavior) {
    using Type = float;
    nn::GradientFilter<Type> filter;
    
    // Store original gradient
    auto orig_grad = param1_->const_matrix_grad<Type>(2, 2);
    Eigen::Tensor<Type, 2> orig_grad_copy = orig_grad;
    
    // Apply EMA
    filter.ApplyEMA(parameters_, 0.98f, 1.0f);
    
    // Verify gradients changed
    auto new_grad = param1_->const_matrix_grad<Type>(2, 2);
    bool gradients_changed = false;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            if (std::abs(new_grad(i,j).to_float() - orig_grad_copy(i,j).to_float()) > EPSILON) {
                gradients_changed = true;
                break;
            }
        }
    }
    EXPECT_TRUE(gradients_changed);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}