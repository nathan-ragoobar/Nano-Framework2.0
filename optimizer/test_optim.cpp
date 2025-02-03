#include <gtest/gtest.h>
#include "./optim.hpp"
#include "./../nn/Parameter.hpp"
#include "./../tensor/fixed_point.hpp"

class OptimizerTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(OptimizerTest, SGDFixedPoint) {
    // 1. Create parameters
    const int num_params = 4;
    std::vector<nn::Parameter*> parameters;
    auto param1 = std::make_unique<nn::Parameter>(nn::DT_FIXED, num_params);
    auto param2 = std::make_unique<nn::Parameter>(nn::DT_FIXED, num_params);
    
    // 2. Set initial values
    auto data1 = param1->flat<fixed_point_7pt8>();
    data1.device(nn::g_device) = data1.constant(fixed_point_7pt8(1.0f));
    
    auto data2 = param2->flat<fixed_point_7pt8>();
    data2.device(nn::g_device) = data2.constant(fixed_point_7pt8(2.0f));

    // 3. Create gradients
    param1->LazyAllocateGradient();
    param2->LazyAllocateGradient();
    
    auto grad1 = param1->flat_grad<fixed_point_7pt8>();
    grad1.device(nn::g_device) = grad1.constant(fixed_point_7pt8(0.1f));
    
    auto grad2 = param2->flat_grad<fixed_point_7pt8>();
    grad2.device(nn::g_device) = grad2.constant(fixed_point_7pt8(0.2f));

    parameters.push_back(param1.get());
    parameters.push_back(param2.get());

    // 4. Create optimizer
    float learning_rate = 0.5f;
    optim::SGD<fixed_point_7pt8> optimizer(parameters, learning_rate);

    // 5. Take one optimization step
    optimizer.Step();

    // 6. Verify updates
    auto updated_data1 = param1->flat<fixed_point_7pt8>();
    auto updated_data2 = param2->flat<fixed_point_7pt8>();

    fixed_point_7pt8 expected1 = fixed_point_7pt8(1.0f) - learning_rate * fixed_point_7pt8(0.1f);
    fixed_point_7pt8 expected2 = fixed_point_7pt8(2.0f) - learning_rate * fixed_point_7pt8(0.2f);

    for(int i = 0; i < num_params; i++) {
        EXPECT_FLOAT_EQ(updated_data1.data()[i].to_float(), expected1.to_float());
        EXPECT_FLOAT_EQ(updated_data2.data()[i].to_float(), expected2.to_float());
    }

    // 7. Test ZeroGrad
    optimizer.ZeroGrad();
    auto zeroed_grad1 = param1->flat_grad<fixed_point_7pt8>();
    auto zeroed_grad2 = param2->flat_grad<fixed_point_7pt8>();

    for(int i = 0; i < num_params; i++) {
        EXPECT_FLOAT_EQ(zeroed_grad1.data()[i].to_float(), 0.0f);
        EXPECT_FLOAT_EQ(zeroed_grad2.data()[i].to_float(), 0.0f);
    }
}

TEST_F(OptimizerTest, AdamWFixedPoint) {
    // 1. Setup parameters
    const int num_params = 4;
    std::vector<nn::Parameter*> parameters;
    auto param1 = std::make_unique<nn::Parameter>(nn::DT_FIXED, num_params);
    
    // 2. Initialize parameter values
    auto data1 = param1->flat<fixed_point_7pt8>();
    data1.device(nn::g_device) = data1.constant(fixed_point_7pt8(1.0f));
    
    // 3. Setup gradients
    param1->LazyAllocateGradient();
    auto grad1 = param1->flat_grad<fixed_point_7pt8>();
    grad1.device(nn::g_device) = grad1.constant(fixed_point_7pt8(0.1f));
    
    parameters.push_back(param1.get());

    // 4. Create optimizer
    float lr = 0.001f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float weight_decay = 0.01f;
    
    optim::AdamW<fixed_point_7pt8> optimizer(parameters, lr, beta1, beta2, eps, weight_decay);

    // 5. Run multiple steps
    for(int t = 1; t <= 3; t++) {
        optimizer.Step(t);
        
        // 6. Verify updates
        auto updated_data = param1->flat<fixed_point_7pt8>();
        
        // Values will change based on optimizer step, weight decay and momentum
        EXPECT_NE(updated_data.data()[0].to_float(), 1.0f);
        
        // Verify gradients still exist
        auto current_grad = param1->flat_grad<fixed_point_7pt8>();
        EXPECT_EQ(current_grad.data()[0].to_float(), 0.1f);
    }

    // 7. Test gradient zero
    optimizer.ZeroGrad();
    auto zeroed_grad = param1->flat_grad<fixed_point_7pt8>();
    for(int i = 0; i < num_params; i++) {
        EXPECT_FLOAT_EQ(zeroed_grad.data()[i].to_float(), 0.0f);
    }
}