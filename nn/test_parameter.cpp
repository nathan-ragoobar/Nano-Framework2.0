#include "./../tensor/fixed_point.hpp"
#include "./Parameter.hpp"
#include <gtest/gtest.h>

using namespace nn;

TEST(ConstantFillTest, FixedPointFill) {
    // Setup
    std::vector<FixedPointQ5_10> data(10);
    absl::Span<FixedPointQ5_10> span(data);
    
    // Test fill with 1.5
    ConstantFill(span, FixedPointQ5_10(1.5f));
    
    // Verify
    for(const auto& val : span) {
        EXPECT_EQ(val.toFloat(), 1.5f);
    }
}

TEST(UniformFillTest, FloatRangeTest) {
    // Setup
    std::vector<float> data(1000);
    absl::Span<float> span(data);
    float min_val = -1.0f;
    float max_val = 1.0f;
    
    // Set random seed for reproducibility
    nn::ManualSeed(42);
    
    // Fill with random values
    nn::UniformFill(span, min_val, max_val);
    
    // Verify bounds
    for(const auto& val : span) {
        EXPECT_GE(val, min_val);
        EXPECT_LE(val, max_val);
    }
    
    // Verify randomness (values aren't all the same)
    float first_val = span[0];
    bool all_same = true;
    for(size_t i = 1; i < span.size(); i++) {
        if(span[i] != first_val) {
            all_same = false;
            break;
        }
    }
    EXPECT_FALSE(all_same);
}

TEST(UniformFillTest, FixedPointRangeTest) {
    // Setup
    std::vector<FixedPointQ5_10> data(1000);
    absl::Span<FixedPointQ5_10> span(data);
    FixedPointQ5_10 min_val(-1.0f);
    FixedPointQ5_10 max_val(1.0f);
    
    // Set seed
    nn::ManualSeed(42);
    
    // Fill with random values
    nn::UniformFill(span, min_val, max_val);
    
    // Verify bounds
    for(const auto& val : span) {
        EXPECT_GE(val, min_val);
        EXPECT_LE(val, max_val);
    }
    
    // Verify randomness
    FixedPointQ5_10 first_val = span[0];
    bool all_same = true;
    for(size_t i = 1; i < span.size(); i++) {
        if(span[i] != first_val) {
            all_same = false;
            break;
        }
    }
    EXPECT_FALSE(all_same);
}

TEST(NormalFillTest, FixedPointDistribution) {
    std::vector<FixedPointQ5_10> data(1000);
    absl::Span<FixedPointQ5_10> span(data);
    
    FixedPointQ5_10 mean(0.0f);
    FixedPointQ5_10 std(1.0f);
    
    nn::ManualSeed(42);
    nn::NormalFill(span, mean, std);
    
    // Verify distribution properties
    float sum = 0.0f;
    for(const auto& val : span) {
        sum += val.toFloat();
    }
    float empirical_mean = sum / span.size();
    
    EXPECT_NEAR(empirical_mean, 0.0f, 0.1f);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}