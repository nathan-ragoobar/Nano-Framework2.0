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

TEST(KaimingUniformFillTest, FixedPointTest) {
    std::vector<FixedPointQ5_10> data(100);
    absl::Span<FixedPointQ5_10> span(data);
    
    nn::ManualSeed(42);
    nn::KaimingUniformFill(span, 10);
    
    FixedPointQ5_10 expected_bound = FixedPointQ5_10::sqrt(FixedPointQ5_10(0.1f));
    
    for(const auto& val : span) {
        EXPECT_TRUE(val >= -expected_bound);
        EXPECT_TRUE(val <= expected_bound);
    }
}

TEST(UpperTriangularTest, FixedPointTest) {
    // Create tensor instead of matrix
    const int size = 3;

    // Q5.10 format has range [-16, 15.999]
    constexpr float kMinValue = -16.0f;  // Changed from -32.0f

    std::vector<FixedPointQ5_10> data(size * size);
    
    // Create tensor map from data
    TTypes<FixedPointQ5_10, 2>::Tensor matrix(data.data(), size, size);
    
    // Initialize to zero
    for (int i = 0; i < size * size; i++) {
        data[i] = FixedPointQ5_10(0.0f);
    }

    UpperTriangularWithNegativeInf(matrix);
    
    // Check diagonal and lower triangle are zero
    for(int i = 0; i < size; i++) {
        for(int j = 0; j <= i; j++) {
            EXPECT_EQ(matrix(i,j).toFloat(), 0.0f);
        }
    }
    
    // Check upper triangle is minimum value
    for(int i = 0; i < size; i++) {
        for(int j = i + 1; j < size; j++) {
            EXPECT_EQ(matrix(i,j).toFloat(), kMinValue);
        }
    }
}

TEST(OneHotTest, FixedPointTest) {
    const int batch_size = 3;
    const int num_classes = 4;
    
    std::vector<int> target_data = {1, 0, 2};
    std::vector<FixedPointQ5_10> label_data(batch_size * num_classes, FixedPointQ5_10(0.0f));
    
    TTypes<int>::ConstFlat target(target_data.data(), batch_size);
    TTypes<FixedPointQ5_10>::Matrix label(label_data.data(), batch_size, num_classes);
    
    OneHot(target, label);
    
    // Verify results
    for(int i = 0; i < batch_size; i++) {
        for(int j = 0; j < num_classes; j++) {
            if(j == target_data[i]) {
                EXPECT_EQ(label(i,j).toFloat(), 1.0f);
            } else {
                EXPECT_EQ(label(i,j).toFloat(), 0.0f);
            }
        }
    }
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}