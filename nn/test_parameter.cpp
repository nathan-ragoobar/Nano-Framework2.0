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