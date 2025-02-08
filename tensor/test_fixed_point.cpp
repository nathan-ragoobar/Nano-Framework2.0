#include <gtest/gtest.h>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include "fixed_point.hpp"
#include <cmath>

class FixedPointTest : public ::testing::Test {
protected:
    static constexpr float EPSILON = 0.001f; // Tolerance for floating-point comparisons

    
};

// Constructor Tests
TEST_F(FixedPointTest, DefaultConstructor) {
    fixed_point_7pt8 fp;
    EXPECT_FLOAT_EQ(fp.to_float(), 0.0f);
}

TEST_F(FixedPointTest, IntegerConstructor) {
    fixed_point_7pt8 fp1(5);
    EXPECT_NEAR(fp1.to_float(), 5.0f, EPSILON);

    fixed_point_7pt8 fp2(-5);
    EXPECT_NEAR(fp2.to_float(), -5.0f, EPSILON);

   // Test clamping
    fixed_point_7pt8 fp3(129);  // Should clamp to max value
    EXPECT_NEAR(fp3.to_float(), fixed_point_7pt8::value_max().to_float(), EPSILON);

    fixed_point_7pt8 fp4(-129);  // Should clamp to min value
    EXPECT_NEAR(fp4.to_float(), fixed_point_7pt8::value_min().to_float(), EPSILON);
}

TEST_F(FixedPointTest, FloatConstructor) {
    fixed_point_7pt8 fp1(5.5f);
    EXPECT_NEAR(fp1.to_float(), 5.5f, EPSILON);

    fixed_point_7pt8 fp2(-5.5f);
    EXPECT_NEAR(fp2.to_float(), -5.5f, EPSILON);

    // Test fractional precision
    fixed_point_7pt8 fp3(0.125f);
    EXPECT_NEAR(fp3.to_float(), 0.125f, EPSILON);

    // Test clamping
    fixed_point_7pt8 fp4(129.0f);  // Should clamp to ~15.999
    EXPECT_LT(fp4.to_float(), 127.0f);
    EXPECT_GT(fp4.to_float(), 128.9f);

    fixed_point_7pt8 fp5(-129.0f);  // Should clamp to -16
    EXPECT_NEAR(fp5.to_float(), fixed_point_7pt8::value_min().to_float(), EPSILON);
}

//Conversion Tests
TEST_F(FixedPointTest, ConversionOperations) {
    // Test to_float
    {
        fixed_point_7pt8 x(1.5f);
        EXPECT_NEAR(x.to_float(), 1.5f, EPSILON);
        
        x = fixed_point_7pt8(-1.5f);
        EXPECT_NEAR(x.to_float(), -1.5f, EPSILON);
        
        x = fixed_point_7pt8(0.0f);
        EXPECT_NEAR(x.to_float(), 0.0f, EPSILON);
    }

    // Test to_double
    {
        fixed_point_7pt8 x(1.5);
        EXPECT_NEAR(x.to_double(), 1.5, EPSILON);
        
        x = fixed_point_7pt8(-1.5);
        EXPECT_NEAR(x.to_double(), -1.5, EPSILON);
        
        x = fixed_point_7pt8(0.0);
        EXPECT_NEAR(x.to_double(), 0.0, EPSILON);
    }

    // Test to_long_double
    {
        fixed_point_7pt8 x(1.5L);
        EXPECT_NEAR(x.to_long_double(), 1.5L, EPSILON);
        
        x = fixed_point_7pt8(-1.5L);
        EXPECT_NEAR(x.to_long_double(), -1.5L, EPSILON);
        
        x = fixed_point_7pt8(0.0L);
        EXPECT_NEAR(x.to_long_double(), 0.0L, EPSILON);
    }

    // Test to_int
    {
        fixed_point_7pt8 x(1.7f);
        EXPECT_EQ(x.to_int(), 1);
        
        x = fixed_point_7pt8(-1.7f);
        EXPECT_EQ(x.to_int(), -1);
        
        x = fixed_point_7pt8(0.3f);
        EXPECT_EQ(x.to_int(), 0);
    }

    // Test to_int8
    {
        fixed_point_7pt8 x(100.0f);
        EXPECT_EQ(x.to_int8(), 100);
        
        x = fixed_point_7pt8(-100.0f);
        EXPECT_EQ(x.to_int8(), -100);
        
        x = fixed_point_7pt8(0.0f);
        EXPECT_EQ(x.to_int8(), 0);
    }

    // Test to_int16
    {
        fixed_point_7pt8 x(100.0f);
        EXPECT_EQ(x.to_int16(), 100);
        
        x = fixed_point_7pt8(-100.0f);
        EXPECT_EQ(x.to_int16(), -100);
        
        x = fixed_point_7pt8(0.0f);
        EXPECT_EQ(x.to_int16(), 0);
    }

    // Test to_int32
    {
        fixed_point_7pt8 x(100.0f);
        EXPECT_EQ(x.to_int32(), 100);
        
        x = fixed_point_7pt8(-100.0f);
        EXPECT_EQ(x.to_int32(), -100);
        
        x = fixed_point_7pt8(0.0f);
        EXPECT_EQ(x.to_int32(), 0);
    }

    // Test Edge Cases
    {
        // Test maximum values
        fixed_point_7pt8 max_val = fixed_point_7pt8::value_max();
        EXPECT_GT(max_val.to_float(), 0.0f);
        
        // Test minimum values
        fixed_point_7pt8 min_val = fixed_point_7pt8::value_min();
        EXPECT_LT(min_val.to_float(), 0.0f);
        
        // Test epsilon
        fixed_point_7pt8 eps_val = fixed_point_7pt8::value_eps();
        EXPECT_GT(eps_val.to_float(), 0.0f);
        EXPECT_LT(eps_val.to_float(), 1.0f);
    }

    // Test Precision
    {
        fixed_point_7pt8 x(1.234567f);
        EXPECT_NEAR(x.to_float(), 1.234567f, 0.01f);
        EXPECT_NEAR(x.to_double(), 1.234567, 0.01);
        EXPECT_EQ(x.to_int(), 1);
    }

    // Test Rounding
    {
        fixed_point_7pt8 x(1.5f);
        EXPECT_EQ(x.to_int(), 1);  // Should truncate, not round
        
        x = fixed_point_7pt8(1.9f);
        EXPECT_EQ(x.to_int(), 1);  // Should truncate, not round
        
        x = fixed_point_7pt8(-1.5f);
        EXPECT_EQ(x.to_int(), -1); // Should truncate towards zero
    }
}





using fixed_point_7pt8 = fixed_point_7pt8;

// Arithmetic Operation Tests
TEST_F(FixedPointTest, Addition) {
    fixed_point_7pt8 fp1(3.5f);
    fixed_point_7pt8 fp2(2.25f);
    fixed_point_7pt8 result = fp1 + fp2;
    EXPECT_NEAR(result.to_float(), 5.75f, EPSILON);

    // Test addition near bounds
    fixed_point_7pt8 fp3(10.0f);
    fixed_point_7pt8 fp4(5.0f);
    result = fp3 + fp4;
    EXPECT_NEAR(result.to_float(), 15.0f, EPSILON);

    // Test negative addition
    fixed_point_7pt8 fp5(-3.5f);
    fixed_point_7pt8 fp6(2.25f);
    result = fp5 + fp6;
    EXPECT_NEAR(result.to_float(), -1.25f, EPSILON);
}

TEST_F(FixedPointTest, Subtraction) {
    fixed_point_7pt8 fp1(3.5f);
    fixed_point_7pt8 fp2(2.25f);
    fixed_point_7pt8 result = fp1 - fp2;
    EXPECT_NEAR(result.to_float(), 1.25f, EPSILON);

    // Test subtraction with negative results
    fixed_point_7pt8 fp3(2.25f);
    fixed_point_7pt8 fp4(3.5f);
    result = fp3 - fp4;
    EXPECT_NEAR(result.to_float(), -1.25f, EPSILON);
}

TEST_F(FixedPointTest, UnaryMinusOperator) {
    fixed_point_7pt8 a(1.5f);
    fixed_point_7pt8 negA = -a;
    EXPECT_NEAR(negA.to_float(), -1.5f, EPSILON);
}

TEST_F(FixedPointTest, Multiplication) {
    fixed_point_7pt8 fp1(3.5f);
    fixed_point_7pt8 fp2(2.0f);
    fixed_point_7pt8 result = fp1 * fp2;
    EXPECT_NEAR(result.to_float(), 7.0f, EPSILON);

    // Test multiplication with fractional numbers
    fixed_point_7pt8 fp3(2.5f);
    fixed_point_7pt8 fp4(0.5f);
    result = fp3 * fp4;
    EXPECT_NEAR(result.to_float(), 1.25f, EPSILON);

    // Test multiplication with negative numbers
    fixed_point_7pt8 fp5(-2.5f);
    fixed_point_7pt8 fp6(2.0f);
    result = fp5 * fp6;
    EXPECT_NEAR(result.to_float(), -5.0f, EPSILON);
}

TEST_F(FixedPointTest, Division) {
    fixed_point_7pt8 fp1(3.5f);
    fixed_point_7pt8 fp2(2.0f);
    fixed_point_7pt8 result = fp1 / fp2;
    EXPECT_NEAR(result.to_float(), 1.75f, EPSILON);

    // Test division with fractional numbers
    fixed_point_7pt8 fp3(2.5f);
    fixed_point_7pt8 fp4(0.5f);
    result = fp3 / fp4;
    EXPECT_NEAR(result.to_float(), 5.0f, EPSILON);

    // Test division with negative numbers
    fixed_point_7pt8 fp5(-2.5f);
    fixed_point_7pt8 fp6(2.0f);
    result = fp5 / fp6;
    EXPECT_NEAR(result.to_float(), -1.25f, EPSILON);
}

// Stream Output Test
TEST_F(FixedPointTest, StreamOutput) {
    fixed_point_7pt8 fp(3.5f);
    std::stringstream ss;
    ss << fp;
    float value;
    ss >> value;
    EXPECT_NEAR(value, 3.5f, EPSILON);
}

// Test the assignment operator
TEST_F(FixedPointTest, AssignmentOperator) {
    fixed_point_7pt8 fp1(5.5f);
    fixed_point_7pt8 fp2;
    fp2 = fp1;
    EXPECT_NEAR(fp2.to_float(), 5.5f, EPSILON);

    fixed_point_7pt8 fp3(-3.75f);
    fp2 = fp3;
    EXPECT_NEAR(fp2.to_float(), -3.75f, EPSILON);

    // Test self-assignment
    fp2 = fp2;
    EXPECT_NEAR(fp2.to_float(), -3.75f, EPSILON);
}

// Equality Operator Test
TEST_F(FixedPointTest, EqualityOperator) {
    fixed_point_7pt8 fp1(3.5f);
    fixed_point_7pt8 fp2(3.5f);
    fixed_point_7pt8 fp3(2.25f);

    EXPECT_TRUE(fp1 == fp2);
    EXPECT_FALSE(fp1 == fp3);

    // Test self-equality
    EXPECT_TRUE(fp1 == fp1);
}
/* */
// Less Than Operator Test
TEST_F(FixedPointTest, LessThanOperator) {
    fixed_point_7pt8 fp1(3.5f);
    fixed_point_7pt8 fp2(4.5f);

    EXPECT_TRUE(fp1 < fp2);
    EXPECT_FALSE(fp2 < fp1);
    EXPECT_FALSE(fp1 < fp1);
}

// Greater Than Operator Test
TEST_F(FixedPointTest, GreaterThanOperator) {
    fixed_point_7pt8 fp1(3.5f);
    fixed_point_7pt8 fp2(4.5f);

    EXPECT_TRUE(fp2 > fp1);
    EXPECT_FALSE(fp1 > fp2);
    EXPECT_FALSE(fp1 > fp1);
}

// Addition Assignment Operator Test
TEST_F(FixedPointTest, AdditionAssignmentOperator) {
    fixed_point_7pt8 fp1(3.5f);
    fixed_point_7pt8 fp2(2.25f);
    fp1 += fp2;
    EXPECT_NEAR(fp1.to_float(), 5.75f, EPSILON);

    // Test self-addition
    fp1 += fp1;
    EXPECT_NEAR(fp1.to_float(), 11.5f, EPSILON);
}

// Subtraction Assignment Operator Test
TEST_F(FixedPointTest, SubtractionAssignmentOperator) {
    fixed_point_7pt8 fp1(3.5f);
    fixed_point_7pt8 fp2(2.25f);
    fp1 -= fp2;
    EXPECT_NEAR(fp1.to_float(), 1.25f, EPSILON);

    // Test self-subtraction
    fp1 -= fp1;
    EXPECT_NEAR(fp1.to_float(), 0.0f, EPSILON);
}

TEST_F(FixedPointTest, NotEqualOperator) {
    fixed_point_7pt8 a(1.5f);
    fixed_point_7pt8 b(2.0f);
    fixed_point_7pt8 c(1.5f);
    
    EXPECT_TRUE(a != b);
    EXPECT_FALSE(a != c);
}

TEST_F(FixedPointTest, GreaterThanOrEqualOperator) {
    fixed_point_7pt8 a(1.5f);
    fixed_point_7pt8 b(1.0f);
    fixed_point_7pt8 c(1.5f);
    
    // Greater than case
    EXPECT_TRUE(a >= b);
    
    // Equal case
    EXPECT_TRUE(a >= c);
    
    // Less than case
    EXPECT_FALSE(b >= a);
}

TEST_F(FixedPointTest, LessThanOrEqualOperator) {
    fixed_point_7pt8 a(1.5f);
    fixed_point_7pt8 b(2.0f);
    fixed_point_7pt8 c(1.5f);
    
    // Less than case
    EXPECT_TRUE(a <= b);
    
    // Equal case
    EXPECT_TRUE(a <= c);
    
    // Greater than case
    EXPECT_FALSE(b <= a);
}

TEST_F(FixedPointTest, MathFunctions) {
    fixed_point_7pt8 x(4.0f);
    
    
    EXPECT_NEAR(sqrt(x).to_float(), 2.0f, EPSILON);
    
    x = fixed_point_7pt8(1.0f);
    EXPECT_NEAR(log(x).to_float(), 0.0f, EPSILON);
    
    EXPECT_NEAR(sin(x).to_float(), std::sin(1.0f), 10*EPSILON);
    EXPECT_NEAR(cos(x).to_float(), std::cos(1.0f), 10*EPSILON);
}

TEST_F(FixedPointTest, LogFunction) {
    static constexpr float EPSILON = 0.15f;
    // Basic cases
    fixed_point_7pt8 x(1.0f);
    EXPECT_NEAR(log(x).to_float(), 0.0f, EPSILON);
    
    x = fixed_point_7pt8(2.718281828f);  // e
    EXPECT_NEAR(log(x).to_float(), 1.0f, EPSILON);
    
    x = fixed_point_7pt8(2.0f);
    EXPECT_NEAR(log(x).to_float(), 0.693147f, EPSILON);
    
    x = fixed_point_7pt8(0.5f);
    EXPECT_NEAR(log(x).to_float(), -0.693147f, EPSILON);
    
    x = fixed_point_7pt8(10.0f);
    EXPECT_NEAR(log(x).to_float(), 2.302585f, EPSILON);

    // Edge cases
    x = fixed_point_7pt8(0.1f);
    EXPECT_NEAR(log(x).to_float(), -2.302585f, EPSILON);
    
    x = fixed_point_7pt8(100.0f);
    EXPECT_NEAR(log(x).to_float(), 4.605170f, EPSILON);

    // Small value test
    x = fixed_point_7pt8(0.01f);
    EXPECT_NEAR(log(x).to_float(), -4.605170f, EPSILON);

    // Verify error handling for invalid inputs
    #ifdef NDEBUG
    x = fixed_point_7pt8(0.0f);
    EXPECT_DEATH(log(x), "");
    
    x = fixed_point_7pt8(-1.0f);
    EXPECT_DEATH(log(x), "");
    #endif
}

class FixedPointTanhTest : public ::testing::Test {
protected:
    static constexpr float tolerance = 0.05f; // 5% tolerance for approximation
};

TEST_F(FixedPointTanhTest, TanhHandlesZero) {
    fixed_point_7pt8 x(0.0f);
    EXPECT_FLOAT_EQ(tanh(x).to_float(), 0.0f);
}

TEST_F(FixedPointTanhTest, TanhHandlesPosValues) {
    float test_values[] = {0.5f, 1.0f, 2.0f, 3.0f};
    for(float val : test_values) {
        fixed_point_7pt8 x(val);
        float expected = std::tanh(val);
        float actual = tanh(x).to_float();
        EXPECT_NEAR(actual, expected, std::abs(expected * tolerance))
            << "Failed for value: " << val;
    }
}

TEST_F(FixedPointTanhTest, TanhHandlesNegValues) {
    float test_values[] = {-0.5f, -1.0f, -2.0f, -3.0f};
    for(float val : test_values) {
        fixed_point_7pt8 x(val);
        float expected = std::tanh(val);
        float actual = tanh(x).to_float();
        EXPECT_NEAR(actual, expected, std::abs(expected * tolerance))
            << "Failed for value: " << val;
    }
}

TEST_F(FixedPointTanhTest, TanhHandlesSaturation) {
    fixed_point_7pt8 pos_large(4.0f);
    fixed_point_7pt8 neg_large(-4.0f);
    
    EXPECT_NEAR(tanh(pos_large).to_float(), 1.0f, tolerance);
    EXPECT_NEAR(tanh(neg_large).to_float(), -1.0f, tolerance);
}

TEST_F(FixedPointTanhTest, TanhEigenIntegration) {
    fixed_point_7pt8 x(1.5f);
    float expected = std::tanh(1.5f);
    float actual = Eigen::numext::tanh(x).to_float();
    EXPECT_NEAR(actual, expected, std::abs(expected * tolerance));
}
/*
/*
TEST_F(FixedPointTest, TanhFunctionTest) {
    // Test with a few sample inputs
    float inputs[] = {-2.0f, -0.5f, 0.0f, 0.5f, 2.0f};
    for (float val : inputs) {
        fixed_point_7pt8 fp(val);
        fixed_point_7pt8 result = fp_ops::tanh(fp);
        float expected = std::tanh(val);
        EXPECT_NEAR(result.to_float(), expected, EPSILON);
    }
}
*/

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}