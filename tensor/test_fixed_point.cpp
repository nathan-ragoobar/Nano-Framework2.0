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
    FixedPointQ5_10 fp;
    EXPECT_FLOAT_EQ(fp.toFloat(), 0.0f);
}

TEST_F(FixedPointTest, IntegerConstructor) {
    FixedPointQ5_10 fp1(5);
    EXPECT_NEAR(fp1.toFloat(), 5.0f, EPSILON);

    FixedPointQ5_10 fp2(-5);
    EXPECT_NEAR(fp2.toFloat(), -5.0f, EPSILON);

    // Test clamping
    FixedPointQ5_10 fp3(20);  // Should clamp to 15
    EXPECT_NEAR(fp3.toFloat(), 15.0f, EPSILON);

    FixedPointQ5_10 fp4(-20);  // Should clamp to -16
    EXPECT_NEAR(fp4.toFloat(), -16.0f, EPSILON);
}

TEST_F(FixedPointTest, FloatConstructor) {
    FixedPointQ5_10 fp1(5.5f);
    EXPECT_NEAR(fp1.toFloat(), 5.5f, EPSILON);

    FixedPointQ5_10 fp2(-5.5f);
    EXPECT_NEAR(fp2.toFloat(), -5.5f, EPSILON);

    // Test fractional precision
    FixedPointQ5_10 fp3(0.125f);
    EXPECT_NEAR(fp3.toFloat(), 0.125f, EPSILON);

    // Test clamping
    FixedPointQ5_10 fp4(20.5f);  // Should clamp to ~15.999
    EXPECT_LT(fp4.toFloat(), 16.0f);
    EXPECT_GT(fp4.toFloat(), 15.9f);

    FixedPointQ5_10 fp5(-20.5f);  // Should clamp to -16
    EXPECT_NEAR(fp5.toFloat(), -16.0f, EPSILON);
}


// Arithmetic Operation Tests
TEST_F(FixedPointTest, Addition) {
    FixedPointQ5_10 fp1(3.5f);
    FixedPointQ5_10 fp2(2.25f);
    FixedPointQ5_10 result = fp1 + fp2;
    EXPECT_NEAR(result.toFloat(), 5.75f, EPSILON);

    // Test addition near bounds
    FixedPointQ5_10 fp3(10.0f);
    FixedPointQ5_10 fp4(5.0f);
    result = fp3 + fp4;
    EXPECT_NEAR(result.toFloat(), 15.0f, EPSILON);

    // Test negative addition
    FixedPointQ5_10 fp5(-3.5f);
    FixedPointQ5_10 fp6(2.25f);
    result = fp5 + fp6;
    EXPECT_NEAR(result.toFloat(), -1.25f, EPSILON);
}

TEST_F(FixedPointTest, Subtraction) {
    FixedPointQ5_10 fp1(3.5f);
    FixedPointQ5_10 fp2(2.25f);
    FixedPointQ5_10 result = fp1 - fp2;
    EXPECT_NEAR(result.toFloat(), 1.25f, EPSILON);

    // Test subtraction with negative results
    FixedPointQ5_10 fp3(2.25f);
    FixedPointQ5_10 fp4(3.5f);
    result = fp3 - fp4;
    EXPECT_NEAR(result.toFloat(), -1.25f, EPSILON);
}

TEST_F(FixedPointTest, UnaryMinusOperator) {
    FixedPointQ5_10 a(1.5f);
    FixedPointQ5_10 negA = -a;
    EXPECT_NEAR(negA.toFloat(), -1.5f, EPSILON);
}

TEST_F(FixedPointTest, Multiplication) {
    FixedPointQ5_10 fp1(3.5f);
    FixedPointQ5_10 fp2(2.0f);
    FixedPointQ5_10 result = fp1 * fp2;
    EXPECT_NEAR(result.toFloat(), 7.0f, EPSILON);

    // Test multiplication with fractional numbers
    FixedPointQ5_10 fp3(2.5f);
    FixedPointQ5_10 fp4(0.5f);
    result = fp3 * fp4;
    EXPECT_NEAR(result.toFloat(), 1.25f, EPSILON);

    // Test multiplication with negative numbers
    FixedPointQ5_10 fp5(-2.5f);
    FixedPointQ5_10 fp6(2.0f);
    result = fp5 * fp6;
    EXPECT_NEAR(result.toFloat(), -5.0f, EPSILON);
}

TEST_F(FixedPointTest, Division) {
    FixedPointQ5_10 fp1(3.5f);
    FixedPointQ5_10 fp2(2.0f);
    FixedPointQ5_10 result = fp1 / fp2;
    EXPECT_NEAR(result.toFloat(), 1.75f, EPSILON);

    // Test division with fractional numbers
    FixedPointQ5_10 fp3(2.5f);
    FixedPointQ5_10 fp4(0.5f);
    result = fp3 / fp4;
    EXPECT_NEAR(result.toFloat(), 5.0f, EPSILON);

    // Test division with negative numbers
    FixedPointQ5_10 fp5(-2.5f);
    FixedPointQ5_10 fp6(2.0f);
    result = fp5 / fp6;
    EXPECT_NEAR(result.toFloat(), -1.25f, EPSILON);
}

// Stream Output Test
TEST_F(FixedPointTest, StreamOutput) {
    FixedPointQ5_10 fp(3.5f);
    std::stringstream ss;
    ss << fp;
    float value;
    ss >> value;
    EXPECT_NEAR(value, 3.5f, EPSILON);
}

// Test the assignment operator
TEST_F(FixedPointTest, AssignmentOperator) {
    FixedPointQ5_10 fp1(5.5f);
    FixedPointQ5_10 fp2;
    fp2 = fp1;
    EXPECT_NEAR(fp2.toFloat(), 5.5f, EPSILON);

    FixedPointQ5_10 fp3(-3.75f);
    fp2 = fp3;
    EXPECT_NEAR(fp2.toFloat(), -3.75f, EPSILON);

    // Test self-assignment
    fp2 = fp2;
    EXPECT_NEAR(fp2.toFloat(), -3.75f, EPSILON);
}

// Equality Operator Test
TEST_F(FixedPointTest, EqualityOperator) {
    FixedPointQ5_10 fp1(3.5f);
    FixedPointQ5_10 fp2(3.5f);
    FixedPointQ5_10 fp3(2.25f);

    EXPECT_TRUE(fp1 == fp2);
    EXPECT_FALSE(fp1 == fp3);

    // Test self-equality
    EXPECT_TRUE(fp1 == fp1);
}

// Less Than Operator Test
TEST_F(FixedPointTest, LessThanOperator) {
    FixedPointQ5_10 fp1(3.5f);
    FixedPointQ5_10 fp2(4.5f);

    EXPECT_TRUE(fp1 < fp2);
    EXPECT_FALSE(fp2 < fp1);
    EXPECT_FALSE(fp1 < fp1);
}

// Greater Than Operator Test
TEST_F(FixedPointTest, GreaterThanOperator) {
    FixedPointQ5_10 fp1(3.5f);
    FixedPointQ5_10 fp2(4.5f);

    EXPECT_TRUE(fp2 > fp1);
    EXPECT_FALSE(fp1 > fp2);
    EXPECT_FALSE(fp1 > fp1);
}

// Addition Assignment Operator Test
TEST_F(FixedPointTest, AdditionAssignmentOperator) {
    FixedPointQ5_10 fp1(3.5f);
    FixedPointQ5_10 fp2(2.25f);
    fp1 += fp2;
    EXPECT_NEAR(fp1.toFloat(), 5.75f, EPSILON);

    // Test self-addition
    fp1 += fp1;
    EXPECT_NEAR(fp1.toFloat(), 11.5f, EPSILON);
}

// Subtraction Assignment Operator Test
TEST_F(FixedPointTest, SubtractionAssignmentOperator) {
    FixedPointQ5_10 fp1(3.5f);
    FixedPointQ5_10 fp2(2.25f);
    fp1 -= fp2;
    EXPECT_NEAR(fp1.toFloat(), 1.25f, EPSILON);

    // Test self-subtraction
    fp1 -= fp1;
    EXPECT_NEAR(fp1.toFloat(), 0.0f, EPSILON);
}

TEST_F(FixedPointTest, NotEqualOperator) {
    FixedPointQ5_10 a(1.5f);
    FixedPointQ5_10 b(2.0f);
    FixedPointQ5_10 c(1.5f);
    
    EXPECT_TRUE(a != b);
    EXPECT_FALSE(a != c);
}

TEST_F(FixedPointTest, GreaterThanOrEqualOperator) {
    FixedPointQ5_10 a(1.5f);
    FixedPointQ5_10 b(1.0f);
    FixedPointQ5_10 c(1.5f);
    
    // Greater than case
    EXPECT_TRUE(a >= b);
    
    // Equal case
    EXPECT_TRUE(a >= c);
    
    // Less than case
    EXPECT_FALSE(b >= a);
}

TEST_F(FixedPointTest, LessThanOrEqualOperator) {
    FixedPointQ5_10 a(1.5f);
    FixedPointQ5_10 b(2.0f);
    FixedPointQ5_10 c(1.5f);
    
    // Less than case
    EXPECT_TRUE(a <= b);
    
    // Equal case
    EXPECT_TRUE(a <= c);
    
    // Greater than case
    EXPECT_FALSE(b <= a);
}

TEST_F(FixedPointTest, MathFunctions) {
    FixedPointQ5_10 x(4.0f);
    
    EXPECT_NEAR(FixedPointQ5_10::sqrt(x).toFloat(), 2.0f, EPSILON);
    
    x = FixedPointQ5_10(1.0f);
    EXPECT_NEAR(FixedPointQ5_10::log(x).toFloat(), 0.0f, EPSILON);
    
    EXPECT_NEAR(FixedPointQ5_10::sin(x).toFloat(), std::sin(1.0f), EPSILON);
    EXPECT_NEAR(FixedPointQ5_10::cos(x).toFloat(), std::cos(1.0f), EPSILON);
}

class FixedPointTanhTest : public ::testing::Test {
protected:
    static constexpr float tolerance = 0.05f; // 5% tolerance for approximation
};

TEST_F(FixedPointTanhTest, HandlesZero) {
    FixedPointQ5_10 x(0.0f);
    EXPECT_FLOAT_EQ(x.tanh().toFloat(), 0.0f);
}

TEST_F(FixedPointTanhTest, HandlesPosValues) {
    float test_values[] = {0.5f, 1.0f, 2.0f, 3.0f};
    for(float val : test_values) {
        FixedPointQ5_10 x(val);
        float expected = std::tanh(val);
        float actual = x.tanh().toFloat();
        EXPECT_NEAR(actual, expected, std::abs(expected * tolerance))
            << "Failed for value: " << val;
    }
}

TEST_F(FixedPointTanhTest, HandlesNegValues) {
    float test_values[] = {-0.5f, -1.0f, -2.0f, -3.0f};
    for(float val : test_values) {
        FixedPointQ5_10 x(val);
        float expected = std::tanh(val);
        float actual = x.tanh().toFloat();
        EXPECT_NEAR(actual, expected, std::abs(expected * tolerance))
            << "Failed for value: " << val;
    }
}

TEST_F(FixedPointTanhTest, HandlesSaturation) {
    FixedPointQ5_10 pos_large(4.0f);
    FixedPointQ5_10 neg_large(-4.0f);
    
    EXPECT_NEAR(pos_large.tanh().toFloat(), 1.0f, tolerance);
    EXPECT_NEAR(neg_large.tanh().toFloat(), -1.0f, tolerance);
}

TEST_F(FixedPointTanhTest, EigenIntegration) {
    FixedPointQ5_10 x(1.5f);
    float expected = std::tanh(1.5f);
    float actual = Eigen::numext::tanh(x).toFloat();
    EXPECT_NEAR(actual, expected, std::abs(expected * tolerance));
}

/*
TEST_F(FixedPointTest, TanhFunctionTest) {
    // Test with a few sample inputs
    float inputs[] = {-2.0f, -0.5f, 0.0f, 0.5f, 2.0f};
    for (float val : inputs) {
        FixedPointQ5_10 fp(val);
        FixedPointQ5_10 result = fp_ops::tanh(fp);
        float expected = std::tanh(val);
        EXPECT_NEAR(result.toFloat(), expected, EPSILON);
    }
}
*/

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}