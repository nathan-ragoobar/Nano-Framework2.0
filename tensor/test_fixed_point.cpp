#include <gtest/gtest.h>
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


int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}