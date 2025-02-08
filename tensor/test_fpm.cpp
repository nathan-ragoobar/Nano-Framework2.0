//#include <fpm/fixed.hpp>  // For fpm::fixed_16_16
//#include <fpm/math.hpp>   // For fpm::cos
//#include <fpm/ios.hpp>    // For fpm::operator<<
//#include <Eigen/Core>
//#include <unsupported/Eigen/CXX11/Tensor>
#include "./fpm/fpm.hpp"
#include <iostream>       // For std::cin, std::cout
#include <gtest/gtest.h>
/*
int main() {
    std::cout << "Please input a number: ";
    fpm::fixed_16_16 x;
    std::cin >> x;
    std::cout << "The cosine of " << x << " radians is: " << cos(x) << std::endl;
    return 0;
}
    */

using Type = fpm::fixed_16_16;

class FixedPointTest : public ::testing::Test {
    protected:
        static constexpr float EPSILON = 0.001f; // Tolerance for floating-point comparisons
    
        
    };
    
    // Constructor Tests
    TEST_F(FixedPointTest, DefaultConstructor) {
        Type fp;
        
        EXPECT_FLOAT_EQ(float(fp), 0.0f);
    }
    
    TEST_F(FixedPointTest, IntegerConstructor) {
        Type fp1(5);
        EXPECT_NEAR(float(fp1), 5.0f, EPSILON);
    
        Type fp2(-5);
        EXPECT_NEAR(float(fp2), -5.0f, EPSILON);
    
        // Test clamping
        Type fp3(32770);  // Should clamp to max value
        Type max_value = std::numeric_limits<Type>::max();
        EXPECT_NEAR(float(fp3), float(max_value), EPSILON);
        
        

        Type fp4(-32770);  // Should clamp to min value
        Type min_value = std::numeric_limits<Type>::min(); // Get the minimum value
        EXPECT_NEAR(float(fp4), float(min_value), EPSILON);
    }
    
    TEST_F(FixedPointTest, FloatConstructor) {
        Type fp1(5.5f);
        EXPECT_NEAR(float(fp1), 5.5f, EPSILON);
    
        Type fp2(-5.5f);
        EXPECT_NEAR(float(fp2), -5.5f, EPSILON);
    
        // Test fractional precision
        Type fp3(0.125f);
        EXPECT_NEAR(float(fp3), 0.125f, EPSILON);
    
        // Test clamping
        Type fp4(32770.0f);  // Should clamp to ~15.999
        Type max_value = std::numeric_limits<Type>::max();
        EXPECT_NEAR(float(fp4), float(max_value), EPSILON);
    
        Type fp5(-32770.0f);  // Should clamp to -16
        Type min_value = std::numeric_limits<Type>::min();
        EXPECT_NEAR(float(fp5), float(min_value), EPSILON);
    }

    // Arithmetic Operation Tests
TEST_F(FixedPointTest, Addition) {
    Type fp1(3.5f);
    Type fp2(2.25f);
    Type result = fp1 + fp2;
    EXPECT_NEAR(float(result), 5.75f, EPSILON);

    // Test addition near bounds
    Type fp3(10.0f);
    Type fp4(5.0f);
    result = fp3 + fp4;
    EXPECT_NEAR(float(result), 15.0f, EPSILON);

    // Test negative addition
    Type fp5(-3.5f);
    Type fp6(2.25f);
    result = fp5 + fp6;
    EXPECT_NEAR(float(result), -1.25f, EPSILON);
}

TEST_F(FixedPointTest, Subtraction) {
    Type fp1(3.5f);
    Type fp2(2.25f);
    Type result = fp1 - fp2;
    EXPECT_NEAR(float(result), 1.25f, EPSILON);

    // Test subtraction with negative results
    Type fp3(2.25f);
    Type fp4(3.5f);
    result = fp3 - fp4;
    EXPECT_NEAR(float(result), -1.25f, EPSILON);
}

TEST_F(FixedPointTest, UnaryMinusOperator) {
    Type a(1.5f);
    Type negA = -a;
    EXPECT_NEAR(float(negA), -1.5f, EPSILON);
}

TEST_F(FixedPointTest, Multiplication) {
    Type fp1(3.5f);
    Type fp2(2.0f);
    Type result = fp1 * fp2;
    EXPECT_NEAR(float(result), 7.0f, EPSILON);

    // Test multiplication with fractional numbers
    Type fp3(2.5f);
    Type fp4(0.5f);
    result = fp3 * fp4;
    EXPECT_NEAR(float(result), 1.25f, EPSILON);

    // Test multiplication with negative numbers
    Type fp5(-2.5f);
    Type fp6(2.0f);
    result = fp5 * fp6;
    EXPECT_NEAR(float(result), -5.0f, EPSILON);
}

TEST_F(FixedPointTest, Division) {
    Type fp1(3.5f);
    Type fp2(2.0f);
    Type result = fp1 / fp2;
    EXPECT_NEAR(float(result), 1.75f, EPSILON);

    // Test division with fractional numbers
    Type fp3(2.5f);
    Type fp4(0.5f);
    result = fp3 / fp4;
    EXPECT_NEAR(float(result), 5.0f, EPSILON);

    // Test division with negative numbers
    Type fp5(-2.5f);
    Type fp6(2.0f);
    result = fp5 / fp6;
    EXPECT_NEAR(float(result), -1.25f, EPSILON);
}

// Stream Output Test
TEST_F(FixedPointTest, StreamOutput) {
    Type fp(3.5f);
    std::stringstream ss;
    ss << fp;
    float value;
    ss >> value;
    EXPECT_NEAR(value, 3.5f, EPSILON);
}

// Test the assignment operator
TEST_F(FixedPointTest, AssignmentOperator) {
    Type fp1(5.5f);
    Type fp2;
    fp2 = fp1;
    EXPECT_NEAR(float(fp2), 5.5f, EPSILON);

    Type fp3(-3.75f);
    fp2 = fp3;
    EXPECT_NEAR(float(fp2), -3.75f, EPSILON);

    // Test self-assignment
    fp2 = fp2;
    EXPECT_NEAR(float(fp2), -3.75f, EPSILON);
}

// Equality Operator Test
TEST_F(FixedPointTest, EqualityOperator) {
    Type fp1(3.5f);
    Type fp2(3.5f);
    Type fp3(2.25f);

    EXPECT_TRUE(fp1 == fp2);
    EXPECT_FALSE(fp1 == fp3);

    // Test self-equality
    EXPECT_TRUE(fp1 == fp1);
}
/* */
// Less Than Operator Test
TEST_F(FixedPointTest, LessThanOperator) {
    Type fp1(3.5f);
    Type fp2(4.5f);

    EXPECT_TRUE(fp1 < fp2);
    EXPECT_FALSE(fp2 < fp1);
    EXPECT_FALSE(fp1 < fp1);
}

// Greater Than Operator Test
TEST_F(FixedPointTest, GreaterThanOperator) {
    Type fp1(3.5f);
    Type fp2(4.5f);

    EXPECT_TRUE(fp2 > fp1);
    EXPECT_FALSE(fp1 > fp2);
    EXPECT_FALSE(fp1 > fp1);
}

// Addition Assignment Operator Test
TEST_F(FixedPointTest, AdditionAssignmentOperator) {
    Type fp1(3.5f);
    Type fp2(2.25f);
    fp1 += fp2;
    EXPECT_NEAR(float(fp1), 5.75f, EPSILON);

    // Test self-addition
    fp1 += fp1;
    EXPECT_NEAR(float(fp1), 11.5f, EPSILON);
}

// Subtraction Assignment Operator Test
TEST_F(FixedPointTest, SubtractionAssignmentOperator) {
    Type fp1(3.5f);
    Type fp2(2.25f);
    fp1 -= fp2;
    EXPECT_NEAR(float(fp1), 1.25f, EPSILON);

    // Test self-subtraction
    fp1 -= fp1;
    EXPECT_NEAR(float(fp1), 0.0f, EPSILON);
}

TEST_F(FixedPointTest, NotEqualOperator) {
    Type a(1.5f);
    Type b(2.0f);
    Type c(1.5f);
    
    EXPECT_TRUE(a != b);
    EXPECT_FALSE(a != c);
}

TEST_F(FixedPointTest, GreaterThanOrEqualOperator) {
    Type a(1.5f);
    Type b(1.0f);
    Type c(1.5f);
    
    // Greater than case
    EXPECT_TRUE(a >= b);
    
    // Equal case
    EXPECT_TRUE(a >= c);
    
    // Less than case
    EXPECT_FALSE(b >= a);
}

TEST_F(FixedPointTest, LessThanOrEqualOperator) {
    Type a(1.5f);
    Type b(2.0f);
    Type c(1.5f);
    
    // Less than case
    EXPECT_TRUE(a <= b);
    
    // Equal case
    EXPECT_TRUE(a <= c);
    
    // Greater than case
    EXPECT_FALSE(b <= a);
}

TEST_F(FixedPointTest, MathFunctions) {
    Type x(4.0f);
    
    
    EXPECT_NEAR(float(sqrt(x)), 2.0f, EPSILON);
    
    x = Type(1.0f);
    EXPECT_NEAR(float(log(x)), 0.0f, EPSILON);
    
    EXPECT_NEAR(float(sin(x)), std::sin(1.0f), 10*EPSILON);
    EXPECT_NEAR(float(cos(x)), std::cos(1.0f), 10*EPSILON);
}

TEST_F(FixedPointTest, LogFunction) {
    static constexpr float EPSILON = 0.15f;
    // Basic cases
    Type x(1.0f);
    EXPECT_NEAR(float(log(x)), 0.0f, EPSILON);
    
    x = Type(2.718281828f);  // e
    EXPECT_NEAR(float(log(x)), 1.0f, EPSILON);
    
    x = Type(2.0f);
    EXPECT_NEAR(float(log(x)), 0.693147f, EPSILON);
    
    x = Type(0.5f);
    EXPECT_NEAR(float(log(x)), -0.693147f, EPSILON);
    
    x = Type(10.0f);
    EXPECT_NEAR(float(log(x)), 2.302585f, EPSILON);

    // Edge cases
    x = Type(0.1f);
    EXPECT_NEAR(float(log(x)), -2.302585f, EPSILON);
    
    x = Type(100.0f);
    EXPECT_NEAR(float(log(x)), 4.605170f, EPSILON);

    // Small value test
    x = Type(0.01f);
    EXPECT_NEAR(float(log(x)), -4.605170f, EPSILON);

    // Verify error handling for invalid inputs
    //#ifdef NDEBUG
    x = Type(0.0f);
    EXPECT_DEATH(log(x), "");
    
    x = Type(-1.0f);
    EXPECT_DEATH(log(x), "");
    //#endif
}

class FixedPointTanhTest : public ::testing::Test {
    protected:
        static constexpr float tolerance = 0.05f; // 5% tolerance for approximation
    };
    
    TEST_F(FixedPointTanhTest, TanhHandlesZero) {
        Type x(0.0f);
        Type result = tanh(x);
        EXPECT_FLOAT_EQ(float(result), 0.0f);
    }

    TEST_F(FixedPointTanhTest, TanhHandlesPosValues) {
        float test_values[] = {0.5f, 1.0f, 2.0f, 3.0f};
        for(float val : test_values) {
            Type x(val);
            float expected = std::tanh(val);
            float actual = float(tanh(x));
            EXPECT_NEAR(actual, expected, std::abs(expected * tolerance))
                << "Failed for value: " << val;
        }
    }
    
    TEST_F(FixedPointTanhTest, TanhHandlesNegValues) {
        float test_values[] = {-0.5f, -1.0f, -2.0f, -3.0f};
        for(float val : test_values) {
            Type x(val);
            float expected = std::tanh(val);
            float actual = float(tanh(x));
            EXPECT_NEAR(actual, expected, std::abs(expected * tolerance))
                << "Failed for value: " << val;
        }
    }
    
    TEST_F(FixedPointTanhTest, TanhHandlesSaturation) {
        Type pos_large(4.0f);
        Type neg_large(-4.0f);
        
        EXPECT_NEAR(float(tanh(pos_large)), 1.0f, tolerance);
        EXPECT_NEAR(float(tanh(neg_large)), -1.0f, tolerance);
    }

    /*
    TEST_F(FixedPointTanhTest, TanhEigenIntegration) {
        Type x(1.5f);
        float expected = std::tanh(1.5f);
        float actual = float(Eigen::numext::tanh(x));
        EXPECT_NEAR(actual, expected, std::abs(expected * tolerance));
    }
*/

    
    int main(int argc, char** argv) {
        testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    }