//#include <fpm/fixed.hpp>  // For fpm::fixed_16_16
//#include <fpm/math.hpp>   // For fpm::cos
//#include <fpm/ios.hpp>    // For fpm::operator<<
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
    
    int main(int argc, char** argv) {
        testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    }