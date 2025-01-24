#ifndef FIXED_POINT_HPP
#define FIXED_POINT_HPP

// Add Eigen includes
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include <iostream>
#include <cmath>
#include <cstdint>





class FixedPointQ5_10 {
private:
    int16_t value; // 16-bit integer to store the fixed-point value

    static constexpr int fractional_bits = 10;
    static constexpr int fractional_mask = (1 << fractional_bits) - 1;
    static constexpr int max_integer_part = 15; // 2^4 - 1 (5 bits for integer part, including sign bit)
    static constexpr int min_integer_part = -16; // -2^4 (5 bits for integer part, including sign bit)


public:
    // Constructors
    FixedPointQ5_10() : value(0) {}
     FixedPointQ5_10(int integer_part) {
        if (integer_part > max_integer_part) {
            integer_part = max_integer_part;
        } else if (integer_part < min_integer_part) {
            integer_part = min_integer_part;
        }
        value = integer_part << fractional_bits;
    }

    FixedPointQ5_10(float float_value) {
        float max_value = static_cast<float>(max_integer_part + 1) - 1.0f / (1 << fractional_bits);
        float min_value = static_cast<float>(min_integer_part);
        if (float_value > max_value) {
            float_value = max_value;
        } else if (float_value < min_value) {
            float_value = min_value;
        }
        value = static_cast<int16_t>(round(float_value * (1 << fractional_bits)));
    }

    FixedPointQ5_10(double double_value) {
        double max_value = static_cast<double>(max_integer_part + 1) - 1.0 / (1 << fractional_bits);
        double min_value = static_cast<double>(min_integer_part);
        if (double_value > max_value) {
            double_value = max_value;
        } else if (double_value < min_value) {
            double_value = min_value;
        }
        value = static_cast<int16_t>(round(double_value * (1 << fractional_bits)));
    }

    // Conversion to float
    float toFloat() const {
        return static_cast<float>(value) / (1 << fractional_bits);
    }

     // Assignment operator
    FixedPointQ5_10& operator=(const FixedPointQ5_10& other) {
        if (this != &other) {
            value = other.value;
        }
        return *this;
    }

    // Equality operator
    bool operator==(const FixedPointQ5_10& other) const {
        return value == other.value;
    }

    // Addition
    FixedPointQ5_10 operator+(const FixedPointQ5_10& other) const {
        FixedPointQ5_10 result;
        result.value = value + other.value;
        return result;
    }

    // Subtraction
    FixedPointQ5_10 operator-(const FixedPointQ5_10& other) const {
        FixedPointQ5_10 result;
        result.value = value - other.value;
        return result;
    }
    
    // Unary minus (negation) - add this
    FixedPointQ5_10 operator-() const {
        FixedPointQ5_10 result;
        result.value = -value;
        return result;
    }

    // Multiplication
    FixedPointQ5_10 operator*(const FixedPointQ5_10& other) const {
        FixedPointQ5_10 result;
        result.value = (value * other.value) >> fractional_bits;
        return result;
    }

    // Division
    FixedPointQ5_10 operator/(const FixedPointQ5_10& other) const {
        FixedPointQ5_10 result;
        result.value = (value << fractional_bits) / other.value;
        return result;
    }

    // Output stream operator
    friend std::ostream& operator<<(std::ostream& os, const FixedPointQ5_10& fp) {
        os << fp.toFloat();
        return os;
    }

    // Less than operator
    bool operator<(const FixedPointQ5_10& other) const {
        return value < other.value;
    }

    // Greater than operator
    bool operator>(const FixedPointQ5_10& other) const {
        return value > other.value;
    }

    // Not equal operator
    bool operator!=(const FixedPointQ5_10& other) const {
    return value != other.value;
    }


    // Addition assignment operator
    FixedPointQ5_10& operator+=(const FixedPointQ5_10& other) {
        value += other.value;
        return *this;
    }

    // Subtraction assignment operator
    FixedPointQ5_10& operator-=(const FixedPointQ5_10& other) {
        value -= other.value;
        return *this;
    }

    bool operator>=(const FixedPointQ5_10& other) const {
        return value >= other.value;
    }
    
    bool operator<=(const FixedPointQ5_10& other) const {
        return value <= other.value;
    }
    
   
    // Static math functions
    static FixedPointQ5_10 sqrt(const FixedPointQ5_10& x) {
        return FixedPointQ5_10(std::sqrt(x.toFloat()));
    }

    static FixedPointQ5_10 log(const FixedPointQ5_10& x) {
        return FixedPointQ5_10(std::log(x.toFloat()));
    }

    static FixedPointQ5_10 sin(const FixedPointQ5_10& x) {
        return FixedPointQ5_10(std::sin(x.toFloat()));
    }

    static FixedPointQ5_10 cos(const FixedPointQ5_10& x) {
        return FixedPointQ5_10(std::cos(x.toFloat()));
    }

    // Add tanh implementation using piecewise linear approximation
    FixedPointQ5_10 tanh() const {
    // Constants for the piece-wise linear approximation
        static const FixedPointQ5_10 X1(1.0f);    // First breakpoint
        static const FixedPointQ5_10 X2(2.0f);    // Second breakpoint
        static const FixedPointQ5_10 A1(1.0f);    // Slope for |x| > 2
        static const FixedPointQ5_10 A2(0.75f);   // Slope for 1 < |x| ≤ 2
        static const FixedPointQ5_10 A3(0.6f);    // Slope for |x| ≤ 1
        
        FixedPointQ5_10 x = *this;
        bool negative = x.value < 0;
        if (negative) {
            x.value = -x.value;
        }
        
        FixedPointQ5_10 result(0);
        
        // Piece-wise linear approximation
        if (x.value >= X2.value) {
            result = FixedPointQ5_10(1.0f); // Saturate at 1.0
        }
        else if (x.value >= X1.value) {
            // Linear interpolation between X1 and X2
            FixedPointQ5_10 dx = x - X1;
            result = X1 * A3 + dx * A2;
        }
        else {
            // Linear approximation for small values
            result = x * A3;
        }
        
        // Apply sign
        if (negative) {
            result.value = -result.value;
        }
        
        return result;
}
    
    // Helper for absolute value
    FixedPointQ5_10 abs() const {
        return FixedPointQ5_10(value >= 0 ? value : -value);
    }




};

// Then add Eigen specialization AFTER the complete class definition
namespace Eigen {
namespace numext {

template<>
EIGEN_DEVICE_FUNC inline FixedPointQ5_10 tanh<FixedPointQ5_10>(const FixedPointQ5_10& x) {
    return x.tanh();
}

} // namespace numext
} // namespace Eigen

#endif // FIXED_POINT_HPP