#ifndef FIXED_POINT_HPP_
#define FIXED_POINT_HPP_

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <cmath>

namespace fixed_point {

template <int FractionalBits>
class FixedPoint {
    static_assert(FractionalBits > 0 && FractionalBits < 32,
                 "FractionalBits must be between 0 and 31");

public:
    static constexpr int32_t SCALE = 1 << FractionalBits;
    static constexpr int32_t FRACTION_MASK = SCALE - 1;
    static constexpr int32_t MAX_VALUE = INT32_MAX >> FractionalBits;
    static constexpr int32_t MIN_VALUE = INT32_MIN >> FractionalBits;

    // Constructors
    constexpr FixedPoint() noexcept : value_(0) {}
    
    explicit constexpr FixedPoint(int32_t value) {
        if (value > MAX_VALUE || value < MIN_VALUE) {
            throw std::overflow_error("Integer value out of range for fixed-point conversion");
        }
        value_ = value << FractionalBits;
    }
    
    explicit FixedPoint(float value) {
        const float scaled = value * SCALE;
        if (scaled > INT32_MAX || scaled < INT32_MIN) {
            throw std::overflow_error("Float value out of range for fixed-point conversion");
        }
        value_ = static_cast<int32_t>(scaled);
    }
    
    explicit FixedPoint(double value) {
        const double scaled = value * SCALE;
        if (scaled > INT32_MAX || scaled < INT32_MIN) {
            throw std::overflow_error("Double value out of range for fixed-point conversion");
        }
        value_ = static_cast<int32_t>(scaled);
    }

    // Copy and move operations
    constexpr FixedPoint(const FixedPoint&) noexcept = default;
    constexpr FixedPoint& operator=(const FixedPoint&) noexcept = default;
    constexpr FixedPoint(FixedPoint&&) noexcept = default;
    constexpr FixedPoint& operator=(FixedPoint&&) noexcept = default;

    // Conversion operators
    explicit constexpr operator float() const noexcept {
        return static_cast<float>(value_) / SCALE;
    }
    
    explicit constexpr operator double() const noexcept {
        return static_cast<double>(value_) / SCALE;
    }

    // Arithmetic operations
    constexpr FixedPoint operator+(const FixedPoint& other) const {
        if (__builtin_add_overflow(value_, other.value_, &value_)) {
            throw std::overflow_error("Addition overflow");
        }
        return fromRaw(value_);
    }

    constexpr FixedPoint operator-(const FixedPoint& other) const {
        if (__builtin_sub_overflow(value_, other.value_, &value_)) {
            throw std::overflow_error("Subtraction overflow");
        }
        return fromRaw(value_);
    }

    FixedPoint operator*(const FixedPoint& other) const {
        int64_t result = (static_cast<int64_t>(value_) * other.value_) >> FractionalBits;
        if (result > INT32_MAX || result < INT32_MIN) {
            throw std::overflow_error("Multiplication overflow");
        }
        return fromRaw(static_cast<int32_t>(result));
    }

    FixedPoint operator/(const FixedPoint& other) const {
        if (other.value_ == 0) {
            throw std::domain_error("Division by zero");
        }
        int64_t temp = (static_cast<int64_t>(value_) << FractionalBits) / other.value_;
        if (temp > INT32_MAX || temp < INT32_MIN) {
            throw std::overflow_error("Division overflow");
        }
        return fromRaw(static_cast<int32_t>(temp));
    }

    // Compound assignment operators
    FixedPoint& operator+=(const FixedPoint& other) {
        if (__builtin_add_overflow(value_, other.value_, &value_)) {
            throw std::overflow_error("Addition overflow");
        }
        return *this;
    }

    FixedPoint& operator-=(const FixedPoint& other) {
        if (__builtin_sub_overflow(value_, other.value_, &value_)) {
            throw std::overflow_error("Subtraction overflow");
        }
        return *this;
    }

    FixedPoint& operator*=(const FixedPoint& other) {
        int64_t result = (static_cast<int64_t>(value_) * other.value_) >> FractionalBits;
        if (result > INT32_MAX || result < INT32_MIN) {
            throw std::overflow_error("Multiplication overflow");
        }
        value_ = static_cast<int32_t>(result);
        return *this;
    }

    FixedPoint& operator/=(const FixedPoint& other) {
        if (other.value_ == 0) {
            throw std::domain_error("Division by zero");
        }
        int64_t temp = (static_cast<int64_t>(value_) << FractionalBits) / other.value_;
        if (temp > INT32_MAX || temp < INT32_MIN) {
            throw std::overflow_error("Division overflow");
        }
        value_ = static_cast<int32_t>(temp);
        return *this;
    }

    // Comparison operators
    constexpr bool operator==(const FixedPoint& other) const noexcept {
        return value_ == other.value_;
    }
    
    constexpr bool operator!=(const FixedPoint& other) const noexcept {
        return value_ != other.value_;
    }
    
    constexpr bool operator<(const FixedPoint& other) const noexcept {
        return value_ < other.value_;
    }
    
    constexpr bool operator<=(const FixedPoint& other) const noexcept {
        return value_ <= other.value_;
    }
    
    constexpr bool operator>(const FixedPoint& other) const noexcept {
        return value_ > other.value_;
    }
    
    constexpr bool operator>=(const FixedPoint& other) const noexcept {
        return value_ >= other.value_;
    }

    // Output stream operator
    friend std::ostream& operator<<(std::ostream& os, const FixedPoint& fp) {
        os << static_cast<double>(fp);
        return os;
    }

    // Get raw value
    constexpr int32_t raw() const noexcept { return value_; }

private:
    int32_t value_;

    static constexpr FixedPoint fromRaw(int32_t rawValue) noexcept {
        FixedPoint fp;
        fp.value_ = rawValue;
        return fp;
    }
};

// Fixed-point specific implementations of mathematical functions
template <int FractionalBits>
FixedPoint<FractionalBits> log(const FixedPoint<FractionalBits>& x) {
    if (x.raw() <= 0) {
        throw std::domain_error("Log of non-positive number");
    }
    
    // This could be replaced with a fixed-point specific algorithm
    // for better precision, but this is a safe fallback
    return FixedPoint<FractionalBits>(std::log(static_cast<double>(x)));
}

template <int FractionalBits>
FixedPoint<FractionalBits> exp(const FixedPoint<FractionalBits>& x) {
    double result = std::exp(static_cast<double>(x));
    if (result > FixedPoint<FractionalBits>::MAX_VALUE || 
        result < FixedPoint<FractionalBits>::MIN_VALUE) {
        throw std::overflow_error("Exp result out of range");
    }
    return FixedPoint<FractionalBits>(result);
}

template <int FractionalBits>
FixedPoint<FractionalBits> tanh(const FixedPoint<FractionalBits>& x) {
    return FixedPoint<FractionalBits>(std::tanh(static_cast<double>(x)));
}

// Common type definitions
using fp16_16 = FixedPoint<16>;
using fp24_8 = FixedPoint<8>;
using fp8_24 = FixedPoint<24>;
using fp32_32 = FixedPoint<32>;

} // namespace fixed_point

#endif // FIXED_POINT_HPP_