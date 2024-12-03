#ifndef FIXED_POINT_HPP_
#define FIXED_POINT_HPP_

#include <cstdint>
#include <iostream>

namespace fixed_point {

template <int FractionalBits>
class FixedPoint {
public:
    // Constructors
    FixedPoint() : value_(0) {}
    FixedPoint(int32_t value) : value_(value << FractionalBits) {}
    FixedPoint(float value) : value_(static_cast<int32_t>(value * (1 << FractionalBits))) {}
    FixedPoint(double value) : value_(static_cast<int32_t>(value * (1 << FractionalBits))) {}

    // Conversion to float
    operator float() const {
        return static_cast<float>(value_) / (1 << FractionalBits);
    }

    // Conversion to double
    operator double() const {
        return static_cast<double>(value_) / (1 << FractionalBits);
    }

    // Arithmetic operations
    FixedPoint operator+(const FixedPoint& other) const {
        return FixedPoint::fromRaw(value_ + other.value_);
    }

    FixedPoint operator-(const FixedPoint& other) const {
        return FixedPoint::fromRaw(value_ - other.value_);
    }

    FixedPoint operator*(const FixedPoint& other) const {
        return FixedPoint::fromRaw((value_ * other.value_) >> FractionalBits);
    }

    FixedPoint operator/(const FixedPoint& other) const {
        return FixedPoint::fromRaw((value_ << FractionalBits) / other.value_);
    }

    FixedPoint& operator+=(const FixedPoint& other) {
        value_ += other.value_;
        return *this;
    }

    FixedPoint& operator-=(const FixedPoint& other) {
        value_ -= other.value_;
        return *this;
    }

    FixedPoint& operator*=(const FixedPoint& other) {
        value_ = (value_ * other.value_) >> FractionalBits;
        return *this;
    }

    FixedPoint& operator/=(const FixedPoint& other) {
        value_ = (value_ << FractionalBits) / other.value_;
        return *this;
    }    

    // Comparison operators
    bool operator==(const FixedPoint& other) const {
        return value_ == other.value_;
    }

    bool operator!=(const FixedPoint& other) const {
        return value_ != other.value_;
    }

    bool operator<(const FixedPoint& other) const {
        return value_ < other.value_;
    }

    bool operator<=(const FixedPoint& other) const {
        return value_ <= other.value_;
    }

    bool operator>(const FixedPoint& other) const {
        return value_ > other.value_;
    }

    bool operator>=(const FixedPoint& other) const {
        return value_ >= other.value_;
    }

    // Output stream operator
    friend std::ostream& operator<<(std::ostream& os, const FixedPoint& fp) {
        os << static_cast<float>(fp);
        return os;
    }



private:
    int32_t value_;

    // Constructor from raw value
    static FixedPoint fromRaw(int32_t rawValue) {
        FixedPoint fp;
        fp.value_ = rawValue;
        return fp;
    }
};

using fp32_32 = FixedPoint<32>;

// Specialize math functions for FixedPoint
template <int FractionalBits>
FixedPoint<FractionalBits> log(const FixedPoint<FractionalBits>& x) {
    return FixedPoint<FractionalBits>(std::log(static_cast<float>(x)));
}

template <int FractionalBits>
FixedPoint<FractionalBits> exp(const FixedPoint<FractionalBits>& x) {
    return FixedPoint<FractionalBits>(std::exp(static_cast<float>(x)));
}

template <int FractionalBits>
FixedPoint<FractionalBits> tanh(const FixedPoint<FractionalBits>& x) {
    return FixedPoint<FractionalBits>(std::tanh(static_cast<float>(x)));
}

}  // namespace fixed_point

#endif  // FIXED_POINT_HPP_