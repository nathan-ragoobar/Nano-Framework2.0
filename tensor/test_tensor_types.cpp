#include <gtest/gtest.h>
#include "tensor_types.hpp"
#include "fixed_point.hpp"

//TEST is a macro used to define and Name a test function
//The paramters are the name of the test and the name of the test case. Starting from more general to more specific.
TEST(TTypesTest, FloatTensor) {
    using T = float;
    using TensorType = TTypes<T>::Tensor;
    using ConstTensorType = TTypes<T>::ConstTensor;

    T data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    TensorType tensor(data, 2, 3);
    ConstTensorType const_tensor(data, 2, 3);

    EXPECT_EQ(tensor.dimension(0), 2);
    EXPECT_EQ(tensor.dimension(1), 3);
    EXPECT_EQ(const_tensor.dimension(0), 2);
    EXPECT_EQ(const_tensor.dimension(1), 3);
}
/*
TEST(TTypesTest, FixedPointTensor) {
    using T = fixed_point::FixedPoint<16>;
    using TensorType = TTypes<T>::Tensor;
    using ConstTensorType = TTypes<T>::ConstTensor;

    T data[6] = {T(1.0f), T(2.0f), T(3.0f), T(4.0f), T(5.0f), T(6.0f)};
    TensorType tensor(data, 2, 3);
    ConstTensorType const_tensor(data, 2, 3);

    EXPECT_EQ(tensor.dimension(0), 2);
    EXPECT_EQ(tensor.dimension(1), 3);
    EXPECT_EQ(const_tensor.dimension(0), 2);
    EXPECT_EQ(const_tensor.dimension(1), 3);
}

TEST(TTypesTest, FixedPointOperations) {
    using T = fixed_point::FixedPoint<16>;

    T a(1.0f);
    T b(2.0f);
    T c = a + b;
    T d = a * b;

    EXPECT_EQ(static_cast<float>(c), 3.0f);
    EXPECT_EQ(static_cast<float>(d), 2.0f);
}
*/


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}