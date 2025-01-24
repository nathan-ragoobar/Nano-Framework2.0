#include "./../tensor/fixed_point.hpp"
#include "./Parameter.hpp"
#include <gtest/gtest.h>

using namespace nn;

TEST(ConstantFillTest, FixedPointFill) {
    // Setup
    std::vector<fixed_point_7pt8> data(10);
    absl::Span<fixed_point_7pt8> span(data);
    
    // Test fill with 1.5
    ConstantFill(span, fixed_point_7pt8(1.5f));
    
    // Verify
    for(const auto& val : span) {
        EXPECT_EQ(val.to_float(), 1.5f);
    }
}

TEST(UniformFillTest, FloatRangeTest) {
    // Setup
    std::vector<float> data(1000);
    absl::Span<float> span(data);
    float min_val = -1.0f;
    float max_val = 1.0f;
    
    // Set random seed for reproducibility
    nn::ManualSeed(42);
    
    // Fill with random values
    nn::UniformFill(span, min_val, max_val);
    
    // Verify bounds
    for(const auto& val : span) {
        EXPECT_GE(val, min_val);
        EXPECT_LE(val, max_val);
    }
    
    // Verify randomness (values aren't all the same)
    float first_val = span[0];
    bool all_same = true;
    for(size_t i = 1; i < span.size(); i++) {
        if(span[i] != first_val) {
            all_same = false;
            break;
        }
    }
    EXPECT_FALSE(all_same);
}

TEST(UniformFillTest, FixedPointRangeTest) {
    // Setup
    std::vector<fixed_point_7pt8> data(1000);
    absl::Span<fixed_point_7pt8> span(data);
    fixed_point_7pt8 min_val(-1.0f);
    fixed_point_7pt8 max_val(1.0f);
    
    // Set seed
    nn::ManualSeed(42);
    
    // Fill with random values
    nn::UniformFill(span, min_val, max_val);
    
    // Verify bounds
    for(const auto& val : span) {
        EXPECT_GE(val, min_val);
        EXPECT_LE(val, max_val);
    }
    
    // Verify randomness
    fixed_point_7pt8 first_val = span[0];
    bool all_same = true;
    for(size_t i = 1; i < span.size(); i++) {
        if(span[i] != first_val) {
            all_same = false;
            break;
        }
    }
    EXPECT_FALSE(all_same);
}

TEST(NormalFillTest, FixedPointDistribution) {
    std::vector<fixed_point_7pt8> data(1000);
    absl::Span<fixed_point_7pt8> span(data);
    
    fixed_point_7pt8 mean(0.0f);
    fixed_point_7pt8 std(1.0f);
    
    nn::ManualSeed(42);
    nn::NormalFill(span, mean, std);
    
    // Verify distribution properties
    float sum = 0.0f;
    for(const auto& val : span) {
        sum += val.to_float();
    }
    float empirical_mean = sum / span.size();
    
    EXPECT_NEAR(empirical_mean, 0.0f, 0.1f);
}

TEST(KaimingUniformFillTest, FixedPointTest) {
    std::vector<fixed_point_7pt8> data(100);
    absl::Span<fixed_point_7pt8> span(data);
    
    nn::ManualSeed(42);
    nn::KaimingUniformFill(span, 10);
    
    fixed_point_7pt8 expected_bound = sqrt(fixed_point_7pt8(0.1f));
    
    for(const auto& val : span) {
        EXPECT_TRUE(val >= -expected_bound);
        EXPECT_TRUE(val <= expected_bound);
    }
}
/*
TEST(UpperTriangularTest, FixedPointTest) {
    // Create tensor instead of matrix
    const int size = 3;

    // Q5.10 format has range [-16, 15.999]
    constexpr float kMinValue = -16.0f;  // Changed from -32.0f

    std::vector<fixed_point_7pt8> data(size * size);
    
    // Create tensor map from data
    TTypes<fixed_point_7pt8, 2>::Tensor matrix(data.data(), size, size);
    
    // Initialize to zero
    for (int i = 0; i < size * size; i++) {
        data[i] = fixed_point_7pt8(0.0f);
    }

    UpperTriangularWithNegativeInf(matrix);
    
    // Check diagonal and lower triangle are zero
    for(int i = 0; i < size; i++) {
        for(int j = 0; j <= i; j++) {
            EXPECT_EQ(matrix(i,j).toFloat(), 0.0f);
        }
    }
    
    // Check upper triangle is minimum value
    for(int i = 0; i < size; i++) {
        for(int j = i + 1; j < size; j++) {
            EXPECT_EQ(matrix(i,j).toFloat(), kMinValue);
        }
    }
}

TEST(OneHotTest, FixedPointTest) {
    const int batch_size = 3;
    const int num_classes = 4;
    
    std::vector<int> target_data = {1, 0, 2};
    std::vector<FixedPointQ5_10> label_data(batch_size * num_classes, FixedPointQ5_10(0.0f));
    
    TTypes<int>::ConstFlat target(target_data.data(), batch_size);
    TTypes<FixedPointQ5_10>::Matrix label(label_data.data(), batch_size, num_classes);
    
    OneHot(target, label);
    
    // Verify results
    for(int i = 0; i < batch_size; i++) {
        for(int j = 0; j < num_classes; j++) {
            if(j == target_data[i]) {
                EXPECT_EQ(label(i,j).toFloat(), 1.0f);
            } else {
                EXPECT_EQ(label(i,j).toFloat(), 0.0f);
            }
        }
    }
}

TEST(SplitRangeTest, BasicFunctionality) {
    // Split 10 items into 3 chunks
    auto range0 = SplitRange(10, 0, 3); // {0, 4}  First chunk
    auto range1 = SplitRange(10, 1, 3); // {4, 7}  Second chunk  
    auto range2 = SplitRange(10, 2, 3); // {7, 10} Third chunk

    EXPECT_EQ(range0.first, 0);
    EXPECT_EQ(range0.second, 4);
    EXPECT_EQ(range1.first, 4);
    EXPECT_EQ(range1.second, 7);
    EXPECT_EQ(range2.first, 7);
    EXPECT_EQ(range2.second, 10);
}

// Add type declaration first
struct InvalidType {};

// Compile-time checks
static_assert(!IsValidDataType<InvalidType>::value,
             "Invalid type should not be supported");

TEST(DataTypeTest, FixedPointDataType) {
    // Runtime checks
    EXPECT_EQ(DataTypeToEnum<FixedPointQ5_10>::v(), DT_FIXED);

    // Compile-time checks within test
    // Use std::is_same instead of std::is_same_v for C++14
    static_assert(std::is_same<
        typename EnumToDataType<DT_FIXED>::Type,
        FixedPointQ5_10
    >::value, "DT_FIXED should map to FixedPointQ5_10");
    
    static_assert(IsValidDataType<FixedPointQ5_10>::value,
                 "FixedPointQ5_10 should be a valid type");
    
    static_assert(DataTypeToEnum<FixedPointQ5_10>::value == DT_FIXED,
                 "Static value should match enum");
}

TEST(TypeDefinitionTest, FixedPointQ5_10TypeExists) {
    // Compile-time check that type exists
    static_assert(sizeof(FixedPointQ5_10) > 0, 
                 "FixedPointQ5_10 type not defined");
    
    // Runtime check type has expected properties
    FixedPointQ5_10 test_val(1.0f);
    EXPECT_EQ(test_val.toFloat(), 1.0f);
}

TEST(Parameter, ConstructorInitializesCorrectly) {
    Parameter p(DT_FIXED, 10);
    EXPECT_EQ(p.size(), 10);
}

TEST(Parameter, DefaultConstructorNoAllocation) {
    Parameter p(DT_FIXED);
    EXPECT_EQ(p.size(), 0);
}
TEST(Parameter, LazyAllocationWorks) {
    Parameter p(DT_FIXED);
    EXPECT_EQ(p.size(), 0);
    p.LazyAllocate(5);
    EXPECT_EQ(p.size(), 5);
}

TEST(Parameter, MultipleLazyAllocWithSameSizeOK) {
    Parameter p(DT_FIXED);
    p.LazyAllocate(5);
    EXPECT_EQ(p.size(), 5);
    p.LazyAllocate(5); // Should not throw
    EXPECT_EQ(p.size(), 5);
}

TEST(Parameter, LazyAllocDifferentSizeFails) {
    Parameter p(DT_FIXED);
    p.LazyAllocate(5);
    EXPECT_DEATH(p.LazyAllocate(10), "Check failed");
}

TEST(Parameter, CopyConstructorDeleted) {
    Parameter p1(DT_FIXED);
    EXPECT_FALSE(std::is_copy_constructible<Parameter>::value);
}

TEST(Parameter, AssignmentOperatorDeleted) {
    Parameter p1(DT_FIXED);
    EXPECT_FALSE(std::is_copy_assignable<Parameter>::value);
}

class ParameterFixedPointTest : public ::testing::Test {
protected:
    void SetUp() override {
        param = new Parameter(DT_FIXED, 5);
    }

    void TearDown() override {
        delete param;
    }

    Parameter* param;
};

// LazyAllocation Tests
TEST_F(ParameterFixedPointTest, LazyAllocationInitializesToZero) {
    Parameter p(DT_FIXED);
    p.LazyAllocate(3);
    auto data = p.span<FixedPointQ5_10>();
    
    for(int i = 0; i < 3; i++) {
        EXPECT_EQ(data[i].toFloat(), 0.0f);
    }
}

TEST_F(ParameterFixedPointTest, GradientLazyAllocation) {
    param->LazyAllocateGradient();
    auto grad = param->span_grad<FixedPointQ5_10>();
    
    EXPECT_EQ(grad.size(), 5);
    for(int i = 0; i < 5; i++) {
        EXPECT_EQ(grad[i].toFloat(), 0.0f);
    }
}

// Zero Operations Tests
TEST_F(ParameterFixedPointTest, ZeroDataOperation) {
    auto data = param->span<FixedPointQ5_10>();
    data[0] = FixedPointQ5_10(1.5f);
    
    param->ZeroData();
    EXPECT_EQ(data[0].toFloat(), 0.0f);
}

TEST_F(ParameterFixedPointTest, ZeroGradOperation) {
    param->LazyAllocateGradient();
    auto grad = param->span_grad<FixedPointQ5_10>();
    grad[0] = FixedPointQ5_10(1.5f);
    
    param->ZeroGrad();
    EXPECT_EQ(grad[0].toFloat(), 0.0f);
}

// Data Access Tests
TEST_F(ParameterFixedPointTest, RawDataAccess) {
    auto* data_ptr = param->data<FixedPointQ5_10>();
    EXPECT_NE(data_ptr, nullptr);
    
    data_ptr[0] = FixedPointQ5_10(2.5f);
    EXPECT_EQ(data_ptr[0].toFloat(), 2.5f);
}

TEST_F(ParameterFixedPointTest, RawGradAccess) {
    param->LazyAllocateGradient();
    auto* grad_ptr = param->grad<FixedPointQ5_10>();
    EXPECT_NE(grad_ptr, nullptr);
    
    grad_ptr[0] = FixedPointQ5_10(3.5f);
    EXPECT_EQ(grad_ptr[0].toFloat(), 3.5f);
}

// Span Access Tests
TEST_F(ParameterFixedPointTest, SpanAccess) {
    auto span = param->span<FixedPointQ5_10>();
    EXPECT_EQ(span.size(), 5);
    
    span[0] = FixedPointQ5_10(4.5f);
    EXPECT_EQ(span[0].toFloat(), 4.5f);
}

TEST_F(ParameterFixedPointTest, GradSpanAccess) {
    param->LazyAllocateGradient();
    auto grad_span = param->span_grad<FixedPointQ5_10>();
    EXPECT_EQ(grad_span.size(), 5);
    
    grad_span[0] = FixedPointQ5_10(5.5f);
    EXPECT_EQ(grad_span[0].toFloat(), 5.5f);
}

// Type Checking Tests
TEST_F(ParameterFixedPointTest, WrongTypeAccess) {
    EXPECT_DEATH(param->span<float>(), "");
    EXPECT_DEATH(param->span_grad<float>(), "");
}

class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 24 elements total - can be reshaped as:
        // - 24 x 1 (flat)
        // - 6 x 4 (matrix)
        // - 2 x 3 x 4 (3D)
        // - 2 x 2 x 2 x 3 (4D)
        param = new Parameter(DT_FIXED, 24);
        param->LazyAllocateGradient();
    }

    void TearDown() override {
        delete param;
    }

    Parameter* param;
};

// Flat Tensor Tests
TEST_F(TensorTest, FlatTensorAccess) {
    auto flat = param->flat<FixedPointQ5_10>();
    EXPECT_EQ(flat.size(), 24);
    
    flat(0) = FixedPointQ5_10(1.5f);
    EXPECT_EQ(flat(0).toFloat(), 1.5f);
}

TEST_F(TensorTest, ConstFlatTensorAccess) {
    auto const_flat = param->const_flat<FixedPointQ5_10>();
    EXPECT_EQ(const_flat.size(), 24);
}

// Matrix Tests
TEST_F(TensorTest, MatrixTensorAccess) {
    auto matrix = param->matrix<FixedPointQ5_10>(6, 4);
    EXPECT_EQ(matrix.dimension(0), 6);
    EXPECT_EQ(matrix.dimension(1), 4);
    
    matrix(0, 0) = FixedPointQ5_10(2.5f);
    EXPECT_EQ(matrix(0, 0).toFloat(), 2.5f);
}

TEST_F(TensorTest, MatrixDimensionMismatch) {
    EXPECT_DEATH(param->matrix<FixedPointQ5_10>(5, 5), "");
}

// 3D Tensor Tests
TEST_F(TensorTest, Tensor3DAccess) {
    auto tensor3d = param->tensor_3d<FixedPointQ5_10>(2, 3, 4);
    EXPECT_EQ(tensor3d.dimension(0), 2);
    EXPECT_EQ(tensor3d.dimension(1), 3);
    EXPECT_EQ(tensor3d.dimension(2), 4);
    
    tensor3d(0, 0, 0) = FixedPointQ5_10(3.5f);
    EXPECT_EQ(tensor3d(0, 0, 0).toFloat(), 3.5f);
}

// 4D Tensor Tests
TEST_F(TensorTest, Tensor4DAccess) {
    auto tensor4d = param->tensor_4d<FixedPointQ5_10>(2, 2, 2, 3);
    EXPECT_EQ(tensor4d.dimension(0), 2);
    EXPECT_EQ(tensor4d.dimension(1), 2);
    EXPECT_EQ(tensor4d.dimension(2), 2);
    EXPECT_EQ(tensor4d.dimension(3), 3);
    
    tensor4d(0, 0, 0, 0) = FixedPointQ5_10(4.5f);
    EXPECT_EQ(tensor4d(0, 0, 0, 0).toFloat(), 4.5f);
}

// Gradient Tests
TEST_F(TensorTest, GradientTensorAccess) {
    auto flat_grad = param->flat_grad<FixedPointQ5_10>();
    auto matrix_grad = param->matrix_grad<FixedPointQ5_10>(6, 4);
    auto tensor3d_grad = param->tensor_3d_grad<FixedPointQ5_10>(2, 3, 4);
    auto tensor4d_grad = param->tensor_4d_grad<FixedPointQ5_10>(2, 2, 2, 3);
    
    flat_grad(0) = FixedPointQ5_10(5.5f);
    EXPECT_EQ(flat_grad(0).toFloat(), 5.5f);
    
    matrix_grad(0, 0) = FixedPointQ5_10(6.5f);
    EXPECT_EQ(matrix_grad(0, 0).toFloat(), 6.5f);
}

// Type Checking Tests
TEST_F(TensorTest, WrongTypeAccess) {
    EXPECT_DEATH(param->flat<float>(), "");
    EXPECT_DEATH(param->matrix<float>(6, 4), "");
    EXPECT_DEATH(param->tensor_3d<float>(2, 3, 4), "");
    EXPECT_DEATH(param->tensor_4d<float>(2, 2, 2, 3), "");
}

TEST(ResidualTest, FixedPointOperations) {
    Parameter x(DT_FIXED, 3);
    Parameter Fx(DT_FIXED, 3);
    Parameter Hx(DT_FIXED, 3);
    
    auto x_span = x.span<FixedPointQ5_10>();
    auto Fx_span = Fx.span<FixedPointQ5_10>();
    
    x_span[0] = FixedPointQ5_10(1.0f);
    Fx_span[0] = FixedPointQ5_10(2.0f);
    
   Residual::Forward(x.const_flat<FixedPointQ5_10>(),    // Changed to const_flat
                     Fx.const_flat<FixedPointQ5_10>(),     // Changed to const_flat
                     Hx.flat<FixedPointQ5_10>());
                     
    EXPECT_EQ(Hx.flat<FixedPointQ5_10>()(0).toFloat(), 3.0f);
}

TEST(ResidualTest, BackwardFixedPoint) {
    // Setup parameters
    Parameter Hx_grad(DT_FIXED, 3);
    Parameter x_grad(DT_FIXED, 3);
    Parameter Fx_grad(DT_FIXED, 3);

    // Initialize gradients
    auto Hx_grad_span = Hx_grad.span<FixedPointQ5_10>();
    Hx_grad_span[0] = FixedPointQ5_10(1.0f);
    Hx_grad_span[1] = FixedPointQ5_10(2.0f);
    Hx_grad_span[2] = FixedPointQ5_10(3.0f);

    // Call backward
    Residual::Backward(Hx_grad.const_flat<FixedPointQ5_10>(),
                      x_grad.flat<FixedPointQ5_10>(),
                      Fx_grad.flat<FixedPointQ5_10>());

    // Verify gradients accumulated correctly
    auto x_grad_span = x_grad.span<FixedPointQ5_10>();
    auto Fx_grad_span = Fx_grad.span<FixedPointQ5_10>();

    for(int i = 0; i < 3; i++) {
        EXPECT_EQ(x_grad_span[i].toFloat(), Hx_grad_span[i].toFloat());
        EXPECT_EQ(Fx_grad_span[i].toFloat(), Hx_grad_span[i].toFloat());
    }
}

TEST(NewGELUTest, ForwardFixedPoint) {
    Parameter x(DT_FIXED, 3);
    Parameter y(DT_FIXED, 3);
    
    auto x_span = x.span<FixedPointQ5_10>();
    x_span[0] = FixedPointQ5_10(1.0f);
    x_span[1] = FixedPointQ5_10(0.0f);
    x_span[2] = FixedPointQ5_10(-1.0f);
    
    NewGELU::Forward(x.const_flat<FixedPointQ5_10>(),
                    y.flat<FixedPointQ5_10>());
    
    auto y_span = y.span<FixedPointQ5_10>();
    EXPECT_NEAR(y_span[0].toFloat(), 0.841f, 0.01f);
    EXPECT_NEAR(y_span[1].toFloat(), 0.0f, 0.01f);
    EXPECT_NEAR(y_span[2].toFloat(), -0.159f, 0.01f);
}

TEST(NewGELUTest, BackwardFixedPoint) {
    Parameter x(DT_FIXED, 2);
    Parameter y_grad(DT_FIXED, 2);
    Parameter x_grad(DT_FIXED, 2);
    
    auto x_span = x.span<FixedPointQ5_10>();
    auto y_grad_span = y_grad.span<FixedPointQ5_10>();
    
    x_span[0] = FixedPointQ5_10(1.0f);
    x_span[1] = FixedPointQ5_10(-1.0f);
    y_grad_span[0] = FixedPointQ5_10(1.0f);
    y_grad_span[1] = FixedPointQ5_10(1.0f);
    
    NewGELU::Backward(x.const_flat<FixedPointQ5_10>(),
                     y_grad.const_flat<FixedPointQ5_10>(),
                     x_grad.flat<FixedPointQ5_10>());
    
    auto x_grad_span = x_grad.span<FixedPointQ5_10>();
    EXPECT_NEAR(x_grad_span[0].toFloat(), 1.083f, 0.01f);
    EXPECT_NEAR(x_grad_span[1].toFloat(), 0.084f, 0.01f);
}
*/

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}