#include <gtest/gtest.h>
#include "tensor_util.hpp"

class TensorUtilTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize test data
        for (int i = 0; i < 24; ++i) {
            test_data[i] = floatX(i);
        }

        

    }

    floatX test_data[24];  // Enough data for up to 4D tensor tests
    const floatX const_test_data[24] {
        floatX(0), floatX(1), floatX(2), floatX(3), floatX(4), floatX(5),
        floatX(6), floatX(7), floatX(8), floatX(9), floatX(10), floatX(11),
        floatX(12), floatX(13), floatX(14), floatX(15), floatX(16), floatX(17),
        floatX(18), floatX(19), floatX(20), floatX(21), floatX(22), floatX(23)};
 
};

// Flat Tensor Tests
TEST_F(TensorUtilTest, FlatTensorCreation) {
    const int length = 6;
    
    // Test non-const flat
    auto flat = MakeFlat(test_data, length);
    EXPECT_EQ(flat.dimension(0), length);
    for (int i = 0; i < length; ++i) {
        EXPECT_FLOAT_EQ(flat(i).to_float(), floatX(i).to_float());
    }

    // Test const flat from non-const pointer
    auto const_flat1 = MakeConstFlat(test_data, length);
    EXPECT_EQ(const_flat1.dimension(0), length);
    for (int i = 0; i < length; ++i) {
        EXPECT_FLOAT_EQ(const_flat1(i).to_float(), floatX(i).to_float());
    }

    // Test const flat from const pointer
    auto const_flat2 = MakeConstFlat(const_test_data, length);
    EXPECT_EQ(const_flat2.dimension(0), length);
    for (int i = 0; i < length; ++i) {
        EXPECT_FLOAT_EQ(const_flat2(i).to_float(), floatX(i).to_float());
    }
}

// Matrix Tests
TEST_F(TensorUtilTest, MatrixCreation) {
    const int rows = 3;
    const int cols = 4;
    
    // Test non-const matrix
    auto matrix = MakeMatrix(test_data, rows, cols);
    EXPECT_EQ(matrix.dimension(0), rows);
    EXPECT_EQ(matrix.dimension(1), cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            EXPECT_FLOAT_EQ(matrix(i, j).to_float(), 
                           floatX(i * cols + j).to_float());
        }
    }

    // Test const matrix from non-const pointer
    auto const_matrix1 = MakeConstMatrix(test_data, rows, cols);
    EXPECT_EQ(const_matrix1.dimension(0), rows);
    EXPECT_EQ(const_matrix1.dimension(1), cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            EXPECT_FLOAT_EQ(const_matrix1(i, j).to_float(), 
                           floatX(i * cols + j).to_float());
        }
    }

    // Test const matrix from const pointer
    auto const_matrix2 = MakeConstMatrix(const_test_data, rows, cols);
    EXPECT_EQ(const_matrix2.dimension(0), rows);
    EXPECT_EQ(const_matrix2.dimension(1), cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            EXPECT_FLOAT_EQ(const_matrix2(i, j).to_float(), 
                           floatX(i * cols + j).to_float());
        }
    }
}

// 3D Tensor Tests
TEST_F(TensorUtilTest, ThreeDTensorCreation) {
    const int dim0 = 2, dim1 = 3, dim2 = 2;
    
    // Test non-const 3D tensor
    auto tensor3d = Make3DTensor(test_data, dim0, dim1, dim2);
    EXPECT_EQ(tensor3d.dimension(0), dim0);
    EXPECT_EQ(tensor3d.dimension(1), dim1);
    EXPECT_EQ(tensor3d.dimension(2), dim2);
    
    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            for (int k = 0; k < dim2; ++k) {
                int index = i * (dim1 * dim2) + j * dim2 + k;
                EXPECT_FLOAT_EQ(tensor3d(i, j, k).to_float(), 
                               floatX(index).to_float());
            }
        }
    }

    // Test const 3D tensor
    auto const_tensor3d = MakeConst3DTensor(const_test_data, dim0, dim1, dim2);
    EXPECT_EQ(const_tensor3d.dimension(0), dim0);
    EXPECT_EQ(const_tensor3d.dimension(1), dim1);
    EXPECT_EQ(const_tensor3d.dimension(2), dim2);

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            for (int k = 0; k < dim2; ++k) {
                int index = i * (dim1 * dim2) + j * dim2 + k;
                EXPECT_FLOAT_EQ(const_tensor3d(i, j, k).to_float(), 
                               floatX(index).to_float());
            }
        }
    }
}

// 4D Tensor Tests
TEST_F(TensorUtilTest, FourDTensorCreation) {
    const int dim0 = 2, dim1 = 2, dim2 = 2, dim3 = 2;
    
    // Test non-const 4D tensor
    auto tensor4d = Make4DTensor(test_data, dim0, dim1, dim2, dim3);
    EXPECT_EQ(tensor4d.dimension(0), dim0);
    EXPECT_EQ(tensor4d.dimension(1), dim1);
    EXPECT_EQ(tensor4d.dimension(2), dim2);
    EXPECT_EQ(tensor4d.dimension(3), dim3);
    
    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            for (int k = 0; k < dim2; ++k) {
                for (int l = 0; l < dim3; ++l) {
                    int index = i * (dim1 * dim2 * dim3) + 
                               j * (dim2 * dim3) + 
                               k * dim3 + l;
                    EXPECT_FLOAT_EQ(tensor4d(i, j, k, l).to_float(), 
                                   floatX(index).to_float());
                }
            }
        }
    }

    // Test const 4D tensor
    auto const_tensor4d = MakeConst4DTensor(const_test_data, dim0, dim1, dim2, dim3);
    EXPECT_EQ(const_tensor4d.dimension(0), dim0);
    EXPECT_EQ(const_tensor4d.dimension(1), dim1);
    EXPECT_EQ(const_tensor4d.dimension(2), dim2);
    EXPECT_EQ(const_tensor4d.dimension(3), dim3);
}

// Edge Cases
TEST_F(TensorUtilTest, EdgeCases) {
    // Test single element tensors
    auto flat_single = MakeFlat(test_data, 1);
    EXPECT_EQ(flat_single.dimension(0), 1);
    
    auto matrix_single = MakeMatrix(test_data, 1, 1);
    EXPECT_EQ(matrix_single.dimension(0), 1);
    EXPECT_EQ(matrix_single.dimension(1), 1);
    
    auto tensor3d_single = Make3DTensor(test_data, 1, 1, 1);
    EXPECT_EQ(tensor3d_single.dimension(0), 1);
    EXPECT_EQ(tensor3d_single.dimension(1), 1);
    EXPECT_EQ(tensor3d_single.dimension(2), 1);
    
    auto tensor4d_single = Make4DTensor(test_data, 1, 1, 1, 1);
    EXPECT_EQ(tensor4d_single.dimension(0), 1);
    EXPECT_EQ(tensor4d_single.dimension(1), 1);
    EXPECT_EQ(tensor4d_single.dimension(2), 1);
    EXPECT_EQ(tensor4d_single.dimension(3), 1);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}