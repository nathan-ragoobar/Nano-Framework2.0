#ifndef LLM_CPP__NN_HPP_
#define LLM_CPP__NN_HPP_

#include "./../tensor/fixed_point.hpp"  // Add at top with other includes

#include <unistd.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>

#include "llmc/rand.h"
#include "tensor/tensor_util.hpp"


/* #include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h" */
#include "abseil-cpp/absl/algorithm/container.h"
#include "abseil-cpp/absl/log/check.h"
#include "abseil-cpp/absl/log/log.h"
#include "abseil-cpp/absl/strings/string_view.h"
#include "abseil-cpp/absl/types/span.h"

namespace nn {



#ifdef EIGEN_USE_GPU
Eigen::GpuStreamDevice g_stream;
Eigen::GpuDevice g_device(&g_stream);
#else
Eigen::ThreadPool g_thread_pool(16 /* number of threads in pool */);
Eigen::ThreadPoolDevice g_device(&g_thread_pool,
                                 12 /* number of threads to use */);
#endif

mt19937_state g_mt19937_state;

inline void ManualSeed(unsigned int seed) {
  manual_seed(&g_mt19937_state, seed);
}

//Fills a span of memory with a constant value
template <typename T>
inline void ConstantFill(absl::Span<T> weight, T C) {
#ifdef EIGEN_USE_GPU
  std::vector<float> w(weight.size(), C);
  g_device.memcpyHostToDevice(weight.data(), w.data(),
                              sizeof(float) * w.size());
#else
  absl::c_fill(weight, C);
#endif
}

//Fills array with random values in range [from, to]
//Uses MT19937 random generator
template <typename T>
inline void UniformFill(absl::Span<T> weight, T from = 0.0,  // For the Fixed Point Datatype
                        T to = 1.0) {
#ifdef EIGEN_USE_GPU
  std::vector<T> w(weight.size());
  uniform_fixed(w.data(), w.size(), from, to, &g_mt19937_state);
  g_device.memcpyHostToDevice(weight.data(), w.data(),
                              sizeof(float) * w.size());
#else
  uniform_fixed(weight.data(), weight.size(), from, to, &g_mt19937_state);
#endif
}

inline void UniformFill(absl::Span<float> weight, float from = 0.0,     // For the float datatype      
                        float to = 1.0) {
#ifdef EIGEN_USE_GPU
  std::vector<float> w(weight.size());
  uniform_(w.data(), w.size(), from, to, &g_mt19937_state);
  g_device.memcpyHostToDevice(weight.data(), w.data(),
                              sizeof(float) * w.size());
#else
  uniform_(weight.data(), weight.size(), from, to, &g_mt19937_state);
#endif
}


//function fills a weight tensor with values sampled from a normal (Gaussian) distribution with specified mean and standard deviation. The function has GPU and CPU implementations.
// Add template for type flexibility
template <typename T>
inline void NormalFill(absl::Span<T> weight, T mean = 0.0,
                       T std = 1.0) {
#ifdef EIGEN_USE_GPU
  std::vector<float> w(weight.size());
  normal_fixed(w.data(), w.size(), mean, std, &g_mt19937_state);
  g_device.memcpyHostToDevice(weight.data(), w.data(),
                              sizeof(float) * w.size());
#else
  normal_fixed(weight.data(), weight.size(), mean, std, &g_mt19937_state);
#endif
}

//Implementation of the Kaiming uniform initialization for a weight tensor. The function fills the weight tensor with values sampled from a uniform distribution with bounds calculated based on the number of input features. The function has GPU and CPU implementations.
inline void KaimingUniformFill(absl::Span<FixedPointQ5_10> weight, int in_features) {
    const FixedPointQ5_10 one(1.0f);
    const FixedPointQ5_10 in_features_fp(static_cast<float>(in_features));
    const FixedPointQ5_10 bound = FixedPointQ5_10::sqrt(one / in_features_fp);

#ifdef EIGEN_USE_GPU
    std::vector<FixedPointQ5_10> w(weight.size());
    uniform_fixed(w.data(), w.size(), -bound, bound, &g_mt19937_state);
    g_device.memcpyHostToDevice(weight.data(), w.data(),
                               sizeof(FixedPointQ5_10) * w.size());
#else
    uniform_fixed(weight.data(), weight.size(), -bound, bound, &g_mt19937_state);
#endif
}


//Creates upper triangle matrix, sets upper triangle to negative infinity, sets diagonal and lower triangle to zero
inline void UpperTriangularWithNegativeInf(
    typename TTypes<FixedPointQ5_10>::Matrix matrix) {
    
    using MatrixFixed = 
        Eigen::Matrix<FixedPointQ5_10, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    
    MatrixFixed m = MatrixFixed::Zero(matrix.dimension(0), matrix.dimension(1));
    
    // Use minimum fixed point value instead of infinity
    m.triangularView<Eigen::StrictlyUpper>().setConstant(
        FixedPointQ5_10(-32.0f));  // Minimum value for Q5.10 format

#ifdef EIGEN_USE_GPU
    g_device.memcpyHostToDevice(matrix.data(), m.data(),
                               sizeof(FixedPointQ5_10) * m.size());
#else
    g_device.memcpy(matrix.data(), m.data(), 
                    sizeof(FixedPointQ5_10) * m.size());
#endif
}

//Performs OneHot encoding of target tensor 
inline void OneHot(typename TTypes<int>::ConstFlat target,
                       typename TTypes<FixedPointQ5_10>::Matrix label) {
    int batch_size = target.size();
    int num_class = label.dimension(1);
    
    CHECK_EQ(batch_size, label.dimension(0));
    
    for (int i = 0; i < batch_size; ++i) {
        int ix = target(i);
        CHECK_LT(ix, num_class);
        label(i, ix) = FixedPointQ5_10(1.0f);
    }
}

//Splits a range into n parts and returns the start and end index of the i-th part
//Used for parallel processing of data
//Deals with array indices so no need to switch datatype to FixedPointQ5_10
inline std::pair<int, int> SplitRange(int total, int idx, int n) {
  int q = total / n;
  int r = total % n;
  if (idx < r) {
    return {(q + 1) * idx, (q + 1) * (idx + 1)};
  } else {
    return {q * idx + r, q * (idx + 1) + r};
  }
}

//Supported data types
enum DataType : int { 
    DT_FLOAT = 1, 
    DT_HALF = 2, 
    DT_INT32 = 3,
    DT_FIXED = 4  // Add new enum value
};

// Validates type T for whether it is a supported DataType.
// Add base template here
template <typename T>
struct IsValidDataType {
    static constexpr bool value = false;  // Default to false for unknown types
};
// DataTypeToEnum<T>::v() and DataTypeToEnum<T>::value are the DataType
// constants for T, e.g. DataTypeToEnum<float>::v() is DT_FLOAT.
template <class T>
struct DataTypeToEnum {
  static_assert(IsValidDataType<T>::value, "Specified Data Type not supported");
};  // Specializations below

// EnumToDataType<VALUE>::Type is the type for DataType constant VALUE, e.g.
// EnumToDataType<DT_FLOAT>::Type is float.
template <DataType VALUE>
struct EnumToDataType {};  // Specializations below

// Template specialization for both DataTypeToEnum and EnumToDataType.
#define MATCH_TYPE_AND_ENUM(TYPE, ENUM)     \
  template <>                               \
  struct DataTypeToEnum<TYPE> {             \
    static DataType v() { return ENUM; }    \
    static constexpr DataType value = ENUM; \
  };                                        \
  template <>                               \
  struct IsValidDataType<TYPE> {            \
    static constexpr bool value = true;     \
  };                                        \
  template <>                               \
  struct EnumToDataType<ENUM> {             \
    typedef TYPE Type;                      \
  }




MATCH_TYPE_AND_ENUM(float, DT_FLOAT);
MATCH_TYPE_AND_ENUM(Eigen::half, DT_HALF);
MATCH_TYPE_AND_ENUM(int, DT_INT32);
MATCH_TYPE_AND_ENUM(FixedPointQ5_10, DT_FIXED);  // Add new mapping



// Parameter weight and its corresponding gradient
struct Parameter {
  Parameter(const Parameter&) = delete;
  Parameter& operator=(const Parameter&) = delete;

  explicit Parameter(DataType dtype, int64_t num_element = 0)
      : dtype_(dtype),
        num_element_(num_element),
        data_(nullptr),
        grad_(nullptr) {
    if (num_element) {
      LazyAllocate(num_element);
    }
  }

  ~Parameter() {
    if (data_ != nullptr) {
      g_device.deallocate(data_);
    }
    if (grad_ != nullptr) {
      g_device.deallocate(grad_);
    }
  }

  int64_t size() const { return num_element_; }

  void LazyAllocate(int num_element) {
    if (data_ == nullptr) {
      data_ = Allocate(dtype_, num_element);
      Zero(data_, dtype_, num_element);
      num_element_ = num_element;
    }
    CHECK_EQ(num_element, num_element_);
  }

 
  void LazyAllocateGradient() {
    if (grad_ == nullptr) {
      CHECK_GT(num_element_, 0);
      grad_ = Allocate(dtype_, num_element_);
      Zero(grad_, dtype_, num_element_);
    }
  }

  void ZeroData() {
    if (data_ != nullptr) {
      Zero(data_, dtype_, num_element_);
    }
  }

  void ZeroGrad() {
    if (grad_ != nullptr) {
      Zero(grad_, dtype_, num_element_);
    }
  }

  template <typename T>
  T* data() const {
    return static_cast<T*>(data_);
  }

  template <typename T>
  T* grad() const {
    return static_cast<T*>(grad_);
  }

  template <typename T>
  absl::Span<T> span() const {
    CHECK_EQ(DataTypeToEnum<T>::value, dtype_);
    return {data<T>(), static_cast<size_t>(num_element_)};
  }

  template <typename T>
  absl::Span<T> span_grad() const {
    CHECK_EQ(DataTypeToEnum<T>::value, dtype_);
    return {grad<T>(), static_cast<size_t>(num_element_)};
  }
 
  template <typename T>
  typename TTypes<T>::Flat flat() const {
    CHECK_EQ(DataTypeToEnum<T>::value, dtype_);
    return {data<T>(), num_element_};
  }
  template <typename T>
  typename TTypes<T>::ConstFlat const_flat() const {
    CHECK_EQ(DataTypeToEnum<T>::value, dtype_);
    return {data<T>(), num_element_};
  }

  template <typename T>
  typename TTypes<T>::Matrix matrix(int rows, int cols) const {
    CHECK_EQ(DataTypeToEnum<T>::value, dtype_);
    CHECK_EQ(rows * cols, num_element_);
    return {data<T>(), rows, cols};
  }
  template <typename T>
  typename TTypes<T>::ConstMatrix const_matrix(int rows, int cols) const {
    CHECK_EQ(DataTypeToEnum<T>::value, dtype_);
    CHECK_EQ(rows * cols, num_element_);
    return {data<T>(), rows, cols};
  }

  template <typename T>
  typename TTypes<T, 3>::Tensor tensor_3d(int dim0, int dim1, int dim2) const {
    CHECK_EQ(DataTypeToEnum<T>::value, dtype_);
    CHECK_EQ(dim0 * dim1 * dim2, num_element_);
    return {data<T>(), dim0, dim1, dim2};
  }
  template <typename T>
  typename TTypes<T, 3>::ConstTensor const_tensor_3d(int dim0, int dim1,
                                                     int dim2) const {
    CHECK_EQ(DataTypeToEnum<T>::value, dtype_);
    CHECK_EQ(dim0 * dim1 * dim2, num_element_);
    return {data<T>(), dim0, dim1, dim2};
  }

  template <typename T>
  typename TTypes<T, 4>::Tensor tensor_4d(int dim0, int dim1, int dim2,
                                          int dim3) const {
    CHECK_EQ(DataTypeToEnum<T>::value, dtype_);
    CHECK_EQ(dim0 * dim1 * dim2 * dim3, num_element_);
    return {data<T>(), dim0, dim1, dim2, dim3};
  }
  template <typename T>
  typename TTypes<T, 4>::ConstTensor const_tensor_4d(int dim0, int dim1,
                                                     int dim2, int dim3) const {
    CHECK_EQ(DataTypeToEnum<T>::value, dtype_);
    CHECK_EQ(dim0 * dim1 * dim2 * dim3, num_element_);
    return {data<T>(), dim0, dim1, dim2, dim3};
  }

  template <typename T>
  typename TTypes<T>::Flat flat_grad() const {
    CHECK_EQ(DataTypeToEnum<T>::value, dtype_);
    return {grad<T>(), num_element_};
  }
  template <typename T>
  typename TTypes<T>::ConstFlat const_flat_grad() const {
    CHECK_EQ(DataTypeToEnum<T>::value, dtype_);
    return {grad<T>(), num_element_};
  }

  template <typename T>
  typename TTypes<T>::Matrix matrix_grad(int rows, int cols) const {
    CHECK_EQ(DataTypeToEnum<T>::value, dtype_);
    CHECK_EQ(rows * cols, num_element_);
    return {grad<T>(), rows, cols};
  }
  template <typename T>
  typename TTypes<T>::ConstMatrix const_matrix_grad(int rows, int cols) const {
    CHECK_EQ(DataTypeToEnum<T>::value, dtype_);
    CHECK_EQ(rows * cols, num_element_);
    return {grad<T>(), rows, cols};
  }

  template <typename T>
  typename TTypes<T, 3>::Tensor tensor_3d_grad(int dim0, int dim1,
                                               int dim2) const {
    CHECK_EQ(DataTypeToEnum<T>::value, dtype_);
    CHECK_EQ(dim0 * dim1 * dim2, num_element_);
    return {grad<T>(), dim0, dim1, dim2};
  }
  template <typename T>
  typename TTypes<T, 3>::ConstTensor const_tensor_3d_grad(int dim0, int dim1,
                                                          int dim2) const {
    CHECK_EQ(DataTypeToEnum<T>::value, dtype_);
    CHECK_EQ(dim0 * dim1 * dim2, num_element_);
    return {grad<T>(), dim0, dim1, dim2};
  }

  template <typename T>
  typename TTypes<T, 4>::Tensor tensor_4d_grad(int dim0, int dim1, int dim2,
                                               int dim3) const {
    CHECK_EQ(DataTypeToEnum<T>::value, dtype_);
    CHECK_EQ(dim0 * dim1 * dim2 * dim3, num_element_);
    return {grad<T>(), dim0, dim1, dim2, dim3};
  }

  template <typename T>
  typename TTypes<T, 4>::ConstTensor const_tensor_4d_grad(int dim0, int dim1,
                                                          int dim2,
                                                          int dim3) const {
    CHECK_EQ(DataTypeToEnum<T>::value, dtype_);
    CHECK_EQ(dim0 * dim1 * dim2 * dim3, num_element_);
    return {grad<T>(), dim0, dim1, dim2, dim3};
  }
  /**/

 private:
  static void* Allocate(DataType dtype, int64_t num_element) {
    if (dtype == DT_FIXED) {
      return g_device.allocate(sizeof(FixedPointQ5_10) * num_element);
    } else if (dtype == DT_FLOAT) {
      return g_device.allocate(sizeof(float) * num_element);
    } else if (dtype == DT_HALF) {
      return g_device.allocate(sizeof(Eigen::half) * num_element);
    } else {
      throw std::invalid_argument("invalid data type: " +
                                  std::to_string(dtype));
    }
  }

  static void Zero(void* data, DataType dtype, int64_t num_element) {
    if (dtype == DT_FIXED) {
      g_device.memset(data, 0, sizeof(FixedPointQ5_10) * num_element);
    } else if (dtype == DT_FLOAT) {
      g_device.memset(data, 0, sizeof(float) * num_element);
    } else if (dtype == DT_HALF) {
      g_device.memset(data, 0, sizeof(Eigen::half) * num_element);
    } else {
      throw std::invalid_argument("invalid data type: " +
                                  std::to_string(dtype));
    }
  }

  DataType dtype_;
  int64_t num_element_;
  void* data_;
  void* grad_;
};

using Activation = Parameter;

struct Residual {
  using T = floatX;

  static void Forward(typename TTypes<T>::ConstFlat x,
                      typename TTypes<T>::ConstFlat Fx,
                      typename TTypes<T>::Flat Hx) {
    int N = x.size();
    CHECK(N == Fx.size() && N == Hx.size());

    // H(x) = x + F(x) -> F(x) = H(x) - x
    Hx.device(g_device) = x + Fx;
  }

  static void Backward(typename TTypes<T>::ConstFlat Hx_grad,
                       typename TTypes<T>::Flat x_grad,
                       typename TTypes<T>::Flat Fx_grad) {
    int N = Hx_grad.size();
    CHECK(N == x_grad.size() && N == Fx_grad.size());

    x_grad.device(g_device) += Hx_grad;
    Fx_grad.device(g_device) += Hx_grad;
  }
};

// Careful there are a few versions of GeLU, this one is the exact one used by
// OpenAI
struct NewGELU {
  using T = FixedPointQ5_10;

  static void Forward(typename TTypes<T>::ConstFlat x,
                      typename TTypes<T>::Flat y) {
    CHECK_EQ(x.size(), y.size());
    const T sqrt_2_over_pi(std::sqrt(M_2_PI));

    // y = 0.5 * x * (1.0 + tanh[sqrt(2/pi) * (x + 0.044715 * x^3)])
    const T half(0.5f);
    const T one(1.0f);
    const T coeff(0.044715f);
    
    y.device(g_device) =
        half * x * (one + ((sqrt_2_over_pi * (x + coeff * x * x * x)).tanh()));
  }

  static void Backward(typename TTypes<T>::ConstFlat x,
                       typename TTypes<T>::ConstFlat y_grad,
                       typename TTypes<T>::Flat x_grad) {
    CHECK_EQ(x.size(), y_grad.size());
    CHECK_EQ(x.size(), x_grad.size());

    const T sqrt_2_over_pi(FixedPointQ5_10::sqrt(M_2_PI));
    const T half(0.5f);
    const T one(1.0f);
    const T three(3.0f);
    const T coeff(0.044715f);

    auto cube = coeff * x * x * x;
    auto tanh_arg = sqrt_2_over_pi * (x + cube);
    auto tanh_out = tanh_arg.tanh();
    auto dydx = half * (one + tanh_out) +
                half * x * (one - tanh_out * tanh_out) *
                    (sqrt_2_over_pi * (one + three * coeff * x * x));
    x_grad.device(g_device) += y_grad * dydx;
  }
};




}  // namespace nn


#endif  // LLM_CPP__NN_HPP_