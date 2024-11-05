#ifndef LLM_CPP__NN_HPP_
#define LLM_CPP__NN_HPP_

#include <unistd.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>

#include "../eigen/unsupported/Eigen/CXX11/Tensor"
#include "../eigen/unsupported/Eigen/CXX11/ThreadPool"

#include "nanolib/utils/rand.h"
#include "nanolib/tensor/tensor_util.hpp"


/* #include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h" */
#include "../abseil-cpp/absl/algorithm/container.h"
#include "../abseil-cpp/absl/log/check.h"
#include "../abseil-cpp/absl/log/log.h"
#include "../abseil-cpp/absl/strings/string_view.h"
#include "../abseil-cpp/absl/types/span.h"



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

inline void ConstantFill(absl::Span<float> weight, float C) {
#ifdef EIGEN_USE_GPU
  std::vector<float> w(weight.size(), C);
  g_device.memcpyHostToDevice(weight.data(), w.data(),
                              sizeof(float) * w.size());
#else
  absl::c_fill(weight, C);
#endif
}

inline void UniformFill(absl::Span<float> weight, float from = 0.0,
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

inline void NormalFill(absl::Span<float> weight, float mean = 0.0,
                       float std = 1.0) {
#ifdef EIGEN_USE_GPU
  std::vector<float> w(weight.size());
  normal_(w.data(), w.size(), mean, std, &g_mt19937_state);
  g_device.memcpyHostToDevice(weight.data(), w.data(),
                              sizeof(float) * w.size());
#else
  normal_(weight.data(), weight.size(), mean, std, &g_mt19937_state);
#endif
}

inline void KaimingUniformFill(absl::Span<float> weight, int in_features) {
  const float bound = std::sqrt(1.0f / in_features);
#ifdef EIGEN_USE_GPU
  std::vector<float> w(weight.size());
  uniform_(w.data(), w.size(), -bound, bound, &g_mt19937_state);
  g_device.memcpyHostToDevice(weight.data(), w.data(),
                              sizeof(float) * w.size());
#else
  uniform_(weight.data(), weight.size(), -bound, bound, &g_mt19937_state);
#endif
}

inline void UpperTriangularWithNegativeInf(
    typename TTypes<float>::Matrix matrix) {
  using MatrixXf =
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  MatrixXf m = MatrixXf::Zero(matrix.dimension(0), matrix.dimension(1));
  m.triangularView<Eigen::StrictlyUpper>().setConstant(
      -std::numeric_limits<float>::infinity());
#ifdef EIGEN_USE_GPU
  g_device.memcpyHostToDevice(matrix.data(), m.data(),
                              sizeof(float) * matrix.size());
#else
  g_device.memcpy(matrix.data(), m.data(), sizeof(float) * matrix.size());
#endif
}

inline void OntHot(typename TTypes<int>::ConstFlat target,
                   typename TTypes<float>::Matrix label) {
  int batch_size = target.size(), num_class = label.dimension(1);
  CHECK_EQ(batch_size, label.dimension(0));
  for (int i = 0; i < batch_size; ++i) {
    int ix = target(i);
    CHECK_LT(ix, num_class);
    label(i, ix) = 1.0f;
  }
}

inline std::pair<int, int> SplitRange(int total, int idx, int n) {
  int q = total / n;
  int r = total % n;
  if (idx < r) {
    return {(q + 1) * idx, (q + 1) * (idx + 1)};
  } else {
    return {q * idx + r, q * (idx + 1) + r};
  }
}

enum DataType : int { DT_FLOAT = 1, DT_HALF = 2, DT_INT32 = 3 };

// Validates type T for whether it is a supported DataType.
template <class T>
struct IsValidDataType;

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

// Parameter weight and its corresponding gradient


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
  using T = floatX;

  static void Forward(typename TTypes<T>::ConstFlat x,
                      typename TTypes<T>::Flat y) {
    CHECK_EQ(x.size(), y.size());
    const float sqrt_2_over_pi = std::sqrt(M_2_PI);

    // y = 0.5 * x * (1.0 + tanh[sqrt(2/pi) * (x + 0.044715 * x^3)])
    float coeff = 0.044715f;
    y.device(g_device) =
        0.5 * x * (1.0 + ((sqrt_2_over_pi * (x + coeff * x * x * x)).tanh()));
  }

  static void Backward(typename TTypes<T>::ConstFlat x,
                       typename TTypes<T>::ConstFlat y_grad,
                       typename TTypes<T>::Flat x_grad) {
    CHECK_EQ(x.size(), y_grad.size());
    CHECK_EQ(x.size(), x_grad.size());

    // dL/dx = dL/dy * dy/dx
    //       = dL/dy * [ 0.5 * (1.0 + tanh[sqrt(2/pi) * (x + 0.044715 * x^3)])
    //                 + 0.5 * x * (1 - (tanh[sqrt(2/pi) * (x + 0.044715 *
    //                 x^3)])^2
    //                           *  (sqrt(2/pi) * (1 + 0.044715 * 3 * x^2))
    //                             )
    //                 ]

    const float sqrt_2_over_pi = std::sqrt(M_2_PI);
    float coeff = 0.044715f;
    auto cube = coeff * x * x * x;
    auto tanh_arg = sqrt_2_over_pi * (x + cube);
    auto tanh_out = tanh_arg.tanh();
    auto dydx = 0.5f * (1.0f + tanh_out) +
                0.5f * x * (1.0f - tanh_out * tanh_out) *
                    (sqrt_2_over_pi * (1.0f + 3.0f * coeff * x * x));
    x_grad.device(g_device) += y_grad * dydx;
  }
};





}  // namespace nn

#endif  // LLM_CPP__NN_HPP_
