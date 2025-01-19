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

}  // namespace nn


#endif  // LLM_CPP__NN_HPP_