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






}  // namespace nn


#endif  // LLM_CPP__NN_HPP_