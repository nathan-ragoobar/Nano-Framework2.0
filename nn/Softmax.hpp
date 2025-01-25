#ifndef LLM_CPP__SOFTMAX_HPP_
#define LLM_CPP__SOFTMAX_HPP_

#include <cmath>
#include "tensor/tensor_util.hpp"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "Parameter.hpp"


namespace nn {

struct Softmax {
  using T = fixed_point_7pt8;

  static void Forward(typename TTypes<T>::ConstMatrix x,
                      typename TTypes<T>::Matrix y) {
    // x: [B, D], y: [B, D]
    CHECK_EQ(x.dimension(0), y.dimension(0));
    CHECK_EQ(x.dimension(1), y.dimension(1));

    int batch_size = x.dimension(0), num_class = x.dimension(1);
    Eigen::array<Eigen::Index, 1> along_class = {1};
    Eigen::array<Eigen::Index, 2> batch_by_one = {batch_size, 1};
    Eigen::array<Eigen::Index, 2> one_by_class = {1, num_class};

    // Compute the maximum value along the class dimension
    auto max_val = x.maximum(along_class).eval().reshape(batch_by_one).broadcast(one_by_class);

    // Compute the exponentials
    y.device(g_device) = (x - max_val).exp();

    // Normalize by the sum of exponentials
    auto sum_exp = y.sum(along_class).inverse().eval().reshape(batch_by_one).broadcast(one_by_class);
    y.device(g_device) = y * sum_exp;
  }

  static void Backward(typename TTypes<T>::ConstMatrix y,
                       typename TTypes<T>::ConstMatrix y_grad,
                       typename TTypes<T>::Matrix x_grad) {
    // y:[B, D], y_grad: [B, D], x_grad: [B, D]
    int B = y.dimension(0), D = y.dimension(1);
    CHECK(B == y_grad.dimension(0) && B == x_grad.dimension(0));
    CHECK(D == y_grad.dimension(1) && D == x_grad.dimension(1));

    // Using alternative formula:
    // dL/dx = dL/dy * y - sum(dL/dy * y) * y
    //    = (dL/dy - sum(dL/dy * y)) * y
    int batch_size = y.dimension(0), num_class = y.dimension(1);
    Eigen::array<Eigen::Index, 2> batch_by_one = {batch_size, 1};
    Eigen::array<Eigen::Index, 2> one_by_class = {1, num_class};
    Eigen::array<Eigen::Index, 1> along_class = {1};

    auto dyy = y_grad * y;
    auto sum = dyy.sum(along_class).reshape(batch_by_one);
    auto sub = y_grad - sum.broadcast(one_by_class);
    x_grad.device(g_device) += sub * y;
  }
};


}  // namespace nn

#endif  // LLM_CPP__NN_HPP_