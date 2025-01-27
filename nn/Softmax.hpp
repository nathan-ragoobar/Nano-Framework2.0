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


//From what I could see, this only uses the Eigen library ~NR
struct SoftmaxCrossEntropy {
  using T = fixed_point_7pt8;
  enum Reduction { MEAN, SUM };

  explicit SoftmaxCrossEntropy(Reduction reduction = MEAN)
      : reduction_(reduction) {}

  void Forward(typename TTypes<T>::ConstMatrix logits,
               absl::Span<const int> targets, typename TTypes<T>::Matrix probs,
               float* loss) {
    // logits: [B, C], targets: [B,], probs:[B, C], loss: scalar
    int B = logits.dimension(0), C = logits.dimension(1);
    CHECK(B == static_cast<int>(targets.size()) && B == probs.dimension(0));
    CHECK_EQ(C, probs.dimension(1));

    // apply softmax to convert logits to (normalized) probabilities
    Softmax::Forward(logits, probs);

    // targets: [B,]
    *loss = 0.0f;
    for (int i = 0; i < B; ++i) {
      int ix = targets[i];
      // Convert probs(i, ix) to float, do log, accumulate
      float p = probs(i, ix).to_float();
      *loss += -std::log(p > 0.0f ? p : 1e-12f); 
    }

    if (reduction_ == Reduction::MEAN) {
      *loss /= static_cast<float>(B);
    }
  }

  void Backward(typename TTypes<T>::ConstMatrix probs,
                absl::Span<const int> targets,
                typename TTypes<T>::Matrix logits_grad) {
    // probs: [B, C], targets: [B,]
    // logits_grad: [B, C]
    int B = probs.dimension(0), C = probs.dimension(1);
    CHECK(B == static_cast<int>(targets.size()) && B == logits_grad.dimension(0));
    CHECK_EQ(C, logits_grad.dimension(1));

    float factor = (reduction_ == Reduction::MEAN) ? (1.0f / static_cast<float>(B)) : 1.0f;

    for (int b = 0; b < B; ++b) {
      int ix = targets[b];
      for (int c = 0; c < C; ++c) {
        // Convert everything to float, do the subtract, convert back
        float p   = probs(b, c).to_float();
        float ind = (c == ix) ? 1.0f : 0.0f;
        float grad_float = (p - ind) * factor;
        logits_grad(b, c) += T(grad_float);
      }
    }
  }

  static void ForwardAndBackward(typename TTypes<T>::ConstMatrix logits,
                                 typename TTypes<T>::ConstMatrix labels,
                                 typename TTypes<T>::Flat scratch,
                                 typename TTypes<T>::Flat loss,
                                 typename TTypes<T>::Matrix logit_grad) {
    // This version does a softmax in a single pass, then cross-entropy
    // For fixed_point_7pt8, you must either define .exp() & .log()
    // or convert to/from float. Below does conversions to float.

    int B = logits.dimension(0), C = logits.dimension(1);
    CHECK(B == labels.dimension(0) && C == labels.dimension(1));
    CHECK(B == logit_grad.dimension(0) && C == logit_grad.dimension(1));
    CHECK_EQ(B, scratch.size());
    CHECK_EQ(B, loss.size());

    Eigen::array<Eigen::Index, 1> along_class = {1};
    Eigen::array<Eigen::Index, 2> batch_by_one = {B, 1};
    Eigen::array<Eigen::Index, 2> one_by_class = {1, C};

    // 1) Compute stable logits = logits - max along each row
    scratch.device(g_device) = logits.maximum(along_class);
    logit_grad.device(g_device) =
        logits - scratch.reshape(batch_by_one).broadcast(one_by_class);

    // 2) sum(exp(stable_logits)) along classes
    scratch.device(g_device) = logit_grad.exp().sum(along_class);

    // 3) negative log likelihood
    //    loss = sum(labels * ( - log softmax ))
    //    or any variation that matches your equation
    loss.device(g_device) =
        (labels * (scratch.log().reshape(batch_by_one).broadcast(one_by_class) -
                   logit_grad))
            .sum(along_class);

    // 4) final gradient wrt logits
    //    logit_grad = softmax(logits) - labels
    logit_grad.device(g_device) =
        (logit_grad.exp() /
         scratch.reshape(batch_by_one).broadcast(one_by_class))
        - labels;

    // NOTE: If your T doesn’t have .exp() / .log(), do conversions, e.g.:
    // logit_grad(i,j) = T(std::exp(logit_grad(i,j).to_float()));
  }

  Reduction reduction_;
};


}  // namespace nn

#endif  // LLM_CPP__NN_HPP_