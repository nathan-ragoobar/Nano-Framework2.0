#ifndef LLM_CPP__LOSS_HPP_
#define LLM_CPP__LOSS_HPP_

#include <cmath>
#include "./../tensor/fixed_point.hpp" 
#include "./Parameter.hpp"  // Includes necessary headers for nn::Parameter, etc.
#include "absl/types/span.h"
#include "absl/log/check.h"
#include "tensor/tensor_util.hpp"

namespace nn {

struct CrossEntropy {
  using T = fixed_point_7pt8;
  enum Reduction { MEAN, SUM };

  CrossEntropy(Reduction reduction = Reduction::MEAN) : reduction_(reduction) {}

  void Forward(typename TTypes<T>::ConstMatrix probs,
               absl::Span<const int> targets, float* loss) {
    // probs:[B, C], targets: [B,] loss: scalar
    int B = probs.dimension(0), C = probs.dimension(1);
    CHECK_EQ(B, targets.size());

    *loss = 0.0f;

    // targets: [B,]
    for (int i = 0; i < targets.size(); ++i) {
      int ix = targets[i];
      // Convert probs(i, ix) to float, do log, accumulate
      float p = probs(i, ix).to_float();
      *loss += -std::log(p > 0.0f ? p : 1e-12f); //If prob is negative or zero, set to 1e-12
    }

    if (reduction_ == Reduction::MEAN) {
      *loss /= static_cast<float>(B);
    }
  }

  void Backward(typename TTypes<T>::ConstMatrix probs,
                absl::Span<const int> targets,
                typename TTypes<T>::Matrix probs_grad) {
    // probs: [B, C], targets: [B,]
    // probs_grad: [B, C]
    int B = probs.dimension(0), C = probs.dimension(1);
    CHECK(B == targets.size() && B == probs_grad.dimension(0));
    CHECK_EQ(C, probs_grad.dimension(1));

    float factor =
        reduction_ == Reduction::MEAN ? 1.0f / static_cast<float>(B) : 1.0f;

    for (int b = 0; b < B; ++b) {
      int ix = targets[b];
      // Convert probs(b, ix) to float, do the division, convert back
      float p = probs(b, ix).to_float();
      float grad_float = -1.0f / (p > 0.0f ? p : 1e-12f) * factor;
      probs_grad(b, ix) += T(grad_float);
    }
  }

  Reduction reduction_;
};


}  // namespace nn

#endif  // LLM_CPP__NN_HPP_