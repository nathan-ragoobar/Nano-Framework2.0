#ifndef LLM_CPP__EMBEDDING_HPP_
#define LLM_CPP__EMBEDDING_HPP_

#include <memory>
#include "tensor/fixed_point.hpp"
#include "tensor/tensor_util.hpp"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "Parameter.hpp"
//#include <unsupported/Eigen/CXX11/Tensor>


namespace nn {

struct Embedding {
    using T = fixed_point_15pt16;

  Embedding(int num_embeddings, int embedding_dim)
      : num_embeddings_(num_embeddings), embedding_dim_(embedding_dim) {
    weight_ =
        std::make_unique<Parameter>(DT_FIXED, num_embeddings * embedding_dim);
    NormalFill(weight_->span<T>());
  }

  void Forward(absl::Span<const int> idx, absl::Span<T> embedding) const {
    CHECK_EQ(embedding.size(), idx.size() * embedding_dim_);
    for (size_t i = 0; i < idx.size(); ++i) {
      CHECK_LT(idx[i], num_embeddings_);
      void* dst = embedding.data() + i * embedding_dim_;
      void* src = weight_->data<float>() + idx[i] * embedding_dim_;
      g_device.memcpy(dst, src, sizeof(T) * embedding_dim_);
    }
  }

  void Backward(absl::Span<const int> idx,
                absl::Span<const T> grad_embedding) {
    CHECK_EQ(grad_embedding.size(), idx.size() * embedding_dim_);

    // Lazily allocate the memory for gradients
    weight_->LazyAllocateGradient();
    for (size_t i = 0; i < idx.size(); ++i) {
      CHECK_LT(idx[i], num_embeddings_);
      const T* g = grad_embedding.data() + i * embedding_dim_;
      T* grad = weight_->grad<T>() + idx[i] * embedding_dim_;
      auto g_1d = TTypes<T>::UnalignedConstFlat(g, embedding_dim_);
      auto grad_1d = TTypes<T>::UnalignedFlat(grad, embedding_dim_);
      grad_1d.device(g_device) += g_1d;
    }
  }

  size_t NumParameters() const { return num_embeddings_ * embedding_dim_; }

  size_t NumActivations() const { return 0; }

  void Parameters(std::vector<Parameter*>* parameters) const {
    parameters->push_back(weight_.get());
  }

  int num_embeddings_;
  int embedding_dim_;
  std::unique_ptr<Parameter> weight_;
};

}  // namespace nn

#endif  // LLM_CPP__NN_HPP_