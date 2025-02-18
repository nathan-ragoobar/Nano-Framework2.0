#ifndef LLM_CPP__FASTFEEDFORWARD_HPP_
#define LLM_CPP__FASTFEEDFORWARD_HPP_

#include "nn.hpp"
#include "./sigmoid.hpp"  // Ensure this header file is included

namespace gpt {

struct FastFeedforward {
  using T = floatX;

  explicit FastFeedforward(int n_embed) : n_embed_(n_embed) {
    // Initialize decision node
    decision_fc_ = std::make_unique<nn::Linear>(n_embed, 1);
    
    // Initialize leaf networks
    left_fc_ = std::make_unique<nn::Linear>(n_embed, n_embed);
    right_fc_ = std::make_unique<nn::Linear>(n_embed, n_embed);

    // Activation tensors
    auto dtype = nn::DataTypeToEnum<T>::value;
    decision_act_ = std::make_unique<nn::Activation>(dtype);
    left_act_ = std::make_unique<nn::Activation>(dtype);
    right_act_ = std::make_unique<nn::Activation>(dtype);
    output_act_ = std::make_unique<nn::Activation>(dtype);
  }

  void Forward(typename TTypes<T>::ConstMatrix x,
              typename TTypes<T>::Matrix y) {
    PROFILE_TRACE_FN("FastFeedforward");

    CHECK_EQ(x.dimension(1), n_embed_);
    CHECK_EQ(x.dimension(0), y.dimension(0));
    CHECK_EQ(y.dimension(1), n_embed_);

    int BT = x.dimension(0);
    
    // Lazy allocation for activations
    decision_act_->LazyAllocate(BT);
    left_act_->LazyAllocate(BT * n_embed_);
    right_act_->LazyAllocate(BT * n_embed_);
    output_act_->LazyAllocate(BT * n_embed_);

    // Decision forward pass
    auto decision = decision_act_->matrix<T>(BT, 1);
    decision_fc_->Forward(x, decision);
    nn::Sigmoid::Forward<float>(MakeConstFlat(decision.data(), decision.size()),
                        MakeFlat(decision.data(), decision.size()));

    // Leaf networks forward pass
    auto left_out = left_act_->matrix<T>(BT, n_embed_);
    auto right_out = right_act_->matrix<T>(BT, n_embed_);
    left_fc_->Forward(x, left_out);
    right_fc_->Forward(x, right_out);

    // Mixture output
    auto decision_const = decision_act_->const_matrix<T>(BT, 1);
    auto left_const = left_act_->const_matrix<T>(BT, n_embed_);
    auto right_const = right_act_->const_matrix<T>(BT, n_embed_);
    
    // y = decision * right_out + (1-decision) * left_out
    for (int i = 0; i < BT; i++) {
      for (int j = 0; j < n_embed_; j++) {
        y(i,j) = decision_const(i,0) * right_const(i,j) + 
                 (T(1.0) - decision_const(i,0)) * left_const(i,j);
      }
    }
  }

  void Backward(typename TTypes<T>::ConstMatrix x,
                typename TTypes<T>::ConstMatrix y_grad,
                typename TTypes<T>::Matrix x_grad) {
    PROFILE_TRACE_FN("FastFeedforward");

    CHECK_EQ(x.dimension(1), n_embed_);
    CHECK_EQ(y_grad.dimension(1), n_embed_);
    CHECK_EQ(x_grad.dimension(1), n_embed_);

    int BT = x.dimension(0);

    // Lazy allocate gradients
    decision_act_->LazyAllocateGradient();
    left_act_->LazyAllocateGradient();
    right_act_->LazyAllocateGradient();
    
    // Get activations from forward pass
    auto decision = decision_act_->const_matrix<T>(BT, 1);
    auto left_out = left_act_->const_matrix<T>(BT, n_embed_);
    auto right_out = right_act_->const_matrix<T>(BT, n_embed_);
    
    // Compute gradients for leaf networks
    auto left_grad = left_act_->matrix_grad<T>(BT, n_embed_);
    auto right_grad = right_act_->matrix_grad<T>(BT, n_embed_);
    auto decision_grad = decision_act_->matrix_grad<T>(BT, 1);

    for (int i = 0; i < BT; i++) {
      for (int j = 0; j < n_embed_; j++) {
        left_grad(i,j) = y_grad(i,j) * (T(1.0) - decision(i,0));
        right_grad(i,j) = y_grad(i,j) * decision(i,0);
        decision_grad(i,0) += y_grad(i,j) * (right_out(i,j) - left_out(i,j));
      }
    }
    /*
    // Backward passes through components
    left_fc_->Backward(x, left_grad, x_grad);
    right_fc_->Backward(x, right_grad, x_grad);
    
    auto decision_grad_flat = decision_act_->const_flat_grad<T>();
    auto temp_grad = decision_act_->flat_grad<T>();
    nn::Sigmoid::Backward<float>(decision, decision_grad_flat, temp_grad);
*/
    // Fix 1: Convert gradients to const matrices when passing to Backward
    left_fc_->Backward(x, left_act_->const_matrix_grad<T>(BT, n_embed_), x_grad);
    right_fc_->Backward(x, right_act_->const_matrix_grad<T>(BT, n_embed_), x_grad);
    
    // Fix 2: Convert decision tensor to flat format for Sigmoid
    auto decision_flat = MakeConstFlat(decision.data(), decision.size());
    auto decision_grad_flat = decision_act_->const_flat_grad<T>();
    auto temp_grad = decision_act_->flat_grad<T>();
    nn::Sigmoid::Backward<float>(decision_flat, decision_grad_flat, temp_grad);
    
    typename TTypes<T>::Matrix temp_x_grad = x_grad;
    decision_fc_->Backward(x, decision_act_->const_matrix_grad<T>(BT, 1), temp_x_grad);
  }

  size_t NumParameters() const {
    return decision_fc_->NumParameters() + 
           left_fc_->NumParameters() + 
           right_fc_->NumParameters();
  }

  size_t NumActivations() const {
    return decision_fc_->NumActivations() + 
           left_fc_->NumActivations() + 
           right_fc_->NumActivations() +
           decision_act_->size() +
           left_act_->size() +
           right_act_->size() +
           output_act_->size();
  }

  void Parameters(std::vector<nn::Parameter*>* parameters) const {
    decision_fc_->Parameters(parameters);
    left_fc_->Parameters(parameters);
    right_fc_->Parameters(parameters);
  }

  int n_embed_;
  std::unique_ptr<nn::Linear> decision_fc_;
  std::unique_ptr<nn::Linear> left_fc_;
  std::unique_ptr<nn::Linear> right_fc_;

  // Activation tensors
  std::unique_ptr<nn::Activation> decision_act_;
  std::unique_ptr<nn::Activation> left_act_;
  std::unique_ptr<nn::Activation> right_act_;
  std::unique_ptr<nn::Activation> output_act_;
};

}  // namespace gpt

#endif  // LLM_CPP__FASTFEEDFORWARD_HPP_