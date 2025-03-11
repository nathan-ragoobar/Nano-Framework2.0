#ifndef LLM_CPP__FAST_FEEDFORWARD_NETWORK_HPP_
#define LLM_CPP__FAST_FEEDFORWARD_NETWORK_HPP_

#include "nn.hpp"
#include "sigmoid.hpp"

#ifdef EIGEN_USE_GPU
#include "cuda_profile_util.hpp"
#define PROFILE_TRACE_FN(prefix) NVTX_RANGE_FN(prefix)
#else
#define PROFILE_TRACE_FN(prefix)
#endif

namespace gpt {

// Decision node that determines routing between leaves
struct DecisionNode {
  using T = floatX;
  
  explicit DecisionNode(int input_size) {
    // Decision layer is a single output linear layer
    decision_ = std::make_unique<nn::Linear>(input_size, 1);
    
    // For storing intermediate values
    auto dtype = nn::DataTypeToEnum<T>::value;
    decision_output_ = std::make_unique<nn::Activation>(dtype);
    sigmoid_output_ = std::make_unique<nn::Activation>(dtype);
  }
  
  void Forward(typename TTypes<T>::ConstMatrix x,
               typename TTypes<T>::Matrix choice) {
    PROFILE_TRACE_FN("DecisionNode");
    
    int BT = x.dimension(0);
    decision_output_->LazyAllocate(BT);
    sigmoid_output_->LazyAllocate(BT);
    
    // Linear projection to scalar
    auto decision_out = decision_output_->matrix<T>(BT, 1);
    decision_->Forward(x, decision_out);
    
    // Apply sigmoid for soft routing decision
    auto sigmoid_out = sigmoid_output_->matrix<T>(BT, 1);
    nn::Sigmoid::Forward<T>(MakeConstFlat(decision_out.data(), decision_out.size()),
                         MakeFlat(sigmoid_out.data(), sigmoid_out.size()));
    
    // Copy to output
    for (int b = 0; b < BT; ++b) {
      choice(b, 0) = sigmoid_out(b, 0);
    }
  }
  
  void Backward(typename TTypes<T>::ConstMatrix x,
                typename TTypes<T>::ConstMatrix choice_grad,
                typename TTypes<T>::Matrix x_grad) {
    PROFILE_TRACE_FN("DecisionNode");
    
    int BT = x.dimension(0);
    decision_output_->LazyAllocateGradient();
    sigmoid_output_->LazyAllocateGradient();
    decision_output_->ZeroGrad();
    sigmoid_output_->ZeroGrad();
    
    // Backprop through sigmoid
    auto sigmoid_out = sigmoid_output_->const_matrix<T>(BT, 1);
    auto sigmoid_grad = sigmoid_output_->matrix_grad<T>(BT, 1);
    
    for (int b = 0; b < BT; ++b) {
      sigmoid_grad(b, 0) = choice_grad(b, 0);
    }
    
    auto decision_out = decision_output_->const_flat<T>();
    auto sigmoid_grad_flat = sigmoid_output_->const_flat_grad<T>();
    auto decision_grad = decision_output_->flat_grad<T>();
    
    nn::Sigmoid::Backward<T>(decision_out, sigmoid_grad_flat, decision_grad);
    
    // Backprop through linear
    auto decision_grad_2d = decision_output_->const_matrix_grad<T>(BT, 1);
    decision_->Backward(x, decision_grad_2d, x_grad);
  }
  
  size_t NumParameters() const {
    return decision_->NumParameters();
  }
  
  size_t NumActivations() const {
    return decision_->NumActivations() + 
           decision_output_->size() + 
           sigmoid_output_->size();
  }
  
  void Parameters(std::vector<nn::Parameter*>* parameters) const {
    decision_->Parameters(parameters);
  }
  
  // Linear projection for decision
  std::unique_ptr<nn::Linear> decision_;
  
  // Activation tensors
  std::unique_ptr<nn::Activation> decision_output_;
  std::unique_ptr<nn::Activation> sigmoid_output_;
};

// Leaf network (standard MLP)
struct LeafNetwork {
  using T = floatX;
  
  explicit LeafNetwork(int input_size, int output_size) {
    // Standard linear layers
    fc_ = std::make_unique<nn::Linear>(input_size, output_size);
    
    // For storing activations
    auto dtype = nn::DataTypeToEnum<T>::value;
    output_ = std::make_unique<nn::Activation>(dtype);
  }
  
  void Forward(typename TTypes<T>::ConstMatrix x,
               typename TTypes<T>::Matrix y) {
    PROFILE_TRACE_FN("LeafNetwork");
    
    int BT = x.dimension(0);
    int output_size = y.dimension(1);
    output_->LazyAllocate(BT * output_size);
    
    // Linear projection
    auto leaf_out = output_->matrix<T>(BT, output_size);
    fc_->Forward(x, leaf_out);
    
    // Copy to output matrix
    for (int b = 0; b < BT; ++b) {
      for (int j = 0; j < output_size; ++j) {
        y(b, j) = leaf_out(b, j);
      }
    }
  }
  
  void Backward(typename TTypes<T>::ConstMatrix x,
                typename TTypes<T>::ConstMatrix y_grad,
                typename TTypes<T>::Matrix x_grad) {
    PROFILE_TRACE_FN("LeafNetwork");
    
    int BT = x.dimension(0);
    int output_size = y_grad.dimension(1);
    output_->LazyAllocateGradient();
    output_->ZeroGrad();
    
    auto output_grad = output_->matrix_grad<T>(BT, output_size);
    
    // Copy gradients to output tensor
    for (int b = 0; b < BT; ++b) {
      for (int j = 0; j < output_size; ++j) {
        output_grad(b, j) = y_grad(b, j);
      }
    }
    
    // Backprop through linear layer
    auto output_grad_const = output_->const_matrix_grad<T>(BT, output_size);
    fc_->Backward(x, output_grad_const, x_grad);
  }
  
  size_t NumParameters() const {
    return fc_->NumParameters();
  }
  
  size_t NumActivations() const {
    return fc_->NumActivations() + output_->size();
  }
  
  void Parameters(std::vector<nn::Parameter*>* parameters) const {
    fc_->Parameters(parameters);
  }
  
  // Linear projection
  std::unique_ptr<nn::Linear> fc_;
  
  // Activation tensor
  std::unique_ptr<nn::Activation> output_;
};

// Fast Feedforward Network (1 decision node + 2 leaf networks)
struct FastFeedforwardNetwork {
  using T = floatX;
  
  explicit FastFeedforwardNetwork(int n_embed, int output_size) 
      : n_embed_(n_embed), output_size_(output_size) {
    
    // Create decision node and leaf networks
    decision_ = std::make_unique<DecisionNode>(n_embed);
    left_leaf_ = std::make_unique<LeafNetwork>(n_embed, output_size);
    right_leaf_ = std::make_unique<LeafNetwork>(n_embed, output_size);
    
    // Allocate tensors for intermediate results
    auto dtype = nn::DataTypeToEnum<T>::value;
    choice_ = std::make_unique<nn::Activation>(dtype);
    left_output_ = std::make_unique<nn::Activation>(dtype);
    right_output_ = std::make_unique<nn::Activation>(dtype);
  }
  
  void Forward(typename TTypes<T>::ConstMatrix x,
               typename TTypes<T>::Matrix y) {
    PROFILE_TRACE_FN("FastFeedforwardNetwork");
    
    CHECK_EQ(x.dimension(1), n_embed_);
    CHECK_EQ(y.dimension(1), output_size_);
    CHECK_EQ(x.dimension(0), y.dimension(0));
    
    int BT = x.dimension(0);
    
    // Allocate tensors
    choice_->LazyAllocate(BT);
    left_output_->LazyAllocate(BT * output_size_);
    right_output_->LazyAllocate(BT * output_size_);
    
    // Compute the routing decision
    auto choice_matrix = choice_->matrix<T>(BT, 1);
    decision_->Forward(x, choice_matrix);
    
    // Compute outputs from both leaves
    auto left_out = left_output_->matrix<T>(BT, output_size_);
    auto right_out = right_output_->matrix<T>(BT, output_size_);
    
    left_leaf_->Forward(x, left_out);
    right_leaf_->Forward(x, right_out);
    
    // Mix outputs according to routing decision
    for (int b = 0; b < BT; ++b) {
      float choice_val = choice_matrix(b, 0);
      for (int j = 0; j < output_size_; ++j) {
        y(b, j) = choice_val * right_out(b, j) + (1.0f - choice_val) * left_out(b, j);
      }
    }
  }
  
  void Backward(typename TTypes<T>::ConstMatrix x,
                typename TTypes<T>::ConstMatrix y_grad,
                typename TTypes<T>::Matrix x_grad) {
    PROFILE_TRACE_FN("FastFeedforwardNetwork");
    
    CHECK_EQ(x.dimension(1), n_embed_);
    CHECK_EQ(y_grad.dimension(1), output_size_);
    CHECK_EQ(x.dimension(0), y_grad.dimension(0));
    CHECK_EQ(x.dimension(0), x_grad.dimension(0));
    CHECK_EQ(x.dimension(1), x_grad.dimension(1));
    
    int BT = x.dimension(0);
    
    // Allocate gradients
    choice_->LazyAllocateGradient();
    left_output_->LazyAllocateGradient();
    right_output_->LazyAllocateGradient();
    choice_->ZeroGrad();
    left_output_->ZeroGrad();
    right_output_->ZeroGrad();
    
    auto choice_val = choice_->const_matrix<T>(BT, 1);
    auto choice_grad = choice_->matrix_grad<T>(BT, 1);
    auto left_out = left_output_->const_matrix<T>(BT, output_size_);
    auto right_out = right_output_->const_matrix<T>(BT, output_size_);
    auto left_grad = left_output_->matrix_grad<T>(BT, output_size_);
    auto right_grad = right_output_->matrix_grad<T>(BT, output_size_);
    
    // Calculate gradients for leaf outputs and routing decision
    for (int b = 0; b < BT; ++b) {
      float c = choice_val(b, 0);
      choice_grad(b, 0) = 0;
      
      for (int j = 0; j < output_size_; ++j) {
        // Gradient w.r.t decision
        choice_grad(b, 0) += y_grad(b, j) * (right_out(b, j) - left_out(b, j));
        
        // Gradient w.r.t leaf outputs
        right_grad(b, j) = y_grad(b, j) * c;
        left_grad(b, j) = y_grad(b, j) * (1.0f - c);
      }
    }
    
    // Zero out input gradients
    for (int b = 0; b < BT; ++b) {
      for (int j = 0; j < n_embed_; ++j) {
        x_grad(b, j) = 0;
      }
    }
    
    // Create temporary gradient buffers
    auto dtype = nn::DataTypeToEnum<T>::value;
    nn::Parameter temp_grad(dtype, BT * n_embed_);
    
    // Backprop through leaves
    auto right_grad_const = right_output_->const_matrix_grad<T>(BT, output_size_);
    auto left_grad_const = left_output_->const_matrix_grad<T>(BT, output_size_);
    
    auto temp_grad_matrix = temp_grad.matrix<T>(BT, n_embed_);
    right_leaf_->Backward(x, right_grad_const, temp_grad_matrix);
    
    // Add right leaf gradients to x_grad
    for (int b = 0; b < BT; ++b) {
      for (int j = 0; j < n_embed_; ++j) {
        x_grad(b, j) += temp_grad_matrix(b, j);
      }
    }
    
    // Reset temp gradients
    temp_grad.ZeroData();
    left_leaf_->Backward(x, left_grad_const, temp_grad_matrix);
    
    // Add left leaf gradients to x_grad
    for (int b = 0; b < BT; ++b) {
      for (int j = 0; j < n_embed_; ++j) {
        x_grad(b, j) += temp_grad_matrix(b, j);
      }
    }
    
    // Reset temp gradients
    temp_grad.ZeroData();
    auto choice_grad_const = choice_->const_matrix_grad<T>(BT, 1);
    decision_->Backward(x, choice_grad_const, temp_grad_matrix);
    
    // Add decision gradients to x_grad
    for (int b = 0; b < BT; ++b) {
      for (int j = 0; j < n_embed_; ++j) {
        x_grad(b, j) += temp_grad_matrix(b, j);
      }
    }
  }
  
  size_t NumParameters() const {
    return decision_->NumParameters() + 
           left_leaf_->NumParameters() + 
           right_leaf_->NumParameters();
  }
  
  size_t NumActivations() const {
    return decision_->NumActivations() + 
           left_leaf_->NumActivations() + 
           right_leaf_->NumActivations() + 
           choice_->size() + 
           left_output_->size() + 
           right_output_->size();
  }
  
  void Parameters(std::vector<nn::Parameter*>* parameters) const {
    decision_->Parameters(parameters);
    left_leaf_->Parameters(parameters);
    right_leaf_->Parameters(parameters);
  }
  
  int n_embed_;
  int output_size_;
  
  // Network components
  std::unique_ptr<DecisionNode> decision_;
  std::unique_ptr<LeafNetwork> left_leaf_;
  std::unique_ptr<LeafNetwork> right_leaf_;
  
  // Activation tensors
  std::unique_ptr<nn::Activation> choice_;
  std::unique_ptr<nn::Activation> left_output_;
  std::unique_ptr<nn::Activation> right_output_;
};

}  // namespace gpt
#endif  // LLM_CPP__FAST_FEEDFORWARD_NETWORK_HPP_



/*
// In your GPT2 implementation where you currently use MLP:

// Before:
std::unique_ptr<gpt::MLP> mlp_ = std::make_unique<gpt::MLP>(n_embed);

// After:
std::unique_ptr<gpt::FastFeedforwardNetwork> mlp_ = 
    std::make_unique<gpt::FastFeedforwardNetwork>(n_embed, n_embed);


*/