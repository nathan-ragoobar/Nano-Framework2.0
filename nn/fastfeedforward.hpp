#pragma once
#include <vector>
#include <memory>
#include <cmath>
#include "./../tensor/tensor_util.hpp"
#include "./Parameter.hpp"

namespace nn {

template<typename T>
T compute_entropy_safe(const T p, const T minus_p) {
    const T EPSILON = 1e-6f;
    T p_clamped = std::max(std::min(p, T(1.0f) - EPSILON), EPSILON);
    T minus_p_clamped = std::max(std::min(minus_p, T(1.0f) - EPSILON), EPSILON);
    return -p_clamped * std::log(p_clamped) - minus_p_clamped * std::log(minus_p_clamped);
}

template<typename T>
class FFF {
public:
    FFF(int input_width, int leaf_width, int output_width, int depth,
        float dropout = 0.0f, bool train_hardened = false,
        float region_leak = 0.0f) 
        : input_width_(input_width),
          leaf_width_(leaf_width), 
          output_width_(output_width),
          depth_(depth),
          dropout_(dropout),
          train_hardened_(train_hardened),
          region_leak_(region_leak),
          training_(true) {
        
        InitializeParameters();
    }

    Tensor<T> Forward(const Tensor<T>& x, bool return_entropies = false, bool use_hard_decisions = false) {
        if (training_) {
            return TrainingForward(x, return_entropies, use_hard_decisions);
        } else {
            return EvalForward(x);
        }
    }

private:
    Tensor<T> TrainingForward(const Tensor<T>& x, bool return_entropies, bool use_hard_decisions) {
        auto batch_size = x.shape(0);
        
        // Initialize mixture weights
        auto current_mixture = Tensor<T>::Ones({batch_size, n_leaves_});
        
        // For each level in the tree
        for (int current_depth = 0; current_depth < depth_; current_depth++) {
            int platform = (1 << current_depth) - 1;
            int next_platform = (1 << (current_depth + 1)) - 1;
            int n_nodes = 1 << current_depth;

            // Get current level weights/biases
            auto current_weights = node_weights_->Slice({platform, next_platform});
            auto current_biases = node_biases_->Slice({platform, next_platform});

            // Compute boundary effects
            auto boundary_scores = x.MatMul(current_weights.Transpose());
            auto boundary_logits = boundary_scores + current_biases;
            auto boundary_effect = Sigmoid(boundary_logits);

            // Apply region leak if training
            if (region_leak_ > 0.0f && training_) {
                auto leak_mask = Tensor<T>::Random({batch_size, n_nodes}) < region_leak_;
                boundary_effect = leak_mask.Select(T(1) - boundary_effect, boundary_effect);
            }

            // Update mixture weights
            auto not_boundary = T(1) - boundary_effect;
            UpdateMixture(current_mixture, boundary_effect, not_boundary);
        }

        // Compute leaf activations and final output
        return ComputeLeafActivations(x, current_mixture);
    }

    Tensor<T> EvalForward(const Tensor<T>& x) {
        auto batch_size = x.shape(0);
        auto current_nodes = Tensor<T>::Zeros({batch_size});

        // Traverse tree efficiently during evaluation
        for (int i = 0; i < depth_; i++) {
            int platform = (1 << i) - 1;
            int next_platform = (1 << (i + 1)) - 1;

            auto node_indices = current_nodes.AsType<int>();
            auto weights = node_weights_->IndexSelect(node_indices);
            auto biases = node_biases_->IndexSelect(node_indices);
            
            auto scores = x.MatMul(weights.Transpose()) + biases;
            auto choices = (scores >= 0).AsType<int>();
            
            current_nodes = (current_nodes - platform) * 2 + choices + next_platform;
        }

        return ComputeFinalOutput(x, current_nodes);
    }

    void InitializeParameters() {
        int n_leaves = 1 << depth_;  // 2^depth
        int n_nodes = n_leaves - 1;

        // Initialize node parameters
        float l1_init = 1.0f / std::sqrt(input_width_);
        node_weights_ = std::make_unique<Parameter<T>>(
            Tensor<T>::Random({n_nodes, input_width_}, -l1_init, l1_init));
        node_biases_ = std::make_unique<Parameter<T>>(
            Tensor<T>::Random({n_nodes, 1}, -l1_init, l1_init));

        // Initialize leaf parameters
        float l2_init = 1.0f / std::sqrt(leaf_width_);
        w1s_ = std::make_unique<Parameter<T>>(
            Tensor<T>::Random({n_leaves, input_width_, leaf_width_}, -l1_init, l1_init));
        b1s_ = std::make_unique<Parameter<T>>(
            Tensor<T>::Random({n_leaves, leaf_width_}, -l1_init, l1_init));
        w2s_ = std::make_unique<Parameter<T>>(
            Tensor<T>::Random({n_leaves, leaf_width_, output_width_}, -l2_init, l2_init));
        b2s_ = std::make_unique<Parameter<T>>(
            Tensor<T>::Random({n_leaves, output_width_}, -l2_init, l2_init));
    }

    // Model parameters
    std::unique_ptr<Parameter<T>> node_weights_;
    std::unique_ptr<Parameter<T>> node_biases_;
    std::unique_ptr<Parameter<T>> w1s_;
    std::unique_ptr<Parameter<T>> b1s_; 
    std::unique_ptr<Parameter<T>> w2s_;
    std::unique_ptr<Parameter<T>> b2s_;

    // Configuration
    int input_width_;
    int leaf_width_;
    int output_width_;
    int depth_;
    int n_leaves_;
    float dropout_;
    bool train_hardened_;
    float region_leak_;
    bool training_;
};


template<typename T>
Tensor<T> Sigmoid(const Tensor<T>& x) {
    return T(1) / (T(1) + (-x).Exp());
}

template<typename T>
void UpdateMixture(Tensor<T>& mixture, const Tensor<T>& effect, const Tensor<T>& not_effect) {
    // Implement mixture update logic
}

template<typename T>
Tensor<T> ComputeLeafActivations(const Tensor<T>& x, const Tensor<T>& mixture) {
    // Implement leaf computation logic
}

template<typename T>
Tensor<T> ComputeFinalOutput(const Tensor<T>& x, const Tensor<T>& nodes) {
    // Implement final output computation
}


template<typename T>
class AutogradContext {
public:
    void save_for_backward(const Tensor<T>& tensor) {
        saved_tensors_.push_back(tensor);
    }
    
    const Tensor<T>& get_saved_tensor(size_t idx) const {
        return saved_tensors_[idx];
    }

private:
    std::vector<Tensor<T>> saved_tensors_;
};

template<typename T>
class AutogradFunction {
public:
    virtual ~AutogradFunction() = default;
    
    Tensor<T> apply(const Tensor<T>& input) {
        // Forward pass
        auto result = forward(input);
        if (input.requires_grad()) {
            result.set_grad_fn(this);
        }
        return result;
    }

    virtual Tensor<T> forward(const Tensor<T>& input) = 0;
    virtual Tensor<T> backward(const Tensor<T>& grad_output) = 0;

protected:
    AutogradContext<T> ctx_;
};

// Example autograd function for sigmoid
template<typename T>
class SigmoidFunction : public AutogradFunction<T> {
public:
    Tensor<T> forward(const Tensor<T>& input) override {
        auto output = T(1) / (T(1) + (-input).Exp());
        this->ctx_.save_for_backward(output);
        return output;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        auto output = this->ctx_.get_saved_tensor(0);
        return grad_output * output * (T(1) - output);
    }
};

// Helper function to create sigmoid function
template<typename T>
Tensor<T> sigmoid(const Tensor<T>& x) {
    static SigmoidFunction<T> sigmoid_fn;
    return sigmoid_fn.apply(x);
}

} // namespace nn