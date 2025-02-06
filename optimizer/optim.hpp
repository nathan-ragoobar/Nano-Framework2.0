#ifndef LLM_CPP__OPTIM_HPP_
#define LLM_CPP__OPTIM_HPP_

#include "../nn/Parameter.hpp"

namespace optim {

template<typename Type>
struct SGD {
  SGD(std::vector<nn::Parameter*> parameters, float lr)
      : parameters_(std::move(parameters)), lr_(lr) {}

  void ZeroGrad() {
    for (nn::Parameter* parameter : parameters_) {
      parameter->ZeroGrad();
    }
  }

  void Step() {
    for (nn::Parameter* parameter : parameters_) {
      auto param = parameter->flat<Type>();
      auto grad = parameter->flat_grad<Type>();
      param.device(nn::g_device) -= Type(lr_) * grad;
    }
  }

 private:
  std::vector<nn::Parameter*> parameters_;
  float lr_;
};

template<typename Type>
struct AdamW {
  AdamW(std::vector<nn::Parameter*> parameters, float lr, float beta1 = 0.9f,
        float beta2 = 0.999f, float eps = 1e-8f, float weight_decay = 0.0f)
      : parameters_(std::move(parameters)),
        lr_(lr),
        beta1_(beta1),
        beta2_(beta2),
        eps_(eps),
        weight_decay_(weight_decay) {
    for (const auto& parameter : parameters_) {
      m_.emplace_back(
          std::make_unique<nn::Parameter>(nn::DT_FIXED, parameter->size()));
      v_.emplace_back(
          std::make_unique<nn::Parameter>(nn::DT_FIXED, parameter->size()));
    }
  }

  void ZeroGrad() {
    for (nn::Parameter* parameter : parameters_) {
      parameter->ZeroGrad();
    }
  }

  void Step(int t) {
    for (size_t i = 0; i < parameters_.size(); ++i) {
      auto parameter = parameters_[i]->flat<Type>();
      auto grad = parameters_[i]->flat_grad<Type>();
      auto m = m_[i]->flat<Type>();
      auto v = v_[i]->flat<Type>();

      //printf("Parmeter values: %f\n", parameter(0).to_float());

      // update first moment (momentum)
      m.device(nn::g_device) = Type(beta1_) * m + Type(1.0f - beta1_) * grad;
      
      // update second moment (RMSprop)
      v.device(nn::g_device) = Type(beta2_) * v + Type(1.0f - beta2_) * grad * grad;
      
      // bias-correct moments
      auto m_hat = m / Type(1.0f - std::pow(beta1_, t));
      auto v_hat = v / Type(1.0f - std::pow(beta2_, t));

      // update with weight decay
      parameter.device(nn::g_device) -=
          Type(lr_) * (m_hat / (v_hat.sqrt() + Type(eps_)) + Type(weight_decay_) * parameter);
    }
  }

 private:
  std::vector<nn::Parameter*> parameters_;
  std::vector<std::unique_ptr<nn::Parameter>> m_;
  std::vector<std::unique_ptr<nn::Parameter>> v_;
  float lr_;
  float beta1_;
  float beta2_;
  float eps_;
  float weight_decay_;
};

}  // namespace optim

#endif  // LLM_CPP__OPTIM_HPP_
