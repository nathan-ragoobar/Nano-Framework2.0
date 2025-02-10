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
/*
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
          std::make_unique<nn::Parameter>(nn::DT_FLOAT, parameter->size()));
      v_.emplace_back(
          std::make_unique<nn::Parameter>(nn::DT_FLOAT, parameter->size()));
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
          std::make_unique<nn::Parameter>(nn::DT_FLOAT, parameter->size()));
      v_.emplace_back(
          std::make_unique<nn::Parameter>(nn::DT_FLOAT, parameter->size()));
    }
  }

  void ZeroGrad() {
    for (nn::Parameter* parameter : parameters_) {
      parameter->ZeroGrad();
    }
  }

  void Step(int t) {
    for (size_t i = 0; i < parameters_.size(); ++i) {
      auto parameter = parameters_[i]->flat<float>();
      auto grad = parameters_[i]->flat_grad<float>();
      auto m = m_[i]->flat<float>();
      auto v = v_[i]->flat<float>();

      // update the first moment (momentum)
      m.device(nn::g_device) = beta1_ * m + (1.0f - beta1_) * grad;
      // update the second moment (RMSprop)
      v.device(nn::g_device) = beta2_ * v + (1.0f - beta2_) * grad * grad;
      // bias-correct both moments
      auto m_hat = m / (1.0f - static_cast<float>(std::pow(beta1_, t)));
      auto v_hat = v / (1.0f - static_cast<float>(std::pow(beta2_, t)));

      // update
      parameter.device(nn::g_device) -=
          lr_ * (m_hat / (v_hat.sqrt() + eps_) + weight_decay_ * parameter);
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
*/

template<typename Type>
struct AdamW {
    AdamW(std::vector<nn::Parameter*> parameters, float lr, float beta1 = 0.9f,
          float beta2 = 0.999f, float eps = 1e-8f, float weight_decay = 0.0f)
        : parameters_(std::move(parameters)),
          lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), weight_decay_(weight_decay) {
        for (const auto& parameter : parameters_) {
            // Create momentum buffers with same type as parameters
            auto dtype = parameter->dtype();
            m_.emplace_back(std::make_unique<nn::Parameter>(dtype, parameter->size()));
            v_.emplace_back(std::make_unique<nn::Parameter>(dtype, parameter->size()));
        }
    }

    void ZeroGrad() {
      for (nn::Parameter* parameter : parameters_) {
        parameter->ZeroGrad();
      }
    }

  
    void Step(int t) {
      for (size_t i = 0; i < parameters_.size(); ++i) {
          auto dtype = parameters_[i]->dtype();
          
          if (dtype == nn::DT_FIXED) {
              StepParameter<fpm::fixed_16_16>(i, t);
          } else {
              StepParameter<float>(i, t);
          }
      }
  }

        

private:

    template<typename T>
    void StepParameter(size_t i, int t) {
        auto parameter = parameters_[i]->flat<T>();
        auto grad = parameters_[i]->flat_grad<T>();
        auto m = m_[i]->flat<T>();
        auto v = v_[i]->flat<T>();

        // Convert constants to parameter type
        T beta1(beta1_);
        T beta2(beta2_);
        T eps(eps_);
        T lr(lr_);
        T weight_decay(weight_decay_);

        // Bias correction terms
        T bc1(1.0f - std::pow(beta1_, t));
        T bc2(1.0f - std::pow(beta2_, t));
        
        // Update rule remains the same but uses templated type T
        for (int j = 0; j < parameter.size(); j++) {
            // Update biased first moment estimate
            m.data()[j] = beta1 * m.data()[j] + (T(1.0f) - beta1) * grad.data()[j];
            
            // Update biased second raw moment estimate
            v.data()[j] = beta2 * v.data()[j] + (T(1.0f) - beta2) * grad.data()[j] * grad.data()[j];
            
            // Compute bias-corrected first moment estimate
            T m_hat = m.data()[j] / bc1;
            
            // Compute bias-corrected second raw moment estimate
            T v_hat = v.data()[j] / bc2;
            
            // Update parameters
            parameter.data()[j] = parameter.data()[j] - 
                                lr * (m_hat / (T(std::sqrt(float(v_hat))) + eps) + 
                                weight_decay * parameter.data()[j]);
        }
    }

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
