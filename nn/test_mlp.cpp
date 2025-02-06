#include "./MLP.hpp"
#include <gtest/gtest.h>
#include "./../tensor/fixed_point.hpp" // Include the fixed_point header if needed

constexpr float EPSILON = 1e-4;
/*
class MLPTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Constants used across tests
    n_embed = 3;
    batch_size = 2;
    mlp = std::make_unique<gpt::MLP>(n_embed);
    
    // Initialize Linear layer weights/biases to known values
    auto params = std::vector<nn::Parameter*>();
    mlp->Parameters(&params);
    
    // Set c_fc_ weights to 0.1 and biases to 0.01
    auto fc_data = params[0]->flat<fixed_point_7pt8>();
    auto fc_bias = params[1]->flat<fixed_point_7pt8>();
    for(int i = 0; i < fc_data.size(); i++) {
      fc_data.data()[i] = fixed_point_7pt8(0.1f);
    }
    for(int i = 0; i < fc_bias.size(); i++) {
      fc_bias.data()[i] = fixed_point_7pt8(0.01f);
    }

    // Set c_proj_ weights to 0.2 and biases to 0.02
    auto proj_data = params[2]->flat<fixed_point_7pt8>();
    auto proj_bias = params[3]->flat<fixed_point_7pt8>();
    for(int i = 0; i < proj_data.size(); i++) {
      proj_data.data()[i] = fixed_point_7pt8(0.2f);
    }
    for(int i = 0; i < proj_bias.size(); i++) {
      proj_bias.data()[i] = fixed_point_7pt8(0.02f);
    }
  }

  float gelu(float x) {
    const float sqrt_2_over_pi = std::sqrt(2.0f/M_PI);
    return 0.5f * x * (1.0f + std::tanh(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
  }

  int n_embed;
  int batch_size;
  std::unique_ptr<gpt::MLP> mlp;
  const float EPSILON = 1e-4f;
};

TEST_F(MLPTest, ForwardPass) {
  using T = fixed_point_7pt8;
  
  Eigen::Tensor<T, 2> x_data(batch_size, n_embed);
  Eigen::Tensor<T, 2> y_data(batch_size, n_embed);

  // Input values
  x_data(0,0) = T(1.0f); x_data(0,1) = T(2.0f); x_data(0,2) = T(3.0f);
  x_data(1,0) = T(4.0f); x_data(1,1) = T(5.0f); x_data(1,2) = T(6.0f);

  TTypes<T>::ConstMatrix x(x_data.data(), x_data.dimensions());
  TTypes<T>::Matrix y(y_data.data(), y_data.dimensions());

  mlp->Forward(x, y);

  // Expected values calculation (approximate due to fixed point):
  // 1. First linear: 0.1 * input + 0.01
  // 2. GELU
  // 3. Second linear: 0.2 * gelu_output + 0.02
  
 // Expected calculation for each step:
  for(int i = 0; i < batch_size; i++) {
    for(int j = 0; j < n_embed; j++) {
      // Step 1: First linear layer (c_fc_)
      float fc_out = x(i,j).to_float() * 0.1f + 0.01f;
      
      // Step 2: GELU activation
      float gelu_out = gelu(fc_out);
      
      // Step 3: Second linear layer (c_proj_)
      float final_out = gelu_out * 0.2f + 0.02f;
      
      // For example, for x(0,0) = 1.0:
      // fc_out = 1.0 * 0.1 + 0.01 = 0.11
      // gelu_out = gelu(0.11) ≈ 0.0606
      // final_out = 0.0606 * 0.2 + 0.02 ≈ 0.0321
      
      EXPECT_NEAR(y(i,j).to_float(), final_out, EPSILON);
    }
  }
}

TEST_F(MLPTest, BackwardPass) {
  using T = fixed_point_7pt8;
  
  Eigen::Tensor<T, 2> x_data(batch_size, n_embed);
  Eigen::Tensor<T, 2> y_grad_data(batch_size, n_embed);
  Eigen::Tensor<T, 2> x_grad_data(batch_size, n_embed);
  Eigen::Tensor<T, 2> y_data(batch_size, n_embed);

  // Setup input and gradient values
  x_data(0,0) = T(1.0f); x_data(0,1) = T(2.0f); x_data(0,2) = T(3.0f);
  x_data(1,0) = T(4.0f); x_data(1,1) = T(5.0f); x_data(1,2) = T(6.0f);

  y_grad_data(0,0) = T(0.1f); y_grad_data(0,1) = T(0.2f); y_grad_data(0,2) = T(0.3f);
  y_grad_data(1,0) = T(0.4f); y_grad_data(1,1) = T(0.5f); y_grad_data(1,2) = T(0.6f);

  TTypes<T>::ConstMatrix x(x_data.data(), x_data.dimensions());
  TTypes<T>::Matrix y(y_data.data(), y_data.dimensions());
  TTypes<T>::ConstMatrix y_grad(y_grad_data.data(), y_grad_data.dimensions());
  TTypes<T>::Matrix x_grad(x_grad_data.data(), x_grad_data.dimensions());

  // Forward pass first
  mlp->Forward(x, y);
  
  // Backward pass
  mlp->Backward(x, y_grad, x_grad);

  // Expected gradients calculation:
  // 1. Backward through c_proj_: grad * 0.2 (weights of c_proj_)
  // 2. Backward through GELU: grad * GELU'(x)
  // 3. Backward through c_fc_: grad * 0.1 (weights of c_fc_)
  
  float gelu_derivative(float x) {
    const float sqrt_2_over_pi = std::sqrt(2.0f/M_PI);
    float cdf = 0.5f * (1.0f + std::tanh(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
    float pdf = sqrt_2_over_pi * (1.0f + 3.0f * 0.044715f * x * x) * 
                (1.0f - std::tanh(sqrt_2_over_pi * (x + 0.044715f * x * x * x)) * 
                std::tanh(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
    return cdf + x * pdf;
  }

  for(int i = 0; i < batch_size; i++) {
    for(int j = 0; j < n_embed; j++) {
      // Forward pass intermediate values
      float fc_out = x(i,j).to_float() * 0.1f + 0.01f;
      
      // Backward pass:
      // 1. From y_grad through c_proj_
      float proj_grad = y_grad(i,j).to_float() * 0.2f;
      
      // 2. Through GELU
      float gelu_grad = proj_grad * gelu_derivative(fc_out);
      
      // 3. Through c_fc_
      float expected_x_grad = gelu_grad * 0.1f;
      
      EXPECT_NEAR(x_grad(i,j).to_float(), expected_x_grad, EPSILON);
    }
  }
}

*/


class MLPTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }
};

TEST_F(MLPTest, Forward) {
    using T = fixed_point_7pt8;
    int n_embed = 3;
    gpt::MLP mlp_layer(n_embed);

    Eigen::Tensor<T, 2> x_data(2, n_embed);
    Eigen::Tensor<T, 2> y_data(2, n_embed);

    // Initialize input tensor x
    x_data(0, 0) = T(1.0f); x_data(0, 1) = T(2.0f); x_data(0, 2) = T(3.0f);
    x_data(1, 0) = T(4.0f); x_data(1, 1) = T(5.0f); x_data(1, 2) = T(6.0f);

    // Create TensorMap for x and y
    TTypes<T>::ConstMatrix x(x_data.data(), x_data.dimensions());
    TTypes<T>::Matrix y(y_data.data(), y_data.dimensions());

    // Call the Forward function
    mlp_layer.Forward(x, y);

    // Verify the output (this is a placeholder check, replace with actual expected values)
    for (int i = 0; i < y.dimension(0); ++i) {
        for (int j = 0; j < y.dimension(1); ++j) {
            EXPECT_NEAR(y(i, j).to_float(), 0.0f, EPSILON); // Replace 0.0f with actual expected value
        }
    }

    
}


TEST_F(MLPTest, Backward) {
    using T = fixed_point_7pt8;
    int n_embed = 3;
    gpt::MLP mlp_layer(n_embed);

    Eigen::Tensor<T, 2> x_data(2, n_embed);
    Eigen::Tensor<T, 2> y_data(2, n_embed);
    Eigen::Tensor<T, 2> y_grad_data(2, n_embed);
    Eigen::Tensor<T, 2> x_grad_data(2, n_embed);

    // Initialize input tensor x
    x_data(0, 0) = T(1.0f); x_data(0, 1) = T(2.0f); x_data(0, 2) = T(3.0f);
    x_data(1, 0) = T(4.0f); x_data(1, 1) = T(5.0f); x_data(1, 2) = T(6.0f);

    // Create TensorMap for forward pass
    TTypes<T>::ConstMatrix x(x_data.data(), x_data.dimensions());
    TTypes<T>::Matrix y(y_data.data(), y_data.dimensions());

    // Do forward pass first to initialize activations
    mlp_layer.Forward(x, y);

    // Initialize gradient tensors
    y_grad_data(0, 0) = T(1.0f); y_grad_data(0, 1) = T(2.0f); y_grad_data(0, 2) = T(3.0f);
    y_grad_data(1, 0) = T(4.0f); y_grad_data(1, 1) = T(5.0f); y_grad_data(1, 2) = T(6.0f);
    x_grad_data.setZero();

    // Create TensorMap for backward pass
    TTypes<T>::ConstMatrix y_grad(y_grad_data.data(), y_grad_data.dimensions());
    TTypes<T>::Matrix x_grad(x_grad_data.data(), x_grad_data.dimensions());

    // Call the Backward function
    mlp_layer.Backward(x, y_grad, x_grad);

    // Verify gradients
    for (int i = 0; i < x_grad.dimension(0); ++i) {
        for (int j = 0; j < x_grad.dimension(1); ++j) {
            EXPECT_NEAR(x_grad(i, j).to_float(), 0.0f, EPSILON);
        }
    }
}


TEST_F(MLPTest, NumParameters) {
    int n_embed = 3;
    gpt::MLP mlp_layer(n_embed);

    size_t num_parameters = mlp_layer.NumParameters();
    EXPECT_GT(num_parameters, 0); // Should be greater than 0
}

TEST_F(MLPTest, NumActivations) {
   int n_embed = 3;
    gpt::MLP mlp_layer(n_embed);
    
    // Setup input for Forward pass
    Eigen::Tensor<fixed_point_7pt8, 2> x(1, n_embed);
    Eigen::Tensor<fixed_point_7pt8, 2> y(1, n_embed);
    
    // Call Forward to allocate activations
    TTypes<fixed_point_7pt8>::ConstMatrix x_mat(x.data(), x.dimensions());
    TTypes<fixed_point_7pt8>::Matrix y_mat(y.data(), y.dimensions());
    mlp_layer.Forward(x_mat, y_mat);

    size_t num_activations = mlp_layer.NumActivations();
    EXPECT_GT(num_activations, 0);
}

TEST_F(MLPTest, Parameters) {
    int n_embed = 3;
    gpt::MLP mlp_layer(n_embed);

    std::vector<nn::Parameter*> parameters;
    mlp_layer.Parameters(&parameters);

    EXPECT_GT(parameters.size(), 0); // Should be greater than 0
}