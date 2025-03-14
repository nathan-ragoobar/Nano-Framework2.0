// filepath: /home/nathan/Nano-Framework2.0/nn/test_Block.hpp
#include <gtest/gtest.h>
#include "./Block.hpp"
#include <random>
#include <vector>
#include <memory>

namespace gpt {
namespace testing {

class FFFBlockTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Default setup for tests
    batch_size_ = 2;
    seq_length_ = 4;
    n_embed_ = 16;
    n_head_ = 2;
    block_size_ = seq_length_; // Set to match sequence length
    
    // Create random inputs
    std::default_random_engine generator(42); // Fixed seed for reproducibility
    std::normal_distribution<float> distribution(0.0f, 0.1f);
    
    input_data_ = std::make_unique<nn::Parameter>(
      nn::DataTypeToEnum<float>::value, batch_size_ * seq_length_ * n_embed_);
      auto input = input_data_->tensor_3d<float>(batch_size_, seq_length_, n_embed_);

    for (int b = 0; b < batch_size_; ++b) {
      for (int t = 0; t < seq_length_; ++t) {
        for (int e = 0; e < n_embed_; ++e) {
          input(b, t, e) = distribution(generator);
        }
      }
    }
    
    // Create output buffer
    output_data_ = std::make_unique<nn::Parameter>(
      nn::DataTypeToEnum<float>::value, batch_size_ * seq_length_ * n_embed_);
    
    // Create gradient buffers
    output_grad_ = std::make_unique<nn::Parameter>(
      nn::DataTypeToEnum<float>::value, batch_size_ * seq_length_ * n_embed_);
    input_grad_ = std::make_unique<nn::Parameter>(
      nn::DataTypeToEnum<float>::value, batch_size_ * seq_length_ * n_embed_);
    
    // Random gradients
    auto grad = output_grad_->tensor_3d<float>(batch_size_, seq_length_, n_embed_);
    for (int b = 0; b < batch_size_; ++b) {
      for (int t = 0; t < seq_length_; ++t) {
        for (int e = 0; e < n_embed_; ++e) {
          grad(b, t, e) = distribution(generator);
        }
      }
    }
    
    // Create FFFBlock
    block_ = std::make_unique<gpt::FFFBlock>(block_size_, n_head_, n_embed_);
  }
  
  int batch_size_;
  int seq_length_;
  int n_embed_;
  int n_head_;
  int block_size_;
  std::unique_ptr<nn::Parameter> input_data_;
  std::unique_ptr<nn::Parameter> output_data_;
  std::unique_ptr<nn::Parameter> output_grad_;
  std::unique_ptr<nn::Parameter> input_grad_;
  std::unique_ptr<gpt::FFFBlock> block_;
};

TEST_F(FFFBlockTest, ForwardPass) {
  // Test that forward pass completes without errors
  auto input = input_data_->const_tensor_3d<float>(batch_size_, seq_length_, n_embed_);
  auto output = output_data_->tensor_3d<float>(batch_size_, seq_length_, n_embed_);
  
  ASSERT_NO_THROW(block_->Forward(input, output));
  
  // Check that output is not all zeros (as a basic sanity check)
  bool has_nonzero = false;
  for (int b = 0; b < batch_size_ && !has_nonzero; ++b) {
    for (int t = 0; t < seq_length_ && !has_nonzero; ++t) {
      for (int e = 0; e < n_embed_ && !has_nonzero; ++e) {
        if (std::abs(output(b, t, e)) > 1e-6) {
          has_nonzero = true;
          break;
        }
      }
    }
  }
  
  EXPECT_TRUE(has_nonzero) << "Output should not be all zeros";
}

TEST_F(FFFBlockTest, BackwardPass) {
  // Test that backward pass completes without errors
  auto input = input_data_->const_tensor_3d<float>(batch_size_, seq_length_, n_embed_);
  auto output = output_data_->tensor_3d<float>(batch_size_, seq_length_, n_embed_);
  auto output_grad = output_grad_->const_tensor_3d<float>(batch_size_, seq_length_, n_embed_);
  auto input_grad = input_grad_->tensor_3d<float>(batch_size_, seq_length_, n_embed_);
  
  // Forward pass
  block_->Forward(input, output);
  
  // Initialize input gradients to zero
  for (int b = 0; b < batch_size_; ++b) {
    for (int t = 0; t < seq_length_; ++t) {
      for (int e = 0; e < n_embed_; ++e) {
        input_grad(b, t, e) = 0.0f;
      }
    }
  }
  
  // Backward pass
  ASSERT_NO_THROW(block_->Backward(input, output_grad, input_grad));
  
  // Check that at least some gradients are non-zero
  bool has_nonzero_grad = false;
  for (int b = 0; b < batch_size_ && !has_nonzero_grad; ++b) {
    for (int t = 0; t < seq_length_ && !has_nonzero_grad; ++t) {
      for (int e = 0; e < n_embed_ && !has_nonzero_grad; ++e) {
        if (std::abs(input_grad(b, t, e)) > 1e-6) {
          has_nonzero_grad = true;
          break;
        }
      }
    }
  }
  
  EXPECT_TRUE(has_nonzero_grad) << "Backward pass should produce non-zero gradients";
}

TEST_F(FFFBlockTest, ForwardConsistency) {
  // Test that the output is consistent across multiple Forward calls
  auto input = input_data_->const_tensor_3d<float>(batch_size_, seq_length_, n_embed_);
  auto output1 = output_data_->tensor_3d<float>(batch_size_, seq_length_, n_embed_);
  
  // First forward pass
  block_->Forward(input, output1);
  
  // Create a second output buffer
  nn::Parameter output_data2(nn::DataTypeToEnum<float>::value, 
                           batch_size_ * seq_length_ * n_embed_);
  auto output2 = output_data2.tensor_3d<float>(batch_size_, seq_length_, n_embed_);
  
  // Second forward pass
  block_->Forward(input, output2);
  
  // Compare outputs
  for (int b = 0; b < batch_size_; ++b) {
    for (int t = 0; t < seq_length_; ++t) {
      for (int e = 0; e < n_embed_; ++e) {
        EXPECT_NEAR(output1(b, t, e), output2(b, t, e), 1e-6) 
            << "Forward pass should be deterministic";
      }
    }
  }
}

TEST_F(FFFBlockTest, GradientZeroOnZeroInput) {
  // Test that zero gradient input produces zero gradient output
  auto input = input_data_->const_tensor_3d<float>(batch_size_, seq_length_, n_embed_);
  auto output = output_data_->tensor_3d<float>(batch_size_, seq_length_, n_embed_);
  auto input_grad = input_grad_->tensor_3d<float>(batch_size_, seq_length_, n_embed_);
  
  // Forward pass
  block_->Forward(input, output);
  
  // Create zero gradient input
  nn::Parameter zero_grad(nn::DataTypeToEnum<float>::value, 
                         batch_size_ * seq_length_ * n_embed_);
  auto zero_grad_tensor = zero_grad.tensor_3d<float>(batch_size_, seq_length_, n_embed_);
  for (int b = 0; b < batch_size_; ++b) {
    for (int t = 0; t < seq_length_; ++t) {
      for (int e = 0; e < n_embed_; ++e) {
        zero_grad_tensor(b, t, e) = 0.0f;
      }
    }
  }
  
  // Zero out input gradients
  for (int b = 0; b < batch_size_; ++b) {
    for (int t = 0; t < seq_length_; ++t) {
      for (int e = 0; e < n_embed_; ++e) {
        input_grad(b, t, e) = 0.0f;
      }
    }
  }
  
  // Backward pass with zero gradients
  block_->Backward(input, zero_grad.const_tensor_3d<float>(batch_size_, seq_length_, n_embed_), 
                  input_grad);
  
  // Check that all output gradients are zero
  bool all_zeros = true;
  for (int b = 0; b < batch_size_ && all_zeros; ++b) {
    for (int t = 0; t < seq_length_ && all_zeros; ++t) {
      for (int e = 0; e < n_embed_ && all_zeros; ++e) {
        if (std::abs(input_grad(b, t, e)) > 1e-6) {
          all_zeros = false;
          break;
        }
      }
    }
  }
  
  EXPECT_TRUE(all_zeros) << "Zero gradient input should produce zero gradient output";
}

TEST_F(FFFBlockTest, NumParameters) {
  // Test that NumParameters returns the correct count
  // This is a non-trivial check because of the FastFeedforwardNetwork component
  
  std::unique_ptr<gpt::FastFeedforwardNetwork> ffn = std::make_unique<gpt::FastFeedforwardNetwork>(
      n_embed_, n_embed_ / 4, n_embed_, 2);
  
  std::unique_ptr<nn::LayerNorm> ln1 = std::make_unique<nn::LayerNorm>(n_embed_);
  std::unique_ptr<gpt::CausalSelfAttention> attn = 
      std::make_unique<gpt::CausalSelfAttention>(block_size_, n_head_, n_embed_);
  std::unique_ptr<nn::LayerNorm> ln2 = std::make_unique<nn::LayerNorm>(n_embed_);
  
  size_t expected_params = ln1->NumParameters() + 
                          attn->NumParameters() + 
                          ln2->NumParameters() + 
                          ffn->NumParameters();
  
  EXPECT_EQ(block_->NumParameters(), expected_params);
}

TEST_F(FFFBlockTest, ParametersCollection) {
  // Test that Parameters collects all parameters
  std::vector<nn::Parameter*> params;
  block_->Parameters(&params);
  
  // Should have parameters from LayerNorm1, Attention, LayerNorm2, and FastFeedforwardNetwork
  // Check that we have a reasonable number (not precise due to implementation differences)
  EXPECT_GT(params.size(), 10) << "Should collect a reasonable number of parameters";
  
  // Check that all parameters have at least some non-zero values
  // (This is a basic check that they are initialized)
  for (auto param : params) {
    bool has_nonzero = false;
    auto param_data = param->flat<float>();
    for (int i = 0; i < param_data.size(); ++i) {
      if (std::abs(param_data(i)) > 1e-6) {
        has_nonzero = true;
        break;
      }
    }
    EXPECT_TRUE(has_nonzero) << "All parameters should be initialized with some non-zero values";
  }
}

TEST_F(FFFBlockTest, CompareWithRegularBlock) {
  // Test that FFFBlock and regular Block give different results
  // This confirms that we're actually using FastFeedforwardNetwork and not MLP
  auto input = input_data_->const_tensor_3d<float>(batch_size_, seq_length_, n_embed_);
  
  // Create outputs
  nn::Parameter fff_output(nn::DataTypeToEnum<float>::value, 
                         batch_size_ * seq_length_ * n_embed_);
  nn::Parameter reg_output(nn::DataTypeToEnum<float>::value, 
                         batch_size_ * seq_length_ * n_embed_);
  
  auto fff_out = fff_output.tensor_3d<float>(batch_size_, seq_length_, n_embed_);
  auto reg_out = reg_output.tensor_3d<float>(batch_size_, seq_length_, n_embed_);
  
  // Create a regular Block and our FFFBlock
  std::unique_ptr<gpt::Block> reg_block = 
      std::make_unique<gpt::Block>(block_size_, n_head_, n_embed_);
  
  // Run forward passes
  block_->Forward(input, fff_out);
  reg_block->Forward(input, reg_out);
  
  // Check that outputs are different (at least some elements should differ)
  bool found_diff = false;
  for (int b = 0; b < batch_size_ && !found_diff; ++b) {
    for (int t = 0; t < seq_length_ && !found_diff; ++t) {
      for (int e = 0; e < n_embed_ && !found_diff; ++e) {
        if (std::abs(fff_out(b, t, e) - reg_out(b, t, e)) > 1e-4) {
          found_diff = true;
          break;
        }
      }
    }
  }
  
  EXPECT_TRUE(found_diff) << "FFFBlock output should differ from regular Block output";
}

TEST_F(FFFBlockTest, ConstructorCorrectness) {
  // Test that the constructor correctly initializes the FFF with the right depth
  int depth = 2;
  int leaf_width = n_embed_ / (1 << depth); // Should be n_embed_ / 4
  
  // Get FFF parameters from block
  std::vector<nn::Parameter*> params;
  block_->Parameters(&params);
  
  // Now we need to verify FFF is correctly constructed
  // This is a bit of white-box testing since we need to know internals
  // We'll check that the number of parameters is consistent with a depth=2 FFF
  
  // A depth=2 FFF has 2^2-1=3 decision nodes and 2^2=4 leaf networks
  // Each decision node has: input_size + 1 parameters (weights + bias)
  // Each leaf has: input_size*hidden_size + hidden_size + hidden_size*output_size + output_size
  
  // Expected decision node parameters: 3 * (n_embed + 1)
  // Expected leaf parameters: 4 * (n_embed*leaf_width + leaf_width + leaf_width*n_embed + n_embed)
  
  size_t expected_decision_params = 3 * (n_embed_ + 1);
  size_t expected_leaf_params = 4 * (n_embed_*leaf_width + leaf_width + leaf_width*n_embed_ + n_embed_);
  
  // This is an approximation since we're not precisely checking which parameters belong to the FFF
  // Just verify that the total is reasonable
  EXPECT_GE(block_->NumParameters(), expected_decision_params + expected_leaf_params);
}

} // namespace testing
} // namespace gpt