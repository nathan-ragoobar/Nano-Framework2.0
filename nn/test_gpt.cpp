#include "gpt.hpp"
#include <gtest/gtest.h>
#include "./../tensor/fixed_point.hpp" // Include the fixed_point header if needed

class GPTTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }
};

TEST_F(GPTTest, Initialization) {
    using Type = fixed_point_7pt8;
    int block_size = 4;
    int vocab_size = 1000;
    int padded_vocab_size = 1024;
    int n_layer = 2;
    int n_head = 2;
    int n_embed = 8;

    // Create GPT object
    gpt::GPT gpt(block_size, vocab_size, padded_vocab_size, n_layer, n_head, n_embed);
    LOG(INFO) << "The GPT object has been created";

    // Verify the initialization
    EXPECT_EQ(gpt.block_size_, block_size);
        LOG(INFO) << "block_size_ ran";
    EXPECT_EQ(gpt.vocab_size_, vocab_size);
        LOG(INFO) << "vocab_size_ ran";
    EXPECT_EQ(gpt.padded_vocab_size_, padded_vocab_size);
        LOG(INFO) << "padded_vocab_size_ ran";
    EXPECT_EQ(gpt.n_layer_, n_layer);
        LOG(INFO) << "n_layer_ ran";
    EXPECT_EQ(gpt.n_embed_, n_embed);
        LOG(INFO) << "n_embed_ ran";
    EXPECT_NE(gpt.wte_, nullptr);
        LOG(INFO) << "wte_ ran";
    EXPECT_NE(gpt.wpe_, nullptr);
        LOG(INFO) << "wpe_ ran";
    EXPECT_EQ(gpt.h_.size(), n_layer);
        LOG(INFO) << "h_.size() ran";
    EXPECT_NE(gpt.lnf_, nullptr);
        LOG(INFO) << "lnf_ ran";
    EXPECT_NE(gpt.lm_head_unused_, nullptr);
        LOG(INFO) << "lm_head_unused_ ran";
    EXPECT_NE(gpt.lm_head_, nullptr);
        LOG(INFO) << "lm_head_ ran";
    EXPECT_NE(gpt.softmax_cross_entropy_, nullptr);
        LOG(INFO) << "softmax_cross_entropy_ ran";
    EXPECT_NE(gpt.tok_emb_, nullptr);
        LOG(INFO) << "tok_emb_ ran";
    EXPECT_NE(gpt.pos_emb_, nullptr);
        LOG(INFO) << "pos_emb_ ran";
    EXPECT_NE(gpt.encoded_, nullptr);
        LOG(INFO) << "encoded_ ran";
    EXPECT_NE(gpt.block_y_, nullptr);
        LOG(INFO) << "block_y_ ran";
    EXPECT_NE(gpt.lnf_y_, nullptr);
        LOG(INFO) << "lnf_y_ ran";
    EXPECT_NE(gpt.lnf_mean_, nullptr);
        LOG(INFO) << "lnf_mean_ ran";
    EXPECT_NE(gpt.lnf_rstd_, nullptr);
        LOG(INFO) << "lnf_rstd_ ran";
    EXPECT_NE(gpt.scratch_, nullptr);
        LOG(INFO) << "scratch_ ran";
    EXPECT_NE(gpt.loss_, nullptr);
        LOG(INFO) << "loss_ ran";
    EXPECT_NE(gpt.loss_mean_, nullptr);
        LOG(INFO) << "loss_mean_ ran";
    EXPECT_NE(gpt.probs_, nullptr);
        LOG(INFO) << "probs_ ran";
    EXPECT_NE(gpt.logits_grad_, nullptr);
        LOG(INFO) << "logits_grad_ ran";
}

TEST_F(GPTTest, EmbeddingInitialization) {
    using Type = fixed_point_7pt8;
    int block_size = 4;
    int vocab_size = 1000;
    int padded_vocab_size = 1024;
    int n_layer = 2;
    int n_head = 2;
    int n_embed = 8;
    LOG(INFO) << "Embedding  Initialization";
    // Create GPT object
    gpt::GPT gpt(block_size, vocab_size, padded_vocab_size, n_layer, n_head, n_embed);

    // Verify the embedding initialization
    auto wte_weight = gpt.wte_->weight_->data<Type>();
    auto lm_head_weight = gpt.lm_head_;
    for (int i = 0; i < vocab_size * n_embed; ++i) {
        EXPECT_EQ(wte_weight[i], lm_head_weight[i]);
    }
    for (int i = vocab_size * n_embed; i < padded_vocab_size * n_embed; ++i) {
        EXPECT_EQ(wte_weight[i], Type(0));
    }
}


TEST_F(GPTTest, ForwardCPU) {
    // 1. Setup model parameters
    const int batch_size = 2;
    const int seq_len = 4;
    const int n_embed = 8;
    const int vocab_size = 10;
    const int padded_vocab_size = 16;
    const int n_layer = 2;
    const int n_head = 2;

    // 2. Create model
    gpt::GPT gpt(seq_len, vocab_size, padded_vocab_size, 
                             n_layer, n_head, n_embed);

    // 3. Setup input tokens [batch_size, seq_len]
    Eigen::Tensor<int, 2> input_tokens(batch_size, seq_len);
    input_tokens.setValues({
        {1, 2, 3, 4},
        {5, 6, 7, 8}
    });

    // 4. Setup logits tensor [batch_size, seq_len, vocab_size] 
    Eigen::Tensor<fixed_point_7pt8, 3> logits(batch_size, seq_len, vocab_size);
    logits.setZero();

    // 5. Call Forward
    typename TTypes<int>::ConstMatrix idx(input_tokens.data(), batch_size, seq_len);
    typename TTypes<fixed_point_7pt8, 3>::Tensor logits_out(logits.data(), 
        batch_size, seq_len, vocab_size);
    gpt.Forward(idx, logits_out);

    // 6. Verify outputs
    EXPECT_EQ(logits.dimension(0), batch_size);
    EXPECT_EQ(logits.dimension(1), seq_len); 
    EXPECT_EQ(logits.dimension(2), vocab_size);
    
    // Values should not all be zero after forward pass
    bool all_zero = true;
    for(int i = 0; i < batch_size * seq_len * vocab_size; i++) {
        if(logits.data()[i] != fixed_point_7pt8(0)) {
            all_zero = false;
            break;
        }
    }
    EXPECT_FALSE(all_zero);
}


TEST_F(GPTTest, BackwardCPU) {
    // 1. Setup small model
    const int batch_size = 2;
    const int seq_len = 4;
    const int n_embed = 8;
    const int vocab_size = 10;
    const int padded_vocab_size = 16;
    const int n_layer = 2;
    const int n_head = 2;

    gpt::GPT gpt(seq_len, vocab_size, padded_vocab_size, 
                             n_layer, n_head, n_embed);

    // 2. Create input and target data
    Eigen::Tensor<int, 2> input_tokens(batch_size, seq_len);
    input_tokens.setValues({
        {1, 2, 3, 4},
        {5, 6, 7, 8}
    });

    Eigen::Tensor<int, 2> target_tokens(batch_size, seq_len);
    target_tokens.setValues({
        {2, 3, 4, 5},
        {6, 7, 8, 9}
    });

    // 3. Setup output tensors
    Eigen::Tensor<fixed_point_7pt8, 3> logits(batch_size, seq_len, vocab_size);
    logits.setZero();
    float loss = 0.0f;

    // 4. Forward pass
    typename TTypes<int>::ConstMatrix idx(input_tokens.data(), batch_size, seq_len);
    typename TTypes<int>::ConstMatrix targets(target_tokens.data(), batch_size, seq_len);
    typename TTypes<fixed_point_7pt8, 3>::Tensor logits_out(logits.data(), 
        batch_size, seq_len, vocab_size);

    gpt.ForwardCPU(idx, targets, logits_out, &loss);

    // 5. Backward pass
    gpt.BackwardCPU(idx, targets);

    // 6. Verify gradients exist and are non-zero
    EXPECT_TRUE(gpt.wte_->weight_->HasGradient());
    EXPECT_TRUE(gpt.wpe_->weight_->HasGradient());
    
    // Check embedding gradients
    auto wte_grad = gpt.wte_->weight_->grad<fixed_point_7pt8>();
    auto wpe_grad = gpt.wpe_->weight_->grad<fixed_point_7pt8>();
    
    bool has_grad = false;
    for(int i = 0; i < padded_vocab_size * n_embed; i++) {
        if(wte_grad[i] != fixed_point_7pt8(0)) {
            has_grad = true;
            break;
        }
    }
    EXPECT_TRUE(has_grad);
}