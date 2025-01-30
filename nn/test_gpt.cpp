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

    // Verify the initialization
    EXPECT_EQ(gpt.block_size_, block_size);
    EXPECT_EQ(gpt.vocab_size_, vocab_size);
    EXPECT_EQ(gpt.padded_vocab_size_, padded_vocab_size);
    EXPECT_EQ(gpt.n_layer_, n_layer);
    EXPECT_EQ(gpt.n_embed_, n_embed);
    EXPECT_NE(gpt.wte_, nullptr);
    EXPECT_NE(gpt.wpe_, nullptr);
    EXPECT_EQ(gpt.h_.size(), n_layer);
    EXPECT_NE(gpt.lnf_, nullptr);
    EXPECT_NE(gpt.lm_head_unused_, nullptr);
    EXPECT_NE(gpt.lm_head_, nullptr);
    EXPECT_NE(gpt.softmax_cross_entropy_, nullptr);
    EXPECT_NE(gpt.tok_emb_, nullptr);
    EXPECT_NE(gpt.pos_emb_, nullptr);
    EXPECT_NE(gpt.encoded_, nullptr);
    EXPECT_NE(gpt.block_y_, nullptr);
    EXPECT_NE(gpt.lnf_y_, nullptr);
    EXPECT_NE(gpt.lnf_mean_, nullptr);
    EXPECT_NE(gpt.lnf_rstd_, nullptr);
    EXPECT_NE(gpt.scratch_, nullptr);
    EXPECT_NE(gpt.loss_, nullptr);
    EXPECT_NE(gpt.loss_mean_, nullptr);
    EXPECT_NE(gpt.probs_, nullptr);
    EXPECT_NE(gpt.logits_grad_, nullptr);
}

TEST_F(GPTTest, EmbeddingInitialization) {
    using Type = fixed_point_7pt8;
    int block_size = 4;
    int vocab_size = 1000;
    int padded_vocab_size = 1024;
    int n_layer = 2;
    int n_head = 2;
    int n_embed = 8;

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
