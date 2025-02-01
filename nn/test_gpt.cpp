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
