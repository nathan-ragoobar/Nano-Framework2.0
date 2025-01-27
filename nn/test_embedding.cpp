#include "Embedding.hpp"
#include <gtest/gtest.h>
#include "./../tensor/fixed_point.hpp" // Include the fixed_point header if needed

constexpr float EPSILON = 1e-4;

class EmbeddingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }
};

TEST_F(EmbeddingTest, Forward) {
    using T = fixed_point_7pt8;
    int num_embeddings = 3;
    int embedding_dim = 2;
    nn::Embedding embedding_layer(num_embeddings, embedding_dim);

    std::vector<int> idx = {0, 2};
    std::vector<T> embedding_data(idx.size() * embedding_dim);

    // Create Span for embedding
    absl::Span<T> embedding(embedding_data);

    // Call the Forward function
    embedding_layer.Forward(idx, embedding);

    // Verify the output (this is a placeholder check, replace with actual expected values)
    for (size_t i = 0; i < embedding.size(); ++i) {
        EXPECT_NEAR(embedding[i].to_float(), 0.0f, EPSILON); // Replace 0.0f with actual expected value
    }
}

TEST_F(EmbeddingTest, Backward) {
    using T = fixed_point_7pt8;
    int num_embeddings = 3;
    int embedding_dim = 2;
    nn::Embedding embedding_layer(num_embeddings, embedding_dim);

    std::vector<int> idx = {0, 2};
    std::vector<T> grad_embedding_data(idx.size() * embedding_dim, T(1.0f)); // Initialize with gradient of 1.0

    // Create Span for grad_embedding
    absl::Span<const T> grad_embedding(grad_embedding_data);

    // Call the Backward function
    embedding_layer.Backward(idx, grad_embedding);

    // Verify the output (this is a placeholder check, replace with actual expected values)
    for (size_t i = 0; i < grad_embedding.size(); ++i) {
        EXPECT_NEAR(grad_embedding[i].to_float(), 1.0f, EPSILON); // Replace 1.0f with actual expected value
    }
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}