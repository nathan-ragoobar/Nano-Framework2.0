#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <string>
#include <vector>
#include <memory>
#include "../inference.hpp"

// Mock classes to avoid actual model loading during tests
class MockGPT2 : public gpt2::GPT2 {
public:
    MockGPT2() {
        // Set minimum config needed for testing
        config.vocab_size = 50257;
    }
    
    bool BuildFromCheckpoint(const char* filename) override {
        return true;  // Always succeed in test
    }
};

class MockGPT2Tokenizer : public nano::GPT2Tokenizer {
public:
    MockGPT2Tokenizer() : nano::GPT2Tokenizer("", "") {}
    
    std::vector<int> encode(const std::string& text) const override {
        // Simple mock encoding: each char becomes its ASCII value
        std::vector<int> result;
        for (char c : text) {
            result.push_back(static_cast<int>(c));
        }
        return result;
    }
    
    std::string decode(const std::vector<int>& tokens) const override {
        // Simple mock decoding: each token becomes a char
        std::string result;
        for (int token : tokens) {
            if (token < 256) {  // Only convert ASCII range
                result.push_back(static_cast<char>(token));
            } else {
                result.push_back('?');  // Placeholder for non-ASCII
            }
        }
        return result;
    }
};

// Test fixture to set up and tear down the mock environment
class InferenceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Replace the real model and tokenizer with mocks
        Inference::model = std::make_unique<MockGPT2>();
        Inference::tokenizer = std::make_unique<MockGPT2Tokenizer>();
        Inference::isModelLoaded = true;  // Pretend model is loaded
    }
    
    void TearDown() override {
        // Clean up
        Inference::cleanup();
    }
};

// Test basic generation functionality
TEST_F(InferenceTest, BasicGeneration) {
    std::string input = "Hello";
    int maxTokens = 10;
    
    std::string response = Inference::generateResponse(input, maxTokens);
    
    // Check that response contains the input text
    EXPECT_THAT(response, ::testing::HasSubstr("Hello"));
    // Check that response is longer than input (due to generation)
    EXPECT_GT(response.length(), input.length());
}

// Test with empty input
TEST_F(InferenceTest, EmptyInput) {
    std::string input = "";
    int maxTokens = 5;
    
    std::string response = Inference::generateResponse(input, maxTokens);
    
    // Check that some output is generated
    EXPECT_FALSE(response.empty());
}

// Test with zero max tokens
TEST_F(InferenceTest, ZeroMaxTokens) {
    std::string input = "Test";
    int maxTokens = 0;
    
    std::string response = Inference::generateResponse(input, maxTokens);
    
    // Should just return the input with no additional generated text
    EXPECT_EQ(response, input);
}

// Test with large token count
TEST_F(InferenceTest, LargeTokenCount) {
    std::string input = "Short input";
    int maxTokens = 1000;  // Much larger than T limit in the code
    
    std::string response = Inference::generateResponse(input, maxTokens);
    
    // Should not crash and should return something
    EXPECT_FALSE(response.empty());
}

// Test model loading failure
TEST_F(InferenceTest, ModelLoadFailure) {
    // Clean up current mocks
    Inference::cleanup();
    
    // Set up to simulate failure
    Inference::isModelLoaded = false;
    
    // Try to generate with no valid model
    std::string response = Inference::generateResponse("Test", 5);
    
    // Should return error message
    EXPECT_THAT(response, ::testing::HasSubstr("Error: Failed to load"));
}

// Main function to run the tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}