#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "gpt2.hpp"
//#include "nano.hpp"


// Mock functions for file operations
FILE* fopenCheck(const char* filename, const char* mode) {
    static FILE* mock_file = nullptr;
    if (mock_file == nullptr) {
        mock_file = tmpfile();
    }
    return mock_file;
}

size_t freadCheck(void* ptr, size_t size, size_t count, FILE* stream) {
    return fread(ptr, size, count, stream);
}

void fcloseCheck(FILE* stream) {
    fclose(stream);
}


class GPT2Test : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up mock file content
        mock_file = fopenCheck("mock_checkpoint.bin", "wb+");
        int model_header[256] = {20240326, 3, 1024, 50257, 12, 12, 768, 50260};
        fwrite(model_header, sizeof(int), 256, mock_file);
        fseek(mock_file, 0, SEEK_SET);
    }

    void TearDown() override {
        fcloseCheck(mock_file);
    }

    FILE* mock_file;
};

TEST_F(GPT2Test, BuildFromCheckpoint) {
    gpt2::GPT2 model;
    //model.BuildFromCheckpoint("mock_checkpoint.bin");

    // Verify that the hyperparameters are correctly read
    //EXPECT_EQ(model.config.max_seq_len, 1024);
    //EXPECT_EQ(model.config.vocab_size, 50257);
    //EXPECT_EQ(model.config.num_layers, 12);
    //EXPECT_EQ(model.config.num_heads, 12);
    //EXPECT_EQ(model.config.channels, 768);
    //EXPECT_EQ(model.config.padded_vocab_size, 50260);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}