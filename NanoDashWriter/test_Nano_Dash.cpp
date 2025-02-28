#include <gtest/gtest.h>
#include "NanoDash.hpp"


TEST(NanoDash, test_0) {
    // Initialize Qt application
    
    

    EXPECT_EQ(TrainingVisualizer::initialize(argc, argv), 1);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}