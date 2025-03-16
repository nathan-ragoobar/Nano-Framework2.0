#include <gtest/gtest.h>
#include "nano-distributed/parallel/data_parallel.hpp"
#include "nano-distributed/parallel/model_parallel.hpp"

TEST(DataParallelTests, DistributeData) {
    // Setup
    const int num_workers = 4;
    std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<std::vector<int>> distributed_data;

    // Distribute data
    distribute_data(data, num_workers, distributed_data);

    // Check that data is distributed correctly
    ASSERT_EQ(distributed_data.size(), num_workers);
    for (int i = 0; i < num_workers; ++i) {
        ASSERT_FALSE(distributed_data[i].empty());
    }
}

TEST(ModelParallelTests, DistributeModelParameters) {
    // Setup
    Model model;
    std::vector<Model> distributed_models;

    // Distribute model parameters
    distribute_model_parameters(model, distributed_models);

    // Check that models are distributed correctly
    ASSERT_EQ(distributed_models.size(), get_num_workers());
    for (const auto& distributed_model : distributed_models) {
        ASSERT_EQ(distributed_model.get_parameters().size(), model.get_parameters().size());
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}