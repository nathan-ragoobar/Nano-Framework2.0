#include <iostream>
#include <vector>
#include <chrono>
#include "nano-distributed/comm/communicator.hpp"
#include "nano-distributed/parallel/data_parallel.hpp"
#include "nano-distributed/utils/timer.hpp"

void train_model(int rank, int size) {
    // Simulate model training
    std::cout << "Process " << rank << " of " << size << " is training the model." << std::endl;
    // Here you would implement the actual training logic
}

int main(int argc, char** argv) {
    nano_distributed::Communicator communicator;
    communicator.init(argc, argv);

    int rank = communicator.get_rank();
    int size = communicator.get_size();

    // Example data for training
    std::vector<float> data = { /* ... your training data ... */ };

    // Distribute data among processes
    std::vector<float> local_data = nano_distributed::data_parallel::distribute_data(data, rank, size);

    // Start training
    auto start_time = std::chrono::high_resolution_clock::now();
    train_model(rank, size);
    auto end_time = std::chrono::high_resolution_clock::now();

    // Measure elapsed time
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Process " << rank << " finished training in " << elapsed.count() << " seconds." << std::endl;

    communicator.finalize();
    return 0;
}