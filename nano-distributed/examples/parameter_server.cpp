#include <iostream>
#include <vector>
#include "nano-distributed/parallel/parameter_server.hpp"
#include "nano-distributed/comm/communicator.hpp"

int main(int argc, char** argv) {
    // Initialize the communicator
    nano::distributed::Communicator communicator(argc, argv);

    // Create a parameter server
    nano::distributed::ParameterServer<float> parameter_server;

    // Example model parameters
    std::vector<float> model_parameters = {0.1f, 0.2f, 0.3f};

    // Register parameters with the parameter server
    parameter_server.RegisterParameters(model_parameters);

    // Simulate parameter updates from different workers
    for (int worker_id = 0; worker_id < communicator.GetNumWorkers(); ++worker_id) {
        // Simulate some updates
        std::vector<float> updates = {0.01f * worker_id, 0.02f * worker_id, 0.03f * worker_id};
        parameter_server.UpdateParameters(updates);
    }

    // Retrieve updated parameters
    auto updated_parameters = parameter_server.GetParameters();

    // Print updated parameters
    std::cout << "Updated Model Parameters: ";
    for (const auto& param : updated_parameters) {
        std::cout << param << " ";
    }
    std::cout << std::endl;

    return 0;
}