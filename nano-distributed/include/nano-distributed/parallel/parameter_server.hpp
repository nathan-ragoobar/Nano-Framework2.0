#ifndef NANO_DISTRIBUTED_PARAMETER_SERVER_HPP
#define NANO_DISTRIBUTED_PARAMETER_SERVER_HPP

#include <vector>
#include <mutex>
#include <condition_variable>

namespace nano_distributed {

class ParameterServer {
public:
    ParameterServer(size_t num_parameters);
    
    // Update parameters with new values
    void UpdateParameters(const std::vector<float>& updates);
    
    // Retrieve current parameters
    std::vector<float> GetParameters() const;

private:
    mutable std::mutex mtx_;
    std::vector<float> parameters_;
    std::condition_variable cv_;
};

} // namespace nano_distributed

#endif // NANO_DISTRIBUTED_PARAMETER_SERVER_HPP