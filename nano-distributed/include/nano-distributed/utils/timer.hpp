#ifndef NANO_DISTRIBUTED_UTILS_TIMER_HPP
#define NANO_DISTRIBUTED_UTILS_TIMER_HPP

#include <chrono>

namespace nano_distributed {
namespace utils {

class Timer {
public:
    Timer() : start_time_(), end_time_(), is_running_(false) {}

    void Start() {
        start_time_ = std::chrono::high_resolution_clock::now();
        is_running_ = true;
    }

    void Stop() {
        if (is_running_) {
            end_time_ = std::chrono::high_resolution_clock::now();
            is_running_ = false;
        }
    }

    double ElapsedMilliseconds() {
        if (is_running_) {
            Stop();
        }
        return std::chrono::duration<double, std::milli>(end_time_ - start_time_).count();
    }

    double ElapsedSeconds() {
        if (is_running_) {
            Stop();
        }
        return std::chrono::duration<double>(end_time_ - start_time_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
    bool is_running_;
};

} // namespace utils
} // namespace nano_distributed

#endif // NANO_DISTRIBUTED_UTILS_TIMER_HPP