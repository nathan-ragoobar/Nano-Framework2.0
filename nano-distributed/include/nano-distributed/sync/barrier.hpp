#ifndef NANO_DISTRIBUTED_SYNC_BARRIER_HPP
#define NANO_DISTRIBUTED_SYNC_BARRIER_HPP

#include <vector>
#include <mutex>
#include <condition_variable>

namespace nano_distributed {
namespace sync {

class Barrier {
public:
    explicit Barrier(int count) : thread_count(count), waiting_count(0) {}

    void wait() {
        std::unique_lock<std::mutex> lock(mutex);
        waiting_count++;

        if (waiting_count == thread_count) {
            waiting_count = 0;
            condition.notify_all();
        } else {
            condition.wait(lock);
        }
    }

private:
    int thread_count;
    int waiting_count;
    std::mutex mutex;
    std::condition_variable condition;
};

} // namespace sync
} // namespace nano_distributed

#endif // NANO_DISTRIBUTED_SYNC_BARRIER_HPP