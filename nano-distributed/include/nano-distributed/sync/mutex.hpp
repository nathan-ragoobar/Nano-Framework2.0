#ifndef NANO_DISTRIBUTED_SYNC_MUTEX_HPP
#define NANO_DISTRIBUTED_SYNC_MUTEX_HPP

#include <mutex>

namespace nano_distributed {
namespace sync {

class Mutex {
public:
    Mutex() = default;

    void lock() {
        mtx.lock();
    }

    void unlock() {
        mtx.unlock();
    }

private:
    std::mutex mtx;
};

} // namespace sync
} // namespace nano_distributed

#endif // NANO_DISTRIBUTED_SYNC_MUTEX_HPP