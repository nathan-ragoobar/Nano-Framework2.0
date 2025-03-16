#ifndef NANO_DISTRIBUTED_SYNC_ATOMIC_HPP
#define NANO_DISTRIBUTED_SYNC_ATOMIC_HPP

#include <atomic>

namespace nano_distributed {
namespace sync {

class Atomic {
public:
    Atomic() : value(0) {}

    void store(int desired) {
        value.store(desired, std::memory_order_relaxed);
    }

    int load() const {
        return value.load(std::memory_order_relaxed);
    }

    int fetch_add(int arg) {
        return value.fetch_add(arg, std::memory_order_relaxed);
    }

    int fetch_sub(int arg) {
        return value.fetch_sub(arg, std::memory_order_relaxed);
    }

    bool compare_exchange(int expected, int desired) {
        return value.compare_exchange_strong(expected, desired, std::memory_order_acquire, std::memory_order_relaxed);
    }

private:
    std::atomic<int> value;
};

} // namespace sync
} // namespace nano_distributed

#endif // NANO_DISTRIBUTED_SYNC_ATOMIC_HPP