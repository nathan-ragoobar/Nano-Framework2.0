#include <gtest/gtest.h>
#include "nano-distributed/utils/serialization.hpp"
#include "nano-distributed/utils/timer.hpp"

TEST(SerializationTests, SerializeAndDeserialize) {
    int original = 42;
    std::vector<char> buffer;

    // Serialize the integer
    ASSERT_NO_THROW(serialize(original, buffer));

    int deserialized = 0;
    // Deserialize the integer
    ASSERT_NO_THROW(deserialized = deserialize<int>(buffer));

    EXPECT_EQ(original, deserialized);
}

TEST(TimerTests, MeasureElapsedTime) {
    nano_distributed::Timer timer;

    timer.Start();
    // Simulate some work
    for (volatile int i = 0; i < 1000000; ++i);
    timer.Stop();

    double elapsed = timer.Elapsed();
    EXPECT_GE(elapsed, 0.0);
}