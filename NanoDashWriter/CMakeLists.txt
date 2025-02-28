cmake_minimum_required(VERSION 3.14)
project(MetricWriter VERSION 1.0)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Create header-only library target
add_library(metric_writer INTERFACE)
target_include_directories(metric_writer INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

find_package(GTest REQUIRED)


# Enable testing
enable_testing()

# Add test executables
add_executable(test_writer test_writer.cpp)
add_executable(test_writerv2 test_writerv2.cpp)

# Link test executables with Google Test and our library
target_link_libraries(test_writer
    PRIVATE
        metric_writer
        GTest::gtest_main
        GTest::gtest
        pthread
)

target_link_libraries(test_writerv2
    PRIVATE
        metric_writer
        pthread
)

# Register tests
include(GoogleTest)
gtest_discover_tests(test_writer)
#gtest_discover_tests(test_writerv2)