cmake_minimum_required(VERSION 3.14)
project(MetricWriter VERSION 1.0)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Create header-only library target
add_library(metric_writer INTERFACE)
target_include_directories(metric_writer INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# Include FetchContent for downloading GTest
include(FetchContent)

# Download and configure Google Test
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Enable testing
enable_testing()

# Add test executable
add_executable(test_writer test_writer.cpp)

# Link test executable with Google Test and our library
target_link_libraries(test_writer
    PRIVATE
        metric_writer
        GTest::gtest_main
        GTest::gtest
        pthread
)

# Register tests
include(GoogleTest)
gtest_discover_tests(test_writer)