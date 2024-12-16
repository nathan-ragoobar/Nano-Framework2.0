cmake_minimum_required(VERSION 3.10)
project(LLMTests)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find GTest package
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Find Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Fetch and build abseil
include(FetchContent)
FetchContent_Declare(
    absl
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
    GIT_TAG master
)
FetchContent_MakeAvailable(absl)


# Add the executable
add_executable(test_fixed_point
    test_fixed_point.cpp
)

# Add the tensor test executable
add_executable(test_tensor_util
    test_tensor_util.cpp
)

# Link against GTest
target_link_libraries(test_fixed_point
    ${GTEST_LIBRARIES}
    pthread
)

# Link tensor tests
target_link_libraries(test_tensor_util
    ${GTEST_LIBRARIES}
    pthread
    Eigen3::Eigen
    absl::inlined_vector
)

# Set include directories for tensor tests
target_include_directories(test_tensor_util
    PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

# Enable testing
enable_testing()
add_test(NAME fixed_point_tests COMMAND test_fixed_point)
add_test(NAME tensor_tests COMMAND test_tensor_util)