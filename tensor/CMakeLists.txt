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

# Link against GTest and Eigen
target_link_libraries(test_fixed_point
    ${GTEST_LIBRARIES}
    pthread
    Eigen3::Eigen
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

# Set include directories for fixed point tests
target_include_directories(test_fixed_point
    PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

# Add test_tanh executable
add_executable(test_tanh test_tanh.cpp)

# Set include directories for test_tanh
target_include_directories(test_tanh
    PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

# Link libraries for test_tanh
target_link_libraries(test_tanh
    Eigen3::Eigen
)

# Enable testing
enable_testing()
add_test(NAME fixed_point_tests COMMAND test_fixed_point)
add_test(NAME tensor_tests COMMAND test_tensor_util)
#[[

add_executable(test_fpm test_fpm.cpp)
target_link_libraries(test_fpm
    ${GTEST_LIBRARIES}
    pthread
    absl::inlined_vector
    profiler
    #fpm
)
target_compile_options(test_fpm PRIVATE -Ofast -march=native)
]]
#Add test_fpm executable
add_executable(test_fpm test_fpm.cpp)

# Set include directories for test_fpm
target_include_directories(test_fpm
    PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

# Link libraries for test_fpm
target_link_libraries(test_fpm
    ${EIGEN3_INCLUDE_DIR}
    ${GTEST_LIBRARIES}
    pthread
    absl::inlined_vector
    profiler
    #fpm
)

target_compile_options(test_fpm PRIVATE -Ofast -march=native)
