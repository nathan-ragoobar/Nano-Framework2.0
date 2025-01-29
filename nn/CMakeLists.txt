cmake_minimum_required(VERSION 3.10)

# Project name
project(nn_tests)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(GTest REQUIRED)
enable_testing()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/..
    ${CMAKE_SOURCE_DIR}/../abseil-cpp
    ${CMAKE_SOURCE_DIR}/../eigen
)
# Eigen
set(EIGEN3_INCLUDE_DIR /eigen)
add_definitions(-DEIGEN_DONT_PARALLELIZE)
#add_definitions(-DEIGEN_DONT_VECTORIZE)
add_definitions(-DEIGEN_USE_THREADS)
include_directories(${EIGEN3_INCLUDE_DIR})

# Add Abseil as subdirectory or find package
add_subdirectory(${CMAKE_SOURCE_DIR}/../abseil-cpp ${CMAKE_BINARY_DIR}/abseil-cpp)



# Add test executable
add_executable(test_parameter
    test_parameter.cpp
)

# Link libraries
target_link_libraries(test_parameter
    GTest::GTest
    GTest::Main
    pthread
    absl::log
    absl::log_internal_message
    absl::log_internal_check_op
    absl::strings
    absl::span
)

# Add test for test_parameter
add_test(NAME parameter_tests COMMAND test_parameter)

# Add test executable for test_softmax
add_executable(test_softmax
    test_softmax.cpp
)

# Link libraries for test_softmax
target_link_libraries(test_softmax
    GTest::GTest
    GTest::Main
    pthread
    absl::log
    absl::log_internal_message
    absl::log_internal_check_op
    absl::strings
    absl::span
)

# Add test for test_softmax
add_test(NAME softmax_tests COMMAND test_softmax)

# Add test executable for test_loss
add_executable(test_loss
    test_loss.cpp
)

# Link libraries for test_loss
target_link_libraries(test_loss
    GTest::GTest
    GTest::Main
    pthread
    absl::log
    absl::log_internal_message
    absl::log_internal_check_op
    absl::strings
    absl::span
)

# Add test for test_loss
add_test(NAME loss_tests COMMAND test_loss)

# Add test executable for test_embedding
add_executable(test_embedding
    test_embedding.cpp
)

# Link libraries for test_embedding
target_link_libraries(test_embedding
    GTest::GTest
    GTest::Main
    pthread
    absl::log
    absl::log_internal_message
    absl::log_internal_check_op
    absl::strings
    absl::span
)

# Add test for test_embedding
add_test(NAME embedding_tests COMMAND test_embedding)

# Add test executable for test_matmul
add_executable(test_matmul
    test_matmul.cpp
)

# Link libraries for test_matmul
target_link_libraries(test_matmul
    GTest::GTest
    GTest::Main
    pthread
    absl::log
    absl::log_internal_message
    absl::log_internal_check_op
    absl::strings
    absl::span
)

# Add test for test_matmul
add_test(NAME matmul_tests COMMAND test_matmul)


# Add test executable for test_linear
add_executable(test_linear
    test_linear.cpp
)

# Link libraries for test_linear
target_link_libraries(test_linear
    GTest::GTest
    GTest::Main
    pthread
    absl::log
    absl::log_internal_message
    absl::log_internal_check_op
    absl::strings
    absl::span
)

# Add test for test_linear
add_test(NAME linear_tests COMMAND test_linear)