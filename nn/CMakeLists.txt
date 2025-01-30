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

# Add test executable for test_layernorm
add_executable(test_layernorm
    test_layernorm.cpp
)

# Link libraries for test_layernorm
target_link_libraries(test_layernorm
    GTest::GTest
    GTest::Main
    pthread
    absl::log
    absl::log_internal_message
    absl::log_internal_check_op
    absl::strings
    absl::span
)

# Add test for test_layernorm
add_test(NAME layernorm_tests COMMAND test_layernorm)

# Add test executable for test_mlp
add_executable(test_mlp
    test_mlp.cpp
)

# Link libraries for test_mlp
target_link_libraries(test_mlp
    GTest::GTest
    GTest::Main
    pthread
    absl::log
    absl::log_internal_message
    absl::log_internal_check_op
    absl::strings
    absl::span
)

# Add test for test_mlp
add_test(NAME mlp_tests COMMAND test_mlp)

# Add test executable for test_attentionlayer
add_executable(test_attentionlayer
    test_attentionlayer.cpp
)

# Link libraries for test_attentionlayer
target_link_libraries(test_attentionlayer
    GTest::GTest
    GTest::Main
    pthread
    absl::log
    absl::log_internal_message
    absl::log_internal_check_op
    absl::strings
    absl::span
)

# Add test for test_attentionlayer
add_test(NAME attentionlayer_tests COMMAND test_attentionlayer)

# Add test executable for test_block
add_executable(test_block
    test_block.cpp
)

# Link libraries for test_block
target_link_libraries(test_block
    GTest::GTest
    GTest::Main
    pthread
    absl::log
    absl::log_internal_message
    absl::log_internal_check_op
    absl::strings
    absl::span
)

# Add test for test_block
add_test(NAME block_tests COMMAND test_block)

# Add test executable for test_gpt
add_executable(test_gpt
    test_gpt.cpp
)

# Link libraries for test_gpt
target_link_libraries(test_gpt
    GTest::GTest
    GTest::Main
    pthread
    absl::log
    absl::log_internal_message
    absl::log_internal_check_op
    absl::strings
    absl::span
)

# Add test for test_gpt
add_test(NAME gpt_tests COMMAND test_gpt)