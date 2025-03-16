cmake_minimum_required(VERSION 3.10)

project(nano_distributed)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_library(nano_distributed ${SOURCES})

add_executable(data_parallel_training examples/data_parallel_training.cpp)
target_link_libraries(data_parallel_training nano_distributed)

add_executable(parameter_server examples/parameter_server.cpp)
target_link_libraries(parameter_server nano_distributed)

enable_testing()
file(GLOB TEST_SOURCES "tests/*.cpp")
add_executable(run_tests ${TEST_SOURCES})
target_link_libraries(run_tests nano_distributed)
add_test(NAME run_tests COMMAND run_tests)