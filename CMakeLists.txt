cmake_minimum_required(VERSION 3.16)
project(nano LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the Abseil library
add_subdirectory(abseil-cpp)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/abseil-cpp
)

include_directories("/llmc")
include_directories("/eigen")
include_directories("/tensor")
#include_directories("")

include_directories(.)

# Eigen
set(EIGEN3_INCLUDE_DIR /eigen)
add_definitions(-DEIGEN_DONT_PARALLELIZE)
#add_definitions(-DEIGEN_DONT_VECTORIZE)
add_definitions(-DEIGEN_USE_THREADS)
include_directories(${EIGEN3_INCLUDE_DIR})

# Add the fpm library
add_subdirectory(fpm)
include_directories(${CMAKE_SOURCE_DIR}/fpm/include)

# Source files

# Add libraries and executables
add_subdirectory(nn)

add_subdirectory(optimizer)

add_library(nano nano.hpp)
target_link_libraries(nano nn)

#add_library(gpt gpt.hpp)
#target_link_libraries(gpt nn)

add_library(gpt2 gpt2.hpp)
target_link_libraries(gpt2 gpt)

#add_library(optim optim.hpp)
#target_link_libraries(optim nn)

add_executable(train_gpt2_cpu train_gpt2.cpp)
target_link_libraries(train_gpt2_cpu gpt2 optim profiler)
target_compile_options(train_gpt2_cpu PRIVATE -Ofast -march=native)

add_executable(inference_gpt2_cpu inference_gpt2.cpp)
target_link_libraries(inference_gpt2_cpu gpt2 optim profiler)
target_compile_options(inference_gpt2_cpu PRIVATE -Ofast -march=native)

add_executable(test_gpt2 test_gpt2.cpp)
target_link_libraries(test_gpt2
    ${GTEST_LIBRARIES}
    pthread
    absl::inlined_vector
    gpt2
    optim
    profiler
)
target_compile_options(test_gpt2 PRIVATE -Ofast -march=native)

add_executable(test_fpm test_fpm.cpp)
target_link_libraries(test_fpm
    ${GTEST_LIBRARIES}
    pthread
    absl::inlined_vector
    gpt2
    optim
    profiler
    fpm
)
target_compile_options(test_fpm PRIVATE -Ofast -march=native)