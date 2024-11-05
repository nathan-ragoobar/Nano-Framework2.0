cmake_minimum_required(VERSION 3.16)
project(nano LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the path to where nanolib was built
add_subdirectory("./nanolib")


include_directories(.)

include_directories(${CMAKE_SOURCE_DIR}/nanolib/abseil-cpp)

# Source files

# Add libraries and executables

add_library(gpt2 INTERFACE)
target_include_directories(gpt2 INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(train_gpt2_cpu train_gpt2.cpp)
target_link_libraries(train_gpt2_cpu gpt2 optim profiler)
target_compile_options(train_gpt2_cpu PRIVATE -Ofast -march=native)
