cmake_minimum_required(VERSION 3.14)
project(nanolib VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add Eigen
add_subdirectory(eigen)

# Add Abseil
set(ABSL_PROPAGATE_CXX_STD ON)
add_subdirectory(abseil-cpp)

# Add our library subdirectories
add_subdirectory(tensor)
add_subdirectory(nn)
add_subdirectory(optim)
add_subdirectory(utils)

# Create the main library target
add_library(nanolib STATIC
    nano.hpp
)

# Link all components
target_link_libraries(nanolib
    PUBLIC
        nanolib_tensor
        nanolib_nn
        nanolib_optim
        nanolib_utils
        Eigen3::Eigen
        absl::base
        absl::strings
        absl::flat_hash_map
)

target_include_directories(nanolib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)