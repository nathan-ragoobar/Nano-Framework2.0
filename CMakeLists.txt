cmake_minimum_required(VERSION 3.16)
project(nano LANGUAGES CXX)

# Print the compiler ID and path
message("Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
message("Compiler Path: ${CMAKE_CXX_COMPILER}")

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#Static linking
if(MINGW)
    # Enable static linking
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc -static-libstdc++")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")
endif()

# Add MSVC-specific compiler options
#This is for compiling on windows I think
if(MSVC)
    add_compile_options(/bigobj)
endif()

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

include_directories(.)

# Eigen
set(EIGEN3_INCLUDE_DIR /eigen)
add_definitions(-DEIGEN_DONT_PARALLELIZE)
#add_definitions(-DEIGEN_DONT_VECTORIZE)
add_definitions(-DEIGEN_USE_THREADS)
include_directories(${EIGEN3_INCLUDE_DIR})


# Source files

# Add libraries and executables
add_subdirectory(nn)

add_subdirectory(optimizer)

add_library(nano nano.hpp)
target_link_libraries(nano nn)

#add_library(gpt gpt.hpp)
#target_link_libraries(gpt nn)

add_library(gpt2 INTERFACE)
target_link_libraries(gpt2 INTERFACE gpt)

#add_library(optim optim.hpp)
#target_link_libraries(optim nn)

add_executable(train_gpt2_cpu train_gpt2.cpp)
target_link_libraries(train_gpt2_cpu gpt2 optim)

#[[
# Static linking
if(MINGW)
    # Get MinGW DLL directory
    execute_process(
        COMMAND where g++
        OUTPUT_VARIABLE MINGW_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    get_filename_component(MINGW_BIN_DIR ${MINGW_PATH} DIRECTORY)

    # Copy required DLLs
    add_custom_command(TARGET train_gpt2_cpu POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${MINGW_BIN_DIR}/libgcc_s_seh-1.dll"
            "${MINGW_BIN_DIR}/libstdc++-6.dll"
            "${MINGW_BIN_DIR}/libwinpthread-1.dll"
            $<TARGET_FILE_DIR:train_gpt2_cpu>
    )
endif()
]]

# Platform-specific settings
if (MSVC)
    target_compile_definitions(train_gpt2_cpu PRIVATE _CRT_SECURE_NO_WARNINGS)
    
    # Debug configuration
    target_compile_options(train_gpt2_cpu PRIVATE 
        $<$<CONFIG:Debug>:/Od>    # Disable optimization for debug
    )
    
    # Release configuration
    target_compile_options(train_gpt2_cpu PRIVATE 
        $<$<CONFIG:Release>:/O2>     # Maximum optimization
        $<$<CONFIG:Release>:/arch:AVX2>  # Use Advanced Vector Extensions 2
        #$<$<CONFIG:Release>:/GL>     # Whole program optimization
        $<$<CONFIG:Release>:/fp:fast> # Fast floating point model
        $<$<CONFIG:Release>:/favor:AMD64>  # AMD-specific optimizations
    )
    target_link_libraries(train_gpt2_cpu ws2_32)
else()
    target_compile_options(train_gpt2_cpu PRIVATE -Ofast -march=native)
endif()
