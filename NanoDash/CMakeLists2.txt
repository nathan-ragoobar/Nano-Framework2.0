cmake_minimum_required(VERSION 3.16)
project(TrainingVisualizer VERSION 1.0.0 LANGUAGES CXX)

# Qt configuration
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Find Qt packages
find_package(Qt6 COMPONENTS Core Widgets Charts REQUIRED)

# Create executable
add_executable(training_visualizer
    main.cpp
    TrainingVisualizer.cpp
    TrainingVisualizer.hpp
)

# Link Qt libraries
target_link_libraries(training_visualizer
    PRIVATE
        Qt6::Core
        Qt6::Widgets
        Qt6::Charts
)

# Set C++ standard
target_compile_features(training_visualizer PRIVATE cxx_std_17)