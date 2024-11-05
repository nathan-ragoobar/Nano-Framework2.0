cmake_minimum_required(VERSION 3.16)


add_library(optim optim.hpp)
target_link_libraries(optim nn)