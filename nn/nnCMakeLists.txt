cmake_minimum_required(VERSION 3.16)


add_library(nn nn.hpp)
target_link_libraries(nn
    absl::strings absl::log absl::check pthread)

add_library(gpt gpt.hpp)
target_link_libraries(gpt nn)