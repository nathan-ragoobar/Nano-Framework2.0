add_library(nanolib_tensor STATIC
    tensor_types.hpp
    tensor_util.hpp
)

target_include_directories(nanolib_tensor
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}
)

target_link_libraries(nanolib_tensor
    PUBLIC
        Eigen3::Eigen
        absl::base
)