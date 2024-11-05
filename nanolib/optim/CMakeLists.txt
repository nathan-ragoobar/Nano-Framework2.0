add_library(nanolib_optim STATIC
    optim.hpp
)

target_include_directories(nanolib_optim
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}
)

target_link_libraries(nanolib_optim
    PUBLIC
        nanolib_tensor
        nanolib_nn
)
