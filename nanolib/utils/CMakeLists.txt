add_library(nanolib_utils STATIC
    dataloader.h
    rand.h
    tokenizer.h
    utils.h
)

target_include_directories(nanolib_utils
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}
)

target_link_libraries(nanolib_utils
    PUBLIC
        nanolib_tensor
        absl::strings
        absl::flat_hash_map
)
