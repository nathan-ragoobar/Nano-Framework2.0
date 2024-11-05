add_library(nanolib_nn STATIC
    nnhead.hpp
    AttentionLayer.hpp
    Block.hpp
    DropOut.hpp
    Embedding.hpp
    gpt.hpp
    LayerNorm.hpp
    Linear.hpp
    Loss.hpp
    MatMul.hpp
    MLP.hpp
    nn.hpp
    Parameter.hpp
    SoftMax.hpp
)

target_include_directories(nanolib_nn
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}
)

target_link_libraries(nanolib_nn
    PUBLIC
        nanolib_tensor
        
)