#ifndef LLM_CPP__GPT_HPP_
#define LLM_CPP__GPT_HPP_

#include "nn.hpp"
#include "MLP.hpp"
#include "AttentionLayer.hpp"

#ifdef EIGEN_USE_GPU
#include "cuda_profile_util.hpp"
#define PROFILE_TRACE_FN(prefix) NVTX_RANGE_FN(prefix)
#else
#define PROFILE_TRACE_FN(prefix)
#endif

namespace gpt {



struct Block {
  using Type = floatX;

  Block(int block_size, int n_head, int n_embed) {
    ln1_ = std::make_unique<nn::LayerNorm>(n_embed);
    attn_ = std::make_unique<CausalSelfAttention>(block_size, n_head, n_embed);
    ln2_ = std::make_unique<nn::LayerNorm>(n_embed);
    mlp_ = std::make_unique<MLP>(n_embed);

    // activation
    auto dtype = nn::DataTypeToEnum<Type>::value;
    ln1_y_ = std::make_unique<nn::Activation>(dtype);      // [B*T, C]
    ln1_mean_ = std::make_unique<nn::Activation>(dtype);   // [B*T]
    ln1_rstd_ = std::make_unique<nn::Activation>(dtype);   // [B*T]
    att_y_ = std::make_unique<nn::Activation>(dtype);      // [B, T, C]
    residual1_ = std::make_unique<nn::Activation>(dtype);  // [B, T, C]
    ln2_y_ = std::make_unique<nn::Activation>(dtype);      // [B*T, C]
    ln2_mean_ = std::make_unique<nn::Activation>(dtype);   // [B*T]
    ln2_rstd_ = std::make_unique<nn::Activation>(dtype);   // [B*T]
    mlp_y_ = std::make_unique<nn::Activation>(dtype);      // [B, T, C]
  }

  /*
### Attributes
- **Layer Normalization**:
  - `std::unique_ptr<nn::LayerNorm> ln1_`: First layer normalization.
  - `std::unique_ptr<nn::LayerNorm> ln2_`: Second layer normalization.

- **Attention**:
  - `std::unique_ptr<CausalSelfAttention> attn_`: Causal self-attention mechanism.

- **MLP**:
  - `std::unique_ptr<MLP> mlp_`: Multi-layer perceptron.

- **Activation Tensors**:
  - `std::unique_ptr<nn::Activation> ln1_y_`: Activation tensor for the first layer normalization output.
  - `std::unique_ptr<nn::Activation> ln1_mean_`: Mean tensor for the first layer normalization.
  - `std::unique_ptr<nn::Activation> ln1_rstd_`: RSTD tensor for the first layer normalization.
  - `std::unique_ptr<nn::Activation> att_y_`: Activation tensor for the attention output.
  - `std::unique_ptr<nn::Activation> residual1_`: Activation tensor for the first residual connection.
  - `std::unique_ptr<nn::Activation> ln2_y_`: Activation tensor for the second layer normalization output.
  - `std::unique_ptr<nn::Activation> ln2_mean_`: Mean tensor for the second layer normalization.
  - `std::unique_ptr<nn::Activation> ln2_rstd_`: RSTD tensor for the second layer normalization.
  - `std::unique_ptr<nn::Activation> mlp_y_`: Activation tensor for the MLP output.

### Constructor
- [`Block(int block_size, int n_head, int n_embed)`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FC%3A%2FCode3020%2FNano-Framework2.0%2Fgpt.hpp%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A399%2C%22character%22%3A7%7D%7D%5D%2C%222e61b7d5-839e-428e-94e6-c14c1d091738%22%5D "Go to definition"): Initializes the block with the given parameters, creating instances of layer normalization, attention, and MLP, as well as initializing activation tensors.

### Methods
- `void Forward(typename TTypes<Type, 3>::ConstTensor x, typename TTypes<Type, 3>::Tensor y)`: Performs the forward pass through the block.
  - **Layer Normalization 1**: Applies the first layer normalization to the input.
  - **Attention**: Applies the causal self-attention mechanism.
  - **Residual Connection 1**: Adds the attention output to the input.
  - **Layer Normalization 2**: Applies the second layer normalization to the residual output.
  - **MLP**: Applies the multi-layer perceptron.
  - **Residual Connection 2**: Adds the MLP output to the residual output.

- `void Backward(typename TTypes<Type, 3>::ConstTensor x, typename TTypes<Type, 3>::ConstTensor y_grad, typename TTypes<Type, 3>::Tensor x_grad)`: Performs the backward pass through the block, computing gradients for each component.

- `size_t NumParameters() const`: Returns the total number of parameters in the block.
- `size_t NumActivations() const`: Returns the total number of activations in the block.
- `void Parameters(std::vector<nn::Parameter*>* parameters) const`: Collects all parameters of the block into a provided vector.

### Summary
The [`Block`] struct represents a single transformer block in a GPT model. It includes components for layer normalization, causal self-attention, and a multi-layer perceptron, along with activation tensors for intermediate results. The block performs forward and backward passes, computes the number of parameters and activations, and collects parameters for optimization.
   */

  void Forward(typename TTypes<Type, 3>::ConstTensor x,
               typename TTypes<Type, 3>::Tensor y) {
    PROFILE_TRACE_FN("Block");

    // x: [B, T, C], y: [B, T, C]
    const int B = x.dimension(0);  // batch size
    const int T = x.dimension(1);  // sequence length
    const int C = x.dimension(2);  // embedding dimensionality (n_embd)
    CHECK_EQ(B, y.dimension(0));
    CHECK_EQ(T, y.dimension(1));
    CHECK_EQ(C, y.dimension(2));

    ln1_y_->LazyAllocate(B * T * C);
    ln1_mean_->LazyAllocate(B * T);
    ln1_rstd_->LazyAllocate(B * T);
    att_y_->LazyAllocate(B * T * C);
    residual1_->LazyAllocate(B * T * C);
    ln2_y_->LazyAllocate(B * T * C);
    ln2_mean_->LazyAllocate(B * T);
    ln2_rstd_->LazyAllocate(B * T);
    mlp_y_->LazyAllocate(B * T * C);

    // LN1
    auto x_2d = MakeConstMatrix(x.data(), B * T, C);
    auto ln1_y_2d = MakeMatrix(ln1_y_->data<Type>(), B * T, C);
    auto ln1_mean_1d = MakeFlat(ln1_mean_->data<Type>(), B * T);
    auto ln1_rstd_1d = MakeFlat(ln1_rstd_->data<Type>(), B * T);
    ln1_->Forward(x_2d, ln1_y_2d, ln1_mean_1d, ln1_rstd_1d);

    // Attention
    auto ln1_y_3d = MakeConst3DTensor(ln1_y_2d.data(), B, T, C);
    auto att_y_3d = Make3DTensor(att_y_->data<Type>(), B, T, C);
    attn_->Forward(ln1_y_3d, att_y_3d);

    // Residual
    auto x_1d = MakeConstFlat(x.data(), B * T * C);
    auto att_y_1d = MakeConstFlat(att_y_->data<Type>(), B * T * C);
    auto residual1_1d = MakeFlat(residual1_->data<Type>(), residual1_->size());
    nn::Residual::Forward(x_1d, att_y_1d, residual1_1d);

    // LN2
    auto ln2_y_2d = MakeMatrix(ln2_y_->data<Type>(), B * T, C);
    auto ln2_y_2d_const = MakeConstMatrix(ln2_y_->data<Type>(), B * T, C);
    auto ln2_mean_1d = MakeFlat(ln2_mean_->data<Type>(), B * T);
    auto ln2_rstd_1d = MakeFlat(ln2_rstd_->data<Type>(), B * T);
    auto residual1_2d = MakeConstMatrix(residual1_->data<Type>(), B * T, C);
    ln2_->Forward(residual1_2d, ln2_y_2d, ln2_mean_1d, ln2_rstd_1d);

    // MLP
    auto mlp_y_2d = MakeMatrix(mlp_y_->data<Type>(), B * T, C);
    mlp_->Forward(ln2_y_2d_const, mlp_y_2d);

    // Residual
    auto residual1_1d_const =
        MakeConstFlat(residual1_->data<Type>(), residual1_->size());
    auto mlp_y_1d = MakeConstFlat(mlp_y_->data<Type>(), B * T * C);
    auto y_1d = MakeFlat(y.data(), y.size());
    nn::Residual::Forward(residual1_1d_const, mlp_y_1d, y_1d);
  }

  void Backward(typename TTypes<Type, 3>::ConstTensor x,
                typename TTypes<Type, 3>::ConstTensor y_grad,
                typename TTypes<Type, 3>::Tensor x_grad) {
    PROFILE_TRACE_FN("Block");

    // x: [B, T, C], y_grad: [B, T, C], x_grad: [B, T, C]
    const int B = x.dimension(0);  // batch size
    const int T = x.dimension(1);  // sequence length
    const int C = x.dimension(2);  // embedding dimensionality (n_embd)
    CHECK_EQ(B, y_grad.dimension(0));
    CHECK_EQ(T, y_grad.dimension(1));
    CHECK_EQ(C, y_grad.dimension(2));
    CHECK_EQ(B, x_grad.dimension(0));
    CHECK_EQ(T, x_grad.dimension(1));
    CHECK_EQ(C, x_grad.dimension(2));

    ln1_y_->LazyAllocateGradient();
    att_y_->LazyAllocateGradient();
    residual1_->LazyAllocateGradient();
    ln2_y_->LazyAllocateGradient();
    mlp_y_->LazyAllocateGradient();
    ln1_y_->ZeroGrad();
    att_y_->ZeroGrad();
    residual1_->ZeroGrad();
    ln2_y_->ZeroGrad();
    mlp_y_->ZeroGrad();

    // backward residual
    auto y_grad_1d = MakeConstFlat(y_grad.data(), y_grad.size());
    auto residual1_grad_1d =
        MakeFlat(residual1_->grad<Type>(), residual1_->size());
    auto mlp_y_grad_1d = MakeFlat(mlp_y_->grad<Type>(), mlp_y_->size());
    nn::Residual::Backward(y_grad_1d, residual1_grad_1d, mlp_y_grad_1d);

    // backward MLP
    auto ln2_y_2d = MakeConstMatrix(ln2_y_->data<Type>(), B * T, C);
    auto ln2_y_grad_2d = MakeMatrix(ln2_y_->grad<Type>(), B * T, C);
    auto mlp_y_grad_2d = MakeConstMatrix(mlp_y_->grad<Type>(), B * T, C);
    mlp_->Backward(ln2_y_2d, mlp_y_grad_2d, ln2_y_grad_2d);

    // backward LN2
    auto ln2_mean_1d = MakeConstFlat(ln2_mean_->data<Type>(), B * T);
    auto ln2_rstd_1d = MakeConstFlat(ln2_rstd_->data<Type>(), B * T);
    auto residual1_2d = MakeConstMatrix(residual1_->data<Type>(), B * T, C);
    auto ln2_y_grad_2d_const = MakeConstMatrix(ln2_y_->grad<Type>(), B * T, C);
    auto residual1_grad_2d = MakeMatrix(residual1_->grad<Type>(), B * T, C);
    ln2_->Backward(residual1_2d, ln2_y_grad_2d_const, ln2_mean_1d, ln2_rstd_1d,
                   residual1_grad_2d);

    // backward residual
    auto residual1_grad_1d_const =
        MakeConstFlat(residual1_->grad<Type>(), residual1_->size());
    auto x_grad_1d = MakeFlat(x_grad.data(), x_grad.size());
    auto att_y_grad_1d = MakeFlat(att_y_->grad<Type>(), att_y_->size());
    nn::Residual::Backward(residual1_grad_1d_const, x_grad_1d, att_y_grad_1d);

    // backward attention
    auto ln1_y_3d = MakeConst3DTensor(ln1_y_->data<Type>(), B, T, C);
    auto ln1_y_grad_3d = Make3DTensor(ln1_y_->grad<Type>(), B, T, C);
    auto att_y_grad_3d = MakeConst3DTensor(att_y_->grad<Type>(), B, T, C);
    attn_->Backward(ln1_y_3d, att_y_grad_3d, ln1_y_grad_3d);

    // backward LN1
    auto x_2d = MakeConstMatrix(x.data(), B * T, C);
    auto ln1_mean_1d = MakeConstFlat(ln1_mean_->data<Type>(), B * T);
    auto ln1_rstd_1d = MakeConstFlat(ln1_rstd_->data<Type>(), B * T);
    auto ln1_y_grad_2d = MakeConstMatrix(ln1_y_->grad<Type>(), B * T, C);
    auto x_grad_2d = MakeMatrix(x_grad.data(), B * T, C);
    ln1_->Backward(x_2d, ln1_y_grad_2d, ln1_mean_1d, ln1_rstd_1d, x_grad_2d);
  }

  size_t NumParameters() const {
    return ln1_->NumParameters() + attn_->NumParameters() +
           ln2_->NumParameters() + mlp_->NumParameters();
  }

  size_t NumActivations() const {
    return ln1_->NumActivations() + attn_->NumActivations() +
           ln2_->NumActivations() + mlp_->NumActivations() + ln1_y_->size() +
           ln1_mean_->size() + ln2_rstd_->size() + att_y_->size() +
           residual1_->size() + ln2_y_->size() + ln2_mean_->size() +
           ln2_rstd_->size() + mlp_y_->size();
  }

  void Parameters(std::vector<nn::Parameter*>* parameters) const {
    ln1_->Parameters(parameters);
    attn_->Parameters(parameters);
    ln2_->Parameters(parameters);
    mlp_->Parameters(parameters);
  }

  std::unique_ptr<nn::LayerNorm> ln1_;
  std::unique_ptr<CausalSelfAttention> attn_;
  std::unique_ptr<nn::LayerNorm> ln2_;
  std::unique_ptr<MLP> mlp_;

  // activation tensors
  std::unique_ptr<nn::Activation> ln1_y_;                // [B*T, C]
  std::unique_ptr<nn::Activation> ln1_mean_, ln1_rstd_;  // [B*T]
  std::unique_ptr<nn::Activation> att_y_;                // [B, T, C]
  std::unique_ptr<nn::Activation> residual1_;            // [B, T, C]
  std::unique_ptr<nn::Activation> ln2_y_;                // [B*T, C]
  std::unique_ptr<nn::Activation> ln2_mean_, ln2_rstd_;  // [B*T]
  std::unique_ptr<nn::Activation> mlp_y_;                // [B, T, C]
};

struct GPT {
  using Type = floatX;

  GPT(int block_size, int vocab_size, int padded_vocab_size, int n_layer,
      int n_head, int n_embed)
      : block_size_(block_size),
        vocab_size_(vocab_size),
        padded_vocab_size_(padded_vocab_size),
        n_layer_(n_layer),
        n_embed_(n_embed),
        lm_head_(nullptr),
        lm_head_grad_(nullptr) {
    CHECK_GT(n_layer, 0);

    wte_ = std::make_unique<nn::Embedding>(padded_vocab_size, n_embed);
    wpe_ = std::make_unique<nn::Embedding>(block_size, n_embed);
    for (int i = 0; i < n_layer; ++i) {
      h_.emplace_back(std::make_unique<Block>(block_size, n_head, n_embed));
    }
    lnf_ = std::make_unique<nn::LayerNorm>(n_embed);

    lm_head_unused_ = std::make_unique<nn::Linear>(n_embed, vocab_size);
    // https://paperswithcode.com/method/weight-tying
    nn::g_device.memcpy(wte_->weight_->data<Type>(),
                        lm_head_unused_->weight_->template data<Type>(),
                        sizeof(float) * vocab_size * n_embed);
    nn::g_device.memset(
        wte_->weight_->data<Type>() + vocab_size * n_embed, 0,
        sizeof(float) * (padded_vocab_size - vocab_size) * n_embed);
    lm_head_ = wte_->weight_->data<Type>();
    softmax_cross_entropy_ = std::make_unique<nn::SoftmaxCrossEntropy>();

    // activation
    auto dtype = nn::DataTypeToEnum<Type>::value;
    tok_emb_ = std::make_unique<nn::Activation>(dtype);    // [B, T, C]
    pos_emb_ = std::make_unique<nn::Activation>(dtype);    // [T, C]
    encoded_ = std::make_unique<nn::Activation>(dtype);    // [B, T, C]
    block_y_ = std::make_unique<nn::Activation>(dtype);    // [L, B, T, C]
    lnf_y_ = std::make_unique<nn::Activation>(dtype);      // [B*T, C]
    lnf_mean_ = std::make_unique<nn::Activation>(dtype);   // [B*T]
    lnf_rstd_ = std::make_unique<nn::Activation>(dtype);   // [B*T]
    scratch_ = std::make_unique<nn::Activation>(dtype);    // [B*T]
    loss_ = std::make_unique<nn::Activation>(dtype);       // [B*T]
    loss_mean_ = std::make_unique<nn::Activation>(dtype);  // [1]
    probs_ = std::make_unique<nn::Activation>(dtype);      // [B*T, vocab_size]
    logits_grad_ =
        std::make_unique<nn::Activation>(dtype);  // [B*T, vocab_size]
  }

  void __Forward(typename TTypes<int>::ConstMatrix idx) {
    PROFILE_TRACE_FN("GPT");

    const int B = idx.dimension(0), T = idx.dimension(1), C = n_embed_,
              L = n_layer_;
    const int BT = B * T, TC = T * C;
    const int BTC = BT * C;

    CHECK_LE(T, block_size_) << "Cannot forward sequence of length " << T
                             << ", block size is only " << block_size_;
    std::vector<int> pos(T);
    std::iota(pos.begin(), pos.end(), 0);

    // Lazily allocate memory
    tok_emb_->LazyAllocate(B * T * C);
    pos_emb_->LazyAllocate(T * C);
    encoded_->LazyAllocate(B * T * C);
    block_y_->LazyAllocate(L * B * T * C);
    lnf_y_->LazyAllocate(BT * C);
    lnf_mean_->LazyAllocate(BT);
    lnf_rstd_->LazyAllocate(BT);

    wte_->Forward(idx,
                  absl::MakeSpan(tok_emb_->data<Type>(), tok_emb_->size()));
    wpe_->Forward(pos,
                  absl::MakeSpan(pos_emb_->data<Type>(), pos_emb_->size()));

    auto tok_emb = tok_emb_->matrix<Type>(B, TC);
    auto pos_emb = pos_emb_->flat<Type>();
    auto encoded = encoded_->matrix<Type>(B, TC);
    Eigen::array<Eigen::Index, 2> batch_by_one = {B, 1};
    Eigen::array<Eigen::Index, 2> one_by_class = {1, TC};
    encoded.device(nn::g_device) =
        tok_emb + pos_emb.reshape(one_by_class).broadcast(batch_by_one);

    for (int l = 0; l < n_layer_; ++l) {
      const auto& block = h_[l];
      Type* x = l == 0 ? encoded_->data<Type>()
                       : block_y_->data<Type>() + (l - 1) * BTC;
      Type* y = block_y_->data<Type>() + l * BTC;
      auto block_x_3d = MakeConst3DTensor(x, B, T, C);
      auto block_y_3d = Make3DTensor(y, B, T, C);
      block->Forward(block_x_3d, block_y_3d);
    }

    auto block_out_2d =
        MakeConstMatrix(block_y_->data<Type>() + (L - 1) * BTC, BT, C);
    auto lnf_y = MakeMatrix(lnf_y_->data<Type>(), BT, C);
    auto lnf_mean = MakeFlat(lnf_mean_->data<Type>(), BT);
    auto lnf_rstd = MakeFlat(lnf_rstd_->data<Type>(), BT);
    lnf_->Forward(block_out_2d, lnf_y, lnf_mean, lnf_rstd);
  }

  void Forward(typename TTypes<int>::ConstMatrix idx,
               typename TTypes<Type, 3>::Tensor logits) {
    PROFILE_TRACE_FN("GPT");

    const int B = idx.dimension(0), T = idx.dimension(1), C = n_embed_;
    const int BT = B * T;
    CHECK(logits.dimension(0) == B && logits.dimension(1) == T &&
          logits.dimension(2) == vocab_size_)
        << "B: " << B << ", T: " << T << ", vocab_size: " << vocab_size_;
    __Forward(idx);

    // OPTIMIZE:
    // inference-time mini-optimization: only forward the lm_head on the very
    // last position
    //    auto lnf_y_3d = Eigen::TensorMap<nn::Tensor3D>(lnf_y_.data(), B, T,
    //    C); nn::Tensor2D lnf_y_last_t = lnf_y_3d.chip(T - 1, 1);
    auto lnf_y = MakeMatrix(lnf_y_->data<Type>(), BT, C);
    auto lm_head = MakeMatrix(lm_head_, vocab_size_, C);
    auto logits_2d = MakeMatrix(logits.data(), BT, vocab_size_);
    //    nn::MatMul::Forward(lnf_y, lm_head, logits_2d);
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
        Eigen::IndexPair<int>(1, 1)};
    logits_2d.device(nn::g_device) = lnf_y.contract(lm_head, product_dims);
  }

  void SoftmaxForwardCPU(typename TTypes<Type>::ConstMatrix logits,
                         absl::Span<const int> targets, float* loss) {
    PROFILE_TRACE_FN("GPT");

    int BT = logits.dimension(0);
    CHECK_EQ(BT, targets.size());
    CHECK_EQ(vocab_size_, logits.dimension(1));
    probs_->LazyAllocate(BT * vocab_size_);
    auto probs_2d = MakeMatrix(probs_->data<Type>(), BT, vocab_size_);
    softmax_cross_entropy_->Forward(logits, targets, probs_2d, loss);
  }

  void SoftmaxForwardGPU(typename TTypes<Type>::ConstMatrix logits,
                         typename TTypes<Type>::ConstMatrix labels,
                         float* loss) {
    PROFILE_TRACE_FN("GPT");

    int BT = logits.dimension(0);
    CHECK_EQ(BT, labels.dimension(0));
    CHECK_EQ(vocab_size_, logits.dimension(1));
    CHECK_EQ(vocab_size_, labels.dimension(1));
    scratch_->LazyAllocate(BT);
    loss_->LazyAllocate(BT);
    loss_mean_->LazyAllocate(1);
    logits_grad_->LazyAllocate(BT * vocab_size_);
    logits_grad_->ZeroData();
    auto logits_grad = MakeMatrix(logits_grad_->data<Type>(), BT, vocab_size_);
    nn::SoftmaxCrossEntropy::ForwardAndBackward(
        logits, labels, scratch_->template flat<Type>(),
        loss_->template flat<Type>(), logits_grad);
    logits_grad.device(nn::g_device) = logits_grad * (1.0f / BT);

#ifdef EIGEN_USE_GPU
    TTypes<Type>::UnalignedScalar loss_mean(loss_mean_->data<Type>());
    loss_mean.device(nn::g_device) = loss_->template flat<Type>().mean();
    nn::g_device.memcpyDeviceToHost(loss, loss_mean.data(), sizeof(Type));
    nn::g_device.synchronize();
#else
    LOG(FATAL) << "Never reach here!!!";
#endif
    //    TTypes<float>::Scalar loss_scalar(loss);
    //    loss_scalar.device(nn::g_device) = loss_->template
    //    flat<Type>().mean();
  }

  void ForwardCPU(typename TTypes<int>::ConstMatrix idx,
                  typename TTypes<int>::ConstMatrix targets,
                  typename TTypes<Type, 3>::Tensor logits, float* loss) {
    PROFILE_TRACE_FN("GPT");

    // idx: [B, T], targets: [B, T]
    // logits: [B, T, vocab_size]
    const int B = idx.dimension(0), T = idx.dimension(1), C = n_embed_;
    const int BT = B * T;
    CHECK(targets.dimension(0) == B && targets.dimension(1) == T);
    CHECK(logits.dimension(0) == B && logits.dimension(1) == T &&
          logits.dimension(2) == vocab_size_)
        << "B: " << B << ", T: " << T << ", vocab_size: " << vocab_size_;
    __Forward(idx);

    auto lnf_y = MakeMatrix(lnf_y_->data<Type>(), BT, C);
    auto lm_head = MakeMatrix(lm_head_, vocab_size_, n_embed_);
    auto logits_2d = MakeMatrix(logits.data(), BT, vocab_size_);
    auto probs_2d = MakeMatrix(probs_->data<Type>(), BT, vocab_size_);

    // [BT, C] x [C, vocab_size] -> [BT, vocab_size]
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
        Eigen::IndexPair<int>(1, 1)};
    logits_2d.device(nn::g_device) = lnf_y.contract(lm_head, product_dims);

    auto logits_2d_const = MakeConstMatrix(logits.data(), BT, vocab_size_);
    SoftmaxForwardCPU(logits_2d_const, targets, loss);
  }

  void ForwardGPU(typename TTypes<int>::ConstMatrix idx,
                  typename TTypes<Type, 3>::ConstTensor labels,
                  typename TTypes<Type, 3>::Tensor logits, float* loss) {
    PROFILE_TRACE_FN("GPT");

    // idx: [B, T], targets: [B, T]
    // logits: [B, T, vocab_size]
    const int B = idx.dimension(0), T = idx.dimension(1), C = n_embed_;
    const int BT = B * T;
    CHECK(labels.dimension(0) == B && labels.dimension(1) == T &&
          labels.dimension(2) == vocab_size_);
    CHECK(logits.dimension(0) == B && logits.dimension(1) == T &&
          logits.dimension(2) == vocab_size_)
        << "B: " << B << ", T: " << T << ", vocab_size: " << vocab_size_;
    __Forward(idx);

    auto lnf_y = MakeMatrix(lnf_y_->data<Type>(), BT, C);
    auto lm_head = MakeMatrix(lm_head_, vocab_size_, n_embed_);
    auto logits_2d = MakeMatrix(logits.data(), BT, vocab_size_);

    // [BT, C] x [C, vocab_size] -> [BT, vocab_size]
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
        Eigen::IndexPair<int>(1, 1)};
    logits_2d.device(nn::g_device) = lnf_y.contract(lm_head, product_dims);

    auto logits_2d_const = MakeConstMatrix(logits.data(), BT, vocab_size_);
    auto labels_2d_const = MakeConstMatrix(labels.data(), BT, vocab_size_);
    SoftmaxForwardGPU(logits_2d_const, labels_2d_const, loss);
  }

  void SoftmaxBackwardCPU(absl::Span<const int> targets) {
    PROFILE_TRACE_FN("GPT");

    int BT = targets.size();
    logits_grad_->LazyAllocate(BT * vocab_size_);
    logits_grad_->ZeroData();
    auto probs_2d = MakeConstMatrix(probs_->data<Type>(), BT, vocab_size_);
    auto logits_grad_2d =
        MakeMatrix(logits_grad_->data<Type>(), BT, vocab_size_);
    softmax_cross_entropy_->Backward(probs_2d, targets, logits_grad_2d);
  }

  void BackwardCPU(typename TTypes<int>::ConstMatrix idx,
                   typename TTypes<int>::ConstMatrix targets) {
    PROFILE_TRACE_FN("GPT");

    SoftmaxBackwardCPU(targets);
    BackwardGPU(idx);
  }

  void BackwardGPU(typename TTypes<int>::ConstMatrix idx) {
    PROFILE_TRACE_FN("GPT");

    // idx: [B, T], targets: [B, T]
    const int B = idx.dimension(0), T = idx.dimension(1), C = n_embed_,
              L = n_layer_;
    const int BT = B * T, TC = T * C;
    const int BTC = BT * C;

    wte_->weight_->LazyAllocateGradient();
    if (lm_head_grad_ == nullptr) {
      lm_head_grad_ = wte_->weight_->grad<Type>();
    }

    tok_emb_->LazyAllocateGradient();
    pos_emb_->LazyAllocateGradient();
    encoded_->LazyAllocateGradient();
    block_y_->LazyAllocateGradient();
    lnf_y_->LazyAllocateGradient();

    tok_emb_->ZeroGrad();
    pos_emb_->ZeroGrad();
    encoded_->ZeroGrad();
    block_y_->ZeroGrad();
    lnf_y_->ZeroGrad();

    // backward lm_head
    auto logits_grad_2d =
        MakeMatrix(logits_grad_->data<Type>(), BT, vocab_size_);
    auto lnf_y = MakeMatrix(lnf_y_->data<Type>(), BT, C);
    auto lnf_y_grad = MakeMatrix(lnf_y_->grad<Type>(), BT, C);
    auto lm_head = MakeMatrix(lm_head_, vocab_size_, C);
    auto lm_head_grad = MakeMatrix(lm_head_grad_, vocab_size_, C);
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
        Eigen::IndexPair<int>(1, 0)};
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims2 = {
        Eigen::IndexPair<int>(0, 0)};
    lnf_y_grad.device(nn::g_device) += logits_grad_2d.contract(
        lm_head, product_dims);  // [BT, vocab_size] x [vocab_size, C]
    lm_head_grad.device(nn::g_device) += logits_grad_2d.contract(
        lnf_y, product_dims2);  // [vocab_size, BT] x [BT, C]

    // backward LNF
    auto block_out_2d =
        MakeConstMatrix(block_y_->data<Type>() + (L - 1) * BTC, BT, C);
    auto block_out_grad_2d =
        MakeMatrix(block_y_->grad<Type>() + (L - 1) * BTC, BT, C);
    auto lnf_mean = MakeConstFlat(lnf_mean_->data<Type>(), BT);
    auto lnf_rstd = MakeConstFlat(lnf_rstd_->data<Type>(), BT);
    auto lnf_y_grad_2d = MakeConstMatrix(lnf_y_->grad<Type>(), BT, C);
    lnf_->Backward(block_out_2d, lnf_y_grad_2d, lnf_mean, lnf_rstd,
                   block_out_grad_2d);

    // backward blocks
    for (int l = n_layer_ - 1; l >= 0; --l) {
      const auto& block = h_[l];
      Type* x = l == 0 ? encoded_->data<Type>()
                       : block_y_->data<Type>() + (l - 1) * BTC;
      Type* x_grad = l == 0 ? encoded_->grad<Type>()
                            : block_y_->grad<Type>() + (l - 1) * BTC;
      Type* y_grad = block_y_->grad<Type>() + l * BTC;
      auto block_x_3d = MakeConst3DTensor(x, B, T, C);
      auto block_x_grad_3d = Make3DTensor(x_grad, B, T, C);
      auto block_y_grad_3d = MakeConst3DTensor(y_grad, B, T, C);
      block->Backward(block_x_3d, block_y_grad_3d, block_x_grad_3d);
    }

    // backward tok_emb, pos_emb
    auto encoded_grad = encoded_->matrix_grad<Type>(B, TC);
    auto tok_emb_grad = tok_emb_->matrix_grad<Type>(B, TC);
    auto pos_emb_grad = pos_emb_->flat_grad<Type>();
    Eigen::array<Eigen::Index, 1> along_batch = {0};
    tok_emb_grad.device(nn::g_device) = encoded_grad;
    pos_emb_grad.device(nn::g_device) = tok_emb_grad.sum(along_batch);

    // backward wte, wpe
    std::vector<int> pos(T);
    std::iota(pos.begin(), pos.end(), 0);
    wte_->Backward(idx, tok_emb_grad);
    wpe_->Backward(pos, pos_emb_grad);
  }

  size_t NumParameters() const {
    size_t num_parameters = 0;
    num_parameters += wte_->NumParameters();
    num_parameters += wpe_->NumParameters();
    for (const auto& b : h_) {
      num_parameters += b->NumParameters();
    }
    num_parameters += lnf_->NumParameters();
    return num_parameters;
  }

  size_t NumActivations() const {
    size_t num_activations = 0;
    num_activations += wte_->NumActivations();
    num_activations += wpe_->NumActivations();
    for (const auto& b : h_) {
      num_activations += b->NumActivations();
    }
    num_activations += lnf_->NumActivations();
    num_activations += tok_emb_->size();
    num_activations += pos_emb_->size();
    num_activations += encoded_->size();
    num_activations += block_y_->size();
    num_activations += lnf_y_->size();
    num_activations += lnf_mean_->size();
    num_activations += lnf_rstd_->size();
#ifdef EIGEN_USE_GPU
    num_activations += scratch_->size();
    num_activations += loss_->size();
    num_activations += loss_mean_->size();
#else
    num_activations += probs_->size();
#endif
    num_activations += logits_grad_->size();
    return num_activations;
  }

  void Parameters(std::vector<nn::Parameter*>* parameters) const {
    wte_->Parameters(parameters);
    wpe_->Parameters(parameters);
    for (const auto& b : h_) {
      b->Parameters(parameters);
    }
    lnf_->Parameters(parameters);
  }

 public:
  int block_size_;
  int vocab_size_;
  int padded_vocab_size_;
  int n_layer_;
  int n_embed_;

  // transformer
  std::unique_ptr<nn::Embedding> wte_;
  std::unique_ptr<nn::Embedding> wpe_;
  std::vector<std::unique_ptr<Block>> h_;
  std::unique_ptr<nn::LayerNorm> lnf_;
  std::unique_ptr<nn::SoftmaxCrossEntropy> softmax_cross_entropy_;

  // head
  std::unique_ptr<nn::Linear> lm_head_unused_;
  Type *lm_head_, *lm_head_grad_;  // [vocal_size, C]

  // activation tensors and gradients
  std::unique_ptr<nn::Activation> tok_emb_;              // [B, T, C]
  std::unique_ptr<nn::Activation> pos_emb_;              // [T, C]
  std::unique_ptr<nn::Activation> encoded_;              // [B, T, C]
  std::unique_ptr<nn::Activation> block_y_;              // [L, B, T, C]
  std::unique_ptr<nn::Activation> lnf_y_;                // [B*T, C]
  std::unique_ptr<nn::Activation> lnf_mean_, lnf_rstd_;  // [B*T]
  std::unique_ptr<nn::Activation> probs_;                // [B*T, vocab_size]
  std::unique_ptr<nn::Activation> scratch_;              // [B*T]
  std::unique_ptr<nn::Activation> loss_;                 // [B*T]
  std::unique_ptr<nn::Activation> loss_mean_;            // [1]
  std::unique_ptr<nn::Activation> logits_grad_;          // [B*T, vocab_size]
};

}  // namespace gpt

#endif  // LLM_CPP__GPT_HPP_
