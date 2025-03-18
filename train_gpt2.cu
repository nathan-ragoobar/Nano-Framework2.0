#ifdef _WIN32
#include "./llmc/unistd.h"  // Include Windows implementation
#else
#include <unistd.h>     // Include POSIX implementation
#endif

#include <iostream>
#include <memory>
#include <vector>
#include <numeric>
#include <cstring>
#include <string>
#include <cmath> 
#include "NanoDashWriter/writer.hpp" // Add this include

#include <nvtx3/nvToolsExt.h>
#include "./gpt2.hpp"
#include "llmc/dataloader.h"
#include "llmc/tokenizer.h"
//#include "optim.hpp"
#include  "./nano.hpp"
#include "cuda_profile_util.hpp"
#include <eigen/unsupported/Eigen/CXX11/Tensor>
#include <cuda_runtime.h>


// sampler

unsigned int random_u32(unsigned long long* state) {
  // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long* state) {  // random float32 in [0,1)
  return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_mult(float* probabilities, int n, float coin) {
  // sample index from probabilities (they must sum to 1!)
  // coin is a random number in [0, 1), usually from random_f32()
  float cdf = 0.0f;
  for (int i = 0; i < n; i++) {
    cdf += probabilities[i];
    if (coin < cdf) {
      return i;
    }
  }
  return n - 1;  // in case of rounding errors
}

float cosine_learning_rate(int step, int total_steps, float initial_lr) {
  float pi = 3.14159265358979323846;
  return initial_lr * 0.5 * (1 + cos(pi * step / total_steps));
}

void save_model_checkpoint(gpt2::GPT2& model, int step) {
  char filename[100];
  snprintf(filename, sizeof(filename), "c4_gpt2_124M_step_%d.bin", step);
  model.SaveModel(filename);
  printf("Saved model checkpoint: %s\n", filename);
}

// CUDA error checking
void cudaCheck(cudaError_t error, const char* file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

// Update the print_usage function to include the new parameter
void print_usage() {
  printf("Usage: train_gpt2 [options]\n");
  printf("Options:\n");
  printf("  --batch_size N            Batch size (default: 4)\n");
  printf("  --seq_len N               Sequence length (default: 64)\n");
  printf("  --learning_rate N         Initial learning rate (default: 1e-3)\n");
  printf("  --steps N                 Total training steps (default: 35000)\n");
  printf("  --checkpoint_steps N      Save model checkpoint every N steps (default: 5000)\n");
  printf("  --inference_steps N        Generate text every N steps (default: 100)\n");
  printf("  --help                    Display this help message\n");
}


int main(int argc, char** argv) {
  int B = 4;   // batch size 4 (i.e. 4 independent token sequences will be
  // trained on)
  int T = 64;  // sequence length 64 (i.e. each sequence is 64 tokens long).
  // must be <= maxT, which is 1024 for GPT-2
  // Define total training steps and initial learning rate
  int total_steps = 100;
  float initial_lr = 1e-3f;
  int checkpoint_steps = 5000;  // New default value
  int inference_steps = 100;

  // Add the checkpoint_steps parameter to the argument parser
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--help") == 0) {
        print_usage();
        return 0;
    } else if (strcmp(argv[i], "--batch_size") == 0 && i + 1 < argc) {
        B = atoi(argv[++i]);
        if (B <= 0) {
            fprintf(stderr, "Error: Batch size must be positive\n");
            return 1;
        }
    } else if (strcmp(argv[i], "--seq_len") == 0 && i + 1 < argc) {
        T = atoi(argv[++i]);
        if (T <= 0 || T > 1024) {
            fprintf(stderr, "Error: Sequence length must be between 1 and 1024\n");
            return 1;
        }
    } else if (strcmp(argv[i], "--learning_rate") == 0 && i + 1 < argc) {
        initial_lr = atof(argv[++i]);
        if (initial_lr <= 0) {
            fprintf(stderr, "Error: Learning rate must be positive\n");
            return 1;
        }
    } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
        total_steps = atoi(argv[++i]);
        if (total_steps <= 0) {
            fprintf(stderr, "Error: Steps must be positive\n");
            return 1;
        }
    } else if (strcmp(argv[i], "--checkpoint_steps") == 0 && i + 1 < argc) {
        checkpoint_steps = atoi(argv[++i]);
        if (checkpoint_steps <= 0) {
            fprintf(stderr, "Error: Checkpoint steps must be positive\n");
            return 1;
        }
    }else if (strcmp(argv[i], "--inference_steps") == 0 && i + 1 < argc) {
      inference_steps = atoi(argv[++i]);
      if (inference_steps <= 0) {
          fprintf(stderr, "Error: Inference steps must be positive\n");
          return 1;
      } 
    }else {
        fprintf(stderr, "Unknown option: %s\n", argv[i]);
        print_usage();
        return 1;
    }
  }

  // Add checkpoint interval to the configuration output
  printf("Configuration:\n");
  printf("  Batch size: %d\n", B);
  printf("  Sequence length: %d\n", T);
  printf("  Learning rate: %g\n", initial_lr);
  printf("  Training steps: %d\n", total_steps);
  printf("  Checkpoint every: %d steps\n\n", checkpoint_steps);
  printf("  Inference every: %d steps\n\n", inference_steps);


  // Initialize the MetricWriter object
  std::vector<std::string> metrics = {
    "train_loss", 
    "val_loss",
    "perplexity",
    "time_ms",
    "tokens_per_second",  // Add this
    "learning_rate"       // Add this
  };
  MetricWriter writer("gpt2_training", metrics);

  gpt2::GPT2Config config;
  config.max_seq_len = 1024;
  config.vocab_size = 50257;
  config.padded_vocab_size = 50304;
  
  config.num_layers = 12;
  config.num_heads = 12;
  config.channels = 768;
  /*
  config.num_layers = 8;
  config.num_heads = 8;
  config.channels = 64;
*/
  gpt2::GPT2 model;
  //model.InitializeFromScratch(config);
  model.BuildFromCheckpoint("jfleg_attempt3.bin");

  // build the DataLoaders from tokens files. for now use tiny_stories if
  // available, else tiny_shakespeare
  // Only use edu_fineweb dataset
  const char* train_tokens = "c4_tokenized/train_tokenized_output_*.bin";
  const char* val_tokens = "c4_tokenized/val_tokenized_output_*.bin";

  // Check if directory exists and print the paths we're trying to use
  printf("Using training data path: %s\n", train_tokens);
  printf("Using validation data path: %s\n", val_tokens);

  // Initialize dataloaders - glob will handle the wildcard patterns
  printf("Initializing training dataloader...\n");
  DataLoader train_loader, val_loader;
  dataloader_init(&train_loader, train_tokens, B, T, 0, 1, 0);

  printf("Initializing validation dataloader...\n");
  dataloader_init(&val_loader, val_tokens, B, T, 0, 1, 0);

  printf("train dataset num_batches: %zu\n", train_loader.num_tokens / (B * T));
  printf("val dataset num_batches: %zu\n", val_loader.num_tokens / (B * T));
  int val_num_batches = 5;

  // build the Tokenizer
  Tokenizer tokenizer;
  tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

  // some memory for generating samples from the model
  unsigned long long rng_state = 1337;
  int* gen_tokens = (int*)mallocCheck(B * T * sizeof(int));
  const int genT = 64;  // number of steps of inference we will do

  // train
  struct timespec start, end;
  int V = model.config.vocab_size;
  std::unique_ptr<float[]> prob = std::make_unique<float[]>(B * T * V);
  std::unique_ptr<float[]> label = std::make_unique<float[]>(B * T * V);

  // After Parameter creation
    //printf("Device memory info before allocation:\n");
    //size_t free_mem, total_mem;
    //cudaCheck(cudaMemGetInfo(&free_mem, &total_mem));
    //printf("Free: %zu MB, Total: %zu MB\n", free_mem/(1024*1024), total_mem/(1024*1024));



  nn::Parameter d_label(nn::DT_FLOAT, B * T * V),
      d_logit(nn::DT_FLOAT, B * T * V), d_prob(nn::DT_FLOAT, B * T * V);
/*
    // Verify each Parameter's memory
    printf("d_label ptr: %p, size: %zu bytes\n", 
        (void*)d_label.data<float>(), 
        d_label.size() * sizeof(float));
    printf("d_logit ptr: %p, size: %zu bytes\n",
        (void*)d_logit.data<float>(),
        d_logit.size() * sizeof(float));
    printf("d_prob ptr: %p, size: %zu bytes\n",
        (void*)d_prob.data<float>(),
        d_prob.size() * sizeof(float));
*/

  // Add after Parameter creation
  //Check if memory allocation is successful
   if (!d_label.data<float>() || !d_logit.data<float>() || !d_prob.data<float>()) {
    printf("GPU memory allocation failed\n");
    exit(1);
}    


  nn::Softmax softmax;
  std::vector<nn::Parameter*> parameters;
  model.Parameters(&parameters);
  optim::AdamW optimizer(parameters, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.0f);
  std::vector<double> timings;



  for (int step = 0; step <= total_steps; step++) {
    NvtxRange step_range("Train step", step);

    // Calculate the current learning rate using the cosine schedule
    float current_lr = cosine_learning_rate(step, total_steps, initial_lr);


    // once in a while estimate the validation loss
    if (step % 10 == 0) {
      float val_loss = 0.0f;
      dataloader_reset(&val_loader);
      for (int i = 0; i < val_num_batches; i++) {
        NvtxRange validation_range("validation");
        dataloader_next_batch(&val_loader);
        float loss = 0.0f;
        auto idx = TTypes<int>::ConstMatrix(val_loader.inputs, B, T);
        std::memset(label.get(), 0, sizeof(float) * B * T * V);
        nn::OntHot(MakeConstFlat(val_loader.targets, B * T),
                   MakeMatrix(label.get(), B * T, V));


                   

        // Before cudaMemcpy, add synchronization and validation
        cudaCheck(cudaDeviceSynchronize());
        if (!label.get()) {
            printf("Host memory is null\n");
            exit(1);
        }
        /*
        printf("Debug: Copying %zu bytes from host(%p) to device(%p)\n", 
            sizeof(float) * B * T * V, 
            (void*)label.get(), 
            (void*)d_label.data<float>());
          */
        // Before the cudaMemcpy, verify alignment
        if ((reinterpret_cast<std::uintptr_t>(d_label.data<float>()) % 16) != 0) {
            printf("Warning: Device pointer not 16-byte aligned\n");
        }
        if ((reinterpret_cast<std::uintptr_t>(label.get()) % 16) != 0) {
            printf("Warning: Host pointer not 16-byte aligned\n");
}


    // Before the problematic cudaMemcpy
    //cudaCheck(cudaDeviceReset());  // Reset device state
    //cudaCheck(cudaSetDevice(0));   // Ensure we're on the right device

    // Add error checking before memcpy
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error before memcpy: %s\n", cudaGetErrorString(err));
    }

    // Try pinned memory for the host
    float* pinned_label;
    cudaCheck(cudaMallocHost(&pinned_label, sizeof(float) * B * T * V));
    std::memcpy(pinned_label, label.get(), sizeof(float) * B * T * V);

    // Copy from pinned memory to device (correct direction)
    cudaCheck(cudaMemcpy(d_label.data<float>(),  // destination (device)
    pinned_label,            // source (host)
    sizeof(float) * B * T * V,
    cudaMemcpyHostToDevice));

    // Free pinned memory
    cudaCheck(cudaFreeHost(pinned_label));

/*
        cudaCheck(cudaMemcpy(d_label.data<float>(), label.get(),
                            sizeof(float) * B * T * V,
                            cudaMemcpyHostToDevice));
        cudaCheck(cudaDeviceSynchronize());
*/


            /*
        cudaCheck(cudaMemcpy(d_label.data<float>(), label.get(),
                             sizeof(float) * B * T * V,
                             cudaMemcpyHostToDevice));
                             */
        auto label_3d = d_label.const_tensor_3d<float>(B, T, V);
        auto logit_3d = d_logit.tensor_3d<float>(B, T, V);
        model.gpt2_->ForwardGPU(idx, label_3d, logit_3d, &loss);
        val_loss += loss;
      }
      val_loss /= val_num_batches;

      if (step == 0) {
        size_t num_activations = model.gpt2_->NumActivations();
        printf("num_activations: %zu(%zu MB)\n", num_activations,
               num_activations * sizeof(floatX) / 1024 / 1024);
      }
      printf("val loss %f\n", val_loss);
      writer.addValidationLoss(val_loss, step);
    }

    // once in a while do model inference to print generated text
    if (step % inference_steps == 0) {
      NvtxRange generation_range("generation");
      // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
      for (int i = 0; i < B * T; ++i) {
        gen_tokens[i] = tokenizer.eot_token;
      }
      // now sample from the model autoregressively
      printf("generating:\n---\n");
      for (int t = 1; t < genT; t++) {
        // note that inference is very wasteful here because for each token
        // we re-calculate the forward pass for all of (B,T) positions from
        // scratch but the inference here is just for sanity checking anyway and
        // we can maybe optimize a bit more later, with careful tests
        auto gen_tokens_2d = TTypes<int>::ConstMatrix(gen_tokens, B, T);
        auto logit_3d = d_logit.tensor_3d<float>(B, T, V);
        model.gpt2_->Forward(gen_tokens_2d, logit_3d);
        auto logit_2d = d_logit.const_matrix<float>(B * T, V);
        auto prob_2d = d_prob.matrix<float>(B * T, V);
        softmax.Forward(logit_2d, prob_2d);
        nn::g_device.memcpyDeviceToHost(prob.get(), d_prob.data<float>(),
                                        sizeof(float) * B * T * V);
        //nn::g_device.synchronize();
        cudaDeviceSynchronize(); //From the CUDA runtime API
        // furthermore, below we're only using b=0 (i.e. the first row) of all B
        // rows we're in principle running B "inference streams" in parallel
        // here but only using position 0 get the Vp-dimensional vector probs[0,
        // t-1, :]
        float* probs = prob.get() + (t - 1) * V;
        float coin = random_f32(&rng_state);
        // note we're only sampling from the first V elements, ignoring padding
        // (the probabilities in the padded region should be zero anyway)
        int next_token = sample_mult(probs, model.config.vocab_size, coin);
        gen_tokens[t] = next_token;
        // print the generated token, either using the Tokenizer or a fallback
        if (tokenizer.init_ok) {
          const char* token_str = tokenizer_decode(&tokenizer, next_token);
          safe_printf(token_str);
        } else {
          // fall back to printing the token id
          printf("%d ", next_token);
        }
        fflush(stdout);
      }
      printf("\n---\n");
    }

    // do a training step
    clock_gettime(CLOCK_MONOTONIC, &start);
    dataloader_next_batch(&train_loader);
    float loss = 0.0f;
    auto idx = TTypes<int>::ConstMatrix(train_loader.inputs, B, T);
    std::memset(label.get(), 0, sizeof(float) * B * T * V);
    nn::OntHot(MakeConstFlat(train_loader.targets, B * T),
               MakeMatrix(label.get(), B * T, V));
    cudaCheck(cudaMemcpy(d_label.data<float>(), label.get(),
                         sizeof(float) * B * T * V, cudaMemcpyHostToDevice));
    auto label_3d = d_label.const_tensor_3d<float>(B, T, V);
    auto logit_3d = d_logit.tensor_3d<float>(B, T, V);
    model.gpt2_->ForwardGPU(idx, label_3d, logit_3d, &loss);
    optimizer.ZeroGrad();
    model.gpt2_->BackwardGPU(idx);
    optimizer.Step(step + 1, current_lr);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_elapsed_s =
    (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    // Calculate tokens per second
    float tokens_per_second = (B * T) / time_elapsed_s;

    // Calculate perplexity (e^loss for cross-entropy loss)
    float perplexity = exp(loss);

    printf("step %d: train loss %f | perplexity %f | tokens/sec %f (took %f ms)\n", 
          step, loss, perplexity, tokens_per_second, time_elapsed_s * 1000);

    // Add metrics to the writer
    writer.addTrainingLoss(loss, step);
    writer.addScalar("perplexity", perplexity, step);  // Add perplexity metric
    writer.addScalar("time_ms", time_elapsed_s * 1000, step);
    writer.addScalar("tokens_per_second", tokens_per_second, step);
    writer.addScalar("learning_rate", current_lr, step); 
    
    if (step) {
      timings.push_back(time_elapsed_s);
    }

    //Save model checkpoint
    if (step % checkpoint_steps == 0 && step > 0) {
      save_model_checkpoint(model, step);
    }

  }

  double sum = std::accumulate(timings.begin(), timings.end(), 0.0);
  if (!timings.empty()) {
    printf("final %zu iters avg: %.3f ms\n", timings.size(),
           1000 * sum / timings.size());
  }

  //Save model
  model.SaveModel("c4_gpt2_124MFinal.bin");

  writer.close(); //Close the writer

  // free
  dataloader_free(&train_loader);
  dataloader_free(&val_loader);
  tokenizer_free(&tokenizer);
  free(gen_tokens);
  return 0;
}
