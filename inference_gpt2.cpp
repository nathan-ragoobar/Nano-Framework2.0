#ifdef _WIN32
#include "./llmc/unistd.h"  // Include Windows implementation
#else
#include <unistd.h>     // Include POSIX implementation
#endif

#include <iostream>
#include <memory>

#include "gpt2.hpp"
//#include "llmc/dataloader.h"
//#include "llmc/tokenizer.h"
#include "nano.hpp"
#include "llmc/tokenizer.hpp"


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

int main(int argc, char** argv) {

    struct timespec start, end;

    gpt2::GPT2 model;
    model.BuildFromCheckpoint("./gpt2_124M100Steps.bin"); //Loads model

    int B = 4;   // batch size 4 (i.e. 4 independent token sequences will be
               // trained on)
    int T = 64;  // sequence length 64 (i.e. each sequence is 64 tokens long).
               // must be <= maxT, which is 1024 for GPT-2

    // build the Tokenizer
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    //build the Nano Tokenizer. This has encoding and decoding functions
    nano::Tokenizer tokenizer_nano;
    tokenizer_nano.init();

    nano::GPT2Tokenizer tokenizer_gpt2("vocab.bpe", "encoder.json");



    // some memory for generating samples from the model
    unsigned long long rng_state = 1337;
    int* gen_tokens = (int*)mallocCheck(B * T * sizeof(int));
    const int genT = 64;  // number of steps of inference we will do

    int V = model.config.vocab_size;
    std::unique_ptr<float[]> logit = std::make_unique<float[]>(B * T * V);
    std::unique_ptr<float[]> prob = std::make_unique<float[]>(B * T * V);
    //nn::Parameter label(nn::DT_FLOAT, B * T * V);
    nn::Softmax softmax;

    // Ask the user for input
    std::string input;
    std::cout << "Enter a prompt: ";
    std::getline(std::cin, input);

    // Variable to keep track of the last printed length
    size_t last_printed = 0;

    //Start the timer for inference
    clock_gettime(CLOCK_MONOTONIC, &start);

    //Tokenize the input
    //std::vector<uint32_t> input_tokens = tokenizer_nano.encode_string(input);
    std::vector<int> input_tokens = tokenizer_gpt2.encode(input);

    //Print the tokens
    for (int i = 0; i < input_tokens.size(); i++) {
        std::cout << input_tokens[i] << " ";
    }
    
    // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
    // Initialize gen_tokens with the input tokens and pad with EOT token
    for (int i = 0; i < B * T; ++i) {
        if (i < input_tokens.size()) {
            gen_tokens[i] = input_tokens[i];
        } else {
            gen_tokens[i] = tokenizer.eot_token;
        }
    }
      
     
      // now sample from the model autoregressively
      printf("generating:\n---\n");
      for (int t = input_tokens.size(); t < genT; t++) {
        // note that inference is very wasteful here because for each token
        // we re-calculate the forward pass for all of (B,T) positions from
        // scratch but the inference here is just for sanity checking anyway and
        // we can maybe optimize a bit more later, with careful tests
        auto gen_tokens_2d = TTypes<int>::ConstMatrix(gen_tokens, B, T);
        auto logit_3d = Make3DTensor(logit.get(), B, T, V);
        model.gpt2_->Forward(gen_tokens_2d, logit_3d);
        auto logit_2d = MakeConstMatrix(logit.get(), B * T, V);
        auto prob_2d = MakeMatrix(prob.get(), B * T, V);
        softmax.Forward(logit_2d, prob_2d);
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
        // Replace the token printing section with this:
        // Replace the token printing section with this:
        if (tokenizer.init_ok) {
          // Keep track of last token's position
          static std::string current_output;
          
          // Get just the new token
          std::vector<int> new_token{next_token};
          std::string new_text = tokenizer_gpt2.decode(new_token);
          
          // Add to current output
          current_output += new_text;
          
          // Clear line and print current state
          std::cout << "\r" << std::string(last_printed, ' ') << "\r" << current_output << std::flush;
          last_printed = current_output.length();
          
          // Extra newline at natural breaks
          if (new_text.find('.') != std::string::npos || 
              new_text.find('\n') != std::string::npos || 
              next_token == tokenizer.eot_token) {
              std::cout << std::endl;
          }
      } else {
          printf("%d ", next_token);
          fflush(stdout);
      }
        fflush(stdout);
      }
        //std::string nano_token_str = tokenizer_nano.decode_string(gen_tokens,genT);
        //print Nano token
        //std::cout << "Nano Tokenizer: "<< nano_token_str;
      printf("\n---\n");

      clock_gettime(CLOCK_MONOTONIC, &end);

      double time_elapsed_s =
        (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Inference took: %f ms)\n",
           time_elapsed_s * 1000);
    
    const char* tiny_stories_val = "dev/data/tinystories/TinyStories_val.bin";
  
    const char* tiny_shakespeare_val =
           "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
    const char* val_tokens = access(tiny_shakespeare_val, F_OK) != -1
           ? tiny_shakespeare_val
           : tiny_stories_val;
    
    nn::Parameter label(nn::DT_FLOAT, B * T * V);
    DataLoader val_loader;
    dataloader_init(&val_loader, val_tokens, B, T, 0, 1, 0);
    // Add initialization here
    //if (!dataloader_init(&val_loader, "val.bin", B, T)) {  // Make sure val.bin exists
      //fprintf(stderr, "Failed to initialize validation dataloader\n");
      //return 1;
  //}
    int val_num_batches = 5;
    bool USE_FAST_SOFTMAX = true;

    for (int step = 0; step <= 10; step++) {
    
      float val_loss = 0.0f;
      dataloader_reset(&val_loader);
      for (int i = 0; i < val_num_batches; i++) {
        dataloader_next_batch(&val_loader);
        float loss = 0.0f;
        auto idx = TTypes<int>::ConstMatrix(val_loader.inputs, B, T);
        if (USE_FAST_SOFTMAX) {
          auto target = TTypes<int>::ConstMatrix(val_loader.targets, B, T);
          auto logit_3d = Make3DTensor(logit.get(), B, T, V);
          model.gpt2_->ForwardCPU(idx, target, logit_3d, &loss);
        } else {
          label.ZeroData();
          nn::OntHot(MakeConstFlat(val_loader.targets, B * T),
                      label.matrix<float>(B * T, V));
          auto label_3d = label.const_tensor_3d<float>(B, T, V);
          auto logit_3d = Make3DTensor(logit.get(), B, T, V);
          model.gpt2_->ForwardGPU(idx, label_3d, logit_3d, &loss);
        }
        val_loss += loss;
      }
      val_loss /= val_num_batches;

      if (step == 0) {
        size_t num_activations = model.gpt2_->NumActivations();
        printf("num_activations: %zu(%zu MB)\n", num_activations,
                num_activations * sizeof(floatX) / 1024 / 1024);
      }
      printf("val loss %f\n", val_loss);
    }

    // Add cleanup
    dataloader_free(&val_loader);
    return 0;
      

}