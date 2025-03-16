
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
#include "NanoDashWriter/writer.hpp" // Add this include

#include "gpt2.hpp"
//#include "llmc/dataloader.h"
//#include "llmc/tokenizer.h"
#include "nano.hpp"

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
  snprintf(filename, sizeof(filename), "gpt2_124M_step_%d.bin", step);
  model.SaveModel(filename);
  printf("Saved model checkpoint: %s\n", filename);
}

// Define a structure for paired examples
struct GrammarExample {
  std::vector<int> incorrect_tokens;
  std::vector<int> correct_tokens;
};


// Load a batch of grammar examples for fine-tuning
std::vector<GrammarExample> load_grammar_dataset(const char* filename, nano::GPT2Tokenizer& tokenizer) {
  std::vector<GrammarExample> examples;
  std::ifstream file(filename);
  std::string line_incorrect, line_correct;
  
  if (!file.is_open()) {
    std::cerr << "Error: Could not open grammar dataset file: " << filename << std::endl;
    return examples;
  }
  
  while (std::getline(file, line_incorrect) && std::getline(file, line_correct)) {
    GrammarExample example;
    
    // Use the GPT2Tokenizer encode method directly with the strings
    example.incorrect_tokens = tokenizer.encode(line_incorrect);
    example.correct_tokens = tokenizer.encode(line_correct);
    
    examples.push_back(example);
  }
  
  return examples;
}


// Load a batch of grammar examples for fine-tuning
void load_grammar_batch(std::vector<GrammarExample>& examples, int* inputs, int* targets,
                       int batch_size, int seq_len, int batch_idx) {
    
    // Zero out arrays first
    std::memset(inputs, 0, batch_size * seq_len * sizeof(int));
    std::memset(targets, 0, batch_size * seq_len * sizeof(int));
    
    for (int b = 0; b < batch_size; b++) {
        // Get example index with wrapping
        size_t ex_idx = (batch_idx * batch_size + b) % examples.size();
        const auto& example = examples[ex_idx];
        
        // Determine sequence length to use (up to seq_len)
        int incorrect_len = std::min((int)example.incorrect_tokens.size(), seq_len - 1);
        int correct_len = std::min((int)example.correct_tokens.size(), seq_len - 1);
        
        // Fill the input with incorrect text (we'll predict the correct version)
        for (int t = 0; t < incorrect_len; t++) {
            inputs[b * seq_len + t] = example.incorrect_tokens[t];
        }
        
        // Fill the target with correct text (shifted by one token)
        targets[b * seq_len] = example.correct_tokens[0];  // First token
        for (int t = 0; t < correct_len - 1; t++) {
            targets[b * seq_len + t + 1] = example.correct_tokens[t + 1];
        }
    }
}


void evaluate_grammar_correction(gpt2::GPT2& model, nano::GPT2Tokenizer& tokenizer, 
                               const std::vector<GrammarExample>& examples, 
                               int step) {
    
    printf("\n--- Grammar Correction Evaluation (Step %d) ---\n", step);
    
    // Use a small subset for evaluation
    const int num_eval_examples = std::min(5, (int)examples.size());
    const int B = 8;  // Use same batch size as training
    const int T = 64; // Max sequence length
    const int V = model.config.vocab_size;
    
    // Allocate memory
    std::unique_ptr<float[]> logits = std::make_unique<float[]>(B * T * V);
    std::unique_ptr<float[]> probs = std::make_unique<float[]>(B * T * V);
    int* inputs = (int*)mallocCheck(B * T * sizeof(int));
    int* dummy_targets = (int*)mallocCheck(B * T * sizeof(int));
    nn::Softmax softmax;
    
    for (int i = 0; i < num_eval_examples; i++) {
        const auto& example = examples[i];
        
        // Print original incorrect sentence
        printf("Incorrect: ");
        for (int token : example.incorrect_tokens) {
            std::vector<int> token_vec{token};
            std::string token_str = tokenizer.decode(token_vec);
            std::cout << token_str;
        }
        printf("\n");

        // Print expected correct sentence
        printf("Expected:  ");
        for (int token : example.correct_tokens) {
            std::vector<int> token_vec{token};
            std::string token_str = tokenizer.decode(token_vec);
            std::cout << token_str;
        }
        printf("\n");
        
        // Prepare input - fill all batch elements with the same data
        std::memset(inputs, 0, B * T * sizeof(int));
        std::memset(dummy_targets, 0, B * T * sizeof(int));
        
        int len = std::min((int)example.incorrect_tokens.size(), T);
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < len; t++) {
                inputs[b * T + t] = example.incorrect_tokens[t];
            }
        }
        
        // Forward pass with matching dimensions
        auto inputs_2d = TTypes<int>::ConstMatrix(inputs, B, T);
        auto targets_2d = TTypes<int>::ConstMatrix(dummy_targets, B, T);
        auto logits_3d = Make3DTensor(logits.get(), B, T, V);
        
        float dummy_loss;
        model.gpt2_->ForwardCPU(inputs_2d, targets_2d, logits_3d, &dummy_loss);
        
        // Generate corrected output by sampling
        unsigned long long rng_state = 1337;
        std::vector<int> generated;
        generated.push_back(inputs[0]); // Start with first token
        
        for (int t = 1; t < len + 10; t++) { // Generate slightly more tokens
            if (t < len) {
                // For input context part, just use the input
                generated.push_back(inputs[t]);
                continue;
            }
            
            // Set up for generation - use only first batch
            for (int b = 0; b < B; b++) {
                for (int j = 0; j < T; j++) {
                    inputs[b * T + j] = j < generated.size() ? generated[j] : 0;
                }
            }
            
            // Forward pass
            model.gpt2_->ForwardCPU(inputs_2d, targets_2d, logits_3d, &dummy_loss);
            
            // Extract logits for the last position in first batch
            auto logit_2d = MakeConstMatrix(logits.get(), B * T, V);
            auto prob_2d = MakeMatrix(probs.get(), B * T, V);
            softmax.Forward(logit_2d, prob_2d);
            
            // Sample next token from first batch element
            float* token_probs = probs.get() + (generated.size() - 1) * V;
            float coin = random_f32(&rng_state);
            int next_token = sample_mult(token_probs, V, coin);
            
            // Add token to generated sequence
            generated.push_back(next_token);
            
            // Stop if we generate EOT
            if (next_token == tokenizer.eot_token()) break;
        }
        
        // Print model's corrected version
        printf("Generated: ");
        for (int token : generated) {  // Use generated tokens, not incorrect_tokens
            std::vector<int> token_vec{token};
            std::string token_str = tokenizer.decode(token_vec);
            std::cout << token_str;
        }
        printf("\n\n");
    }
    
    printf("--- End Evaluation ---\n\n");
    free(inputs);
    free(dummy_targets);
}

// Update the print_usage function to include the new parameter
void print_usage() {
  printf("Usage: train_gpt2 [options]\n");
  printf("Options:\n");
  printf("  --batch_size N            Batch size (default: 4)\n");
  printf("  --seq_len N               Sequence length (default: 64)\n");
  printf("  --learning_rate N         Initial learning rate (default: 1e-3)\n");
  printf("  --steps N                 Total training steps (default: 35000)\n");
  printf("  --checkpoint_steps N      Save model checkpoint every N steps (default: 5000)\n");
  printf("  --help                    Display this help message\n");
  printf("  --finetune                  Enable fine-tuning mode\n");
  printf("  --grammar_dataset PATH      Path to grammar dataset\n");
  printf("  --finetune_lr N             Fine-tuning learning rate (default: 5e-5)\n");

}

bool USE_FAST_SOFTMAX = true;

int main(int argc, char** argv) {
  int B = 4;   // batch size 4 (i.e. 4 independent token sequences will be
  // trained on)
  int T = 64;  // sequence length 64 (i.e. each sequence is 64 tokens long).
  // must be <= maxT, which is 1024 for GPT-2
  // Define total training steps and initial learning rate
  int total_steps = 35000;
  float initial_lr = 1e-3f;
  int checkpoint_steps = 5000;  // New default value

  //Fine tuning config
  bool finetune_mode = false;
  const char* grammar_dataset_path = nullptr;
  float finetune_lr = 5e-5f;


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
    }  else if (strcmp(argv[i], "--finetune") == 0) {
        finetune_mode = true;
    } else if (strcmp(argv[i], "--grammar_dataset") == 0 && i + 1 < argc) {
        grammar_dataset_path = argv[++i];
    } else if (strcmp(argv[i], "--finetune_lr") == 0 && i + 1 < argc) {
        finetune_lr = atof(argv[++i]);
        if (finetune_lr <= 0) {
            fprintf(stderr, "Error: Fine-tuning learning rate must be positive\n");
            return 1;
        }
    } else {
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


  // Initialize the MetricWriter object
  std::vector<std::string> metrics = {
    "train_loss", 
    "val_loss",
    "time_ms",
    "tokens_per_second",  // Add this
    "learning_rate"       // Add this
  };
  MetricWriter writer("gpt2_training", metrics);

  gpt2::GPT2Config config;
  config.max_seq_len = 1024;
  config.vocab_size = 50257;
  config.padded_vocab_size = 50304;
  
  /*
  config.num_layers = 12;
  config.num_heads = 12;
  config.channels = 768;
  */
  config.num_layers = 8;
  config.num_heads = 8;
  config.channels = 64;
 
  

  gpt2::GPT2 model;
  //model.InitializeFromScratch(config);

  //gpt2::GPT2 model;
  model.BuildFromCheckpoint("./gpt2_124M.bin"); //Loads model

  // build the DataLoaders from tokens files. for now use tiny_stories if
  // available, else tiny_shakespeare
  // Only use edu_fineweb dataset
  const char* train_tokens = "jfleg/jfleg_train.bin";
  const char* val_tokens = "jfleg/jfleg_val.bin";

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

  nano::GPT2Tokenizer tokenizer_gpt2("vocab.bpe", "encoder.json");
  

  // some memory for generating samples from the model
  unsigned long long rng_state = 1337;
  int* gen_tokens = (int*)mallocCheck(B * T * sizeof(int));
  const int genT = 64;  // number of steps of inference we will do

  // train
  struct timespec start, end;
  int V = model.config.vocab_size;
  std::unique_ptr<float[]> logit = std::make_unique<float[]>(B * T * V);
  std::unique_ptr<float[]> prob = std::make_unique<float[]>(B * T * V);
  nn::Parameter label(nn::DT_FLOAT, B * T * V);
  nn::Softmax softmax;
  std::vector<nn::Parameter*> parameters;
  model.Parameters(&parameters);
  optim::AdamW optimizer(parameters, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f); //defines the AdamW optimizer optimizer to be used
  std::vector<double> timings;

 

  for (int step = 0; step <= total_steps; step++) {

    // Calculate the current learning rate using the cosine schedule
    float current_lr = cosine_learning_rate(step, total_steps, initial_lr);

    // once in a while estimate the validation loss
    if (step % 10 == 0) {
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
        size_t num_parameters = model.gpt2_->NumParameters();
        printf("num_parameters: %zu\n", num_parameters);
      }
      printf("val loss %f\n", val_loss);
      writer.addValidationLoss(val_loss, step);
    }

    // once in a while do model inference to print generated text
    if (step % 10 == 0) {

      // Ask the user for input
      std::string input;
      std::cout << "Enter a prompt: ";
      std::getline(std::cin, input);

      //Tokenize  the input 
      std::vector<int> input_tokens = tokenizer_gpt2.encode(input);

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

      clock_gettime(CLOCK_MONOTONIC, &start);
      printf("generating:\n---\n");
      for (int t = 1; t < genT; t++) {
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
      // Add after the generation loop:
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    int tokens_generated = genT - 1;  // Exclude the EOT token
    printf("Generated %d tokens in %.3f seconds (%.1f tokens/sec)\n", 
          tokens_generated, time_elapsed_s, tokens_generated/time_elapsed_s);


    }

    if (finetune_mode) {
      // Load grammar dataset
      if (!grammar_dataset_path) {
          fprintf(stderr, "Error: Grammar dataset path required for fine-tuning\n");
          return 1;
      }
      
      printf("Loading grammar dataset from: %s\n", grammar_dataset_path);
      std::vector<GrammarExample> grammar_examples = 
          load_grammar_dataset(grammar_dataset_path, tokenizer_gpt2);
      printf("Loaded %zu grammar examples\n", grammar_examples.size());
      
      // Allocate memory for custom batch loading
      int* finetune_inputs = (int*)mallocCheck(B * T * sizeof(int));
      int* finetune_targets = (int*)mallocCheck(B * T * sizeof(int));
      
      // Use lower learning rate for fine-tuning
      initial_lr = finetune_lr;
      
      // Fine-tuning loop
      for (int step = 0; step <= total_steps; step++) {
          float current_lr = cosine_learning_rate(step, total_steps, initial_lr);
          
          // Load a batch of grammar examples
          load_grammar_batch(grammar_examples, finetune_inputs, finetune_targets, 
                            B, T, step);
          
          clock_gettime(CLOCK_MONOTONIC, &start);
          float loss = 0.0f;
          auto idx = TTypes<int>::ConstMatrix(finetune_inputs, B, T);
          
          if (USE_FAST_SOFTMAX) {
              auto target = TTypes<int>::ConstMatrix(finetune_targets, B, T);
              auto logit_3d = Make3DTensor(logit.get(), B, T, V);
              model.gpt2_->ForwardCPU(idx, target, logit_3d, &loss);
              optimizer.ZeroGrad();
              model.gpt2_->BackwardCPU(idx, target);
          } else {
              label.ZeroData();
              nn::OntHot(MakeConstFlat(finetune_targets, B * T),
                        label.matrix<float>(B * T, V));
              auto label_3d = label.const_tensor_3d<float>(B, T, V);
              auto logit_3d = Make3DTensor(logit.get(), B, T, V);
              model.gpt2_->ForwardGPU(idx, label_3d, logit_3d, &loss);
              optimizer.ZeroGrad();
              model.gpt2_->BackwardGPU(idx);
          }
          
          optimizer.Step(step + 1, current_lr);
          
          clock_gettime(CLOCK_MONOTONIC, &end);
          double time_elapsed_s = 
              (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

          // Calculate tokens per second
          float tokens_per_second = (B * T) / time_elapsed_s;

          printf("step %d: train loss %f | tokens/sec %f (took %f ms)\n", step, loss, tokens_per_second,
           time_elapsed_s * 1000);
          
          // Add metrics to the writer
          writer.addTrainingLoss(loss, step);
          writer.addScalar("time_ms", time_elapsed_s * 1000, step);
          writer.addScalar("tokens_per_second", tokens_per_second, step);  // Add this
          writer.addScalar("learning_rate", current_lr, step); 
          
          // Periodically evaluate on validation grammar examples
          if (step % 50 == 0) {
              // Implement grammar correction evaluation
              evaluate_grammar_correction(model, tokenizer_gpt2, grammar_examples, step);
          }
          
          // Save checkpoints
          if (step % checkpoint_steps == 0 && step > 0) {
              char filename[100];
              snprintf(filename, sizeof(filename), "grammar_gpt2_step_%d.bin", step);
              model.SaveModel(filename);
              printf("Saved grammar fine-tuned model: %s\n", filename);
          }
      }
      
      // Free fine-tuning specific resources
      free(finetune_inputs);
      free(finetune_targets);
      
      // Save final fine-tuned model
      model.SaveModel("grammar_gpt2_final.bin");
  } else {
      // do a training step
    clock_gettime(CLOCK_MONOTONIC, &start);
    dataloader_next_batch(&train_loader);
    float loss = 0.0f;
    auto idx = TTypes<int>::ConstMatrix(train_loader.inputs, B, T);
    if (USE_FAST_SOFTMAX) {
      auto target = TTypes<int>::ConstMatrix(train_loader.targets, B, T);
      auto logit_3d = Make3DTensor(logit.get(), B, T, V);
      model.gpt2_->ForwardCPU(idx, target, logit_3d, &loss);  //This calls the forward pass on the model.gpt2_ object 
      optimizer.ZeroGrad();
      model.gpt2_->BackwardCPU(idx, target);
    } else {
      label.ZeroData();
      nn::OntHot(MakeConstFlat(train_loader.targets, B * T),
                 label.matrix<float>(B * T, V));
      auto label_3d = label.const_tensor_3d<float>(B, T, V);
      auto logit_3d = Make3DTensor(logit.get(), B, T, V);
      model.gpt2_->ForwardGPU(idx, label_3d, logit_3d, &loss);
      optimizer.ZeroGrad();
      model.gpt2_->BackwardGPU(idx);
    }
    optimizer.Step(step + 1, current_lr);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_elapsed_s =
        (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    // Calculate tokens per second
    float tokens_per_second = (B * T) / time_elapsed_s;
    
     printf("step %d: train loss %f | tokens/sec %f (took %f ms)\n", step, loss, tokens_per_second,
           time_elapsed_s * 1000);

    // Add metrics to the writer
    writer.addTrainingLoss(loss, step);
    writer.addScalar("time_ms", time_elapsed_s * 1000, step);
    writer.addScalar("tokens_per_second", tokens_per_second, step);  // Add this
    writer.addScalar("learning_rate", current_lr, step); 

    if (step) {
      timings.push_back(time_elapsed_s);
    }

    // Save model checkpoint
    
    //Save model checkpoint
    if (step % checkpoint_steps == 0 && step > 0) {
      save_model_checkpoint(model, step);
    }

  }

    
  }

  double sum = std::accumulate(timings.begin(), timings.end(), 0.0);
  if (!timings.empty()) {
    printf("final %zu iters avg: %.3f ms\n", timings.size(),
           1000 * sum / timings.size());
  }

  //Save model
  model.SaveModel("gpt2_124MFinal.bin");

  writer.close(); //Close the writer

  // free
  dataloader_free(&train_loader);
  dataloader_free(&val_loader);
  tokenizer_free(&tokenizer);
  free(gen_tokens);
  return 0;
}
