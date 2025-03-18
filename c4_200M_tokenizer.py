#!/usr/bin/env python
"""
Downloads, processes, and tokenizes part of the C4_200M dataset for grammar correction.
- Streams data from HuggingFace datasets
- Formats examples into the instruction style:
    ### Instruction: Correct the grammar in this text
    ### Input: [incorrect sentence]
    ### Output: [corrected sentence]
    ### End
- Tokenizes using tiktoken with GPT‑2 encoding
- Splits the data into training (80%) and validation (20%) splits
- Places examples continuously one after another, separated by EOT_TOKEN
- Outputs each split in the TinyStories binary format
- Processes data in batches to avoid memory issues
"""

import os
import argparse
import numpy as np
import tiktoken
from tqdm import tqdm
from datasets import load_dataset

# -----------------------------------------------------------------------------
# Constants and configuration
# -----------------------------------------------------------------------------
INSTRUCTION_PREFIX = "### Instruction: Correct the grammar in this text\n### Input: "
INSTRUCTION_SEPARATOR = "\n### Output: "
TASK_COMPLETE = "\n### End"
MAGIC_NUMBER = 20240520
VERSION = 1

# -----------------------------------------------------------------------------
# Tokenization setup using tiktoken (GPT-2 encoding)
# -----------------------------------------------------------------------------
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode_ordinary(s)
EOT_TOKEN = enc._special_tokens['<|endoftext|>']

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def ensure_dir_exists(dir_path):
    """Ensure that the directory exists."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    return dir_path

def write_tokenized_data(tokens, output_path, file_idx=0, total_files=1):
    """Write tokens to a binary file in TinyStories format with header."""
    # Create header: 256 int32 values
    header = np.zeros(256, dtype=np.int32)
    header[0] = MAGIC_NUMBER    # Magic number
    header[1] = VERSION         # Version
    header[2] = len(tokens)     # Total token count
    header[3] = total_files     # Total files in this split
    header[4] = file_idx + 1    # Current file number (1-indexed)

    print(f"Writing {len(tokens):,} tokens to {output_path}")
    with open(output_path, 'wb') as f:
        # Write header as bytes
        f.write(header.tobytes())
        # Convert tokens to uint16 and write them
        tokens_np = np.array(tokens, dtype=np.uint16)
        tokens_np.tofile(f)
    
    print(f"Successfully wrote tokenized data to {output_path}")

def concatenate_examples(tokenized_examples):
    """
    Concatenate tokenized examples into one continuous sequence.
    
    Args:
        tokenized_examples: List of tokenized examples, each ending with an EOT_TOKEN
        
    Returns:
        List of tokens with examples concatenated
    """
    # Flatten the list of examples into a single list of tokens
    return [token for example in tokenized_examples for token in example]

# -----------------------------------------------------------------------------
# Main processing function
# -----------------------------------------------------------------------------
def main(args):
    print("Tokenizing C4_200M dataset with continuous examples...")

    ensure_dir_exists(args.output_dir)
    
    # Load the C4_200M dataset in streaming mode
    print("Loading C4_200M dataset in streaming mode...")
    ds = load_dataset("liweili/c4_200m", streaming=True)
    ds_train = ds["train"]

    # Total number of examples to process
    max_examples = args.sample_limit
    
    # Batch processing setup
    max_tokens_per_file = args.max_tokens_per_file
    
    # Track examples and files
    processed_examples = 0
    train_examples_count = 0
    val_examples_count = 0
    train_file_idx = 0
    val_file_idx = 0
    
    # Print a few examples in the desired formatted style before tokenization
    print("\nPrinting a few formatted examples:\n")
    print_count = min(3, max_examples)
    
    # Process examples from the dataset
    examples = ds_train
    
    # Split ratio (80% train, 20% val)
    train_ratio = 0.8
    train_limit = int(max_examples * train_ratio)
    
    # Initialize batch storage
    train_batch = []
    val_batch = []
    
    with tqdm(total=max_examples, desc="Processing examples") as pbar:
        for i, item in enumerate(examples):
            # Each item has 'input' and 'output'
            incorrect = item["input"].strip()
            corrected = item["output"].strip()
            
            # Format using the instruction template
            formatted_text = f"{INSTRUCTION_PREFIX}{incorrect}{INSTRUCTION_SEPARATOR}{corrected}{TASK_COMPLETE}"
            
            # Tokenize the formatted text
            tokens = encode(formatted_text)
            # Append the end-of-text token
            tokens.append(EOT_TOKEN)
            
            # Print a few example formats with token information
            if processed_examples < print_count:
                print(f"Example {processed_examples+1}:\n{formatted_text}\n")
                print(f"Token count: {len(tokens)}")
                print(f"First 10 tokens: {tokens[:10]}")
                print(f"Last 10 tokens: {tokens[-10:] if len(tokens) >= 10 else tokens}")
                
                # Optionally display token-to-text mapping for better understanding
                print("\nSample token-to-text mapping:")
                sample_size = min(5, len(tokens))
                for i in range(sample_size):
                    token_id = tokens[i]
                    token_bytes = enc.decode_single_token_bytes(token_id)
                    token_text = token_bytes.decode('utf-8', errors='replace')
                    print(f"Token {i}: ID={token_id}, Text='{token_text}'")
                
                print(f"{'-'*50}\n")
            
            # Determine if this example goes to training or validation
            # 80% for training, 20% for validation
            if processed_examples < train_limit:
                train_batch.append(tokens)
                train_examples_count += 1
                
                # Check if train batch is large enough to write
                train_token_count = sum(len(ex) for ex in train_batch)
                if train_token_count >= max_tokens_per_file:
                    print(f"\nProcessing train batch {train_file_idx + 1}...")
                    train_tokens = concatenate_examples(train_batch)
                    
                    # Write training batch to file
                    train_output_path = os.path.join(
                        args.output_dir, 
                        f"train_tokenized_output_{train_file_idx+1:03d}.bin"
                    )
                    
                    # We don't know total files yet, but we'll update this in post-processing
                    write_tokenized_data(train_tokens, train_output_path, 
                                        file_idx=train_file_idx, total_files=1)
                    
                    # Clear batch to free memory
                    train_batch = []
                    train_file_idx += 1
            else:
                val_batch.append(tokens)
                val_examples_count += 1
                
                # Check if val batch is large enough to write
                val_token_count = sum(len(ex) for ex in val_batch)
                if val_token_count >= max_tokens_per_file:
                    print(f"\nProcessing validation batch {val_file_idx + 1}...")
                    val_tokens = concatenate_examples(val_batch)
                    
                    # Write validation batch to file
                    val_output_path = os.path.join(
                        args.output_dir, 
                        f"val_tokenized_output_{val_file_idx+1:03d}.bin"
                    )
                    
                    # We don't know total files yet, but we'll update this in post-processing
                    write_tokenized_data(val_tokens, val_output_path, 
                                       file_idx=val_file_idx, total_files=1)
                    
                    # Clear batch to free memory
                    val_batch = []
                    val_file_idx += 1
            
            processed_examples += 1
            pbar.update(1)
            
            # Stop once we have enough examples
            if processed_examples >= max_examples:
                break
    
    # Process any remaining examples in the batches
    if train_batch:
        print(f"\nProcessing final train batch...")
        train_tokens = concatenate_examples(train_batch)
        train_output_path = os.path.join(
            args.output_dir, 
            f"train_tokenized_output_{train_file_idx+1:03d}.bin"
        )
        write_tokenized_data(train_tokens, train_output_path, 
                           file_idx=train_file_idx, total_files=train_file_idx+2)
        train_file_idx += 1
    
    if val_batch:
        print(f"\nProcessing final validation batch...")
        val_tokens = concatenate_examples(val_batch)
        val_output_path = os.path.join(
            args.output_dir, 
            f"val_tokenized_output_{val_file_idx+1:03d}.bin"
        )
        write_tokenized_data(val_tokens, val_output_path, 
                           file_idx=val_file_idx, total_files=val_file_idx+2)
        val_file_idx += 1
    
    # Update file headers with correct total file count
    print("\nUpdating file headers with correct total counts...")
    
    # Update train file headers
    for i in range(train_file_idx):
        train_path = os.path.join(args.output_dir, f"train_tokenized_output_{i+1:03d}.bin")
        with open(train_path, 'rb+') as f:
            header = np.fromfile(f, dtype=np.int32, count=256)
            header[3] = train_file_idx  # Update total files
            f.seek(0)
            header.tofile(f)
    
    # Update validation file headers
    for i in range(val_file_idx):
        val_path = os.path.join(args.output_dir, f"val_tokenized_output_{i+1:03d}.bin")
        with open(val_path, 'rb+') as f:
            header = np.fromfile(f, dtype=np.int32, count=256)
            header[3] = val_file_idx  # Update total files
            f.seek(0)
            header.tofile(f)
    
    print("\nTokenization complete!")
    print(f"Training examples processed: {train_examples_count}")
    print(f"Training files created: {train_file_idx}")
    print(f"Validation examples processed: {val_examples_count}")
    print(f"Validation files created: {val_file_idx}")

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tokenize part of the C4_200M dataset for grammar correction with continuous examples."
    )
    parser.add_argument("--output_dir", type=str, default="c4_tokenized", 
                        help="Output directory for tokenized files")
    parser.add_argument("--sample_limit", type=int, default=100, 
                        help="Total number of examples to process from the dataset")
    parser.add_argument("--max_tokens_per_file", type=int, default=100000000, 
                        help="Maximum number of tokens per output file (default: 100M tokens)")
    parser.add_argument("--batch_size", type=int, default=10000,
                        help="Number of examples to process before checking for file writing")
    
    args = parser.parse_args()
    main(args)