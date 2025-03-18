#!/usr/bin/env python
"""
Downloads, processes, and tokenizes part of the C4_200M dataset for grammar correction.
- First downloads the entire dataset portion
- Uses multiprocessing for efficient parallel tokenization
- Formats examples into the instruction style:
    ### Instruction: Correct the grammar in this text
    ### Input: [incorrect sentence]
    ### Output: [corrected sentence]
    ### End
- Tokenizes using tiktoken with GPT‑2 encoding
- Splits the data into training (80%) and validation (20%) splits
- Places examples continuously one after another, separated by EOT_TOKEN
- Outputs each split in the TinyStories binary format with fixed file sizes
"""

import os
import argparse
import numpy as np
import tiktoken
from tqdm import tqdm
from datasets import load_dataset
import multiprocessing as mp
from functools import partial
import time
import math

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
    """Concatenate tokenized examples into one continuous sequence."""
    return [token for example in tokenized_examples for token in example]

def process_example(item):
    """Process a single example (format text and tokenize)."""
    incorrect = item["input"].strip()
    corrected = item["output"].strip()
    
    # Format using the instruction template
    formatted_text = f"{INSTRUCTION_PREFIX}{incorrect}{INSTRUCTION_SEPARATOR}{corrected}{TASK_COMPLETE}"
    
    # Tokenize the formatted text
    tokens = encode(formatted_text)
    # Append the end-of-text token
    tokens.append(EOT_TOKEN)
    
    return tokens

def chunk_list(input_list, chunk_size):
    """Split a list into chunks of specified size."""
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]

# -----------------------------------------------------------------------------
# Main processing function
# -----------------------------------------------------------------------------
def main(args):
    print("Starting C4_200M dataset processing for grammar correction...")
    start_time = time.time()

    ensure_dir_exists(args.output_dir)
    
    # Step 1: Download the dataset
    print(f"Downloading {args.sample_limit} examples from the C4_200M dataset...")
    ds = load_dataset("liweili/c4_200m", split=f"train[:{args.sample_limit}]")
    print(f"Downloaded {len(ds)} examples")
    
    # Print a few examples for verification
    print_count = min(3, len(ds))
    for i in range(print_count):
        print(f"\nExample {i+1}:")
        print(f"Input: {ds[i]['input']}")
        print(f"Output: {ds[i]['output']}")
    
    # Step 2: Split data into train/val
    train_ratio = 0.8
    train_size = int(len(ds) * train_ratio)
    
    train_ds = ds[:train_size]
    val_ds = ds[train_size:]
    
    print(f"\nSplit dataset into {len(train_ds)} training and {len(val_ds)} validation examples")
    
    # Step 3: Process training data with multiprocessing
    print("\nProcessing training data in parallel...")
    num_processes = min(mp.cpu_count(), args.num_processes)
    print(f"Using {num_processes} processes for tokenization")
    
    # Process in parallel
    with mp.Pool(processes=num_processes) as pool:
        all_train_tokens = list(tqdm(
            pool.imap(process_example, train_ds, chunksize=100),
            total=len(train_ds),
            desc="Tokenizing training examples"
        ))
    
    # Step 4: Process validation data with multiprocessing
    print("\nProcessing validation data in parallel...")
    with mp.Pool(processes=num_processes) as pool:
        all_val_tokens = list(tqdm(
            pool.imap(process_example, val_ds, chunksize=100),
            total=len(val_ds),
            desc="Tokenizing validation examples"
        ))
    
    # Step 5: Calculate token counts and file divisions
    train_token_count = sum(len(tokens) for tokens in all_train_tokens)
    val_token_count = sum(len(tokens) for tokens in all_val_tokens)
    
    print(f"\nTotal train tokens: {train_token_count:,}")
    print(f"Total validation tokens: {val_token_count:,}")
    
    max_tokens_per_file = args.max_tokens_per_file
    
    # Calculate number of files needed
    train_files_needed = max(1, math.ceil(train_token_count / max_tokens_per_file))
    val_files_needed = max(1, math.ceil(val_token_count / max_tokens_per_file))
    
    print(f"Will create {train_files_needed} training files and {val_files_needed} validation files")
    
    # Step 6: Organize examples into evenly-sized files
    # Distribute examples to get approximately equal sized files
    train_examples_per_file = math.ceil(len(all_train_tokens) / train_files_needed)
    val_examples_per_file = math.ceil(len(all_val_tokens) / val_files_needed)
    
    # Split into chunks
    train_batches = chunk_list(all_train_tokens, train_examples_per_file)
    val_batches = chunk_list(all_val_tokens, val_examples_per_file)
    
    print(f"\nDistributed training examples into {len(train_batches)} batches")
    print(f"Distributed validation examples into {len(val_batches)} batches")
    
    # Step 7: Write training files
    print("\nWriting training files...")
    for i, batch in enumerate(train_batches):
        tokens = concatenate_examples(batch)
        output_path = os.path.join(args.output_dir, f"train_tokenized_output_{i+1:03d}.bin")
        write_tokenized_data(tokens, output_path, file_idx=i, total_files=len(train_batches))
    
    # Step 8: Write validation files
    print("\nWriting validation files...")
    for i, batch in enumerate(val_batches):
        tokens = concatenate_examples(batch)
        output_path = os.path.join(args.output_dir, f"val_tokenized_output_{i+1:03d}.bin")
        write_tokenized_data(tokens, output_path, file_idx=i, total_files=len(val_batches))
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\nTokenization complete!")
    print(f"Training examples processed: {len(train_ds)}")
    print(f"Training files created: {len(train_batches)}")
    print(f"Validation examples processed: {len(val_ds)}")
    print(f"Validation files created: {len(val_batches)}")
    print(f"Total processing time: {elapsed_time:.2f} seconds")

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tokenize part of the C4_200M dataset for grammar correction with continuous examples."
    )
    parser.add_argument("--output_dir", type=str, default="c4_tokenizedv2", 
                        help="Output directory for tokenized files")
    parser.add_argument("--sample_limit", type=int, default=100, 
                        help="Total number of examples to process from the dataset")
    parser.add_argument("--max_tokens_per_file", type=int, default=100000000, 
                        help="Maximum number of tokens per output file (default: 100M tokens)")
    parser.add_argument("--num_processes", type=int, default=mp.cpu_count(),
                        help=f"Number of processes to use (default: all available cores, {mp.cpu_count()})")
    
    args = parser.parse_args()
    main(args)