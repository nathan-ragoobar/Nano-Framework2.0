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
- Packs multiple examples together up to specified sequence length
- Skips examples exceeding max sequence length
- Uses EOT_TOKEN as PAD token for any remaining space
- Outputs each split in the TinyStories binary format
- Processes data in batches to avoid memory issues
- Uses multiprocessing for parallel processing of batches
"""

import os
import argparse
import numpy as np
import tiktoken
from tqdm import tqdm
import itertools
from datasets import load_dataset
import multiprocessing as mp
from functools import partial
import time
import sys

# -----------------------------------------------------------------------------
# Constants and configuration
# -----------------------------------------------------------------------------
INSTRUCTION_PREFIX = "### Instruction: Correct the grammar in this text\n### Input: "
INSTRUCTION_SEPARATOR = "\n### Output: "
TASK_COMPLETE = "\n### End"
MAGIC_NUMBER = 20240520
VERSION = 1
DEFAULT_SEQ_LEN = 256

# -----------------------------------------------------------------------------
# Tokenization setup using tiktoken (GPT-2 encoding)
# -----------------------------------------------------------------------------
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode_ordinary(s)
EOT_TOKEN = enc._special_tokens['<|endoftext|>']

# We're now using EOT_TOKEN as our PAD token as well
PAD_TOKEN = EOT_TOKEN

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

def pack_examples(tokenized_examples, fixed_len):
    """
    Pack multiple tokenized examples together to make efficient use of the fixed sequence length.
    
    Args:
        tokenized_examples: List of tokenized examples, each ending with an EOT_TOKEN
        fixed_len: The fixed sequence length
        
    Returns:
        List of packed token sequences, each of exactly fixed_len tokens
    """
    packed_sequences = []
    current_sequence = []
    
    for example in tokenized_examples:
        # If adding this example would exceed the fixed length, finalize current sequence
        if len(current_sequence) + len(example) > fixed_len:
            # Pad the current sequence to fixed_len
            if len(current_sequence) < fixed_len:
                current_sequence.extend([PAD_TOKEN] * (fixed_len - len(current_sequence)))
            
            packed_sequences.append(current_sequence)
            current_sequence = []  # Start a new sequence
        
        # If this single example is too large, skip it (should be caught earlier)
        if len(example) > fixed_len:
            continue
            
        # Add the example to the current sequence
        current_sequence.extend(example)
    
    # Handle the final sequence if it's not empty
    if current_sequence:
        if len(current_sequence) < fixed_len:
            current_sequence.extend([PAD_TOKEN] * (fixed_len - len(current_sequence)))
        packed_sequences.append(current_sequence)
    
    # Flatten the list of sequences into a single list of tokens
    return [token for sequence in packed_sequences for token in sequence]

def process_batch(batch_items, seq_len, is_training=True, show_examples=False, output_dir=None, file_idx=0):
    """Process a batch of examples in parallel."""
    tokenized_examples = []
    skipped = 0
    
    # Process each item in the batch
    for i, item in enumerate(batch_items):
        incorrect = item["input"].strip()
        corrected = item["output"].strip()
        
        # Format using the instruction template
        formatted_text = f"{INSTRUCTION_PREFIX}{incorrect}{INSTRUCTION_SEPARATOR}{corrected}{TASK_COMPLETE}"
        
        # Tokenize the formatted text
        tokens = encode(formatted_text)
        tokens.append(EOT_TOKEN)
        
        # Skip if too long
        if len(tokens) > seq_len:
            skipped += 1
            continue
        
        # Print sample if requested
        if show_examples and i < 3:
            print(f"\nExample {i+1}:")
            print(formatted_text)
            print(f"Token count: {len(tokens)}")
            print(f"First 10 tokens: {tokens[:10]}")
            print(f"Last 10 tokens: {tokens[-10:] if len(tokens) >= 10 else tokens}")
        
        tokenized_examples.append(tokens)
    
    # Pack examples if we have any
    if tokenized_examples:
        packed_tokens = pack_examples(tokenized_examples, seq_len)
        
        # Write to file if output_dir is provided
        if output_dir:
            split_name = "train" if is_training else "val"
            output_path = os.path.join(output_dir, f"{split_name}_tokenized_output_{file_idx+1:03d}.bin")
            write_tokenized_data(packed_tokens, output_path, file_idx, 1)  # Will update total_files later
            
        return {
            'tokens': packed_tokens if not output_dir else None,  # Only return tokens if not writing to file
            'count': len(tokenized_examples),
            'skipped': skipped,
            'file_idx': file_idx if output_dir else None
        }
    
    return {
        'tokens': [],
        'count': 0,
        'skipped': skipped,
        'file_idx': None
    }

# -----------------------------------------------------------------------------
# Main processing function
# -----------------------------------------------------------------------------
def main(args):
    start_time = time.time()
    seq_len = args.seq_len
    print(f"Tokenizing C4_200M dataset with fixed sequence length of {seq_len}...")
    print(f"Using {args.num_workers} parallel workers")

    ensure_dir_exists(args.output_dir)
    
    # Load the C4_200M dataset in streaming mode
    print("Loading C4_200M dataset in streaming mode...")
    ds = load_dataset("liweili/c4_200m", streaming=True)
    ds_train = ds["train"]

    # Total number of examples to process
    max_examples = args.sample_limit
    
    # Split ratio (80% train, 20% val)
    train_ratio = 0.8
    train_limit = int(max_examples * train_ratio)
    
    # Batch sizes for parallel processing
    batch_size = args.batch_size
    max_tokens_per_file = args.max_tokens_per_file
    
    # Setup multiprocessing pool
    pool = mp.Pool(processes=args.num_workers)
    
    # Print a few examples for reference
    print("\nPrinting a few formatted examples:")
    sample_batch = []
    for i, item in enumerate(ds_train):
        if i < 3:
            sample_batch.append(item)
        else:
            break
    
    # Process sample batch to show examples
    process_batch(sample_batch, seq_len, show_examples=True)
    
    # Initialize counters
    processed_examples = 0
    skipped_examples = 0
    train_examples_count = 0
    val_examples_count = 0
    train_file_idx = 0
    val_file_idx = 0
    
    # Setup batching for parallel processing
    train_batches = []
    val_batches = []
    current_train_batch = []
    current_val_batch = []
    
    print("\nCollecting batches for processing...")
    with tqdm(total=max_examples, desc="Collecting examples") as pbar:
        for i, item in enumerate(ds_train):
            # Determine if this goes to train or validation
            if processed_examples < train_limit:
                current_train_batch.append(item)
                
                # Check if we have a full batch
                if len(current_train_batch) >= batch_size:
                    train_batches.append(current_train_batch)
                    current_train_batch = []
            else:
                current_val_batch.append(item)
                
                # Check if we have a full batch
                if len(current_val_batch) >= batch_size:
                    val_batches.append(current_val_batch)
                    current_val_batch = []
            
            processed_examples += 1
            pbar.update(1)
            
            if processed_examples >= max_examples:
                break
    
    # Add any remaining items to batches
    if current_train_batch:
        train_batches.append(current_train_batch)
    
    if current_val_batch:
        val_batches.append(current_val_batch)
    
    # Process training batches in parallel
    print(f"\nProcessing {len(train_batches)} training batches in parallel...")
    train_results = []
    
    # Process each batch and write directly to files
    process_train_batch = partial(
        process_batch, 
        seq_len=seq_len, 
        is_training=True, 
        output_dir=args.output_dir
    )
    
    # Map batch index to each batch
    batch_with_idx = [(batch, idx) for idx, batch in enumerate(train_batches)]
    
    with tqdm(total=len(train_batches), desc="Processing train batches") as pbar:
        for i, (batch, idx) in enumerate(batch_with_idx):
            result = process_train_batch(batch, file_idx=idx)
            train_results.append(result)
            train_examples_count += result['count']
            skipped_examples += result['skipped']
            train_file_idx = max(train_file_idx, idx + 1)
            pbar.update(1)
    
    # Process validation batches in parallel
    print(f"\nProcessing {len(val_batches)} validation batches in parallel...")
    val_results = []
    
    process_val_batch = partial(
        process_batch, 
        seq_len=seq_len, 
        is_training=False, 
        output_dir=args.output_dir
    )
    
    batch_with_idx = [(batch, idx) for idx, batch in enumerate(val_batches)]
    
    with tqdm(total=len(val_batches), desc="Processing val batches") as pbar:
        for i, (batch, idx) in enumerate(batch_with_idx):
            result = process_val_batch(batch, file_idx=idx)
            val_results.append(result)
            val_examples_count += result['count']
            skipped_examples += result['skipped']
            val_file_idx = max(val_file_idx, idx + 1)
            pbar.update(1)
    
    # Close the pool
    pool.close()
    pool.join()
    
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
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    print("\nTokenization complete!")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Training examples processed: {train_examples_count}")
    print(f"Training files created: {train_file_idx}")
    print(f"Validation examples processed: {val_examples_count}")
    print(f"Validation files created: {val_file_idx}")
    print(f"Skipped examples (exceeded max length): {skipped_examples}")
    print(f"Processing speed: {processed_examples/elapsed_time:.2f} examples/second")

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tokenize part of the C4_200M dataset for grammar correction with fixed-length examples."
    )
    parser.add_argument("--output_dir", type=str, default="c4_tokenized", 
                        help="Output directory for tokenized files")
    parser.add_argument("--sample_limit", type=int, default=100, 
                        help="Total number of examples to process from the dataset")
    parser.add_argument("--max_tokens_per_file", type=int, default=100000000, 
                        help="Maximum number of tokens per output file (default: 100M tokens)")
    parser.add_argument("--batch_size", type=int, default=1000,
                        help="Number of examples per batch for parallel processing")
    parser.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN,
                        help=f"Sequence length for tokenized examples (default: {DEFAULT_SEQ_LEN})")
    parser.add_argument("--num_workers", type=int, default=mp.cpu_count(),
                        help=f"Number of parallel workers (default: {mp.cpu_count()})")
    
    args = parser.parse_args()
    main(args)