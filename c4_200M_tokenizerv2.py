#!/usr/bin/env python
"""
Analysis script for C4_200M tokenizer that calculates average examples per sequence.
- Processes a sample of the C4_200M dataset
- Formats and tokenizes examples with the same instruction style
- Tracks how many examples fit into each fixed-length sequence
- Reports detailed packing efficiency statistics
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
import statistics

# -----------------------------------------------------------------------------
# Constants and configuration
# -----------------------------------------------------------------------------
INSTRUCTION_PREFIX = "### Instruction: Correct the grammar in this text\n### Input: "
INSTRUCTION_SEPARATOR = "\n### Output: "
TASK_COMPLETE = "\n### End"
DEFAULT_SEQ_LEN = 256

# -----------------------------------------------------------------------------
# Tokenization setup using tiktoken (GPT-2 encoding)
# -----------------------------------------------------------------------------
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode_ordinary(s)
EOT_TOKEN = enc._special_tokens['<|endoftext|>']
PAD_TOKEN = EOT_TOKEN

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
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

def pack_examples_with_stats(tokenized_examples, fixed_len):
    """
    Pack multiple tokenized examples together and track statistics.
    
    Args:
        tokenized_examples: List of tokenized examples, each ending with an EOT_TOKEN
        fixed_len: The fixed sequence length
        
    Returns:
        Dictionary with tokens, count of sequences, and examples per sequence
    """
    packed_sequences = []
    current_sequence = []
    examples_in_current_sequence = 0
    examples_per_sequence = []  # Track examples in each sequence
    tokens_used_per_sequence = []  # Track token utilization
    
    for example in tokenized_examples:
        # If adding this example would exceed the fixed length, finalize current sequence
        if len(current_sequence) + len(example) > fixed_len:
            # Calculate utilization before padding
            tokens_used = len(current_sequence)
            utilization_pct = (tokens_used / fixed_len) * 100
            tokens_used_per_sequence.append((tokens_used, utilization_pct))
            
            # Pad the current sequence to fixed_len
            if len(current_sequence) < fixed_len:
                current_sequence.extend([PAD_TOKEN] * (fixed_len - len(current_sequence)))
            
            packed_sequences.append(current_sequence)
            examples_per_sequence.append(examples_in_current_sequence)
            
            # Start a new sequence
            current_sequence = []
            examples_in_current_sequence = 0
        
        # If this single example is too large, skip it
        if len(example) > fixed_len:
            continue
            
        # Add the example to the current sequence
        current_sequence.extend(example)
        examples_in_current_sequence += 1
    
    # Handle the final sequence if it's not empty
    if current_sequence:
        # Calculate utilization before padding
        tokens_used = len(current_sequence)
        utilization_pct = (tokens_used / fixed_len) * 100
        tokens_used_per_sequence.append((tokens_used, utilization_pct))
        
        if len(current_sequence) < fixed_len:
            current_sequence.extend([PAD_TOKEN] * (fixed_len - len(current_sequence)))
        packed_sequences.append(current_sequence)
        examples_per_sequence.append(examples_in_current_sequence)
    
    # Flatten the list of sequences into a single list of tokens
    all_tokens = [token for sequence in packed_sequences for token in sequence]
    
    return {
        'tokens': all_tokens,
        'num_sequences': len(packed_sequences),
        'examples_per_sequence': examples_per_sequence,
        'tokens_used_per_sequence': tokens_used_per_sequence
    }

# -----------------------------------------------------------------------------
# Main analysis function
# -----------------------------------------------------------------------------
def main(args):
    start_time = time.time()
    seq_len = args.seq_len
    print(f"Analyzing packing efficiency for C4_200M dataset with sequence length {seq_len}...")

    # Step 1: Download the dataset sample
    print(f"Downloading {args.sample_limit} examples from the C4_200M dataset...")
    ds = load_dataset("liweili/c4_200m", split=f"train[:{args.sample_limit}]")
    print(f"Downloaded {len(ds)} examples")
    
    # Step 2: Process examples with multiprocessing
    print("\nProcessing examples in parallel...")
    num_processes = min(mp.cpu_count(), args.num_processes)
    print(f"Using {num_processes} processes for tokenization")
    
    with mp.Pool(processes=num_processes) as pool:
        all_tokens = list(tqdm(
            pool.imap(process_example, ds, chunksize=100),
            total=len(ds),
            desc="Tokenizing examples"
        ))
    
    # Step 3: Calculate token statistics
    token_lengths = [len(tokens) for tokens in all_tokens]
    
    print(f"\nToken statistics from {len(token_lengths)} examples:")
    print(f"  Min tokens per example: {min(token_lengths)}")
    print(f"  Max tokens per example: {max(token_lengths)}")
    print(f"  Average tokens per example: {sum(token_lengths) / len(token_lengths):.2f}")
    print(f"  Median tokens per example: {statistics.median(token_lengths)}")
    
    # Count examples that would be skipped
    skipped = sum(1 for length in token_lengths if length > seq_len)
    print(f"  Examples exceeding seq_len ({seq_len}): {skipped} ({skipped/len(token_lengths)*100:.2f}%)")
    
    # Step 4: Pack examples and calculate packing efficiency
    print("\nAnalyzing packing efficiency...")
    # Filter out examples that are too long
    valid_examples = [tokens for tokens in all_tokens if len(tokens) <= seq_len]
    
    packing_result = pack_examples_with_stats(valid_examples, seq_len)
    
    examples_per_seq = packing_result['examples_per_sequence']
    tokens_used = packing_result['tokens_used_per_sequence']
    
    # Calculate statistics
    avg_examples_per_seq = sum(examples_per_seq) / len(examples_per_seq) if examples_per_seq else 0
    min_examples_per_seq = min(examples_per_seq) if examples_per_seq else 0
    max_examples_per_seq = max(examples_per_seq) if examples_per_seq else 0
    
    # Token utilization
    avg_tokens_used = sum(t[0] for t in tokens_used) / len(tokens_used) if tokens_used else 0
    avg_utilization = sum(t[1] for t in tokens_used) / len(tokens_used) if tokens_used else 0
    
    print("\nPacking efficiency results:")
    print(f"  Total sequences created: {packing_result['num_sequences']}")
    print(f"  Valid examples packed: {len(valid_examples)}")
    print(f"  Average examples per sequence: {avg_examples_per_seq:.2f}")
    print(f"  Min examples in a sequence: {min_examples_per_seq}")
    print(f"  Max examples in a sequence: {max_examples_per_seq}")
    print(f"  Average tokens used per sequence: {avg_tokens_used:.2f}/{seq_len} ({avg_utilization:.2f}%)")
    
    # Sequence distribution histogram
    counts = {}
    for count in examples_per_seq:
        counts[count] = counts.get(count, 0) + 1
    
    print("\nDistribution of examples per sequence:")
    for count in sorted(counts.keys()):
        percentage = (counts[count] / len(examples_per_seq)) * 100
        print(f"  {count} examples: {counts[count]} sequences ({percentage:.2f}%)")
    
    elapsed_time = time.time() - start_time
    print(f"\nAnalysis completed in {elapsed_time:.2f} seconds")

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze packing efficiency for C4_200M dataset tokenization."
    )
    parser.add_argument("--sample_limit", type=int, default=1000, 
                        help="Number of examples to analyze from the dataset")
    parser.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN,
                        help=f"Sequence length to use for analysis (default: {DEFAULT_SEQ_LEN})")
    parser.add_argument("--num_processes", type=int, default=mp.cpu_count(),
                        help=f"Number of processes to use for tokenization (default: {mp.cpu_count()})")
    
    args = parser.parse_args()
    main(args)