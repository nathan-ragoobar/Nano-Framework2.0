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
- Packs multiple examples together up to 256 tokens per sequence
- Skips examples exceeding 256 tokens
- Uses EOT_TOKEN as PAD token for any remaining space
- Outputs each split in the TinyStories binary format
"""

import os
import argparse
import numpy as np
import tiktoken
from tqdm import tqdm
import itertools
from datasets import load_dataset

# -----------------------------------------------------------------------------
# Constants and configuration
# -----------------------------------------------------------------------------
INSTRUCTION_PREFIX = "### Instruction: Correct the grammar in this text\n### Input: "
INSTRUCTION_SEPARATOR = "\n### Output: "
TASK_COMPLETE = "\n### End"
MAGIC_NUMBER = 20240520
VERSION = 1
FIXED_SEQ_LEN = 256

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

def write_tokenized_data(tokens, output_path):
    """Write tokens to a binary file in TinyStories format with header."""
    # Create header: 256 int32 values
    header = np.zeros(256, dtype=np.int32)
    header[0] = MAGIC_NUMBER    # Magic number
    header[1] = VERSION         # Version
    header[2] = len(tokens)     # Total token count

    print(f"Writing {len(tokens):,} tokens to {output_path}")
    with open(output_path, 'wb') as f:
        # Write header as bytes
        f.write(header.tobytes())
        # Convert tokens to uint16 and write them
        tokens_np = np.array(tokens, dtype=np.uint16)
        tokens_np.tofile(f)
    
    print(f"Successfully wrote tokenized data to {output_path}")

def pack_examples(tokenized_examples, fixed_len=FIXED_SEQ_LEN):
    """
    Pack multiple tokenized examples together to make efficient use of the fixed sequence length.
    
    Args:
        tokenized_examples: List of tokenized examples, each ending with an EOT_TOKEN
        fixed_len: The fixed sequence length (default: 256)
        
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

# -----------------------------------------------------------------------------
# Main processing function
# -----------------------------------------------------------------------------
def main(args):
    ensure_dir_exists(args.output_dir)
    
    # Load the C4_200M dataset in streaming mode
    print("Loading C4_200M dataset in streaming mode...")
    ds = load_dataset("liweili/c4_200m", streaming=True)
    ds_train = ds["train"]

    # Total number of examples to process
    max_examples = args.sample_limit
    
    train_examples = []  # Store tokenized training examples
    val_examples = []    # Store tokenized validation examples
    
    # Track examples
    processed_examples = 0
    skipped_examples = 0
    
    # Print a few examples in the desired formatted style before tokenization
    print("\nPrinting a few formatted examples:\n")
    print_count = min(3, max_examples)
    
    # Process examples from the dataset
    examples = ds_train
    
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
            
            # Check if the tokenized example is too long
            if len(tokens) > FIXED_SEQ_LEN:
                skipped_examples += 1
                continue  # Skip this example
            
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
            if processed_examples < int(max_examples * 0.8):
                train_examples.append(tokens)
            else:
                val_examples.append(tokens)
            
            processed_examples += 1
            pbar.update(1)
            
            # Stop once we have enough examples
            if processed_examples >= max_examples:
                break
    
    # Pack tokenized examples together to maximize space usage
    print("\nPacking examples...")
    train_tokens = pack_examples(train_examples)
    val_tokens = pack_examples(val_examples)
    
    # Calculate actual sequence counts
    train_sequences = len(train_tokens) // FIXED_SEQ_LEN
    val_sequences = len(val_tokens) // FIXED_SEQ_LEN
    
    # Write tokenized data to binary files for each split
    train_output_path = os.path.join(args.output_dir, "train_tokenized_output.bin")
    val_output_path = os.path.join(args.output_dir, "val_tokenized_output.bin")
    
    write_tokenized_data(train_tokens, train_output_path)
    write_tokenized_data(val_tokens, val_output_path)
    
    print("\nTokenization complete!")
    print(f"Training examples processed: {len(train_examples)}")
    print(f"Training sequences after packing: {train_sequences} ({len(train_tokens)} tokens)")
    print(f"Validation examples processed: {len(val_examples)}")
    print(f"Validation sequences after packing: {val_sequences} ({len(val_tokens)} tokens)")
    print(f"Skipped examples (exceeded max length): {skipped_examples}")
    print(f"Packing efficiency: {(len(train_examples) + len(val_examples)) / (train_sequences + val_sequences):.2f} examples per sequence")

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tokenize part of the C4_200M dataset for grammar correction with fixed-length examples."
    )
    parser.add_argument("--output_dir", type=str, default="c4_tokenized", help="Output directory for tokenized files")
    parser.add_argument("--sample_limit", type=int, default=100, help="Total number of examples to process from the dataset")
    
    args = parser.parse_args()
    main(args)