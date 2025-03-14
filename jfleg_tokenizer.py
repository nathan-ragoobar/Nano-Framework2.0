"""
Downloads, processes, and tokenizes the JFLEG dataset for grammar correction.
- Downloads data from HuggingFace datasets
- Adds special tokens for instruction format
- Tokenizes using GPT-2 tokenizer with tiktoken
- Outputs in the same binary format as TinyStories

The output is written to a newly created jfleg/ folder.
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
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "jfleg")
INSTRUCTION_PREFIX = "### Instruction: Correct the grammar in this text\n### Input: "
INSTRUCTION_SEPARATOR = "\n### Output: "
TASK_COMPLETE = "\n### End"

# -----------------------------------------------------------------------------
# Tokenization setup
# -----------------------------------------------------------------------------
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode_ordinary(s)
EOT_TOKEN = enc._special_tokens['<|endoftext|>']

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def ensure_dir_exists(dir_path):
    """Make sure the directory exists"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    return dir_path

def process_split(dataset, name, sample_limit=None):
    """Process a dataset split and convert to tokens with instruction format."""
    print(f"\nProcessing {name} split...")
    
    # Statistics tracking
    total_examples = 0
    total_tokens = 0
    skipped_examples = 0
    all_tokens = []
    
    # Limit dataset size if requested
    if sample_limit is not None:
        dataset = dataset.select(range(min(sample_limit, len(dataset))))
    
    # Use tqdm for progress bar
    for item in tqdm(dataset, desc=f"Processing {name}"):
        sentence = item['sentence'].strip()
        corrections = item['corrections']
        
        # Skip sentences without corrections
        if not corrections:
            skipped_examples += 1
            continue
            
        # For each correction, create a separate example
        for correction in corrections:
            correction = correction.strip()
            
            # Skip if correction is identical to original (no grammar change)
            if sentence == correction:
                continue
                
            # Format with instruction structure
            formatted_text = f"{INSTRUCTION_PREFIX}{sentence}{INSTRUCTION_SEPARATOR}{correction}{TASK_COMPLETE}"
            
            # Tokenize the formatted text
            tokens = encode(formatted_text)
            
            # Add EOT token
            tokens.append(EOT_TOKEN)
            
            # Add to token list
            all_tokens.extend(tokens)
            total_tokens += len(tokens)
            total_examples += 1
    
    print(f"Processed {total_examples} examples with {total_tokens} tokens")
    print(f"Skipped {skipped_examples} examples")
    
    return all_tokens, total_examples, total_tokens

def write_tokenized_data(tokens, output_path):
    """Write tokens to binary file in TinyStories format."""
    # Create header
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520  # magic number (same as TinyStories)
    header[1] = 1         # version
    header[2] = len(tokens)  # total token count
    
    print(f"Writing {len(tokens):,} tokens to {output_path}")
    with open(output_path, 'wb') as f:
        # Write header
        f.write(header.tobytes())
        
        # Write tokens
        tokens_np = np.array(tokens, dtype=np.uint16)
        tokens_np.tofile(f)
    
    print(f"Successfully wrote tokenized data to {output_path}")
    return len(tokens)

def create_tokenized_jfleg(output_dir=None, train_fraction=0.8, sample_limit=None):
    """Main function to download, process, and tokenize JFLEG dataset."""
    if output_dir is None:
        output_dir = DATA_CACHE_DIR
    
    ensure_dir_exists(output_dir)
    
    print("Loading JFLEG datasets...")
    
    # Load validation set and split into train/val
    val_ds = load_dataset("jhu-clsp/jfleg", split="validation")
    test_ds = load_dataset("jhu-clsp/jfleg", split="test")
    
    # Determine split point for validation set
    total_examples = len(val_ds)
    train_size = int(total_examples * train_fraction)
    
    # Shuffle the dataset
    val_ds = val_ds.shuffle(seed=42)
    
    # Split the dataset
    train_ds = val_ds.select(range(train_size))
    val_ds_subset = val_ds.select(range(train_size, total_examples))
    
    print(f"Created splits: train ({len(train_ds)} examples), validation ({len(val_ds_subset)} examples)")
    
    # Process and tokenize each split
    train_tokens, train_examples, train_token_count = process_split(train_ds, "train", sample_limit)
    val_tokens, val_examples, val_token_count = process_split(val_ds_subset, "validation", sample_limit)
    test_tokens, test_examples, test_token_count = process_split(test_ds, "test", sample_limit)
    
    # Write tokenized data
    train_path = os.path.join(output_dir, "jfleg_train.bin")
    val_path = os.path.join(output_dir, "jfleg_val.bin")
    test_path = os.path.join(output_dir, "jfleg_test.bin")
    
    write_tokenized_data(train_tokens, train_path)
    write_tokenized_data(val_tokens, val_path)
    write_tokenized_data(test_tokens, test_path)
    
    # Print summary
    print("\n=== JFLEG Tokenization Complete ===")
    print(f"Train: {train_examples} examples, {train_token_count} tokens")
    print(f"Validation: {val_examples} examples, {val_token_count} tokens")
    print(f"Test: {test_examples} examples, {test_token_count} tokens")
    print(f"Total: {train_examples + val_examples + test_examples} examples, {train_token_count + val_token_count + test_token_count} tokens")
    
    return train_path, val_path, test_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JFLEG dataset to tokenized binary format")
    parser.add_argument("--output_dir", default=DATA_CACHE_DIR, help="Output directory")
    parser.add_argument("--train_fraction", type=float, default=0.8, 
                       help="Fraction of validation data to use for training")
    parser.add_argument("--sample_limit", type=int, default=None, 
                       help="Maximum number of examples to include per split")
    args = parser.parse_args()
    
    create_tokenized_jfleg(args.output_dir, args.train_fraction, args.sample_limit)



    '''
    To run this script, use the following command:
    python jfleg_tokenizer.py --output_dir jfleg --train_fraction 0.8 --sample_limit 1000
    '''