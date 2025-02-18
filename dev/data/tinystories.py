"""
Downloads and tokenizes the TinyStories dataset.
- The download is from HuggingFace datasets.
- The tokenization is GPT-2 tokenizer with tiktoken

The output is written to a newly created tinystories/ folder.
The script prints:

Tokenizing val split...
Saved 19043638 tokens to tinystories/TinyStories_val.bin
Tokenizing train split...
Saved 925653391 tokens to tinystories/TinyStories_train.bin

And runs in 1-2 minutes two depending on your internet
connection and computer. The .bin files are raw byte
streams of int32 numbers indicating the token ids.
"""

import os
import glob
import json
import random
import requests
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import tiktoken
import numpy as np
from data_common import download_file, write_datafile

# -----------------------------------------------------------------------------
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "tinystories")

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode_ordinary(s)
'''
def write_datafile(file_obj, tokens, append=False):
    """Write tokens to a binary file as uint16 values"""
    tokens = np.array(tokens, dtype=np.uint16)
    tokens.tofile(file_obj)
'''
def download():
    """Downloads the TinyStories dataset to DATA_CACHE_DIR"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # download the TinyStories dataset, unless it's already downloaded
    data_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    data_filename = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data.tar.gz")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")

    # unpack the tar.gz file into all the data shards (json files)
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Unpacking {data_filename}...")
        os.system(f"tar -xzf {data_filename} -C {data_dir}")
    else:
        print(f"{data_dir} already exists, skipping unpacking...")

    # print a single example just for debugging and such
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    print("Download done.")
    print(f"Number of shards: {len(shard_filenames)}")
    # with open(shard_filenames[0], "r") as f:
    #     data = json.load(f)
    # print(f"Example story:\n{data[0]}")

def process_shard(shard_index, shard_filename):
    eot = enc._special_tokens['<|endoftext|>']
    all_tokens = []
    stories_processed = 0
    
    print(f"Processing shard file: {shard_filename}")
    with open(shard_filename, "r") as f:
        try:
            # Load the entire JSON array at once
            stories = json.load(f)
            
            # Process each story in the array
            for story_json in stories:
                text = story_json["story"].strip()
                tokens = encode(text)
                all_tokens.extend(tokens)  # Add story tokens
                all_tokens.append(eot)     # Add EOT token
                stories_processed += 1
                
                if len(all_tokens) >= 1000000:
                    print(f"Yielding chunk of {len(all_tokens)} tokens from {stories_processed} stories")
                    yield all_tokens
                    all_tokens = []
                    stories_processed = 0
                    
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON in {shard_filename}: {e}")
            return
    
    if all_tokens:
        print(f"Yielding final chunk of {len(all_tokens)} tokens from {stories_processed} stories")
        yield all_tokens

def tokenize():
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if not shard_filenames:
        raise RuntimeError(f"No .json files found in {data_dir}")
    
    print(f"Found {len(shard_filenames)} shards")
    
    val_shards = [shard_filenames[0]]  # First shard for validation
    train_shards = shard_filenames[1:] # Rest for training
    
    for split_name, split_shards in [("val", val_shards), ("train", train_shards)]:
        print(f"\nProcessing {split_name} split...")
        split_filename = os.path.join(DATA_CACHE_DIR, f"TinyStories_{split_name}.bin")
        
        # First pass: count total tokens
        total_tokens = 0
        for shard_index, shard_filename in enumerate(split_shards):
            print(f"\nCounting tokens in shard {shard_index + 1}/{len(split_shards)}")
            for token_chunk in process_shard(shard_index, shard_filename):
                total_tokens += len(token_chunk)
        
        # Create and write header
        header = np.zeros(256, dtype=np.int32)
        header[0] = 20240520  # magic
        header[1] = 1         # version
        header[2] = total_tokens
        
        print(f"\nWriting {total_tokens:,} tokens to {split_filename}")
        with open(split_filename, 'wb') as f:
            # Write header first
            f.write(header.tobytes())
            
            # Second pass: write all tokens
            for shard_index, shard_filename in enumerate(split_shards):
                print(f"Writing tokens from shard {shard_index + 1}/{len(split_shards)}")
                for token_chunk in process_shard(shard_index, shard_filename):
                    tokens_np = np.array(token_chunk, dtype=np.uint16)
                    tokens_np.tofile(f)
        
        print(f"Finished {split_name} split: {total_tokens:,} tokens")


if __name__ == "__main__":
    try:
        download()
        tokenize()
    except Exception as e:
        print(f"Error: {e}")
        raise
    # Prints:
    # Tokenizing val split...
    # Saved 19043638 tokens to data/TinyStories_val.bin
    # Tokenizing train split...
    # Saved 925653391 tokens to data/TinyStories_train.bin
