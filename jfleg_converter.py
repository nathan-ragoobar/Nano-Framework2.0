from datasets import load_dataset
import random
import argparse
import os
import tqdm

def process_split(dataset, output_dir, name, sample_limit=None):
    """Process a dataset split and convert to line-by-line format."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, f"jfleg_{name}.txt")
    
    # Statistics tracking
    total_examples = 0
    skipped_examples = 0
    
    print(f"Converting {len(dataset) if sample_limit is None else min(sample_limit, len(dataset))} examples for {name} split...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # Use tqdm for progress bar
        for item in tqdm.tqdm(dataset, total=sample_limit if sample_limit else len(dataset)):
            if sample_limit is not None and total_examples >= sample_limit:
                break
                
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
                    
                # Write as pairs of lines: incorrect, then correct
                f.write(f"{sentence}\n")
                f.write(f"{correction}\n")
                total_examples += 1
    
    print(f"Conversion complete for {name}!")
    print(f"Total examples written: {total_examples}")
    print(f"Skipped examples: {skipped_examples}")
    print(f"Output saved to: {output_path}")
    
    return output_path

def convert_jfleg_with_splits(output_dir, train_fraction=0.8, sample_limit=None):
    """Create train/val splits from the validation data."""
    print("Loading JFLEG validation dataset...")
    ds = load_dataset("jhu-clsp/jfleg", split="validation")
    
    # Determine split point
    total_examples = len(ds)
    train_size = int(total_examples * train_fraction)
    
    # Shuffle the dataset
    ds = ds.shuffle(seed=42)
    
    # Split the dataset
    train_ds = ds.select(range(train_size))
    val_ds = ds.select(range(train_size, total_examples))
    
    print(f"Created splits: train ({len(train_ds)} examples), validation ({len(val_ds)} examples)")
    
    # Process each split
    train_path = process_split(train_ds, output_dir, "train", sample_limit)
    val_path = process_split(val_ds, output_dir, "val", sample_limit)
    
    # Also process the test set normally
    print("Loading JFLEG test dataset...")
    test_ds = load_dataset("jhu-clsp/jfleg", split="test")
    test_path = process_split(test_ds, output_dir, "test", sample_limit)
    
    return train_path, val_path, test_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JFLEG dataset to line-by-line format")
    parser.add_argument("--output_dir", default="./data", help="Output directory")
    parser.add_argument("--train_fraction", type=float, default=0.8, 
                       help="Fraction of validation data to use for training")
    parser.add_argument("--sample_limit", type=int, default=None, 
                       help="Maximum number of examples to include per split")
    args = parser.parse_args()
    
    convert_jfleg_with_splits(args.output_dir, args.train_fraction, args.sample_limit)