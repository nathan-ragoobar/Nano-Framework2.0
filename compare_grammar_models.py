import subprocess
import os

# Configuration
INFERENCE_EXECUTABLE = "./inference_gpt2_cpu"
MODEL_PATHS = [
    "./jfleg_attempt1.bin",
    "./jfleg_attempt2.bin",
    "./jfleg_attempt3.bin",
]

# Test sentences with grammatical errors
TEST_SENTENCES = [
    "For example they can play football whenever they want but the olders can not .",
    "For not use car .",
    "And young people spend time more ther lifestile .",
    "But YOU have to create these opportunities .",
]

# Run inference on a model with the given prompt
def run_inference(model_path, prompt):
    formatted_prompt = f"### Instruction: Correct the grammar in this text\n### Input: {prompt}\n"
    
    try:
        print(f"Running inference with {os.path.basename(model_path)} on: {prompt[:30]}...")
        
        result = subprocess.run(
            [INFERENCE_EXECUTABLE, "--model", model_path, "--genlen", "100"],
            input=formatted_prompt,
            text=True,
            capture_output=True,
            check=False
        )
        
        if result.returncode != 0:
            print(f"Error: Process returned code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return None
        
        # Extract the generated text from output
        output = result.stdout
        
        # Find the generated text between "---" markers
        start_marker = "---"
        end_marker = "---"
        
        start_idx = output.find(start_marker)
        if start_idx >= 0:
            # Start after the marker
            start_idx += len(start_marker)
            end_idx = output.rfind(end_marker)
            if end_idx >= 0:
                generated_text = output[start_idx:end_idx].strip()
                return generated_text
        
        return "ERROR: Could not parse output"
        
    except Exception as e:
        print(f"Exception during inference: {e}")
        return None

# Main processing loop
def compare_models():
    # Create output directory
    output_dir = "grammar_corrections"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each model
    for model_path in MODEL_PATHS:
        model_name = os.path.basename(model_path).replace('.bin', '')
        output_file = os.path.join(output_dir, f"{model_name}_results.txt")
        
        print(f"\nProcessing model: {model_name}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Grammar correction results for {model_name}\n")
            f.write("=" * 80 + "\n\n")
            
            for sentence in TEST_SENTENCES:
                print(f"\nProcessing sentence: {sentence}")
                
                # Run the inference
                full_output = run_inference(model_path, sentence)
                
                if full_output:
                    # Simply write input and full output
                    f.write(f"{sentence}\n")
                    f.write(f"{full_output}\n\n")
                    print(f"Output received (length: {len(full_output)} chars)")
                else:
                    f.write(f"{sentence}\n")
                    f.write("[ERROR: No output generated]\n\n")
                    print("Error: Failed to generate output")
            
        print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    print("Starting grammar correction comparison across models")
    compare_models()
    print("\nComparison complete!")