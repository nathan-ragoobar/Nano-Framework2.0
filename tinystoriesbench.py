import subprocess
import json

# Define the path and arguments separately
INFERENCE_EXECUTABLE = "./inference_gpt2_cpu"
MODEL_PATH = "gpt2_124MFinal.bin"

# List of Tiny Stories benchmark prompts
TINY_STORIES_PROMPTS = [
    "Tom and Jane are friends. One day, Jane goes to Tom's house. Tom has a big pot of soup. He wants to share it with Jane. \"Jane, do you want some soup?\" Tom asks.",
    "Lily wanted to get either a cat or a dog. Her mother didn't let her get a dog so instead she...",
    "Jack was hungry, so he went looking for...",
    # Add more prompts from the Tiny Stories dataset.
]

# Function to call the inference executable
def run_inference(prompt):
    try:
        print(f"Running inference on prompt: {prompt[:30]}...")
        
        # Pass executable and arguments as separate list items
        result = subprocess.run(
            [INFERENCE_EXECUTABLE, "--model", MODEL_PATH],
            input=prompt,
            text=True,
            capture_output=True,
            check=False  # Don't raise exception on non-zero exit
        )
        
        # Check for errors
        if result.returncode != 0:
            print(f"Error: Process returned exit code {result.returncode}")
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
            end_idx = output.find(end_marker, start_idx)
            if end_idx >= 0:
                generated_text = output[start_idx:end_idx].strip()
                return generated_text
        
        # If markers not found, return the whole output
        print("Warning: Could not extract text between markers, returning full output")
        return output
        
    except Exception as e:
        print(f"Exception running inference: {e}")
        return None

# Collect outputs
generated_stories = []
for i, prompt in enumerate(TINY_STORIES_PROMPTS):
    print(f"\nRunning prompt {i+1}/{len(TINY_STORIES_PROMPTS)}")
    generated_text = run_inference(prompt)
    if generated_text:
        generated_stories.append({"prompt": prompt, "completion": generated_text})
    print(f"Completed {i+1}/{len(TINY_STORIES_PROMPTS)} prompts")

# Format output for GPT-4 grading
gpt4_grading_prompt = """
You are evaluating the performance of a student who was given a story prompt and asked to complete it. 
Please assess their completion based on the following criteria:
- **Grammar (Score: 1-10)**
- **Creativity (Score: 1-10)**
- **Consistency with the prompt (Score: 1-10)**
- **Overall coherence (Score: 1-10)**
- **Approximate age of the student based on writing style (Choose: A: 3 or under, B: 4-5, C: 6-7, D: 8-9, E: 10-12, F: 13-16)**

Here are the student's responses:

""" + "\n\n".join(
    [f"**Prompt:** {story['prompt']}\n**Completion:** {story['completion']}" for story in generated_stories]
)

# Save the grading prompt to a file for easy access
with open("gpt4_grading_prompt.txt", "w", encoding="utf-8") as f:
    f.write(gpt4_grading_prompt)

print("GPT-4 grading prompt saved to 'gpt4_grading_prompt.txt'.")