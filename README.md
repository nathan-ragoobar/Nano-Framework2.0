# Nano-Framework2.0

## Overview
Nano-Framework2.0 is a lightweight framework designed for efficient and scalable machine learning model training and inference. This project includes implementations for training and running GPT-2 models.

## Prerequisites
- **Python 3.8+**
- **CUDA Toolkit** (for GPU support)
- **CMake** (for building the project)
- **GCC** (for compiling C++ code)
- **Python Libraries**: Install the required Python libraries using `requirements.txt`.

NB: The following instructions are for Linux based systems. If you're on windows the easiest thing is to use run it using WSL.

## Installation

### Step 1: Clone the Repository
1.1 Clone all branchs of the repo using the `--mirror` flag.\
1.2 Change to the Nano-Framework directory.\
1.3 Change to the inference branch.\

```sh
git clone --mirror https://github.com/yourusername/Nano-Framework2.0.git
cd Nano-Framework2.0
git switch inference
```

### Step 2: Downloading and tokenizing the dataset
2.1 Create a python virtual environment and activate it
2.2 pip install all the necessary libraries
2.3 Run either the tinyshakespeare or tinystories scripts to get a dataset to use for now
Note: Nano does have it's own tokenizer but it's easier to just write python scripts to download and process these dataset
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python dev/data/tinyshakespeare.py
python dev/data/tinystories.py
```

### Step 3: Compiling
3.1 Create a build directory in the root folder and switch to it
3.2 Run the cmake command to generate the make files
3.3 Run the make command to generate the three executables for:
- inference_gpt2_cpu
- train_gpt2_cpu
- train_gpt2_gpu

Note: If you only want to generate one executable, then run `make <executable_name>`
```bash
mkdir build
cd build
cmake ..
make
```

### Step 4: Running
4.1 Copy the generate file from build directory to root
4.2 Run
```bash
cd ..

cp build/inference_gpt2_cpu ..
cp build/train_gpt2_cpu ..
cp build/train_gpt2_gpu ..

./inference_gpt2_cpu
./train_gpt2_cpu
./train_gpt2_gpu
```