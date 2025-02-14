#!/bin/bash

# Ensure necessary directories exist
mkdir -p /opt/atulyaai

# Install dependencies
echo "Installing Python packages..."

# Create and activate a Python virtual environment
python3 -m venv /opt/atulyaai/venv
source /opt/atulyaai/venv/bin/activate

# Install necessary dependencies from the requirements.txt (if available)
if [ -f /opt/atulyaai/requirements.txt ]; then
    pip install -r /opt/atulyaai/requirements.txt
else
    echo "requirements.txt not found, skipping installation from requirements file."
fi

# Install HuggingFace transformers and torch for DeepSeek model
pip install transformers torch

# Create the Python script to use the model
echo "Creating Python script 'use_deepseek_model.py'..."

cat <<EOL > /opt/atulyaai/use_deepseek_model.py
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")

# Prepare the input text and tokenize it
input_text = "Who are you?"
inputs = tokenizer(input_text, return_tensors="pt")

# Generate the response from the model
outputs = model.generate(inputs['input_ids'])

# Decode the generated tokens and print the result
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
EOL

# Check if DeepSeek model exists and download
MODEL_DIR="/opt/atulyaai/deepseek_model"

echo "Checking if the model is already downloaded..."

if [ ! -d "$MODEL_DIR" ]; then
    echo "Downloading DeepSeek 14b model..."

    # Correct model URL from Hugging Face
    MODEL_URL="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B/resolve/main/pytorch_model.bin"
    mkdir -p "$MODEL_DIR"

    # Download the model using wget
    wget -P "$MODEL_DIR" "$MODEL_URL"
    if [ $? -eq 0 ]; then
        echo "Model downloaded successfully."
    else
        echo "Failed to download the model. Please check the URL."
        exit 1
    fi
else
    echo "DeepSeek 14b model already exists."
fi

# Run the Python script to generate output
echo "Running the Python script 'use_deepseek_model.py'..."
python /opt/atulyaai/use_deepseek_model.py

# Final message
echo "Installation and execution completed successfully!"
