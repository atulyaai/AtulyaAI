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

# Install HuggingFace transformers for DeepSeek model
pip install transformers

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

# Final message
echo "Installation completed successfully!"
