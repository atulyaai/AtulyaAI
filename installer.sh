#!/bin/bash

# Set your Hugging Face API token (replace with your token)
HUGGINGFACE_TOKEN="your_hugging_face_token_here"  # Replace with your actual Hugging Face token

# Define model names for DeepSeek models
MODEL_14B_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"  # Qwen 14B model
MODEL_70B_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"  # Llama 70B model

# Install Hugging Face libraries (if not installed)
echo "Installing Hugging Face libraries..."
pip install huggingface_hub transformers

# Function to download models from Hugging Face
download_model() {
    MODEL_NAME=$1
    OUTPUT_DIR=$2

    echo "Downloading model: $MODEL_NAME..."

    # Use huggingface_hub library to download the model with the API token
    python3 -c "
import os
from huggingface_hub import hf_hub_download

# Set Hugging Face token for authentication
os.environ['HF_HOME'] = '$HUGGINGFACE_TOKEN'

# Download model
hf_hub_download(repo_id='$MODEL_NAME', cache_dir='$OUTPUT_DIR', use_auth_token=True)
" 

    if [ $? -eq 0 ]; then
        echo "Model $MODEL_NAME downloaded successfully!"
    else
        echo "Failed to download model $MODEL_NAME."
        exit 1
    fi
}

# Download DeepSeek-Qwen-14B model
download_model $MODEL_14B_NAME "./models"

# Download DeepSeek-Llama-70B model
download_model $MODEL_70B_NAME "./models"

echo "Model download complete!"
