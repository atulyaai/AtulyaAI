#!/bin/bash

# Update the system and install dependencies
echo "Updating system and installing dependencies..."
sudo apt update -y
sudo apt upgrade -y
sudo apt install -y python3 python3-pip curl git wget build-essential

# Install Hugging Face libraries
echo "Installing Hugging Face dependencies..."
pip3 install huggingface_hub transformers

# Check if Python and pip are installed
if ! command -v python3 &> /dev/null
then
    echo "Python3 not found, installing..."
    sudo apt install -y python3
fi

if ! command -v pip3 &> /dev/null
then
    echo "pip3 not found, installing..."
    sudo apt install -y python3-pip
fi

# Set Hugging Face Token (ensure to replace with your actual Hugging Face Token)
HUGGINGFACE_TOKEN="your-huggingface-token"

# Function to download models
download_model() {
    local model_url=$1
    local model_name=$2
    echo "Downloading model from $model_url..."
    mkdir -p models
    # Download model using Hugging Face hub
    python3 -m huggingface_hub download $model_url --token $HUGGINGFACE_TOKEN --cache-dir ./models
    # Move the downloaded model to the proper location if necessary
    mv ./models/$model_name/* ./models/
}

# Model URLs and names (replace with your actual model URLs and names)
MODEL_14B_URL="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
MODEL_70B_URL="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
MODEL_14B_NAME="DeepSeek-R1-Distill-Qwen-14B"
MODEL_70B_NAME="DeepSeek-R1-Distill-Llama-70B"

# Download models
download_model $MODEL_14B_URL $MODEL_14B_NAME
download_model $MODEL_70B_URL $MODEL_70B_NAME

# Done
echo "Model download complete."
echo "AtulyaAI installation complete! You can now use your AI system."
