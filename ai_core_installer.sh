#!/bin/bash

# AI Core Installer
# Purpose: Install AI models, indexing, and optimizations

# Exit on error
set -e

# Log file
LOG_FILE="/var/log/atulyaai_ai_core_install.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# GitHub repository URL
GITHUB_REPO="https://raw.githubusercontent.com/atulyaai/AtulyaAI/main"

# Install Dependencies
echo "Installing AI Core dependencies..."
sudo apt-get install -y python3-pip git

# Download requirements.txt
echo "Downloading requirements.txt..."
wget -O requirements.txt "${GITHUB_REPO}/requirements.txt"

# Install Python Dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing Python dependencies..."
    pip3 install -r requirements.txt
else
    echo "Error: requirements.txt not found!"
    exit 1
fi

# Download AI Models from Hugging Face
echo "Downloading AI models from Hugging Face..."

# Create a directory for models
mkdir -p ./models

# Python script to download models
echo "Creating Python script to download models..."
cat <<EOL > download_models.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Define models to download
models = {
    "deepseek-14b": "DeepSeek/DeepSeek-14B",
    "deepseek-70b": "DeepSeek/DeepSeek-70B"
}

# Download models
for model_name, model_path in models.items():
    print(f"Downloading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Save model and tokenizer
    model.save_pretrained(f"./models/{model_name}")
    tokenizer.save_pretrained(f"./models/{model_name}")
    print(f"{model_name} downloaded and saved to ./models/{model_name}")
EOL

# Run the Python script
echo "Running Python script to download models..."
python3 download_models.py

# Optimize AI Models
echo "Optimizing AI models..."
python3 ./scripts/optimize_models.py

echo "AI Core installation complete!"
