#!/bin/bash

# AI Core Installer
# Purpose: Install AI models, indexing, and optimizations

# Exit on error
set -e

# Log file
LOG_FILE="/var/log/atulyaai_ai_core_install.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Install Dependencies
echo "Installing AI Core dependencies..."
sudo apt-get install -y python3-pip git

# Install Python Dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

# Download AI Models
echo "Downloading AI models..."
MODEL_14B_URL="https://example.com/deepseek-14b"
MODEL_70B_URL="https://example.com/deepseek-70b"
wget -O ./models/deepseek-14b "$MODEL_14B_URL"
wget -O ./models/deepseek-70b "$MODEL_70B_URL"

# Optimize AI Models
echo "Optimizing AI models..."
python3 ./scripts/optimize_models.py

echo "AI Core installation complete!"
