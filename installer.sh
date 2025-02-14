#!/bin/bash
set -e  # Exit on error

INSTALL_DIR="/opt/atulyaai"
WEB_UI_DIR="$INSTALL_DIR/web_ui"
CORE_DIR="$INSTALL_DIR/core"
MODULES_DIR="$INSTALL_DIR/modules"
VENV_DIR="$INSTALL_DIR/venv"
MODEL_REPO="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"  # Updated model repo from Hugging Face
MODEL_DIR="$CORE_DIR/models/deepseek14b"

# Ensure required directories exist
mkdir -p "$WEB_UI_DIR/backend" "$WEB_UI_DIR/frontend" "$WEB_UI_DIR/admin"
mkdir -p "$CORE_DIR" "$MODULES_DIR"

# Update system and install dependencies
echo "Updating system and installing dependencies..."
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git wget

# Set up virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "Setting up virtual environment..."
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install huggingface_hub to download the model
pip install huggingface_hub

# Clone or update repository
if [ ! -d "$INSTALL_DIR/.git" ]; then
    echo "Cloning repository..."
    git clone https://github.com/atulyaai/AtulyaAI "$INSTALL_DIR"
else
    echo "Updating repository..."
    cd "$INSTALL_DIR"
    git reset --hard
    git pull origin main
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r "$INSTALL_DIR/requirements.txt"

# Ensure DeepSeek14B model is installed from Hugging Face
if [ ! -d "$MODEL_DIR" ]; then
    echo "Downloading DeepSeek14B model from Hugging Face..."
    mkdir -p "$MODEL_DIR"
    cd "$MODEL_DIR"
    # Log into Hugging Face and download the model
    hf_token="your_huggingface_token"  # Replace with your Hugging Face API token if required
    huggingface-cli login "$hf_token"
    huggingface-cli download "$MODEL_REPO" --cache-dir="$MODEL_DIR"
fi

# Start Web UI setup
if [ ! -d "$WEB_UI_DIR/backend" ]; then
    echo "Creating backend directory..."
    mkdir -p "$WEB_UI_DIR/backend"
fi

cd "$WEB_UI_DIR/backend"
if [ ! -f "manage.py" ]; then
    echo "Initializing Django backend..."
    django-admin startproject backend .
fi

# Final cleanup
cd "$INSTALL_DIR"
echo "Installation complete! Run 'source $VENV_DIR/bin/activate' to use Atulya AI."
