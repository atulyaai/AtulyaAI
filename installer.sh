#!/bin/bash
set -e  # Exit on error

INSTALL_DIR="/opt/atulyaai"
WEB_UI_DIR="$INSTALL_DIR/web_ui"
CORE_DIR="$INSTALL_DIR/core"
MODULES_DIR="$INSTALL_DIR/modules"
VENV_DIR="$INSTALL_DIR/venv"

# Ensure required directories exist
echo "Creating necessary directories..."
mkdir -p "$WEB_UI_DIR/backend" "$WEB_UI_DIR/frontend" "$WEB_UI_DIR/admin"
mkdir -p "$CORE_DIR" "$MODULES_DIR"

# Update system and install dependencies
echo "Updating system packages..."
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git

# Set up virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "Setting up virtual environment..."
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists."
fi
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Clone or update repository
if [ ! -d "$INSTALL_DIR/.git" ]; then
    echo "Cloning Atulya AI repository..."
    git clone https://github.com/atulyaai/AtulyaAI "$INSTALL_DIR"
else
    cd "$INSTALL_DIR"
    echo "Pulling the latest changes from the repository..."
    git reset --hard
    git pull origin main
fi

# Install Python dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r "$INSTALL_DIR/requirements.txt"

# Ensure DeepSeek14B model is installed
MODEL_DIR="$CORE_DIR/models/deepseek14b"
if [ ! -d "$MODEL_DIR" ]; then
    echo "Downloading DeepSeek14B model from Hugging Face..."
    mkdir -p "$MODEL_DIR"
    cd "$MODEL_DIR"
    # Replace with actual download command for Hugging Face model
    huggingface-cli login --token your_huggingface_token
    huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
else
    echo "DeepSeek14B model already exists."
fi

# Start Web UI setup
echo "Setting up the Django backend..."
cd "$WEB_UI_DIR/backend"
if [ ! -f "manage.py" ]; then
    echo "Initializing Django project..."
    django-admin startproject backend .
else
    echo "Django project already initialized."
fi

# Final cleanup and activation
echo "Installation complete! Run 'source /opt/atulyaai/venv/bin/activate' to use Atulya AI."
