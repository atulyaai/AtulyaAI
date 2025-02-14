#!/bin/bash
set -e  # Exit on error

INSTALL_DIR="/opt/atulyaai"
WEB_UI_DIR="$INSTALL_DIR/web_ui"
CORE_DIR="$INSTALL_DIR/core"
MODULES_DIR="$INSTALL_DIR/modules"
VENV_DIR="$INSTALL_DIR/venv"

# Ensure required directories exist
mkdir -p "$WEB_UI_DIR/backend" "$WEB_UI_DIR/frontend" "$WEB_UI_DIR/admin"
mkdir -p "$CORE_DIR" "$MODULES_DIR"

# Update system and install dependencies
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git

# Set up virtual environment
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Upgrade pip only if it's a compatible version for Atulya AI
CURRENT_PIP=$(pip --version | awk '{print $2}')
if [[ "$CURRENT_PIP" != "25.0.1" ]]; then
    echo "Upgrading pip to version 25.0.1 for Atulya AI..."
    pip install --upgrade "pip==25.0.1"
fi

# Clone or update repository
if [ ! -d "$INSTALL_DIR/.git" ]; then
    git clone https://github.com/atulyaai/AtulyaAI "$INSTALL_DIR"
else
    cd "$INSTALL_DIR"
    git reset --hard
    git pull origin main
fi

# Install Python dependencies while preventing pip upgrade
PIP_NO_PYTHON_VERSION_WARNING=1 pip install -r "$INSTALL_DIR/requirements.txt" --upgrade-strategy only-if-needed

# Ensure DeepSeek14B model is installed
MODEL_DIR="$CORE_DIR/models/deepseek14b"
if [ ! -d "$MODEL_DIR" ]; then
    echo "Downloading DeepSeek14B model from Hugging Face..."
    mkdir -p "$MODEL_DIR"
    cd "$MODEL_DIR"
    
    # Ensure Hugging Face authentication token is available
    if [ -z "$HF_TOKEN" ]; then
        echo "Please set your Hugging Face token as HF_TOKEN environment variable."
        exit 1
    fi
    
    huggingface-cli login --token $HF_TOKEN
    huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --cache-dir "$MODEL_DIR"
fi

# Start Web UI setup
cd "$WEB_UI_DIR/backend"
if [ ! -f "manage.py" ]; then
    echo "Initializing Django backend..."
    django-admin startproject backend .
fi

# Final cleanup
cd "$INSTALL_DIR"
echo "Installation complete! Run 'source $VENV_DIR/bin/activate' to use Atulya AI."
