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

# Upgrade pip
pip install --upgrade pip

# Clone or update repository
if [ ! -d "$INSTALL_DIR/.git" ]; then
    git clone https://github.com/atulyaai/AtulyaAI "$INSTALL_DIR"
else
    cd "$INSTALL_DIR"
    git reset --hard
    git pull origin main
fi

# Install Python dependencies
pip install -r "$INSTALL_DIR/requirements.txt"

# Ensure DeepSeek14B model is installed
MODEL_DIR="$CORE_DIR/models/deepseek14b"
if [ ! -d "$MODEL_DIR" ]; then
    echo "Downloading DeepSeek14B model..."
    mkdir -p "$MODEL_DIR"
    cd "$MODEL_DIR"
    wget -O model.tar.gz "https://example.com/deepseek14b.tar.gz"  # Replace with actual link
    tar -xzf model.tar.gz && rm model.tar.gz
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
