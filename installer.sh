#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status
LOGFILE="/var/log/atulyaai_installer.log"
exec > >(tee -a "$LOGFILE") 2>&1  # Log output

INSTALL_DIR="/opt/atulyaai"
GIT_REPO="https://github.com/atulyaai/AtulyaAI.git"
MODEL_DIR="$INSTALL_DIR/core/models"
MODEL_URL="https://huggingface.co/deepseek-ai/deepseek-llm-14b-chat"

# Function to check if a package is installed
is_installed() {
    dpkg -l | grep -q "$1"
}

# Ensure script is run as root
if [[ $EUID -ne 0 ]]; then
    echo "Please run as root"
    exit 1
fi

# Install required system packages if missing
REQUIRED_PKGS=(git python3 python3-venv python3-pip nginx)
for pkg in "${REQUIRED_PKGS[@]}"; do
    if ! is_installed "$pkg"; then
        echo "Installing $pkg..."
        apt update && apt install -y "$pkg"
    else
        echo "$pkg is already installed. Skipping."
    fi
done

# Clone or update repository
if [ -d "$INSTALL_DIR" ]; then
    echo "Updating Atulya AI..."
    cd "$INSTALL_DIR"
    git reset --hard  # Prevent conflicts
    git pull origin main || { echo "Git pull failed"; exit 1; }
else
    echo "Cloning Atulya AI..."
    git clone "$GIT_REPO" "$INSTALL_DIR"
fi

# Create virtual environment
if [ ! -d "$INSTALL_DIR/venv" ]; then
    echo "Setting up virtual environment..."
    python3 -m venv "$INSTALL_DIR/venv"
fi
source "$INSTALL_DIR/venv/bin/activate"

# Install Python dependencies
pip install --upgrade pip
pip install -r "$INSTALL_DIR/requirements.txt"

# Ensure Core & Web UI are installed first
cd "$INSTALL_DIR/web_ui/backend"
uvicorn main:app --host 0.0.0.0 --port 8000 &
cd "$INSTALL_DIR/web_ui/frontend"
npm install && npm run build

# Install DeepSeek 14B model if missing
if [ ! -d "$MODEL_DIR/deepseek-14b" ]; then
    echo "Downloading DeepSeek 14B model..."
    mkdir -p "$MODEL_DIR"
    git clone "$MODEL_URL" "$MODEL_DIR/deepseek-14b"
else
    echo "DeepSeek 14B model already installed. Skipping."
fi

# Start services
systemctl restart nginx

echo "✅ Atulya AI installed successfully!"
