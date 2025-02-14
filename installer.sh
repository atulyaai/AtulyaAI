#!/bin/bash
set -e  # Exit on error

INSTALL_DIR="/opt/atulyaai"
VENV_DIR="$INSTALL_DIR/venv"

# Ensure script runs as root
if [[ $EUID -ne 0 ]]; then
    echo "Please run as root or use sudo"
    exit 1
fi

# Update package list
echo "Updating package list..."
apt update -y

# Install required dependencies if missing
deps=(git python3 python3-venv python3-pip unzip)
for pkg in "${deps[@]}"; do
    if ! dpkg -s $pkg &> /dev/null; then
        echo "Installing $pkg..."
        apt install -y $pkg
    fi
done

# Clone or update the repository
if [ -d "$INSTALL_DIR" ]; then
    echo "Updating existing AtulyaAI installation..."
    cd "$INSTALL_DIR"
    git reset --hard
    git pull origin main
else
    echo "Cloning AtulyaAI repository..."
    git clone https://github.com/atulyaai/AtulyaAI "$INSTALL_DIR"
fi

# Set up Python virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r "$INSTALL_DIR/requirements.txt"

# Ensure core and web UI are installed first
mkdir -p "$INSTALL_DIR/core" "$INSTALL_DIR/web_ui"

# Set permissions
echo "Setting permissions..."
chown -R $(whoami):$(whoami) "$INSTALL_DIR"
chmod -R 755 "$INSTALL_DIR"

# Enable and restart the service
systemctl daemon-reload
systemctl enable atulyaai.service
systemctl restart atulyaai.service

echo "AtulyaAI installation complete."
