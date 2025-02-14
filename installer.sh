#!/bin/bash

# AtulyaAI One-Click Installer for Ubuntu 22.04 & 24.04
# This script installs all necessary dependencies and sets up AtulyaAI.

echo "🚀 Starting AtulyaAI Installation..."

# Ensure the script is run as root
if [[ $EUID -ne 0 ]]; then
   echo "❌ This script must be run as root. Use: sudo bash installer.sh"
   exit 1
fi

# Update System
echo "🔄 Updating system packages..."
apt update && apt upgrade -y

# Install Dependencies
echo "📦 Installing required dependencies..."
apt install -y python3 python3-pip git curl wget unzip libpq-dev

# Define installation directory
INSTALL_DIR="/opt/atulyaai"
REPO_URL="https://github.com/atulyaai/AtulyaAI.git"

# Check if AtulyaAI directory exists
if [ -d "$INSTALL_DIR" ]; then
    echo "⚠️ AtulyaAI directory already exists. Updating existing installation..."
    cd "$INSTALL_DIR"
    
    # Check if it's a valid Git repository
    if [ -d ".git" ]; then
        git reset --hard origin/main
        git pull origin main
    else
        echo "❌ Existing directory is not a valid Git repository. Removing and re-cloning..."
        cd /opt
        rm -rf "$INSTALL_DIR"
        git clone "$REPO_URL" "$INSTALL_DIR"
    fi
else
    echo "📥 Cloning AtulyaAI source code..."
    git clone "$REPO_URL" "$INSTALL_DIR"
fi

# Navigate to installation directory
cd "$INSTALL_DIR"

# Ensure requirements.txt is downloaded
echo "📄 Downloading latest requirements.txt..."
wget -O requirements.txt https://raw.githubusercontent.com/atulyaai/AtulyaAI/main/requirements.txt

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Ensure main.py exists
if [ ! -f "main.py" ]; then
    echo "❌ main.py not found in $INSTALL_DIR. Please check the repository."
    exit 1
fi

# Setup AtulyaAI Service
echo "⚙️ Setting up AtulyaAI as a service..."
cat <<EOF > /etc/systemd/system/atulyaai.service
[Unit]
Description=AtulyaAI Service
After=network.target

[Service]
ExecStart=/usr/bin/python3 $INSTALL_DIR/main.py
WorkingDirectory=$INSTALL_DIR
Restart=always
User=root

[Install]
WantedBy=multi-user.target
EOF

# Enable & Start Service
systemctl daemon-reload
systemctl enable atulyaai
systemctl restart atulyaai

# Final Check
echo "✅ AtulyaAI Installation Completed!"
echo "🚀 Run the following command to check status: sudo systemctl status atulyaai"
