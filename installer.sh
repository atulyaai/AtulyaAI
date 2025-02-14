#!/bin/bash

set -e  # Exit on any error

LOG_FILE="/var/log/atulyaai_install.log"
echo "Starting AtulyaAI Installation..." | tee -a $LOG_FILE

# Check if running as root
if [ "$(id -u)" -ne 0 ]; then
    echo "Please run as root or use sudo." | tee -a $LOG_FILE
    exit 1
fi

# Update system packages
echo "Updating system packages..." | tee -a $LOG_FILE
apt update -y && apt upgrade -y | tee -a $LOG_FILE

# Check & Install Dependencies
deps=(git python3 python3-pip python3-venv)
for dep in "${deps[@]}"; do
    if ! dpkg -l | grep -q "${dep}"; then
        echo "Installing $dep..." | tee -a $LOG_FILE
        apt install -y "$dep" | tee -a $LOG_FILE
    else
        echo "$dep is already installed, skipping..." | tee -a $LOG_FILE
    fi
done

# Clone or Update Repository
INSTALL_DIR="/opt/atulyaai"
if [ -d "$INSTALL_DIR" ]; then
    echo "Updating AtulyaAI repository..." | tee -a $LOG_FILE
    cd $INSTALL_DIR
    git reset --hard HEAD  # Reset local changes
    git pull origin main --rebase | tee -a $LOG_FILE
else
    echo "Cloning AtulyaAI repository..." | tee -a $LOG_FILE
    git clone https://github.com/atulyaai/AtulyaAI.git "$INSTALL_DIR" | tee -a $LOG_FILE
fi
cd "$INSTALL_DIR"

# Set permissions
chmod +x installer.sh

# Create virtual environment
if [ ! -d "$INSTALL_DIR/venv" ]; then
    echo "Setting up Python virtual environment..." | tee -a $LOG_FILE
    python3 -m venv "$INSTALL_DIR/venv"
fi
source "$INSTALL_DIR/venv/bin/activate"

# Install Python dependencies
pip install --upgrade pip | tee -a $LOG_FILE
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt | tee -a $LOG_FILE
else
    echo "No requirements.txt found, skipping..." | tee -a $LOG_FILE
fi

# Ensure AI model installation
echo "Checking AI model installation..." | tee -a $LOG_FILE
if [ ! -f "$INSTALL_DIR/models/deepseek-14b.bin" ]; then
    echo "Downloading DeepSeek-14B model..." | tee -a $LOG_FILE
    mkdir -p "$INSTALL_DIR/models"
    wget -O "$INSTALL_DIR/models/deepseek-14b.bin" "https://example.com/deepseek-14b.bin" | tee -a $LOG_FILE
else
    echo "AI model already installed, skipping..." | tee -a $LOG_FILE
fi

# Restart AtulyaAI Service
echo "Restarting AtulyaAI service..." | tee -a $LOG_FILE
systemctl restart atulyaai.service | tee -a $LOG_FILE

echo "Installation complete!" | tee -a $LOG_FILE

