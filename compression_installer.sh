#!/bin/bash

# Compression Installer
# Purpose: Install DNA compression and AI storage

# Exit on error
set -e

# Log file
LOG_FILE="/var/log/atulyaai_compression_install.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Install Dependencies
echo "Installing compression dependencies..."
sudo apt-get install -y python3-pip

# Install Python Dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

echo "Compression installation complete!"
