#!/bin/bash

# IoT Control Installer
# Purpose: Connect external devices and enable remote control

# Exit on error
set -e

# Log file
LOG_FILE="/var/log/atulyaai_iot_control_install.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Install Dependencies
echo "Installing IoT Control dependencies..."
sudo apt-get install -y python3-pip

# Install Python Dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

echo "IoT Control installation complete!"
