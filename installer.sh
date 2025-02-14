#!/bin/bash

# AtulyaAI Fully Automated Installer for Ubuntu 22.04 & 24.04

set -e  # Exit immediately if a command exits with a non-zero status

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
apt install -y python3 python3-pip git curl wget unzip libpq-dev ffmpeg portaudio19-dev 

# Clone AtulyaAI Repository
echo "📥 Downloading AtulyaAI source code..."
if [ -d "/opt/atulyaai" ]; then
    echo "⚠️ Existing installation detected. Removing old version..."
    rm -rf /opt/atulyaai
fi
git clone https://github.com/atulyaai/AtulyaAI.git /opt/atulyaai

# Navigate to installation directory
cd /opt/atulyaai

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip3 install --no-cache-dir -r requirements.txt

# Ensure main.py exists
if [ ! -f "main.py" ]; then
    echo "❌ main.py not found in /opt/atulyaai. Installation failed."
    exit 1
fi

# Setup AtulyaAI as a systemd Service
echo "⚙️ Setting up AtulyaAI as a service..."
cat <<EOF > /etc/systemd/system/atulyaai.service
[Unit]
Description=AtulyaAI Service
After=network.target

[Service]
ExecStart=/usr/bin/python3 /opt/atulyaai/main.py
WorkingDirectory=/opt/atulyaai
Restart=always
User=root

[Install]
WantedBy=multi-user.target
EOF

# Enable & Start Service
systemctl daemon-reload
systemctl enable atulyaai
systemctl start atulyaai

# Final Status Check
echo "✅ AtulyaAI Installation Completed!"
echo "🚀 Run the following command to check status: sudo systemctl status atulyaai"
