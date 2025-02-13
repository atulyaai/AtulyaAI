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
apt install -y python3 python3-pip git curl wget unzip

# Clone AtulyaAI Repository (Replace with your repo URL)
echo "📥 Downloading AtulyaAI source code..."
git clone https://github.com/atulyaai/AtulyaAI-Installer.git /opt/atulyaai

# Navigate to installation directory
cd /opt/atulyaai

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip3 install -r requirements.txt

# Setup AtulyaAI Service (Optional: Add systemd service)
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
systemctl enable atulyaai
systemctl start atulyaai

echo "✅ AtulyaAI Installation Completed!"
echo "🚀 Run the following command to check status: sudo systemctl status atulyaai"
