#!/bin/bash

# AtulyaAI One-Click Installer for Ubuntu 22.04 & 24.04

echo "🚀 Starting AtulyaAI Installation..."

# Ensure script is run as root
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

# Clone AtulyaAI Repository (Correct Repository)
echo "📥 Downloading AtulyaAI source code..."
git clone https://github.com/atulyaai/AtulyaAI.git /opt/atulyaai || { echo "❌ Git clone failed!"; exit 1; }

# Navigate to installation directory
if [[ ! -d "/opt/atulyaai" ]]; then
  echo "❌ Installation directory not found! Exiting..."
  exit 1
fi
cd /opt/atulyaai

# Download requirements.txt if missing
if [[ ! -f "requirements.txt" ]]; then
  echo "📥 Fetching requirements.txt from GitHub..."
  wget -O requirements.txt https://raw.githubusercontent.com/atulyaai/AtulyaAI/main/requirements.txt || { echo "❌ Failed to download requirements.txt!"; exit 1; }
fi

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip3 install -r requirements.txt || { echo "❌ Failed to install dependencies!"; exit 1; }

# Setup AtulyaAI Service
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
