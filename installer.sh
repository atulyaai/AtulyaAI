#!/bin/bash
echo "🚀 Starting AtulyaAI Installation..."

if [[ $EUID -ne 0 ]]; then
   echo "❌ This script must be run as root. Use: sudo bash installer.sh"
   exit 1
fi

echo "🔄 Updating system packages..."
apt update && apt upgrade -y

echo "📦 Installing dependencies..."
apt install -y python3 python3-pip git curl wget unzip

echo "📥 Cloning AtulyaAI Repository..."
rm -rf /opt/atulyaai
git clone https://github.com/atulyaai/AtulyaAI.git /opt/atulyaai
cd /opt/atulyaai

echo "🐍 Installing Python dependencies..."
pip3 install -r requirements.txt

echo "⚙️ Setting up AtulyaAI service..."
cat <<EOF > /etc/systemd/system/atulyaai.service
[Unit]
Description=AtulyaAI Service
After=network.target

[Service]
ExecStart=/usr/bin/python3 /opt/atulyaai/server.py
WorkingDirectory=/opt/atulyaai
Restart=always
User=root

[Install]
WantedBy=multi-user.target
EOF

systemctl enable atulyaai
systemctl start atulyaai

echo "✅ Installation Completed! Run: sudo systemctl status atulyaai"
