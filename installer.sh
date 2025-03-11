#!/bin/bash
echo "🔄 Installing AtulyaAI..."
git clone https://github.com/atulyaai/AtulyaAI.git /opt/atulyaai
cd /opt/atulyaai
chmod +x install/install_system.py
sudo python3 install/install_system.py
