#!/bin/bash

# Update script for AtulyaAI

REPO_URL="https://github.com/atulyaai/AtulyaAI.git"
INSTALL_DIR="/opt/atulyaai"

echo "🔄 Checking for updates..."

# Pull the latest changes from GitHub
if [ -d "$INSTALL_DIR" ]; then
    cd "$INSTALL_DIR"
    git pull origin main
else
    echo "❌ Installation directory not found!"
    exit 1
fi

echo "✅ Update completed!"
