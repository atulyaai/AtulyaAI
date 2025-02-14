#!/bin/bash

set -e  # Exit on error

echo "🔄 Checking dependencies..."

# Check and install missing dependencies
REQUIRED_PACKAGES=("python3" "python3-venv" "git" "curl" "fastapi" "uvicorn")
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if ! dpkg -s "$pkg" &> /dev/null && ! python3 -c "import $pkg" &> /dev/null; then
        echo "Installing missing package: $pkg"
        sudo apt-get install -y "$pkg" || pip install "$pkg"
    fi
done

# Activate virtual environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt

# Handle Git updates safely
echo "🔄 Syncing with GitHub repository..."
cd /opt/atulyaai || exit 1

git reset --hard HEAD  # Reset local changes
git pull origin main || (echo "❌ Git pull failed! Check manually."; exit 1)

# Restart AtulyaAI if necessary
if systemctl is-active --quiet atulyaai; then
    echo "🔄 Restarting AtulyaAI service..."
    sudo systemctl restart atulyaai
else
    echo "✅ Starting AtulyaAI service..."
    sudo systemctl start atulyaai
fi

sudo systemctl status atulyaai

