#!/bin/bash

echo "------------------------------------------"
echo "     🚀 Atulya AI - Auto Installer       "
echo "------------------------------------------"

# Check & Install Basic Dependencies
echo "[1/8] Checking system dependencies..."
if [ -f /etc/debian_version ]; then
    sudo apt update && sudo apt install -y python3 python3-pip python3-venv git curl unzip nodejs npm
elif [ -f /etc/redhat-release ]; then
    sudo yum update && sudo yum install -y python3 python3-pip python3-virtualenv git curl unzip nodejs npm
else
    echo "Unsupported OS. Exiting..."
    exit 1
fi

# Install Caddy for Reverse Proxy
echo "[2/8] Installing & Configuring Caddy..."
sudo apt install -y caddy
sudo systemctl enable --now caddy

# Create Virtual Environment & Install Python Dependencies
echo "[3/8] Setting up Python environment..."
python3 -m venv atulya_env
source atulya_env/bin/activate

echo "[4/8] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Clone or Update Atulya AI Repository
if [ ! -d "AtulyaAI" ]; then
    echo "[5/8] Cloning Atulya AI repository..."
    git clone https://github.com/your-repo/AtulyaAI.git
else
    echo "[5/8] Updating Atulya AI repository..."
    cd AtulyaAI && git pull origin main
fi

# Install & Set Up Web UI
echo "[6/8] Installing frontend dependencies..."
cd frontend
npm install
npm run build
cp -r dist/* /var/www/html/atulya_ai/
cd ..

# Configure Backend (Django + FastAPI)
echo "[7/8] Configuring Backend..."
cd backend
python manage.py migrate
python manage.py collectstatic --noinput
cd ..

# Start Web UI & API Server
echo "[8/8] Starting Atulya AI Services..."
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &

echo "------------------------------------------"
echo " ✅ Installation Complete!"
echo " 🌍 Access the Web UI at: http://your-ip:8000"
echo "------------------------------------------"
