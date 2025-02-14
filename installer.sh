#!/bin/bash
set -e

# AtulyaAI Installer & Auto-Updater

# Update and install system dependencies
echo "🔄 Updating system packages..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv git cron

# Create AtulyaAI directory if missing
if [ ! -d "/opt/atulyaai" ]; then
    echo "📂 Creating AtulyaAI directory..."
    sudo mkdir -p /opt/atulyaai
    sudo chown $USER:$USER /opt/atulyaai
fi

cd /opt/atulyaai

# Clone or update repository
echo "🔄 Syncing with GitHub repository..."
if [ ! -d ".git" ]; then
    git clone https://github.com/atulyaai/AtulyaAI.git .
else
    git pull origin main
fi

# Set up virtual environment
echo "🐍 Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Install dependencies
echo "📦 Installing required Python packages..."
pip install -r requirements.txt

# Check for missing model script
echo "🛠 Checking AI model files..."
if [ ! -f "core/models/check_model.py" ]; then
    echo "🚨 Missing check_model.py, restoring..."
    wget -O core/models/check_model.py https://raw.githubusercontent.com/atulyaai/AtulyaAI/main/core/models/check_model.py
fi

# Create auto-update script
echo "⚙️ Setting up auto-update system..."
cat <<EOF > /opt/atulyaai/auto_update.sh
#!/bin/bash
cd /opt/atulyaai
source venv/bin/activate
git pull origin main
pip install -r requirements.txt
sudo systemctl restart atulyaai
EOF
chmod +x /opt/atulyaai/auto_update.sh

# Add cron job for auto-update
echo "🕒 Setting up cron job for updates..."
(crontab -l 2>/dev/null; echo "0 * * * * /opt/atulyaai/auto_update.sh >> /var/log/atulyaai_update.log 2>&1") | crontab -

# Set up systemd service
echo "🔧 Setting up AtulyaAI service..."
cat <<EOF | sudo tee /etc/systemd/system/atulyaai.service
[Unit]
Description=AtulyaAI Service
After=network.target

[Service]
ExecStart=/opt/atulyaai/venv/bin/python /opt/atulyaai/server.py
Restart=always
User=$USER
WorkingDirectory=/opt/atulyaai

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable atulyaai
sudo systemctl restart atulyaai

# Installation complete
echo "✅ Installation complete! Run: sudo systemctl status atulyaai"


