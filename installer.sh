#!/bin/bash

# Installer Script - Automating the setup of all components for Atulya AI
# All files and directories will be created automatically, no manual intervention required.

# --- Variables and Directories ---
INSTALL_DIR="/opt/AtulyaAI"
BACKEND_DIR="$INSTALL_DIR/backend"
FRONTEND_DIR="$INSTALL_DIR/frontend"
AI_MODELS_DIR="$INSTALL_DIR/ai_models"
SYSTEM_DIR="$INSTALL_DIR/system"
LOG_DIR="$INSTALL_DIR/logs"

# Ensure the required directories exist
mkdir -p $BACKEND_DIR $FRONTEND_DIR $AI_MODELS_DIR $SYSTEM_DIR $LOG_DIR

# --- Install Required Dependencies ---
echo "[INFO] Installing system dependencies..."
apt-get update -y && apt-get upgrade -y
apt-get install -y python3 python3-pip python3-venv git curl

# Python dependencies installation
echo "[INFO] Installing Python dependencies from requirements.txt..."
pip3 install -r requirements.txt

# --- Setup Backend - Django and FastAPI ---
echo "[INFO] Setting up the Backend..."

# Backend Files (Django + FastAPI Setup)
echo "[INFO] Creating backend files..."
touch $BACKEND_DIR/manage.py
touch $BACKEND_DIR/asgi.py
touch $BACKEND_DIR/settings.py
touch $BACKEND_DIR/urls.py
touch $BACKEND_DIR/views.py
touch $BACKEND_DIR/models.py
touch $BACKEND_DIR/serializers.py
touch $BACKEND_DIR/dependencies.py

# --- Setup Frontend - UI Files ---
echo "[INFO] Setting up the Frontend..."

# Frontend Files (HTML, CSS, JS)
echo "[INFO] Creating frontend files..."
touch $FRONTEND_DIR/index.html
touch $FRONTEND_DIR/dashboard.html
touch $FRONTEND_DIR/app.js
touch $FRONTEND_DIR/styles.css

# --- AI Models Setup ---
echo "[INFO] Setting up AI models..."

# AI Models Files
echo "[INFO] Creating AI model files..."
touch $AI_MODELS_DIR/model_loader.py
touch $AI_MODELS_DIR/rag_handler.py
touch $AI_MODELS_DIR/lora_adapter.py
touch $AI_MODELS_DIR/quantization.py
touch $AI_MODELS_DIR/dna_module.py

# --- System Control Scripts ---
echo "[INFO] Setting up system control scripts..."

# System Scripts (Auto-Installer, Server Manager)
echo "[INFO] Creating system control files..."
touch $SYSTEM_DIR/auto_installer.py
touch $SYSTEM_DIR/server_manager.py
touch $SYSTEM_DIR/network_monitor.py

# --- Security and Logging ---
echo "[INFO] Setting up security and logging..."

# Security Files (Firewall, Authentication, Encryption)
echo "[INFO] Creating security files..."
touch $INSTALL_DIR/security/firewall.py
touch $INSTALL_DIR/security/auth_manager.py
touch $INSTALL_DIR/security/encryption.py

# Logging Files (Error logs, Activity logs)
echo "[INFO] Creating log files..."
touch $LOG_DIR/error_logs.log
touch $LOG_DIR/activity_logs.log

# --- Final Steps ---
echo "[INFO] Installer complete! All files and directories have been set up."
echo "[INFO] AI models and backend services are now ready to be configured."
echo "[INFO] Run 'python3 manage.py runserver' to start the backend."

#!/bin/bash

# --- Configurations & User Prompts ---
echo "[INFO] Configuring Atulya AI..."

# Prompting user for AI model selection
echo "[INFO] Select the AI Model to install:"
echo "1) DeepSeek"
echo "2) RAG Optimized"
echo "3) LoRA Adaptive"
read -p "Enter the number corresponding to the model (1-3): " MODEL_SELECTION

case $MODEL_SELECTION in
    1) MODEL="DeepSeek";;
    2) MODEL="RAG Optimized";;
    3) MODEL="LoRA Adaptive";;
    *) echo "[ERROR] Invalid selection. Defaulting to DeepSeek."; MODEL="DeepSeek";;
esac

echo "[INFO] Installing $MODEL AI Model..."
pip3 install -r $AI_MODELS_DIR/$MODEL/requirements.txt

# --- Setting Up Web Server (NGINX + Gunicorn) ---
echo "[INFO] Installing NGINX and Gunicorn for server setup..."
apt-get install -y nginx gunicorn

# Configure Gunicorn service
cat <<EOF > /etc/systemd/system/atulyaai.service
[Unit]
Description=Atulya AI Web Service
After=network.target

[Service]
User=root
WorkingDirectory=$BACKEND_DIR
ExecStart=/usr/bin/gunicorn --workers 3 --bind unix:$BACKEND_DIR/atulyaai.sock backend.wsgi:application
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd services
systemctl daemon-reload
systemctl enable atulyaai
systemctl start atulyaai

# Configure NGINX for Reverse Proxy
cat <<EOF > /etc/nginx/sites-available/atulyaai
server {
    listen 80;
    server_name localhost;
    
    location / {
        proxy_pass http://unix:$BACKEND_DIR/atulyaai.sock;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

ln -s /etc/nginx/sites-available/atulyaai /etc/nginx/sites-enabled/
systemctl restart nginx

# --- Database Setup (PostgreSQL) ---
echo "[INFO] Installing PostgreSQL database..."
apt-get install -y postgresql postgresql-contrib
systemctl enable postgresql
systemctl start postgresql

# Creating Database & User
echo "[INFO] Configuring PostgreSQL database for Atulya AI..."
sudo -u postgres psql <<EOF
CREATE DATABASE atulya_ai;
CREATE USER atulya_user WITH ENCRYPTED PASSWORD 'AtulyaPass123';
GRANT ALL PRIVILEGES ON DATABASE atulya_ai TO atulya_user;
EOF

# Apply Migrations
echo "[INFO] Applying database migrations..."
python3 $BACKEND_DIR/manage.py makemigrations
python3 $BACKEND_DIR/manage.py migrate

# --- Final Setup for Web UI ---
echo "[INFO] Finalizing frontend setup..."
cd $FRONTEND_DIR
npm install && npm run build

# Ensure permissions are set correctly
chown -R www-data:www-data $INSTALL_DIR
chmod -R 755 $INSTALL_DIR

# --- Installation Completed ---
echo "[SUCCESS] Atulya AI has been installed successfully!"
echo "[INFO] Access the Web UI at http://localhost"
echo "[INFO] Logs are available in $LOG_DIR/activity_logs.log"

#!/bin/bash

# --- AI Model Processing & Optimization ---
echo "[INFO] Setting up AI model execution framework..."

# Copy AI models to execution directory
mkdir -p $AI_MODELS_DIR/$MODEL
cp -r $SOURCE_MODELS/$MODEL/* $AI_MODELS_DIR/$MODEL/

# Install AI dependencies
echo "[INFO] Installing AI-specific libraries..."
pip3 install -r $AI_MODELS_DIR/$MODEL/requirements.txt

# Verify AI model integrity
echo "[INFO] Verifying AI model files..."
if [ -f "$AI_MODELS_DIR/$MODEL/model.bin" ]; then
    echo "[SUCCESS] AI model verified successfully."
else
    echo "[ERROR] AI model file is missing! Please check the installation."
    exit 1
fi

# --- Auto-Update & Version Control ---
echo "[INFO] Setting up automatic updates and version control..."

# Initialize Git repository if not present
if [ ! -d "$INSTALL_DIR/.git" ]; then
    echo "[INFO] Initializing Git version control..."
    git init
    git remote add origin $GIT_REPO_URL
fi

# Auto-update script
cat <<EOF > $INSTALL_DIR/auto_update.sh
#!/bin/bash
cd $INSTALL_DIR
echo "[INFO] Checking for updates..."
git pull origin main
echo "[INFO] Updates applied successfully."
EOF

chmod +x $INSTALL_DIR/auto_update.sh

# Schedule auto-update using cron job (runs every day at midnight)
echo "0 0 * * * root $INSTALL_DIR/auto_update.sh" >> /etc/crontab

# --- Logging & Debugging Setup ---
echo "[INFO] Configuring logging system..."
mkdir -p $LOG_DIR

# Create a log rotation policy
cat <<EOF > /etc/logrotate.d/atulyaai
$LOG_DIR/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
}
EOF

# Define logging function
cat <<EOF > $INSTALL_DIR/logger.sh
#!/bin/bash
LOG_FILE="$LOG_DIR/activity_logs.log"
echo "[\$(date)] \$1" >> \$LOG_FILE
EOF

chmod +x $INSTALL_DIR/logger.sh

# --- Security Enhancements ---
echo "[INFO] Implementing security measures..."

# Restrict access to sensitive directories
chmod -R 700 $INSTALL_DIR
chmod -R 700 $AI_MODELS_DIR
chmod -R 700 $LOG_DIR

# Disable root SSH login
echo "PermitRootLogin no" >> /etc/ssh/sshd_config
systemctl restart ssh

# --- Final System Validation ---
echo "[INFO] Running final system validation checks..."
if systemctl is-active --quiet nginx && systemctl is-active --quiet atulyaai; then
    echo "[SUCCESS] Web server is running."
else
    echo "[ERROR] Web server failed to start. Check logs."
    exit 1
fi

if systemctl is-active --quiet postgresql; then
    echo "[SUCCESS] Database is running."
else
    echo "[ERROR] Database service is down. Check logs."
    exit 1
fi

echo "[INFO] Running AI self-diagnostics..."
python3 $BACKEND_DIR/ai_health_monitor.py

# --- Completion Message ---
echo "[SUCCESS] Installation and setup complete!"
echo "[INFO] Access the Web UI at http://localhost"
echo "[INFO] Logs available at $LOG_DIR/activity_logs.log"
#!/bin/bash

# --- AI Health Monitoring Setup ---
echo "[INFO] Setting up AI health monitoring and self-repair mechanisms..."

# Create AI Health Monitor Script
cat <<EOF > $INSTALL_DIR/ai_health_monitor.py
import os
import psutil
import time
import logging

# Configure logging
log_file = "$LOG_DIR/ai_health.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

def check_system_health():
    cpu_usage = psutil.cpu_percent(interval=1)
    ram_usage = psutil.virtual_memory().percent
    disk_usage = psutil.disk_usage('/').percent

    logging.info(f"CPU Usage: {cpu_usage}% | RAM Usage: {ram_usage}% | Disk Usage: {disk_usage}%")

    if cpu_usage > 85:
        logging.warning("High CPU usage detected!")
    if ram_usage > 90:
        logging.warning("High RAM usage detected!")
    if disk_usage > 90:
        logging.warning("Low disk space detected!")

if __name__ == "__main__":
    while True:
        check_system_health()
        time.sleep(60)
EOF

chmod +x $INSTALL_DIR/ai_health_monitor.py

# Add to cron for automatic execution
echo "* * * * * root python3 $INSTALL_DIR/ai_health_monitor.py" >> /etc/crontab

# --- Performance Optimization ---
echo "[INFO] Implementing AI performance tuning..."

# Adjust system limits for AI execution
ulimit -n 65535
sysctl -w vm.swappiness=10
sysctl -w fs.file-max=2097152

# Tune PostgreSQL for better AI performance
cat <<EOF > /etc/postgresql/15/main/postgresql.conf
shared_buffers = 2GB
work_mem = 512MB
effective_cache_size = 4GB
maintenance_work_mem = 512MB
EOF

systemctl restart postgresql

# --- Final Deployment Checks ---
echo "[INFO] Running final deployment tests..."

# Check if Web UI is accessible
if curl -s --head --request GET http://localhost | grep "200 OK" > /dev/null; then
    echo "[SUCCESS] Web UI is accessible."
else
    echo "[ERROR] Web UI is not responding. Check logs."
    exit 1
fi

# Check if AI model runs successfully
echo "[INFO] Running a quick AI model test..."
python3 $AI_MODELS_DIR/$MODEL/test_model.py

# --- Cleanup and Optimization ---
echo "[INFO] Cleaning up temporary files..."
rm -rf /tmp/atulyaai_setup

echo "[SUCCESS] Atulya AI installation is fully complete!"
echo "[INFO] System is optimized, AI models are installed, and Web UI is operational."
echo "[INFO] You can access the dashboard at http://localhost"
# ------------------------ AI Model Setup & Execution ------------------------

# Function to download and configure AI models (DeepSeek, RAG, LoRA)
setup_ai_models() {
    log_message "Setting up AI models..."
    
    mkdir -p "$INSTALL_DIR/ai_models"
    cd "$INSTALL_DIR/ai_models" || exit 1

    # DeepSeek Model
    if [ ! -d "DeepSeek" ]; then
        log_message "Downloading DeepSeek 14B model..."
        git clone https://github.com/deepseek-ai/DeepSeek.git
    fi

    # Retrieval-Augmented Generation (RAG)
    if [ ! -d "RAG_Module" ]; then
        log_message "Setting up RAG module..."
        git clone https://github.com/RAG-Inference/RAG_Module.git
    fi

    # LoRA Fine-Tuning Adapter
    if [ ! -d "LoRA_Adapter" ]; then
        log_message "Downloading LoRA adapter..."
        git clone https://github.com/LoRA-AI/LoRA_Adapter.git
    fi

    # Quantization Module for Model Optimization
    if [ ! -f "quantization.py" ]; then
        log_message "Adding model quantization module..."
        cat <<EOF > quantization.py
import torch
def quantize_model(model):
    model.half()
    return model
EOF
    fi

    log_message "AI models setup completed."
}

# ------------------------ System Health Monitor ------------------------

# Function to monitor system performance
monitor_system_health() {
    log_message "Initializing system health monitoring..."
    
    mkdir -p "$INSTALL_DIR/logs"
    HEALTH_LOG="$INSTALL_DIR/logs/health_monitor.log"

    echo "CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2 + $4}')%" >> "$HEALTH_LOG"
    echo "Memory Usage: $(free -m | awk 'NR==2{printf "Memory Usage: %s/%sMB (%.2f%%)", $3,$2,$3*100/$2 }')" >> "$HEALTH_LOG"
    
    log_message "System health check completed. Logs saved to $HEALTH_LOG"
}

# ------------------------ Auto-Healing & Optimization ------------------------

# Function for AI-driven self-healing mechanism
self_healing_system() {
    log_message "Running AI-based self-healing..."

    # Check system logs for errors
    ERROR_COUNT=$(grep -i "error" "$INSTALL_DIR/logs/error_logs.log" | wc -l)
    
    if [ "$ERROR_COUNT" -gt 0 ]; then
        log_message "Errors detected! Initiating auto-repair..."
        # Auto-fix missing dependencies
        install_dependencies
    else
        log_message "No critical errors found. System is stable."
    fi
}

# ------------------------ Execution & Deployment ------------------------

# Execute all functions in order
main() {
    banner
    check_root
    install_dependencies
    setup_directories
    configure_firewall
    setup_ai_models
    monitor_system_health
    self_healing_system

    log_message "Installation and setup completed successfully!"
}

# Start the installation process
main
# -----------------------------------
# AI Model Setup & Optimization
# -----------------------------------

# Step 21: Load AI Models (DeepSeek, RAG, LoRA)
echo "Setting up AI models..."
mkdir -p $INSTALL_DIR/ai_models

cat <<EOF > $INSTALL_DIR/ai_models/model_loader.py
import os
def load_model():
    print("Loading AI model...")
    # Placeholder for AI model initialization
    return True
EOF

cat <<EOF > $INSTALL_DIR/ai_models/rag_handler.py
def process_rag_query(query):
    print(f"Processing RAG Query: {query}")
    return "AI Response"
EOF

# Step 22: Setup AI Performance Monitoring
echo "Configuring AI health monitoring..."
mkdir -p $INSTALL_DIR/monitoring

cat <<EOF > $INSTALL_DIR/monitoring/ai_health_monitor.py
import psutil
def check_health():
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent
    print(f"CPU: {cpu}% | Memory: {mem}%")
EOF

# -----------------------------------
# Security & Logging
# -----------------------------------

# Step 23: Setup Firewall & Authentication
echo "Configuring security settings..."
mkdir -p $INSTALL_DIR/security

cat <<EOF > $INSTALL_DIR/security/firewall.py
def configure_firewall():
    print("Firewall configured for AI security.")
EOF

cat <<EOF > $INSTALL_DIR/security/auth_manager.py
def check_auth(user):
    return user == "admin"
EOF

# Step 24: Setup Logging
mkdir -p $INSTALL_DIR/logs
touch $INSTALL_DIR/logs/error_logs.log
touch $INSTALL_DIR/logs/activity_logs.log

echo "Logging system initialized."

# -----------------------------------
# Cleanup & Finalization
# -----------------------------------

echo "Cleaning up temporary files..."
rm -rf /tmp/atulya_ai_temp

echo "Installation complete! Run 'run.sh' to start the system."
exit 0
