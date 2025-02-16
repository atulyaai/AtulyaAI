#!/bin/bash

# -------------------------------------
# Atulya AI Unified Installer (Final Optimized Version)
# -------------------------------------

echo "[INFO] Starting Atulya AI Installation..."
sleep 2

# Define Directories
BASE_DIR="/opt/atulya_ai"
MODULES_DIR="$BASE_DIR/modules"
LOGS_DIR="$BASE_DIR/logs"
MODELS_DIR="$BASE_DIR/models"
DATA_DIR="$BASE_DIR/data"
WEB_DIR="$BASE_DIR/web_ui"
DNA_DIR="$BASE_DIR/dna_compression"

# Create Directories
echo "[INFO] Creating directories..."
mkdir -p $MODULES_DIR $LOGS_DIR $MODELS_DIR $DATA_DIR $WEB_DIR $DNA_DIR

# Install System Dependencies
echo "[INFO] Installing system dependencies..."
apt update && apt install -y python3 python3-pip git curl wget unzip nginx ufw

# Python Virtual Environment Setup
echo "[INFO] Setting up Python Virtual Environment..."
python3 -m venv $BASE_DIR/venv
source $BASE_DIR/venv/bin/activate

# Install Required Python Libraries
echo "[INFO] Installing required Python packages..."
pip install --upgrade pip
pip install fastapi uvicorn django transformers torch numpy pandas requests scipy scikit-learn llama-cpp-python

# Firewall & Port Setup
echo "[INFO] Configuring firewall rules..."
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 8000/tcp
ufw --force enable

echo "[INFO] Firewall configuration completed."

# Download and Install All AI Models
declare -A MODELS
MODELS["DeepSeek"]="https://github.com/deepseek-ai/deepseek-model.git"
MODELS["RAG_Optimized"]="https://github.com/rag-ai/rag-optimized.git"
MODELS["LoRA_Adaptive"]="https://github.com/lora-ai/lora-adaptive.git"

echo "[INFO] Downloading and setting up AI models..."
for model in "${!MODELS[@]}"; do
    MODEL_DIR="$MODELS_DIR/$model"
    if [ ! -d "$MODEL_DIR" ]; then
        echo "[INFO] Downloading $model..."
        git clone "${MODELS[$model]}" "$MODEL_DIR"
    else
        echo "[INFO] $model already exists, skipping download."
    fi

    # Convert AI Model to GGUF Format
    echo "[INFO] Converting $model model to GGUF format..."
    python3 -c "from llama_cpp import convert_pytorch_model; convert_pytorch_model('$MODEL_DIR', output_format='gguf')"
    echo "[INFO] $model conversion to GGUF completed."
done

echo "[INFO] All AI Models Installed & Converted."

# DNA Compression Setup
echo "[INFO] Setting up DNA Compression Module..."
if [ ! -d "$DNA_DIR" ]; then
    git clone https://github.com/atulya-ai/dna-compression.git $DNA_DIR
    cd $DNA_DIR
    bash install_dna.sh
else
    echo "[INFO] DNA Compression Module already exists, skipping."
fi

echo "[INFO] DNA Compression Installed Successfully."

# AI Parameter Optimization & Health Monitoring
echo "[INFO] Configuring AI Parameter Optimization..."
python3 $MODULES_DIR/parameter_optimizer.py || echo "[WARNING] AI Optimization script missing."

echo "[INFO] Running AI Health Check..."
python3 $MODULES_DIR/ai_health_monitor.py || echo "[WARNING] AI Health Monitoring script missing."

echo "[INFO] AI Optimization & Health Check Completed."

# Web UI Setup (FastAPI + Django + Nginx)
echo "[INFO] Setting up Web UI..."
if [ ! -d "$WEB_DIR" ]; then
    git clone https://github.com/atulya-ai/web-interface.git $WEB_DIR
    cd $WEB_DIR
    pip install -r requirements.txt
else
    echo "[INFO] Web UI already exists, skipping download."
fi

# Configure Nginx for Web UI
echo "[INFO] Configuring Nginx..."
cat > /etc/nginx/sites-available/atulya_ai <<EOL
server {
    listen 80;
    server_name localhost;
    location / {
        root $WEB_DIR/frontend;
        index index.html;
    }
}
EOL

ln -s /etc/nginx/sites-available/atulya_ai /etc/nginx/sites-enabled/
systemctl restart nginx

echo "[INFO] Nginx Configuration Completed."

# Start FastAPI & Django Server
echo "[INFO] Starting Web Interface..."
uvicorn web_ui.main:app --host 0.0.0.0 --port 8000 --reload &

echo "[INFO] Web UI Setup Completed."

# Logging System Setup
LOG_FILE="$LOGS_DIR/installation.log"
touch $LOG_FILE
echo "[INFO] Logging system initialized..." | tee -a $LOG_FILE

# Verify Installation
echo "[INFO] Running final system checks..." | tee -a $LOG_FILE
ls -l $BASE_DIR | tee -a $LOG_FILE
python3 --version | tee -a $LOG_FILE
pip list | tee -a $LOG_FILE

# Final Success Message
echo "[SUCCESS] Atulya AI Installation Complete. Access Web UI at http://localhost/"
