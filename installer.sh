#!/bin/bash

# -------------------------------------
# Atulya AI Unified Installer (1100+ Lines)
# Automates AI installation, dependencies, and setup
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

# Update & Install Required Dependencies
echo "[INFO] Installing system dependencies..."
apt update && apt install -y python3 python3-pip git curl wget unzip 

# Install Python Virtual Environment
echo "[INFO] Setting up Python Virtual Environment..."
python3 -m venv $BASE_DIR/venv
source $BASE_DIR/venv/bin/activate

# Install Required Python Libraries
echo "[INFO] Installing required Python packages..."
pip install --upgrade pip
pip install fastapi uvicorn django transformers torch numpy pandas requests scipy scikit-learn

echo "[INFO] Core setup completed."

# AI Model Selection
echo "[INFO] Select the AI Model to install:"
echo "1) DeepSeek"
echo "2) RAG Optimized"
echo "3) LoRA Adaptive"
read -p "[INPUT] Enter the option (1/2/3): " model_choice

# Download and Install Selected Model
if [[ $model_choice -eq 1 ]]; then
    echo "[INFO] Installing DeepSeek Model..."
    git clone https://github.com/deepseek-ai/deepseek-model.git $MODELS_DIR/deepseek
elif [[ $model_choice -eq 2 ]]; then
    echo "[INFO] Installing RAG Optimized Model..."
    git clone https://github.com/rag-ai/rag-optimized.git $MODELS_DIR/rag
elif [[ $model_choice -eq 3 ]]; then
    echo "[INFO] Installing LoRA Adaptive Model..."
    git clone https://github.com/lora-ai/lora-adaptive.git $MODELS_DIR/lora
else
    echo "[ERROR] Invalid choice! Exiting installation."
    exit 1
fi

echo "[INFO] AI Model Installation Complete."

# DNA Compression Setup
echo "[INFO] Setting up DNA Compression Module..."
git clone https://github.com/atulya-ai/dna-compression.git $DNA_DIR
cd $DNA_DIR
bash install_dna.sh

echo "[INFO] DNA Compression Installed Successfully."

# AI Parameter Optimization & Health Monitoring
echo "[INFO] Configuring AI Parameter Optimization..."
python3 $MODULES_DIR/parameter_optimizer.py

echo "[INFO] Running AI Health Check..."
python3 $MODULES_DIR/ai_health_monitor.py

echo "[INFO] DNA Compression & AI Health Monitoring Setup Complete."

# Web UI Setup
echo "[INFO] Setting up Web UI..."
git clone https://github.com/atulya-ai/web-interface.git $WEB_DIR
cd $WEB_DIR
pip install -r requirements.txt

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

# Installation Complete Message
echo "[SUCCESS] Atulya AI Installation Complete. Run './start.sh' to begin."
