#!/bin/bash

# Variables
PROJECT_DIR="./AtulyaAI_v1.2"
REPO_URL="https://raw.githubusercontent.com/atulyaai/AtulyaAI/main"
LOG_FILE="$PROJECT_DIR/installer.log"
VENV_DIR="$PROJECT_DIR/venv"
MODEL_DIR="$PROJECT_DIR/core/models"
QUANTIZED_MODEL_DIR="$MODEL_DIR/quantized"

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Error handling function
handle_error() {
    log "Error: $1"
    exit 1
}

# Check for root privileges
check_root() {
    if [ "$EUID" -ne 0 ]; then
        handle_error "Please run as root."
    fi
}

# Create project directory
setup_directories() {
    log "Setting up project directory..."
    mkdir -p "$PROJECT_DIR" || handle_error "Failed to create project directory."
    mkdir -p "$PROJECT_DIR/logs" || handle_error "Failed to create logs directory."
    mkdir -p "$QUANTIZED_MODEL_DIR" || handle_error "Failed to create quantized models directory."
}

# Install system dependencies
install_dependencies() {
    log "Installing dependencies..."
    apt-get update || handle_error "Failed to update package list."
    apt-get install -y python3 python3-venv python3-pip git curl || handle_error "Failed to install dependencies."
}

# Set up Python virtual environment
setup_venv() {
    log "Setting up Python virtual environment..."
    python3 -m venv "$VENV_DIR" || handle_error "Failed to create virtual environment."
    source "$VENV_DIR/bin/activate" || handle_error "Failed to activate virtual environment."
}

# Install Python packages
install_python_packages() {
    log "Installing Python packages..."
    pip install torch transformers llama-cpp-python ggml || handle_error "Failed to install Python packages."
}

# Download and quantize models
setup_models() {
    log "Downloading and quantizing models..."

    # Download the standard model (e.g., PyTorch model)
    mkdir -p "$MODEL_DIR" || handle_error "Failed to create models directory."
    curl -o "$MODEL_DIR/original_model.pth" "$REPO_URL/core/models/original_model.pth" || handle_error "Failed to download original model."

    # Quantize the model to 8-bit GGUF format
    log "Quantizing model to 8-bit GGUF format..."
    cat <<EOL > "$MODEL_DIR/quantize_model.py"
import torch
from transformers import AutoModelForCausalLM
from ggml import quantize

# Load the original model
model = AutoModelForCausalLM.from_pretrained("$MODEL_DIR/original_model.pth")

# Quantize the model to 8-bit
quantized_model = quantize(model, bits=8)

# Save the quantized model in GGUF format
quantized_model.save_pretrained("$QUANTIZED_MODEL_DIR/quantized_model_8bit.gguf")
EOL

    # Run the quantization script
    python3 "$MODEL_DIR/quantize_model.py" || handle_error "Failed to quantize model."

    log "Model quantized and saved to $QUANTIZED_MODEL_DIR/quantized_model_8bit.gguf"
}

# Set up Web UI
setup_web_ui() {
    log "Setting up Web UI..."
    mkdir -p "$PROJECT_DIR/web_ui/backend" || handle_error "Failed to create backend directory."
    mkdir -p "$PROJECT_DIR/web_ui/frontend" || handle_error "Failed to create frontend directory."
    mkdir -p "$PROJECT_DIR/web_ui/admin" || handle_error "Failed to create admin directory."

    log "Downloading Web UI files..."
    curl -o "$PROJECT_DIR/web_ui/backend/app.py" "$REPO_URL/web_ui/backend/app.py" || handle_error "Failed to download backend script."
    curl -o "$PROJECT_DIR/web_ui/frontend/index.html" "$REPO_URL/web_ui/frontend/index.html" || handle_error "Failed to download frontend template."
    curl -o "$PROJECT_DIR/web_ui/admin/admin.html" "$REPO_URL/web_ui/admin/admin.html" || handle_error "Failed to download admin template."
}

# Set up core AI functionality
setup_core_ai() {
    log "Setting up core AI functionality..."
    mkdir -p "$PROJECT_DIR/core/nlp" || handle_error "Failed to create NLP directory."
    mkdir -p "$PROJECT_DIR/core/automation" || handle_error "Failed to create automation directory."
    mkdir -p "$PROJECT_DIR/core/voice" || handle_error "Failed to create voice directory."

    log "Downloading core AI scripts..."
    curl -o "$PROJECT_DIR/core/nlp/nlp_processor.py" "$REPO_URL/core/nlp/nlp_processor.py" || handle_error "Failed to download NLP script."
    curl -o "$PROJECT_DIR/core/automation/task_scheduler.py" "$REPO_URL/core/automation/task_scheduler.py" || handle_error "Failed to download automation script."
    curl -o "$PROJECT_DIR/core/voice/wake_word.py" "$REPO_URL/core/voice/wake_word.py" || handle_error "Failed to download voice script."

    # Add code to load quantized GGUF models
    cat <<EOL > "$PROJECT_DIR/core/load_models.py"
from llama_cpp import Llama

def load_model(model_path):
    return Llama(model_path=model_path)

# Load the quantized 8-bit GGUF model
quantized_model = load_model("$QUANTIZED_MODEL_DIR/quantized_model_8bit.gguf")
EOL
}

# Set up modules
setup_modules() {
    log "Setting up modules..."
    mkdir -p "$PROJECT_DIR/modules/cybersecurity" || handle_error "Failed to create cybersecurity directory."
    mkdir -p "$PROJECT_DIR/modules/smart_home" || handle_error "Failed to create smart home directory."
    mkdir -p "$PROJECT_DIR/modules/file_tools" || handle_error "Failed to create file tools directory."
    mkdir -p "$PROJECT_DIR/modules/automation" || handle_error "Failed to create automation directory."
    mkdir -p "$PROJECT_DIR/modules/ai_learning" || handle_error "Failed to create AI learning directory."

    log "Downloading module scripts..."
    curl -o "$PROJECT_DIR/modules/cybersecurity/threat_detection.py" "$REPO_URL/modules/cybersecurity/threat_detection.py" || handle_error "Failed to download cybersecurity script."
    curl -o "$PROJECT_DIR/modules/smart_home/iot_control.py" "$REPO_URL/modules/smart_home/iot_control.py" || handle_error "Failed to download smart home script."
    curl -o "$PROJECT_DIR/modules/file_tools/file_manager.py" "$REPO_URL/modules/file_tools/file_manager.py" || handle_error "Failed to download file tools script."
    curl -o "$PROJECT_DIR/modules/automation/os_automation.py" "$REPO_URL/modules/automation/os_automation.py" || handle_error "Failed to download automation script."
    curl -o "$PROJECT_DIR/modules/ai_learning/rag_lora.py" "$REPO_URL/modules/ai_learning/rag_lora.py" || handle_error "Failed to download AI learning script."
}

# Set up logs, data, and updates
setup_logs_data_updates() {
    log "Setting up logs, data, and updates..."
    mkdir -p "$PROJECT_DIR/data" || handle_error "Failed to create data directory."
    mkdir -p "$PROJECT_DIR/updates" || handle_error "Failed to create updates directory."

    log "Downloading logs, data, and updates scripts..."
    curl -o "$PROJECT_DIR/logs/system_logs.py" "$REPO_URL/logs/system_logs.py" || handle_error "Failed to download logging script."
    curl -o "$PROJECT_DIR/data/data_manager.py" "$REPO_URL/data/data_manager.py" || handle_error "Failed to download data management script."
    curl -o "$PROJECT_DIR/updates/auto_updater.py" "$REPO_URL/updates/auto_updater.py" || handle_error "Failed to download auto-updater script."
}

# Set permissions
set_permissions() {
    log "Setting permissions..."
    chmod -R 755 "$PROJECT_DIR" || handle_error "Failed to set permissions."
}

# Main function
main() {
    check_root
    setup_directories
    install_dependencies
    setup_venv
    install_python_packages
    setup_models
    setup_web_ui
    setup_core_ai
    setup_modules
    setup_logs_data_updates
    set_permissions

    log "Installation completed successfully!"
    log "Access the web-based control panel at: http://localhost:5000"
}

# Run the main function
main
