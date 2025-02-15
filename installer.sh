#!/bin/bash

# Variables
PROJECT_DIR="./AtulyaAI_v1.2"
REPO_URL="https://raw.githubusercontent.com/atulyaai/AtulyaAI/main"
LOG_DIR="$PROJECT_DIR/logs"
VENV_DIR="$PROJECT_DIR/venv"

# Error handling
handle_error() {
    echo "Error: $1"
    exit 1
}

# Check for root privileges
if [ "$EUID" -ne 0 ]; then
    handle_error "Please run as root."
fi

# Create project directory
echo "Setting up project directory..."
mkdir -p "$PROJECT_DIR" || handle_error "Failed to create project directory."
mkdir -p "$LOG_DIR" || handle_error "Failed to create logs directory."

# Install dependencies
echo "Installing dependencies..."
apt-get update || handle_error "Failed to update package list."
apt-get install -y python3 python3-venv python3-pip git curl || handle_error "Failed to install dependencies."

# Set up Python virtual environment
echo "Setting up Python virtual environment..."
python3 -m venv "$VENV_DIR" || handle_error "Failed to create virtual environment."
source "$VENV_DIR/bin/activate" || handle_error "Failed to activate virtual environment."

# Install Python packages
echo "Installing Python packages..."
pip install fastapi uvicorn django htmx torch transformers psutil pandas || handle_error "Failed to install Python packages."

# Download and set up DeepSeek models
echo "Downloading DeepSeek models..."
mkdir -p "$PROJECT_DIR/core/models" || handle_error "Failed to create models directory."
curl -o "$PROJECT_DIR/core/models/deepseek_14b.gguf" "$REPO_URL/core/models/deepseek_14b.gguf" || handle_error "Failed to download DeepSeek 14B model."
curl -o "$PROJECT_DIR/core/models/deepseek_70b.gguf" "$REPO_URL/core/models/deepseek_70b.gguf" || handle_error "Failed to download DeepSeek 70B model."

# Set up Web UI
echo "Setting up Web UI..."
mkdir -p "$PROJECT_DIR/web_ui/backend" || handle_error "Failed to create backend directory."
mkdir -p "$PROJECT_DIR/web_ui/frontend" || handle_error "Failed to create frontend directory."
mkdir -p "$PROJECT_DIR/web_ui/admin" || handle_error "Failed to create admin directory."

# Download Web UI files
curl -o "$PROJECT_DIR/web_ui/backend/app.py" "$REPO_URL/web_ui/backend/app.py" || handle_error "Failed to download backend script."
curl -o "$PROJECT_DIR/web_ui/frontend/index.html" "$REPO_URL/web_ui/frontend/index.html" || handle_error "Failed to download frontend template."
curl -o "$PROJECT_DIR/web_ui/admin/admin.html" "$REPO_URL/web_ui/admin/admin.html" || handle_error "Failed to download admin template."

# Set up core AI functionality
echo "Setting up core AI functionality..."
mkdir -p "$PROJECT_DIR/core/nlp" || handle_error "Failed to create NLP directory."
mkdir -p "$PROJECT_DIR/core/automation" || handle_error "Failed to create automation directory."
mkdir -p "$PROJECT_DIR/core/voice" || handle_error "Failed to create voice directory."

curl -o "$PROJECT_DIR/core/nlp/nlp_processor.py" "$REPO_URL/core/nlp/nlp_processor.py" || handle_error "Failed to download NLP script."
curl -o "$PROJECT_DIR/core/automation/task_scheduler.py" "$REPO_URL/core/automation/task_scheduler.py" || handle_error "Failed to download automation script."
curl -o "$PROJECT_DIR/core/voice/wake_word.py" "$REPO_URL/core/voice/wake_word.py" || handle_error "Failed to download voice script."

# Set up modules
echo "Setting up modules..."
mkdir -p "$PROJECT_DIR/modules/cybersecurity" || handle_error "Failed to create cybersecurity directory."
mkdir -p "$PROJECT_DIR/modules/smart_home" || handle_error "Failed to create smart home directory."
mkdir -p "$PROJECT_DIR/modules/file_tools" || handle_error "Failed to create file tools directory."
mkdir -p "$PROJECT_DIR/modules/automation" || handle_error "Failed to create automation directory."
mkdir -p "$PROJECT_DIR/modules/ai_learning" || handle_error "Failed to create AI learning directory."

curl -o "$PROJECT_DIR/modules/cybersecurity/threat_detection.py" "$REPO_URL/modules/cybersecurity/threat_detection.py" || handle_error "Failed to download cybersecurity script."
curl -o "$PROJECT_DIR/modules/smart_home/iot_control.py" "$REPO_URL/modules/smart_home/iot_control.py" || handle_error "Failed to download smart home script."
curl -o "$PROJECT_DIR/modules/file_tools/file_manager.py" "$REPO_URL/modules/file_tools/file_manager.py" || handle_error "Failed to download file tools script."
curl -o "$PROJECT_DIR/modules/automation/os_automation.py" "$REPO_URL/modules/automation/os_automation.py" || handle_error "Failed to download automation script."
curl -o "$PROJECT_DIR/modules/ai_learning/rag_lora.py" "$REPO_URL/modules/ai_learning/rag_lora.py" || handle_error "Failed to download AI learning script."

# Set up logs, data, and updates
echo "Setting up logs, data, and updates..."
mkdir -p "$PROJECT_DIR/data" || handle_error "Failed to create data directory."
mkdir -p "$PROJECT_DIR/updates" || handle_error "Failed to create updates directory."

curl -o "$PROJECT_DIR/logs/system_logs.py" "$REPO_URL/logs/system_logs.py" || handle_error "Failed to download logging script."
curl -o "$PROJECT_DIR/data/data_manager.py" "$REPO_URL/data/data_manager.py" || handle_error "Failed to download data management script."
curl -o "$PROJECT_DIR/updates/auto_updater.py" "$REPO_URL/updates/auto_updater.py" || handle_error "Failed to download auto-updater script."

# Set permissions
echo "Setting permissions..."
chmod -R 755 "$PROJECT_DIR" || handle_error "Failed to set permissions."

echo "Installation completed successfully!"
echo "Access the web-based control panel at: http://localhost:5000"
