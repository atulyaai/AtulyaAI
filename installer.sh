#!/bin/bash

# Constants for Hugging Face Model Download URL (replace with your model name or URL)
MODEL_14B_NAME="facebook/opt-13b"  # Use 13B model as an example, replace with actual model
MODEL_70B_NAME="EleutherAI/gpt-j-6B"  # Use 6B model as an example, replace with actual model
MODEL_DIRECTORY="/home/atulya_ai/models"
INSTALL_DIR="/home/atulya_ai"
LOG_FILE="$INSTALL_DIR/installation.log"

# Function to install system dependencies
install_dependencies() {
    echo "Installing system dependencies..."
    sudo apt-get update -y
    sudo apt-get install -y python3-pip python3-venv git curl

    # Install Python dependencies
    echo "Installing Python dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
}

# Function to create necessary directories
create_directories() {
    echo "Creating required directories..."
    mkdir -p $MODEL_DIRECTORY
    mkdir -p $INSTALL_DIR/logs
}

# Function to download the models from Hugging Face or custom server
download_models() {
    echo "Downloading DeepSeek 14B model..."
    python3 -c "from transformers import AutoModel; model = AutoModel.from_pretrained('$MODEL_14B_NAME')" >> $LOG_FILE 2>&1

    echo "Downloading DeepSeek 70B model..."
    python3 -c "from transformers import AutoModel; model = AutoModel.from_pretrained('$MODEL_70B_NAME')" >> $LOG_FILE 2>&1

    echo "Model download complete!"
}

# Function to configure system for dynamic model switching
setup_dynamic_model_switching() {
    echo "Setting up dynamic model switching..."
    cat <<EOL > $INSTALL_DIR/core/models/model_selector.py
import psutil
from transformers import AutoModel

# Define the model names
MODEL_14B_NAME = "$MODEL_14B_NAME"
MODEL_70B_NAME = "$MODEL_70B_NAME"

# Function to select model based on system resources
def select_model():
    ram = psutil.virtual_memory().available / (1024 ** 3)  # Convert bytes to GB
    cpu_usage = psutil.cpu_percent()

    # Select model based on resources
    if ram > 16 and cpu_usage < 80:
        return AutoModel.from_pretrained(MODEL_70B_NAME)  # Use 70B for higher resource availability
    else:
        return AutoModel.from_pretrained(MODEL_14B_NAME)  # Use 14B when resources are low

print("Dynamic model selection is set up!")
EOL
}

# Function to set up the main installer and system configuration
setup_installer() {
    echo "Setting up installer script and configurations..."

    # Create config.ini for system configuration
    cat <<EOL > $INSTALL_DIR/config.ini
[system]
model_directory = "$MODEL_DIRECTORY"
log_file = "$LOG_FILE"
EOL

    # Set permissions
    chmod +x $INSTALL_DIR/installer.sh
}

# Function to log installation details
log_installation() {
    echo "Logging installation details..."
    echo "Installation started at $(date)" >> $LOG_FILE
    echo "Installation complete at $(date)" >> $LOG_FILE
}

# Main installer function
main_installation() {
    # Log start time
    log_installation

    # Install dependencies
    install_dependencies

    # Create necessary directories
    create_directories

    # Download models
    download_models

    # Set up dynamic model switching
    setup_dynamic_model_switching

    # Set up installer configuration
    setup_installer

    # Log completion
    log_installation

    echo "Installation completed successfully!"
    echo "Please check the logs at $LOG_FILE for more details."
}

# Run the installation process
main_installation
