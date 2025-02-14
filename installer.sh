#!/bin/bash

# Set variables
MODEL_DIR="/opt/atulyaai/models"
DEEPSEEK_MODEL_URL="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B/resolve/main/deepseek14b_model.zip"
DEEPSEEK_MODEL_PATH="$MODEL_DIR/deepseek14b_model.zip"

# Check if the required directories exist, if not, create them
echo "Checking if the necessary directories exist..."

if [ ! -d "$MODEL_DIR" ]; then
    echo "Creating model directory: $MODEL_DIR"
    mkdir -p $MODEL_DIR
fi

# Download DeepSeek 14b model
echo "Downloading DeepSeek 14b model..."
wget -O $DEEPSEEK_MODEL_PATH $DEEPSEEK_MODEL_URL

# Check if download was successful
if [ $? -eq 0 ]; then
    echo "Model downloaded successfully."
else
    echo "Failed to download the model. Please check the URL."
    exit 1
fi

# Unzip the model
echo "Unzipping the model..."
unzip -o $DEEPSEEK_MODEL_PATH -d $MODEL_DIR

# Check if unzipping was successful
if [ $? -eq 0 ]; then
    echo "Model unzipped successfully."
else
    echo "Failed to unzip the model. Please check the file."
    exit 1
fi

# Set up a virtual environment
echo "Setting up the virtual environment..."

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Activate the virtual environment
source venv/bin/activate

# Install required Python packages
echo "Installing Python dependencies..."
pip install -r /opt/atulyaai/requirements.txt

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo "Python dependencies installed successfully."
else
    echo "Failed to install Python dependencies."
    exit 1
fi

# Start the application (replace with your actual start command)
echo "Starting the application..."
# Add the command to start your application here, e.g.
# python3 /opt/atulyaai/web_ui/backend/app.py

# Success message
echo "Atulya AI installation completed successfully!"
