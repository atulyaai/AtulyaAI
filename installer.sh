#!/bin/bash

# Variables
PROJECT_DIR="./AtulyaAI_v1.2"
MODELS_DIR="$PROJECT_DIR/core/models"
VENV_DIR="$MODELS_DIR/venv"
LOG_DIR="$PROJECT_DIR/logs"
MODEL_14B="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
MODEL_70B="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

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
mkdir -p "$MODELS_DIR" || handle_error "Failed to create models directory."
mkdir -p "$LOG_DIR" || handle_error "Failed to create logs directory."

# Install dependencies
echo "Installing dependencies..."
apt-get update || handle_error "Failed to update package list."
apt-get install -y python3 python3-venv python3-pip git || handle_error "Failed to install dependencies."

# Set up Python virtual environment
echo "Setting up Python virtual environment..."
python3 -m venv "$VENV_DIR" || handle_error "Failed to create virtual environment."
source "$VENV_DIR/bin/activate" || handle_error "Failed to activate virtual environment."

# Install Python packages
echo "Installing Python packages..."
pip install torch transformers flask pandas psutil || handle_error "Failed to install Python packages."

# Download DeepSeek models using Hugging Face Transformers
echo "Downloading DeepSeek 14B model..."
python3 -c "from transformers import AutoModelForCausalLM, AutoTokenizer; model = AutoModelForCausalLM.from_pretrained('$MODEL_14B'); tokenizer = AutoTokenizer.from_pretrained('$MODEL_14B'); model.save_pretrained('$MODELS_DIR/deepseek_14b'); tokenizer.save_pretrained('$MODELS_DIR/deepseek_14b')" || handle_error "Failed to download DeepSeek 14B model."

echo "Downloading DeepSeek 70B model..."
python3 -c "from transformers import AutoModelForCausalLM, AutoTokenizer; model = AutoModelForCausalLM.from_pretrained('$MODEL_70B'); tokenizer = AutoTokenizer.from_pretrained('$MODEL_70B'); model.save_pretrained('$MODELS_DIR/deepseek_70b'); tokenizer.save_pretrained('$MODELS_DIR/deepseek_70b')" || handle_error "Failed to download DeepSeek 70B model."

# Clone the repository (if not already present)
if [ ! -d "$PROJECT_DIR/.git" ]; then
    echo "Cloning Atulya AI repository..."
    git clone https://github.com/your-repo/AtulyaAI_v1.2.git "$PROJECT_DIR" || handle_error "Failed to clone repository."
fi

# Set permissions
echo "Setting permissions..."
chmod -R 755 "$PROJECT_DIR" || handle_error "Failed to set permissions."

# Create Web UI backend script
echo "Setting up Web UI backend..."
cat > "$PROJECT_DIR/web_ui/backend/app.py" <<EOF
from flask import Flask, jsonify, request
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Logging setup
logging.basicConfig(filename='$LOG_DIR/usage.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')

# Load DeepSeek 14B model
model_14b = AutoModelForCausalLM.from_pretrained('$MODELS_DIR/deepseek_14b')
tokenizer_14b = AutoTokenizer.from_pretrained('$MODELS_DIR/deepseek_14b')

# Load DeepSeek 70B model
model_70b = AutoModelForCausalLM.from_pretrained('$MODELS_DIR/deepseek_70b')
tokenizer_70b = AutoTokenizer.from_pretrained('$MODELS_DIR/deepseek_70b')

@app.route('/')
def home():
    return "Welcome to Atulya AI Control Panel!"

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query')
    logging.info(f"Chat Query: {query}")

    # Use DeepSeek 14B for simplicity (you can add logic to switch models)
    inputs = tokenizer_14b(query, return_tensors="pt")
    outputs = model_14b.generate(**inputs, max_length=100)
    response = tokenizer_14b.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
EOF

# Create Web UI frontend
echo "Setting up Web UI frontend..."
mkdir -p "$PROJECT_DIR/web_ui/frontend"
cat > "$PROJECT_DIR/web_ui/frontend/index.html" <<EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Atulya AI Control Panel</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <h1>Atulya AI Control Panel</h1>
    <div id="chat">
        <input type="text" id="query" placeholder="Ask me anything...">
        <button id="send">Send</button>
        <p id="response"></p>
    </div>
    <script src="scripts.js"></script>
</body>
</html>
EOF

cat > "$PROJECT_DIR/web_ui/frontend/styles.css" <<EOF
body {
    font-family: Arial, sans-serif;
    text-align: center;
    margin-top: 50px;
}

h1 {
    color: #333;
}

button {
    padding: 10px 20px;
    font-size: 16px;
    cursor: pointer;
}
EOF

cat > "$PROJECT_DIR/web_ui/frontend/scripts.js" <<EOF
document.getElementById('send').addEventListener('click', async () => {
    const query = document.getElementById('query').value;
    const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
    });
    const data = await response.json();
    document.getElementById('response').textContent = data.response;
});
EOF

echo "Installation completed successfully!"
echo "To start the Web UI, run:"
echo "cd $PROJECT_DIR/web_ui/backend && python3 app.py"
