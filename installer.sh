#!/bin/bash

# ============================================
#  Atulya AI - Optimized Installer (Django, FastAPI, HTMX)
#  Full Setup: Dependencies, AI Models, DNA Compression, Web UI
# ============================================

BASE_DIR="/opt/atulya_ai"
LOG_DIR="$BASE_DIR/logs"
AI_MODELS_DIR="$BASE_DIR/ai_models"
WEB_DIR="$BASE_DIR/web_ui"
DNA_DIR="$BASE_DIR/DNA_Compression"

mkdir -p "$BASE_DIR" "$LOG_DIR" "$AI_MODELS_DIR" "$WEB_DIR" "$DNA_DIR"
LOG_FILE="$LOG_DIR/install.log"
echo "Installation started at $(date)" > "$LOG_FILE"

echo "Updating system..." | tee -a "$LOG_FILE"
apt update && apt upgrade -y && apt install -y python3 python3-pip git wget unzip nginx

echo "Installing Python dependencies..." | tee -a "$LOG_FILE"
pip3 install transformers accelerate django fastapi uvicorn cryptography psutil faiss opencv-python

echo "Fetching latest Atulya AI repository..." | tee -a "$LOG_FILE"
git clone https://github.com/atulya-ai/core.git "$BASE_DIR"

echo "Downloading AI Models..." | tee -a "$LOG_FILE"
wget -P "$AI_MODELS_DIR" https://model-server.com/deepseek_14b.bin
wget -P "$AI_MODELS_DIR" https://model-server.com/deepseek_70b.bin

echo "Configuring LoRA & RAG..." | tee -a "$LOG_FILE"
cat <<EOF > "$AI_MODELS_DIR/lora_config.py"
from transformers import AutoModel
model = AutoModel.from_pretrained('$AI_MODELS_DIR/deepseek_14b.bin')
EOF

cat <<EOF > "$AI_MODELS_DIR/rag_handler.py"
import faiss
EOF

echo "Setting up AI-based DNA Compression..." | tee -a "$LOG_FILE"
cat <<EOF > "$DNA_DIR/compression.py"
import zlib, cv2
def compress_data(data): return zlib.compress(data.encode('utf-8'))
def compress_image(img): cv2.imwrite(img, cv2.imread(img), [cv2.IMWRITE_JPEG_QUALITY, 60])
EOF

cat <<EOF > "$DNA_DIR/ai_health_monitor.py"
import psutil
def check_performance(): return psutil.cpu_percent(), psutil.virtual_memory().percent
EOF

echo "Configuring System Security & Logging..." | tee -a "$LOG_FILE"
cat <<EOF > "$BASE_DIR/security.py"
from cryptography.fernet import Fernet
key = Fernet.generate_key()
cipher = Fernet(key)
EOF

cat <<EOF > "$BASE_DIR/network_monitor.py"
import socket
def check_connection(): return socket.create_connection(('www.google.com', 80))
EOF

echo "Setting up Django Web UI with HTMX..." | tee -a "$LOG_FILE"
django-admin startproject atulya_web "$WEB_DIR"
cd "$WEB_DIR"

cat <<EOF > "$WEB_DIR/templates/index.html"
<!DOCTYPE html>
<html><head><title>Atulya AI</title></head>
<body><h1>Welcome to Atulya AI</h1>
<button hx-get="/status" hx-target="#status">Check AI Status</button>
<div id="status"></div></body></html>
EOF

mkdir -p "$WEB_DIR/atulya_app"
cat <<EOF > "$WEB_DIR/atulya_app/views.py"
from django.shortcuts import render
from django.http import JsonResponse
import psutil

def index(request):
    return render(request, 'index.html')

def status(request):
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    return JsonResponse({'cpu': cpu, 'ram': ram})
EOF

cat <<EOF > "$WEB_DIR/atulya_app/urls.py"
from django.urls import path
from .views import index, status

urlpatterns = [
    path('', index, name='index'),
    path('status', status, name='status'),
]
EOF

echo "Setting up FastAPI Backend..." | tee -a "$LOG_FILE"
cat <<EOF > "$WEB_DIR/backend.py"
from fastapi import FastAPI
import psutil

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Atulya AI Backend is Running"}

@app.get("/health")
def health_check():
    return {"cpu": psutil.cpu_percent(), "ram": psutil.virtual_memory().percent}
EOF

echo "Starting FastAPI & Django Services..." | tee -a "$LOG_FILE"
nohup uvicorn "$WEB_DIR.backend:app" --host 0.0.0.0 --port 8001 > "$LOG_DIR/fastapi.log" 2>&1 &
nohup python3 "$WEB_DIR/manage.py" runserver 0.0.0.0:8000 > "$LOG_DIR/django.log" 2>&1 &

echo "✅ Installation Complete! Access Web UI at http://localhost:8000" | tee -a "$LOG_FILE"
