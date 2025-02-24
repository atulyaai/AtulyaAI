# AtulyaAI Installer
🚀 **About AtulyaAI**  
AtulyaAI is an advanced AI system designed for automation, security, smart home integration, and AI-powered decision-making. This installer provides a one-click setup for Ubuntu 22.04 & 24.04.

## 📜 Features
✅ **Automated Installation** – Sets up AtulyaAI with all dependencies.  
✅ **Self-Updating** – Updates every 8 hours to stay optimized.  
✅ **Modular & Expandable** – Supports adding AI models & features.  
✅ **Web-Based Management** – No SSH needed, manage everything from UI.  

## 🛠️ Installation
### 1️⃣ One-Click Install (Recommended)
Run this command in your terminal:
```
curl -o installer.sh https://raw.githubusercontent.com/atulyaai/AtulyaAI/main/installer.sh && chmod +x installer.sh && sudo ./installer.sh
```
### 2️⃣ Manual Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/atulyaai/AtulyaAI.git /opt/atulyaai
   cd /opt/atulyaai
   ```
2. Install dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```
3. Start AtulyaAI:
   ```bash
   python3 server.py
   ```
