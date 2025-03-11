# AtulyaAI Installer

## 🚀 About AtulyaAI
AtulyaAI is an advanced AI system designed for automation, security, smart home integration, and AI-powered decision-making. This installer provides a one-click setup for **Ubuntu 22.04 & 24.04**.

## 📜 Features
✅ **Automated Installation** – Sets up AtulyaAI with all dependencies.
✅ **Self-Updating** – Updates every 8 hours to stay optimized.
✅ **Modular & Expandable** – Supports adding AI models & features.
✅ **Web-Based Management** – No SSH needed, manage everything from UI.

## 🛠️ Installation

### 1️⃣ One-Click Install (Recommended)
Run this command in your terminal:

```bash
curl -o installer.sh https://raw.githubusercontent.com/atulyaai/AtulyaAI/main/installer.sh && chmod +x installer.sh && sudo ./installer.sh
```

### 2️⃣ Manual Installation

#### Clone the repository:
```bash
git clone https://github.com/atulyaai/AtulyaAI.git /opt/atulyaai
cd /opt/atulyaai
```

#### Install dependencies:
```bash
pip3 install -r requirements.txt
```

#### Start AtulyaAI:
```bash
python3 server.py
```

## 📂 Directory Structure
```
AtulyaAI/
├── configs/          # Configuration files
├── install/          # Installation scripts
├── models/           # AI models
├── src/              # Core AI source code
├── scripts/          # Utility scripts
├── logs/             # Logs & debug info
├── README.md         # This document
└── installer.sh      # Auto-installation script
```

## 🌎 Stay Updated
AtulyaAI is updated every **8 hours** automatically. To manually update:
```bash
cd /opt/atulyaai
git pull origin main
```

## 🔗 Links
- **GitHub**: [https://github.com/atulyaai/AtulyaAI](https://github.com/atulyaai/AtulyaAI)
- **Documentation**: Coming soon!

## 🛠️ Troubleshooting
If you face any issues, run:
```bash
docker logs atulyaai --tail 100
```
Or open an issue on GitHub.

---
💡 **Developed by AtulyaAI Team**

