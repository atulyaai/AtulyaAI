#!/bin/bash

# Set project root directory
PROJECT_DIR="$HOME/AtulyaAI"

# Create AtulyaAI directory
echo "📁 Creating AtulyaAI project directory at $PROJECT_DIR..."
mkdir -p "$PROJECT_DIR"

# Navigate to project directory
cd "$PROJECT_DIR" || exit

# Create required directories
echo "📂 Setting up project structure..."
mkdir -p install src/core/ai src/core/data src/core/utils src/web src/security src/monitoring tests/unit tests/integration configs scripts logs models datasets backups

# Create placeholder files
echo "📜 Adding placeholder scripts and configs..."
touch install/install_system.sh install/install_server.sh
touch src/core/ai/model_loader.py src/core/ai/fine_tuning.py
touch src/core/data/dataset_manager.py src/core/data/compression.py
touch src/core/utils/logger.py src/core/utils/error_handling.py
touch src/web/api.py src/web/admin_dashboard.py
touch src/security/firewall.py src/security/malware_detector.py
touch src/monitoring/health_monitor.py src/monitoring/log_manager.py
touch tests/unit/test_sample.py tests/integration/test_sample.py
touch configs/ai_config.yaml configs/paths.yaml
touch scripts/backup_manager.py scripts/task_scheduler.py
touch logs/.gitkeep models/.gitkeep datasets/.gitkeep backups/.gitkeep

# Create a README file
cat <<EOL > README.md
# 🚀 AtulyaAI

AtulyaAI is an advanced AI system designed for automation, security, smart home integration, and AI-powered decision-making.

## 📜 Features
✅ Automated Installation  
✅ Self-Updating  
✅ Modular & Expandable  
✅ Web-Based Management  

## 🛠️ Installation
Run the following command:
\`\`\`bash
curl -o installer.sh https://raw.githubusercontent.com/atulyaai/AtulyaAI/main/installer.sh && chmod +x installer.sh && sudo ./installer.sh
\`\`\`
EOL

# Initialize Git repository
echo "🛠️ Initializing Git repository..."
git init
git add .
git commit -m "Initial commit with project structure"

# Run system and server installation scripts
echo "🚀 Running installation scripts..."
chmod +x install/install_system.sh install/install_server.sh
./install/install_system.sh
./install/install_server.sh

echo "✅ Setup complete! Navigate to $PROJECT_DIR and push your code to GitHub."
