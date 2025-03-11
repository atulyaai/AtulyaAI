#!/bin/bash

# Set project root directory
PROJECT_DIR="$HOME/AtulyaAI"

# Colors for terminal output
GREEN="\033[92m"
RED="\033[91m"
RESET="\033[0m"

# Ensure script is run as root
if [[ $EUID -ne 0 ]]; then
    echo -e "${RED}⚠️ This script must be run as root. Try running with sudo.${RESET}"
    exit 1
fi

# Function to run commands with error handling
run_command() {
    echo -e "${GREEN}Running: $2...${RESET}"
    if eval "$1"; then
        echo -e "${GREEN}✔ Success: $2${RESET}\n"
    else
        echo -e "${RED}✖ Error: $2 failed.${RESET}\n"
        exit 1
    fi
}

# Install system dependencies
run_command "apt update && apt install -y python3 python3-pip python3-venv git wget curl build-essential" "Installing system dependencies"

# Create project directory
mkdir -p "$PROJECT_DIR" && cd "$PROJECT_DIR" || exit

# Setup virtual environment
if [ ! -d "atulya_env" ]; then
    run_command "python3 -m venv atulya_env && source atulya_env/bin/activate && pip install --upgrade pip" "Setting up Python virtual environment"
fi
echo -e "${GREEN}✔ Virtual environment is ready. Activate it with: source $PROJECT_DIR/atulya_env/bin/activate${RESET}"

# Install Python libraries
run_command "source atulya_env/bin/activate && pip install torch transformers accelerate bitsandbytes pydantic fastapi" "Installing Python libraries"

# Configure environment variables
cat <<EOL > ".env"
PYTHONUNBUFFERED=1
TOKENIZERS_PARALLELISM=false
TRANSFORMERS_CACHE=$PROJECT_DIR/models/cache
EOL
echo -e "${GREEN}✔ Environment variables configured.${RESET}"

# Create project structure and placeholder files
mkdir -p install src/{core/{ai,data,utils},web,security,monitoring} tests/{unit,integration} configs scripts logs models datasets backups
touch install/{install_system.sh,setup_server.sh} \
      src/core/ai/{model_loader.py,fine_tuning.py} \
      src/core/data/{dataset_manager.py,compression.py} \
      src/core/utils/{logger.py,error_handling.py} \
      src/web/{api.py,admin_dashboard.py} \
      src/security/{firewall.py,malware_detector.py} \
      src/monitoring/{health_monitor.py,log_manager.py} \
      tests/unit/test_sample.py tests/integration/test_sample.py \
      configs/{ai_config.yaml,paths.yaml} \
      scripts/{backup_manager.py,task_scheduler.py} \
      logs/.gitkeep models/.gitkeep datasets/.gitkeep backups/.gitkeep

# Create README file
cat <<EOL > "README.md"
# 🚀 AtulyaAI

AtulyaAI is an advanced AI system for automation, security, and AI-powered decision-making.

## 📜 Features
✅ Automated Installation  
✅ Self-Updating  
✅ Modular & Expandable  
✅ Web-Based Management  

## 🛠️ Installation
\`\`\`bash
curl -o installer.sh https://raw.githubusercontent.com/atulyaai/AtulyaAI/main/installer.sh && chmod +x installer.sh && sudo ./installer.sh
\`\`\`
EOL
echo -e "${GREEN}✔ Project structure and README created.${RESET}"

# Initialize Git repository if not already initialized
if [ ! -d ".git" ]; then
    run_command "git init && git add . && git commit -m 'Initial commit with project structure'" "Initializing Git repository"
fi

echo -e "${GREEN}🎉 Installation complete! Navigate to $PROJECT_DIR and push your code to GitHub.${RESET}"
echo -e "${GREEN}Activate the virtual environment with: source $PROJECT_DIR/atulya_env/bin/activate${RESET}"
