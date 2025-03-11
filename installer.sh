#!/bin/bash

# Set project root directory
PROJECT_DIR="$HOME/AtulyaAI"

# Colors for terminal output
GREEN="\033[92m"
RED="\033[91m"
RESET="\033[0m"

# Function to check if the script is run with root privileges
check_root() {
    if [[ $EUID -ne 0 ]]; then
        echo -e "${RED}⚠️  This script must be run as root. Try running with sudo.${RESET}"
        exit 1
    fi
}

# Function to run commands with error handling
run_command() {
    local command="$1"
    local description="$2"
    echo -e "${GREEN}Running: ${description}...${RESET}"
    if eval "$command"; then
        echo -e "${GREEN}Success: ${description}${RESET}\n"
    else
        echo -e "${RED}Error: ${description} failed.${RESET}\n"
        exit 1
    fi
}

# Step 1: Install system dependencies
install_system_dependencies() {
    echo -e "${GREEN}📦 Installing system dependencies...${RESET}"
    run_command "sudo apt update && sudo apt install -y python3 python3-pip python3-venv git wget curl build-essential" "Installing system dependencies"
}

# Step 2: Check and set up Python virtual environment
setup_virtualenv() {
    echo -e "${GREEN}🐍 Setting up Python virtual environment...${RESET}"
    if [ ! -d "$PROJECT_DIR/atulya_env" ]; then
        run_command "python3 -m venv $PROJECT_DIR/atulya_env" "Creating virtual environment"
    fi
    echo -e "${GREEN}✅ Virtual environment is set up. Activate it using: source $PROJECT_DIR/atulya_env/bin/activate${RESET}"
}

# Step 3: Install Python libraries
install_python_libraries() {
    echo -e "${GREEN}📚 Installing Python libraries...${RESET}"
    local libraries=(
        "torch>=2.2.1"
        "transformers>=4.40.0"
        "accelerate>=0.29.3"
        "bitsandbytes>=0.43.0"
        "pydantic>=2.5.0"
        "fastapi>=0.109.0"
    )
    run_command "source $PROJECT_DIR/atulya_env/bin/activate && pip install --upgrade pip && pip install ${libraries[*]}" "Installing Python libraries"
}

# Step 4: Configure environment variables
configure_environment() {
    echo -e "${GREEN}⚙️ Configuring environment variables...${RESET}"
    cat <<EOL > "$PROJECT_DIR/.env"
PYTHONUNBUFFERED=1
TOKENIZERS_PARALLELISM=false
TRANSFORMERS_CACHE=$PROJECT_DIR/models/cache
EOL
    echo -e "${GREEN}✅ Environment variables configured in .env file.${RESET}"
}

# Step 5: Create project structure
create_project_structure() {
    echo -e "${GREEN}📂 Creating project structure...${RESET}"
    mkdir -p "$PROJECT_DIR/install" "$PROJECT_DIR/src/core/ai" "$PROJECT_DIR/src/core/data" "$PROJECT_DIR/src/core/utils" \
             "$PROJECT_DIR/src/web" "$PROJECT_DIR/src/security" "$PROJECT_DIR/src/monitoring" \
             "$PROJECT_DIR/tests/unit" "$PROJECT_DIR/tests/integration" "$PROJECT_DIR/configs" \
             "$PROJECT_DIR/scripts" "$PROJECT_DIR/logs" "$PROJECT_DIR/models" "$PROJECT_DIR/datasets" "$PROJECT_DIR/backups"

    # Create placeholder files
    touch "$PROJECT_DIR/install/install_system.sh" "$PROJECT_DIR/install/setup_server.sh"
    touch "$PROJECT_DIR/src/core/ai/model_loader.py" "$PROJECT_DIR/src/core/ai/fine_tuning.py"
    touch "$PROJECT_DIR/src/core/data/dataset_manager.py" "$PROJECT_DIR/src/core/data/compression.py"
    touch "$PROJECT_DIR/src/core/utils/logger.py" "$PROJECT_DIR/src/core/utils/error_handling.py"
    touch "$PROJECT_DIR/src/web/api.py" "$PROJECT_DIR/src/web/admin_dashboard.py"
    touch "$PROJECT_DIR/src/security/firewall.py" "$PROJECT_DIR/src/security/malware_detector.py"
    touch "$PROJECT_DIR/src/monitoring/health_monitor.py" "$PROJECT_DIR/src/monitoring/log_manager.py"
    touch "$PROJECT_DIR/tests/unit/test_sample.py" "$PROJECT_DIR/tests/integration/test_sample.py"
    touch "$PROJECT_DIR/configs/ai_config.yaml" "$PROJECT_DIR/configs/paths.yaml"
    touch "$PROJECT_DIR/scripts/backup_manager.py" "$PROJECT_DIR/scripts/task_scheduler.py"
    touch "$PROJECT_DIR/logs/.gitkeep" "$PROJECT_DIR/models/.gitkeep" "$PROJECT_DIR/datasets/.gitkeep" "$PROJECT_DIR/backups/.gitkeep"

    # Create README file
    cat <<EOL > "$PROJECT_DIR/README.md"
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

    echo -e "${GREEN}✅ Project structure created.${RESET}"
}

# Step 6: Initialize Git repository
initialize_git_repo() {
    echo -e "${GREEN}🛠️ Initializing Git repository...${RESET}"
    cd "$PROJECT_DIR" || exit

    if [ -d "$PROJECT_DIR/.git" ]; then
        echo -e "${GREEN}✅ Git repository already initialized.${RESET}"
    else
        git init
        git add .
        git commit -m "Initial commit with project structure"
        echo -e "${GREEN}✅ Git repository initialized.${RESET}"
    fi
}

# Main function
main() {
    echo -e "${GREEN}🚀 Starting AtulyaAI installation...${RESET}"

    # Ensure script is run with root privileges
    check_root

    # Create project directory
    mkdir -p "$PROJECT_DIR"
    cd "$PROJECT_DIR" || exit

    # Run installation steps
    install_system_dependencies
    setup_virtualenv
    install_python_libraries
    configure_environment
    create_project_structure
    initialize_git_repo

    echo -e "${GREEN}🎉 Installation complete! Navigate to $PROJECT_DIR and push your code to GitHub.${RESET}"
    echo -e "${GREEN}Activate the virtual environment with: source $PROJECT_DIR/atulya_env/bin/activate${RESET}"
}

# Execute main function
main
