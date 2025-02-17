#!/bin/bash

# AtulyaAI_v1.3 Main Installer
# Purpose: Run all sub-installers for a complete setup

# Exit on error
set -e

# Log file
LOG_FILE="/var/log/atulyaai_install.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Sub-Installers
SUB_INSTALLERS=(
    "ai_core_installer.sh"
    "web_ui_installer.sh"
    "automation_installer.sh"
    "cybersecurity_installer.sh"
    "compression_installer.sh"
    "iot_control_installer.sh"
)

# Run Sub-Installers
for installer in "${SUB_INSTALLERS[@]}"; do
    echo "Running $installer..."
    if [ -f "./$installer" ]; then
        bash "./$installer"
    else
        echo "Error: $installer not found!"
        exit 1
    fi
done

echo "AtulyaAI_v1.3 installation complete!"
