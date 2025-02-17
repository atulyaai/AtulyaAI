#!/bin/bash

# Automation Installer
# Purpose: Set up self-learning, updates, and task automation

# Exit on error
set -e

# Log file
LOG_FILE="/var/log/atulyaai_automation_install.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Set Up Cron Job for Updates
echo "Setting up automatic updates..."
(crontab -l 2>/dev/null; echo "0 */8 * * * /opt/AtulyaAI/scripts/update.sh") | crontab -

# Set Up Self-Learning Scripts
echo "Setting up self-learning scripts..."
cp -r ./automation/self_learning /opt/AtulyaAI/

echo "Automation installation complete!"
