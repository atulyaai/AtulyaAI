#!/bin/bash

# Web UI Installer
# Purpose: Install frontend, backend, and admin panel

# Exit on error
set -e

# Log file
LOG_FILE="/var/log/atulyaai_web_ui_install.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Install Dependencies
echo "Installing Web UI dependencies..."
sudo apt-get install -y docker.io docker-compose

# Set Up Docker Containers
echo "Setting up Docker containers..."
sudo docker-compose up -d

echo "Web UI installation complete!"
