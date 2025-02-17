#!/bin/bash

# Cybersecurity Installer
# Purpose: Set up security, firewall, and monitoring

# Exit on error
set -e

# Log file
LOG_FILE="/var/log/atulyaai_cybersecurity_install.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Install Dependencies
echo "Installing cybersecurity dependencies..."
sudo apt-get install -y ufw fail2ban

# Set Up Firewall
echo "Setting up firewall..."
sudo ufw allow ssh
sudo ufw allow http
sudo ufw allow https
sudo ufw enable

# Set Up Fail2Ban
echo "Setting up Fail2Ban..."
sudo systemctl enable fail2ban
sudo systemctl start fail2ban

echo "Cybersecurity installation complete!"
