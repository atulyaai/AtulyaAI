#!/bin/bash

echo "Starting Atulya AI setup..."

# Update package lists
sudo apt update

# Install necessary dependencies
sudo apt install -y python3 python3-pip python3-venv curl ufw

# Create a Python virtual environment
echo "Creating virtual environment..."
python3 -m venv /opt/atulyaai/venv
source /opt/atulyaai/venv/bin/activate

# Upgrade pip inside the virtual environment
echo "Upgrading pip..."
pip install --upgrade pip

# Install required Python packages from requirements.txt
echo "Installing Python dependencies..."
pip install -r /opt/atulyaai/requirements.txt

# Apply database migrations for Django
echo "Applying database migrations..."
cd /opt/atulyaai/web_ui/backend
python3 manage.py migrate

# Allow traffic on port 8080 through the firewall (if UFW is enabled)
echo "Configuring firewall to allow port 8080..."
sudo ufw allow 8080/tcp

# Starting Django server on 0.0.0.0:8080
echo "Starting the Django development server..."
nohup python3 manage.py runserver 0.0.0.0:8080 &

# Check if Django server started successfully
sleep 5
if netstat -tuln | grep :8080; then
  echo "Django server started successfully on port 8080."
else
  echo "Error: Django server failed to start."
fi

# Completion message
echo "Atulya AI setup complete!"
echo "You can access the AI server at http://<your-server-ip>:8080"
