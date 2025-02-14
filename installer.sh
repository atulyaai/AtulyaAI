#!/bin/bash

# Install necessary packages
echo "Installing dependencies..."
sudo apt update
sudo apt install -y python3-pip python3-dev libpq-dev postgresql postgresql-contrib nginx curl

# Install virtual environment and activate it
echo "Setting up virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r /opt/atulyaai/web_ui/backend/requirements.txt

# Set up Django
echo "Setting up Django project..."
cd /opt/atulyaai/web_ui/backend

# Apply Django migrations (if you have any)
echo "Applying migrations..."
python manage.py migrate

# Create a superuser (optional but recommended)
echo "Creating superuser..."
python manage.py createsuperuser  # Follow the prompts to create a user

# Update systemd service to run Django app on boot
echo "Setting up system service..."

# Create a systemd service file for Django
sudo tee /etc/systemd/system/atulyaai_django.service > /dev/null <<EOF
[Unit]
Description=Atulya AI Django Server
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/opt/atulyaai/web_ui/backend
ExecStart=/opt/atulyaai/web_ui/backend/venv/bin/python3 /opt/atulyaai/web_ui/backend/manage.py runserver 0.0.0.0:8000
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable atulyaai_django
sudo systemctl start atulyaai_django

# Set up Nginx to reverse proxy to the Django app (optional but recommended)
echo "Setting up Nginx reverse proxy..."

# Create Nginx configuration for the project
sudo tee /etc/nginx/sites-available/atulyaai > /dev/null <<EOF
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# Enable the site and restart Nginx
sudo ln -s /etc/nginx/sites-available/atulyaai /etc/nginx/sites-enabled
sudo systemctl restart nginx

# Finish installation
echo "Django setup complete and service is running!"
