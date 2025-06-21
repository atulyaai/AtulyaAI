#!/usr/bin/env python3
"""
AtulyaAI Web Interface Startup Script
Launches the complete web application with frontend and backend
"""

import os
import sys
import subprocess
import threading
import time
import webbrowser
from pathlib import Path

def start_backend():
    """Start the Flask backend server"""
    print("🚀 Starting Flask backend...")
    backend_path = Path("web_ui/backend/app.py")
    if backend_path.exists():
        subprocess.run([sys.executable, str(backend_path)], cwd=os.getcwd())
    else:
        print("❌ Backend not found!")

def start_frontend():
    """Start the React frontend development server"""
    print("🎨 Starting React frontend...")
    frontend_path = Path("web_ui/frontend")
    if frontend_path.exists():
        # Install dependencies if needed
        if not (frontend_path / "node_modules").exists():
            print("📦 Installing frontend dependencies...")
            subprocess.run(["npm", "install"], cwd=frontend_path)
        
        # Start development server
        subprocess.run(["npm", "start"], cwd=frontend_path)
    else:
        print("❌ Frontend not found!")

def main():
    print("🤖 AtulyaAI Web Interface")
    print("=" * 50)
    
    # Check if required files exist
    backend_exists = Path("web_ui/backend/app.py").exists()
    frontend_exists = Path("web_ui/frontend/package.json").exists()
    
    if not backend_exists:
        print("❌ Backend not found! Please ensure web_ui/backend/app.py exists.")
        return
    
    if not frontend_exists:
        print("❌ Frontend not found! Please ensure web_ui/frontend/package.json exists.")
        return
    
    print("✅ All components found!")
    print("\n🌐 Starting AtulyaAI Web Interface...")
    print("📡 Backend will be available at: http://localhost:5000")
    print("🎨 Frontend will be available at: http://localhost:3000")
    print("🔗 Opening browser in 5 seconds...")
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=start_backend, daemon=True)
    backend_thread.start()
    
    # Wait a moment for backend to start
    time.sleep(3)
    
    # Start frontend in a separate thread
    frontend_thread = threading.Thread(target=start_frontend, daemon=True)
    frontend_thread.start()
    
    # Wait and open browser
    time.sleep(5)
    try:
        webbrowser.open("http://localhost:3000")
        print("🌐 Browser opened!")
    except:
        print("⚠️  Could not open browser automatically. Please visit http://localhost:3000")
    
    print("\n🔄 Press Ctrl+C to stop all servers...")
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down AtulyaAI...")
        print("👋 Goodbye!")

if __name__ == '__main__':
    main() 