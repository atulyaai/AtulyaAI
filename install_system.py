#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

# Colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

def run_command(command: str, description: str = "") -> bool:
    """Executes a shell command and handles errors."""
    try:
        print(f"{GREEN}Running: {description}...{RESET}")
        subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"{GREEN}Success: {description}{RESET}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{RED}Error: {description} failed.{RESET}")
        print(f"Details: {e}")
        return False

def install_system_dependencies() -> bool:
    """Installs essential system dependencies, including python3-venv."""
    packages = "python3 python3-pip python3-venv git wget curl build-essential"
    return run_command(
        f"sudo apt update && sudo apt install -y {packages}",
        "Installing system dependencies"
    )

def setup_virtualenv() -> bool:
    """Sets up a Python virtual environment."""
    venv_dir = "atulya_env"
    if not Path(venv_dir).exists():
        if not run_command(f"python3 -m venv {venv_dir}", "Creating virtual environment"):
            return False
    print(f"{GREEN}Virtual environment exists. Activating it...{RESET}")
    
    # Update PATH to activate the virtual environment
    os.environ["PATH"] = f"{os.getcwd()}/{venv_dir}/bin:{os.environ['PATH']}"
    
    # Upgrade pip inside the virtual environment
    return run_command(f"{venv_dir}/bin/pip install --upgrade pip", "Upgrading pip")

def install_python_libraries() -> bool:
    """Installs required Python libraries inside the virtual environment."""
    libraries = [
        "torch>=2.2.1",
        "transformers>=4.40.0",
        "accelerate>=0.29.3",
        "bitsandbytes>=0.43.0",
        "pydantic>=2.5.0",
        "fastapi>=0.109.0"
    ]
    return run_command(
        f"atulya_env/bin/pip install {' '.join(libraries)}",
        "Installing Python libraries"
    )

def configure_environment() -> bool:
    """Configures environment variables."""
    env_vars = {
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_CACHE": "./models/cache"
    }
    with open(".env", "w") as f:
        for var, value in env_vars.items():
            f.write(f"{var}={value}\n")
    print(f"{GREEN}Environment variables configured in .env file.{RESET}")
    return True

def main():
    print(f"{GREEN}Starting Atulya AI System Installation...{RESET}\n")

    if not install_system_dependencies():
        sys.exit(1)

    if not setup_virtualenv():
        sys.exit(1)

    if not install_python_libraries():
        sys.exit(1)

    if not configure_environment():
        sys.exit(1)

    print(f"\n{GREEN}Installation complete! Activate the virtual environment with:{RESET}")
    print(f"source atulya_env/bin/activate")

if __name__ == "__main__":
    main()
