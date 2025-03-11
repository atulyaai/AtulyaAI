import os
import subprocess

def install_system():
    print("[+] Updating system and installing dependencies...")
    os.system("sudo apt update && sudo apt upgrade -y")
    os.system("sudo apt install -y git curl python3 python3-pip python3-venv")
    
    print("[+] System setup complete.")

def setup_server():
    print("[+] Setting up AtulyaAI server...")
    
    if not os.path.exists("atulya_env"):
        os.system("python3 -m venv atulya_env")
    
    os.system("source atulya_env/bin/activate && pip install --upgrade pip")
    os.system("source atulya_env/bin/activate && pip install -r requirements.txt")
    
    print("[+] Server setup complete.")

def sync_with_github():
    print("[+] Syncing with GitHub...")
    
    repo_url = "git@github.com:atulyaai/AtulyaAI.git"
    repo_dir = os.path.expanduser("~/AtulyaAI")
    
    if not os.path.exists(repo_dir):
        os.system(f"git clone {repo_url} {repo_dir}")
    else:
        os.system(f"cd {repo_dir} && git pull origin main")
    
    print("[+] GitHub sync complete.")

def main():
    install_system()
    setup_server()
    sync_with_github()
    print("[✔] Installation complete! Run 'python3 server.py' to start AtulyaAI.")

if __name__ == "__main__":
    main()
