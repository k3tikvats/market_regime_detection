import subprocess
import os
import sys
import time
import threading
import webbrowser
from pathlib import Path

# Get project root directory
ROOT_DIR = Path(__file__).parent

def run_backend():
    # Check if uvicorn is installed
    try:
        import uvicorn
        import fastapi
    except ImportError:
        print("\n[ERROR] Missing backend dependencies.")
        print("Please install required packages with:")
        print("pip install -r backend/api/requirements.txt\n")
        return False
    
    os.environ["PYTHONPATH"] = str(ROOT_DIR)
    
    # Change working directory to backend/api
    os.chdir(ROOT_DIR / "backend" / "api")
    
    # Start backend server
    print("\n[INFO] Starting backend server...")
    try:
        # Create and run the backend process
        backend_process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        
        # Wait for backend to start
        time.sleep(2)
        if backend_process.poll() is not None:
            print("\n[ERROR] Failed to start backend server.")
            return False
            
        print("\n[SUCCESS] Backend API running at http://localhost:8000")
        print("\n[INFO] API Documentation available at http://localhost:8000/docs")
        return backend_process
    except Exception as e:
        print(f"\n[ERROR] Failed to start backend: {str(e)}")
        return False

def run_frontend():
    # Change working directory to frontend
    os.chdir(ROOT_DIR / "frontend")
    
    # Check if npm is installed
    try:
        subprocess.run(["npm", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("\n[ERROR] npm is not installed or not in PATH.")
        print("Please install Node.js and npm to run the frontend.")
        return False
    
    # Start frontend server
    print("\n[INFO] Starting frontend development server...")
    try:
        # Create and run the frontend process
        frontend_process = subprocess.Popen(
            ["npm", "run", "dev"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        
        # Wait for frontend to start
        time.sleep(5)
        if frontend_process.poll() is not None:
            print("\n[ERROR] Failed to start frontend server.")
            return False
            
        print("\n[SUCCESS] Frontend running at http://localhost:3000")
        return frontend_process
    except Exception as e:
        print(f"\n[ERROR] Failed to start frontend: {str(e)}")
        return False

def monitor_process(process, name):
    """Monitor and print output from a process"""
    for line in iter(process.stdout.readline, ''):
        print(f"[{name}] {line.strip()}")
    for line in iter(process.stderr.readline, ''):
        print(f"[{name} ERROR] {line.strip()}")

def main():
    print("=" * 60)
    print("Market Regime Detection - Application Launcher")
    print("=" * 60)
    
    # Start backend
    backend_process = run_backend()
    if not backend_process:
        print("\n[ERROR] Failed to start the backend server. Exiting...")
        sys.exit(1)
    
    # Monitor backend in a separate thread
    backend_thread = threading.Thread(target=monitor_process, args=(backend_process, "BACKEND"), daemon=True)
    backend_thread.start()
        
    # Start frontend
    frontend_process = run_frontend()
    if not frontend_process:
        print("\n[ERROR] Failed to start the frontend server. Exiting...")
        backend_process.terminate()
        sys.exit(1)
        
    # Monitor frontend in a separate thread
    frontend_thread = threading.Thread(target=monitor_process, args=(frontend_process, "FRONTEND"), daemon=True)
    frontend_thread.start()
    
    # Open browser after a short delay
    time.sleep(2)
    print("\n[INFO] Opening application in web browser...")
    webbrowser.open("http://localhost:3000")
    
    print("\n[INFO] Press Ctrl+C to stop the servers and exit")
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down servers...")
        frontend_process.terminate()
        backend_process.terminate()
        print("[INFO] Servers stopped. Goodbye!")

if __name__ == "__main__":
    main()