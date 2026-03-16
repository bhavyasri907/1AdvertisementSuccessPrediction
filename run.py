# run.py (updated version)
import subprocess
import sys
import os
import time
import signal
import platform

def print_header():
    print("=" * 70)
    print("🚀 Advertisement Success Prediction System")
    print("=" * 70)
    print(f"📊 Frontend: Streamlit on port 8501")
    print(f"🔌 Backend API: FastAPI on port 8000")
    print("=" * 70)

def run_command(command, description, shell=True):
    print(f"\n📌 {description}...")
    try:
        result = subprocess.run(command, shell=shell, check=True)
        print(f"✅ {description} completed")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}: {e}")
        return None

def main():
    print_header()
    
    # Check Python version
    python_version = sys.version_info
    print(f"🐍 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    run_command(f"{sys.executable} -m pip install -r requirements.txt", "Dependency installation")
    
    # Check if data directory exists
    if not os.path.exists("data"):
        os.makedirs("data", exist_ok=True)
        print("📁 Created data directory")
    
    # Fix column name in dataset
    print("\n🔧 Fixing dataset column name...")
    run_command(f"{sys.executable} fix_column_name.py", "Column name fix")
    
    # Train models
    print("\n🤖 Training models...")
    run_command(f"{sys.executable} train_model.py", "Model training")
    
    # Create model directory if it doesn't exist
    os.makedirs("model", exist_ok=True)
    
    print("\n" + "=" * 70)
    print("🚀 Starting services...")
    print("=" * 70)
    
    # Start FastAPI backend
    print("\n🌐 Starting FastAPI backend on http://localhost:8000")
    
    # Use different commands based on platform
    if platform.system() == "Windows":
        backend_process = subprocess.Popen(
            ["uvicorn", "api:app", "--reload", "--port", "8000", "--host", "0.0.0.0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        backend_process = subprocess.Popen(
            ["uvicorn", "api:app", "--reload", "--port", "8000", "--host", "0.0.0.0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            preexec_fn=os.setsid
        )
    
    # Wait for backend to start
    print("   Waiting for backend to start...", end="", flush=True)
    time.sleep(3)
    print(" ✅")
    
    # Check if backend is running
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', 8000))
    if result == 0:
        print("   Backend is accessible")
    else:
        print("   ⚠️ Backend might not be accessible yet")
    sock.close()
    
    # Start Streamlit frontend
    print("\n🎨 Starting Streamlit frontend on http://localhost:8501")
    frontend_process = subprocess.Popen(
        ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Wait for frontend to start
    time.sleep(2)
    
    print("\n" + "=" * 70)
    print("✅ SYSTEM IS RUNNING!")
    print("=" * 70)
    print("📊 Frontend: http://localhost:8501")
    print("🔌 Backend API: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("📋 API Schema: http://localhost:8000/redoc")
    print("=" * 70)
    print("\n📝 Available API endpoints:")
    print("   GET  /          - Health check")
    print("   GET  /health    - Detailed health check")
    print("   POST /predict   - Make prediction")
    print("   POST /predict-with-video - Predict with video analysis")
    print("   GET  /model-info - Get model information")
    print("   POST /analyze-video-only - Analyze video only")
    print("   GET  /feature-importance - Get feature importance")
    print("\n💡 Press Ctrl+C to stop all services\n")
    
    try:
        # Keep the script running
        while True:
            # Check if processes are still running
            if backend_process.poll() is not None:
                print("\n❌ Backend process stopped unexpectedly!")
                break
            if frontend_process.poll() is not None:
                print("\n❌ Frontend process stopped unexpectedly!")
                break
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping services...")
        
        # Terminate processes
        if platform.system() == "Windows":
            backend_process.terminate()
            frontend_process.terminate()
        else:
            # On Unix, use process groups to kill child processes
            try:
                os.killpg(os.getpgid(backend_process.pid), signal.SIGTERM)
                os.killpg(os.getpgid(frontend_process.pid), signal.SIGTERM)
            except:
                backend_process.terminate()
                frontend_process.terminate()
        
        # Wait for processes to terminate
        time.sleep(2)
        print("✅ All services stopped")
    
    print("\n👋 Goodbye!\n")

if __name__ == "__main__":
    main()