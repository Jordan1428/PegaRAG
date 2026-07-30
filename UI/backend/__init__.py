import sys
import os
import time
import subprocess

def start_backend():
    print("🚀 啟動 FastAPI 後端伺服器 (Port 8000)...")
    subprocess.run([sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"])

def start_frontend():
    print("⏳ 等待 3 秒以確保後端準備就緒...")
    time.sleep(3)
    print("🌐 啟動 Gradio 前端 UI (Port 7860)...")
    subprocess.run([sys.executable, "frontend/app.py"])

if __name__ == "__main__":
    from threading import Thread
    t = Thread(target=start_backend, daemon=True)
    t.start()
    start_frontend()
