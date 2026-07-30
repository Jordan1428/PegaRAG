import sys
import os
import time
import socket
import threading
import subprocess
import webbrowser

def wait_for_port(port, timeout=60):
    """探測特定 Port 服務是否已就緒接收 TCP 連線"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except (OSError, ConnectionRefusedError):
            time.sleep(0.5)
    return False

def wait_for_backend(url="http://127.0.0.1:8000/docs", timeout=60):
    """探測 FastAPI 後端 HTTP 服務是否已真正完全啟動並可接收 HTTP 請求"""
    import urllib.request
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "HealthCheck"})
            with urllib.request.urlopen(req, timeout=1) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(0.8)
    return False

def open_browser_when_ready(url="http://127.0.0.1:7860", port=7860, timeout=60):
    """背景探測：直到 Gradio 前端服務真正開啟且監聽 Port 後，才喚醒瀏覽器"""
    if wait_for_port(port, timeout):
        time.sleep(1.5)  # 預留 1.5 秒讓 Gradio 完成 UI 畫面掛載
        print("✨ 前端服務就緒！自動開啟瀏覽器...")
        webbrowser.open(url)

def main():
    print("=" * 60)
    print("🚀 正在啟動 「AI文件對話系統」...")
    print(f"🐍 當前 Python 環境: {sys.executable}")
    print("=" * 60)

    ui_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. 啟動 FastAPI 後端伺服器 (Port 8000)
    backend_cmd = [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"]
    print("🔹 正在啟動 FastAPI 後端 (http://127.0.0.1:8000)...")
    backend_proc = subprocess.Popen(backend_cmd, cwd=ui_dir)

    # 2. 等待後端 Port 8000 HTTP 服務真正完全初始化完畢
    if wait_for_backend("http://127.0.0.1:8000/docs", timeout=60):
        print("✅ FastAPI 後端服務已成功啟動就緒！")
        time.sleep(1.0)
    else:
        print("⚠️ 後端啟動時間較長，繼續準備前端服務...")

    # 3. 啟動 Gradio 前端 UI (Port 7860)
    frontend_cmd = [sys.executable, "frontend/app.py"]
    print("🌐 正在啟動 Gradio 前端 UI (http://127.0.0.1:7860)...")
    frontend_proc = subprocess.Popen(frontend_cmd, cwd=ui_dir)

    # 4. 啟動背景線程探測前端 Port 7860 服務狀態，就緒後開啟瀏覽器
    threading.Thread(target=open_browser_when_ready, daemon=True).start()

    try:
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("\n🛑 正在關閉系統服務...")
    finally:
        backend_proc.terminate()
        frontend_proc.terminate()
        print("✅ 系統已關閉。")

if __name__ == "__main__":
    main()
