#!/usr/bin/env python3
"""
VisionAI Platform - Quick Demo Launcher
Starts the backend and opens the frontend in the browser.
Run: python start.py
"""

import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


def check_env():
    env_file = Path("backend/.env")
    if not env_file.exists():
        example = Path(".env.example")
        if example.exists():
            import shutil
            shutil.copy(example, env_file)
            print("[✓] Created backend/.env from .env.example")
        else:
            print("[!] No .env file found. Creating minimal config...")
            env_file.write_text(
                "SECRET_KEY=dev-secret-key-change-in-production\n"
                "DATABASE_URL=sqlite+aiosqlite:///./visionai.db\n"
                "DEBUG=true\n"
            )

def install_deps():
    print("[•] Checking / installing Python dependencies...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"],
        cwd="backend",
        capture_output=False,
    )
    if result.returncode != 0:
        print("[!] Some dependencies failed to install. Check requirements.txt.")


def start_backend():
    print("[•] Starting FastAPI backend on http://localhost:8000 ...")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "app.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
            "--log-level", "info",
        ],
        cwd="backend",
    )
    return proc


def open_frontend():
    frontend = Path("frontend/index.html").resolve()
    print(f"[•] Opening frontend: {frontend}")
    time.sleep(3)
    webbrowser.open(str(frontend))


def main():
    print("=" * 60)
    print("   VisionAI Platform — Startup")
    print("=" * 60)

    check_env()

    if "--no-install" not in sys.argv:
        install_deps()

    backend = start_backend()

    print("\n[✓] Backend starting...")
    print("[✓] API Docs: http://localhost:8000/api/docs")
    print("[✓] Health:   http://localhost:8000/health\n")

    if "--no-browser" not in sys.argv:
        open_frontend()

    print("[✓] VisionAI Platform running. Press Ctrl+C to stop.\n")

    try:
        backend.wait()
    except KeyboardInterrupt:
        print("\n[•] Shutting down...")
        backend.terminate()
        backend.wait()
        print("[✓] Goodbye!")


if __name__ == "__main__":
    main()
