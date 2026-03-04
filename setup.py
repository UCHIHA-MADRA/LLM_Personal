#!/usr/bin/env python3
"""
🧠 Personal LLM — Cross-Platform Setup
Works on Windows, macOS, and Linux.

Usage:
    python setup.py              # Full interactive setup
    python setup.py --check      # Check environment only
    python setup.py --headless   # Non-interactive (CI/CD)
"""

import os
import sys
import subprocess
import platform
import shutil
import argparse

# ─── Constants ────────────────────────────────────────────────────────────────

MIN_PYTHON = (3, 10)
REQUIREMENTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "personal_llm", "requirements.txt")
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ─── Utilities ────────────────────────────────────────────────────────────────

def print_banner():
    print()
    print("=" * 60)
    print("  🧠 Personal LLM — Cross-Platform Setup")
    print("  Fully private AI that runs 100% on YOUR hardware.")
    print("  No cloud. No Ollama. No Docker. Just Python + GGUF.")
    print("=" * 60)
    print()

def print_step(num, msg):
    print(f"\n{'─' * 60}")
    print(f"  Step {num}: {msg}")
    print(f"{'─' * 60}")

def run_pip(*args):
    """Run pip with the current Python interpreter."""
    cmd = [sys.executable, "-m", "pip"] + list(args)
    return subprocess.run(cmd, capture_output=True, text=True)

def ask_yes_no(prompt, default=True):
    suffix = " [Y/n]: " if default else " [y/N]: "
    try:
        answer = input(prompt + suffix).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    if not answer:
        return default
    return answer in ("y", "yes")

# ─── Checks ──────────────────────────────────────────────────────────────────

def check_python_version():
    """Verify Python version is sufficient."""
    v = sys.version_info
    if (v.major, v.minor) < MIN_PYTHON:
        print(f"❌ Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required.")
        print(f"   You have: Python {v.major}.{v.minor}.{v.micro}")
        print(f"   Download: https://python.org/downloads/")
        return False
    print(f"✅ Python {v.major}.{v.minor}.{v.micro} ({platform.architecture()[0]})")
    return True

def check_pip():
    """Verify pip is available."""
    result = run_pip("--version")
    if result.returncode != 0:
        print("❌ pip is not available.")
        print("   Install it: python -m ensurepip --upgrade")
        return False
    # Extract version from output
    ver = result.stdout.strip().split(" ")[1] if result.stdout else "unknown"
    print(f"✅ pip {ver}")
    return True

def detect_os():
    """Detect and display the operating system."""
    system = platform.system()
    release = platform.release()
    machine = platform.machine()
    if system == "Windows":
        print(f"✅ OS: Windows {release} ({machine})")
    elif system == "Darwin":
        mac_ver = platform.mac_ver()[0]
        # Check for Apple Silicon
        if machine == "arm64":
            print(f"✅ OS: macOS {mac_ver} (Apple Silicon — Metal GPU acceleration available)")
        else:
            print(f"✅ OS: macOS {mac_ver} (Intel)")
    elif system == "Linux":
        print(f"✅ OS: Linux {release} ({machine})")
    else:
        print(f"⚠️  OS: {system} {release} ({machine}) — untested platform")
    return system, machine

def detect_gpu():
    """Detect NVIDIA GPU availability."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_info = result.stdout.strip().split("\n")[0]
            parts = [p.strip() for p in gpu_info.split(",")]
            name = parts[0] if len(parts) > 0 else "Unknown"
            vram = parts[1] if len(parts) > 1 else "?"
            driver = parts[2] if len(parts) > 2 else "?"
            print(f"✅ GPU: {name} ({vram}, Driver {driver})")
            return "nvidia"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check for Apple Silicon Metal
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        print("✅ GPU: Apple Silicon (Metal acceleration — built-in)")
        return "metal"

    print("ℹ️  GPU: None detected (CPU-only mode — slower but works fine)")
    return None

# ─── Installation ─────────────────────────────────────────────────────────────

def install_requirements():
    """Install Python dependencies from requirements.txt."""
    if not os.path.exists(REQUIREMENTS):
        print(f"❌ Requirements file not found: {REQUIREMENTS}")
        return False

    print("   Installing dependencies from requirements.txt...")
    result = run_pip("install", "-r", REQUIREMENTS, "--quiet")
    if result.returncode != 0:
        print(f"❌ Failed to install dependencies:")
        print(result.stderr[-500:] if result.stderr else "Unknown error")
        return False

    print("✅ All Python dependencies installed")
    return True

def install_gpu_llama(gpu_type):
    """Install GPU-accelerated llama-cpp-python."""
    if gpu_type == "nvidia":
        print("\n🎮 Installing CUDA-accelerated llama-cpp-python...")
        print("   (This may take a few minutes — it compiles C++ code)")
        env = os.environ.copy()
        env["CMAKE_ARGS"] = "-DGGML_CUDA=on"
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "llama-cpp-python", "--force-reinstall", "--no-cache-dir", "--quiet"],
            env=env, capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            print("✅ CUDA-accelerated llama-cpp-python installed (5-10x faster inference)")
            return True
        else:
            print("⚠️  CUDA build failed. Falling back to CPU version.")
            print("   (GPU acceleration requires CUDA Toolkit: https://developer.nvidia.com/cuda-downloads)")
            return False
    elif gpu_type == "metal":
        print("\n🍎 Installing Metal-accelerated llama-cpp-python...")
        env = os.environ.copy()
        env["CMAKE_ARGS"] = "-DGGML_METAL=on"
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "llama-cpp-python", "--force-reinstall", "--no-cache-dir", "--quiet"],
            env=env, capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            print("✅ Metal-accelerated llama-cpp-python installed")
            return True
        else:
            print("⚠️  Metal build failed. Using CPU version.")
            return False
    return False

# ─── Model Download ───────────────────────────────────────────────────────────

def check_and_download_models():
    """Check for existing models and offer to download one."""
    sys.path.insert(0, PROJECT_ROOT)
    try:
        from personal_llm.model_manager import ModelManager
        manager = ModelManager()
        models = manager.list_local_models()

        if models:
            print(f"\n📦 Found {len(models)} downloaded model(s):")
            for m in models:
                print(f"   • {m['filename']} ({m['size_gb']} GB)")
            return True
        else:
            print("\n⚠️  No models found.")
            if ask_yes_no("   Would you like to download a model now?"):
                manager.download_model_interactive()
                return bool(manager.list_local_models())
            else:
                print("   You can download models later with:")
                print("   python -m personal_llm.setup_models")
                return True
    except ImportError as e:
        print(f"⚠️  Could not check models: {e}")
        print("   Try running: python -m personal_llm.setup_models")
        return True

# ─── Hardware Report ──────────────────────────────────────────────────────────

def show_hardware_report():
    """Show hardware compatibility report."""
    sys.path.insert(0, PROJECT_ROOT)
    try:
        from personal_llm.hardware import print_hardware_report
        print_hardware_report()
    except ImportError:
        print("⚠️  Could not load hardware module.")
    except Exception as e:
        print(f"⚠️  Hardware report error: {e}")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="🧠 Personal LLM — Cross-Platform Setup")
    parser.add_argument("--check", action="store_true", help="Only check environment, don't install")
    parser.add_argument("--headless", action="store_true", help="Non-interactive mode (no prompts)")
    parser.add_argument("--no-gpu", action="store_true", help="Skip GPU detection and CUDA/Metal install")
    args = parser.parse_args()

    print_banner()

    # ── Step 1: System Check ──
    print_step(1, "Checking your system")

    if not check_python_version():
        sys.exit(1)
    if not check_pip():
        sys.exit(1)

    system, machine = detect_os()
    gpu_type = None if args.no_gpu else detect_gpu()

    if args.check:
        print("\n✅ System check complete. Run without --check to install.")
        sys.exit(0)

    # ── Step 2: Install Dependencies ──
    print_step(2, "Installing Python dependencies")

    if not install_requirements():
        print("\n❌ Setup failed at dependency installation.")
        sys.exit(1)

    # ── Step 3: GPU Acceleration (optional) ──
    if gpu_type in ("nvidia", "metal"):
        print_step(3, "GPU Acceleration")
        if args.headless:
            install_gpu_llama(gpu_type)
        elif ask_yes_no(f"   Install GPU-accelerated inference ({gpu_type.upper()})? (5-10x faster)"):
            install_gpu_llama(gpu_type)
        else:
            print("   Skipped. Using CPU inference.")
    else:
        print_step(3, "GPU Acceleration")
        print("   No supported GPU detected. Using CPU inference (works fine, just slower).")

    # ── Step 4: Hardware Report ──
    print_step(4, "Your hardware")
    show_hardware_report()

    # ── Step 5: Download Models ──
    if not args.headless:
        print_step(5, "AI Models")
        check_and_download_models()
    else:
        print_step(5, "AI Models (skipped in headless mode)")
        print("   Run later: python -m personal_llm.setup_models")

    # ── Done ──
    print()
    print("=" * 60)
    print("  ✅ Setup Complete!")
    print("=" * 60)
    print()
    print("  Launch options:")
    print()
    print("  1️⃣  Desktop App (Electron — recommended):")
    print("     cd ui && npm install && npm run electron:dev")
    print()
    print("  2️⃣  Gradio Web UI (standalone, no Electron needed):")
    print("     python launch_personal_llm.py")
    print()
    print("  3️⃣  API Server only (for mobile app / custom frontend):")
    print("     python -c \"from personal_llm.api import launch_api; launch_api()\"")
    print()
    print("  📱 Mobile App: Connect from your phone over LAN")
    print("     (see mobile/ directory for Expo build)")
    print()
    print("  📖 Full documentation: PROJECT_REPORT.md")
    print()


if __name__ == "__main__":
    main()
