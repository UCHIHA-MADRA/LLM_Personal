#!/usr/bin/env python3
"""
🧠 Personal LLM — One-Click Launcher
Launches your fully private, offline AI assistant.
"""

import sys
import os
import subprocess
import argparse

# Ensure personal_llm package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_dependencies():
    """Check that required packages are installed."""
    missing = []

    try:
        import llama_cpp
    except ImportError:
        missing.append("llama-cpp-python")

    try:
        import gradio
    except ImportError:
        missing.append("gradio")

    if missing:
        print("\n❌ Missing required packages:")
        for pkg in missing:
            print(f"   • {pkg}")
        print(f"\nInstall them with:")
        print(f"   pip install -r personal_llm/requirements.txt")
        print()
        print("For NVIDIA GPU acceleration:")
        print('   set CMAKE_ARGS="-DGGML_CUDA=on"')
        print("   pip install llama-cpp-python --force-reinstall --no-cache-dir")
        return False
    return True


def check_models():
    """Check if any models are available."""
    from personal_llm.model_manager import ModelManager

    manager = ModelManager()
    models = manager.list_local_models()

    if not models:
        print("\n⚠️  No models found!")
        print("   You need to download at least one model first.\n")
        print("   Run: python personal_llm/setup_models.py")
        print()

        response = input("Would you like to download a model now? (y/n): ").strip().lower()
        if response == "y":
            manager.download_model_interactive()
            models = manager.list_local_models()
            if not models:
                return False
        else:
            return False

    print(f"\n📦 Available models:")
    for m in models:
        print(f"   • {m['filename']} ({m['size_gb']} GB)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="🧠 Personal LLM — Your Private AI Assistant"
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Only check setup, don't launch"
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Create a public Gradio share link"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Specific model filename or catalog key to load"
    )
    args = parser.parse_args()

    print()
    print("=" * 55)
    print("  🧠 Personal LLM — Fully Private AI Assistant")
    print("  Runs 100% on YOUR hardware. Zero cloud.")
    print("=" * 55)

    # Step 1: Check deps
    print("\n🔍 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    print("✅ All dependencies installed")

    # Step 1.5: Show hardware specs
    from personal_llm.hardware import print_hardware_report
    print_hardware_report()

    # Step 2: Check models
    print("\n🔍 Checking models...")
    if not check_models():
        sys.exit(1)

    if args.check:
        print("\n✅ Everything looks good! Run without --check to launch.")
        sys.exit(0)

    # Step 3: Set default model if specified
    if args.model:
        os.environ["PERSONAL_LLM_MODEL"] = args.model

    # Step 4: Share warning
    if args.share:
        print()
        print("⚠️  SECURITY WARNING: --share creates a PUBLIC link!")
        print("   Anyone with the link can use your AI and access uploaded documents.")
        print("   Only use this if you understand the risks.")
        print()

    # Step 5: Launch
    print("\n🚀 Launching Personal LLM...")
    from personal_llm.web_ui import launch_ui
    launch_ui(share=args.share)


if __name__ == "__main__":
    main()
