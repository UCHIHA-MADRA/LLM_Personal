"""
Setup Models — Interactive tool to download GGUF models for your Personal LLM.
Run this once to download models, then everything works offline.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from personal_llm.model_manager import ModelManager
from personal_llm import config


def main():
    print()
    print("=" * 65)
    print("  🧠 Personal LLM — Model Setup")
    print("  Download open-source LLM models to run on YOUR hardware.")
    print("  After download, everything runs 100% offline.")
    print("=" * 65)

    manager = ModelManager()

    # Show current status
    models = manager.list_local_models()
    if models:
        print(f"\n📦 You already have {len(models)} model(s):")
        for m in models:
            print(f"   ✅ {m['filename']} ({m['size_gb']} GB)")

    print(f"\n📁 Models directory: {config.MODELS_DIR}")

    # Show hardware specs so user knows what fits
    try:
        from personal_llm.hardware import print_hardware_report
        print_hardware_report()
    except Exception as e:
        print(f"(Hardware detection skipped: {e})")

    # Check dependencies
    try:
        from huggingface_hub import hf_hub_download
        print("✅ huggingface-hub is installed")
    except ImportError:
        print("\n❌ huggingface-hub is not installed!")
        print("   Run: pip install huggingface-hub")
        print("   Then re-run this script.")
        return

    # Interactive download
    while True:
        result = manager.download_model_interactive()
        if result is None:
            break

        print("\nWould you like to download another model?")
        again = input("(y/n): ").strip().lower()
        if again != "y":
            break

    # Final status
    models = manager.list_local_models()
    if models:
        print(f"\n✅ You have {len(models)} model(s) ready!")
        print(f"\n🚀 Launch your Personal LLM:")
        print(f"   python launch_personal_llm.py")
    else:
        print("\n⚠️ No models downloaded. You need at least one model to chat.")
        print("   Re-run this script to download a model.")

    print()


if __name__ == "__main__":
    main()
