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

    def print_progress(percent: float, msg: str):
        bar_len = 30
        filled = int(bar_len * percent)
        bar = '=' * filled + '-' * (bar_len - filled)
        sys.stdout.write(f"\r[{bar}] {percent*100:.1f}% | {msg}")
        sys.stdout.flush()

    # Interactive download
    while True:
        print("\n" + "=" * 65)
        print("📦  Available Models to Download")
        print("=" * 65)

        keys = list(config.MODEL_CATALOG.keys())
        for i, key in enumerate(keys, 1):
            entry = config.MODEL_CATALOG[key]
            # Check if already downloaded
            downloaded = (manager.models_dir / entry["filename"]).exists()
            status = "✅ Downloaded" if downloaded else f"📥 ~{entry['size_gb']} GB"
            print(f"\n  [{i}] {entry['name']}")
            print(f"      {entry['description']}")
            print(f"      Status: {status}")

        print(f"\n  [0] Cancel")
        print("=" * 65)
        
        try:
            choice = input("\nSelect model to download (number): ").strip()
            if choice == "0":
                break
                
            idx = int(choice) - 1
            if 0 <= idx < len(keys):
                print()
                result = manager.download_model_stream(keys[idx], progress_callback=print_progress)
                print()
                if not result:
                    print("\n❌ Download failed or cancelled.")
            else:
                print("Invalid choice. Try again.")
        except (ValueError, KeyboardInterrupt):
            print("\nCancelled.")
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
