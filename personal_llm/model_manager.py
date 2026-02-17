"""
Model Manager — Download, list, and manage GGUF model files.
Downloads from HuggingFace Hub (one-time), then everything is offline.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any

from . import config

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages GGUF model files on your local disk."""

    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = Path(models_dir) if models_dir else config.MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def list_local_models(self) -> List[Dict[str, Any]]:
        """List all GGUF model files in the models directory."""
        models = []
        for f in sorted(self.models_dir.glob("*.gguf")):
            size_gb = f.stat().st_size / (1024**3)
            # Check if this matches a catalog entry
            catalog_info = self._find_catalog_entry(f.name)
            models.append({
                "filename": f.name,
                "path": str(f),
                "size_gb": round(size_gb, 2),
                "description": catalog_info.get("description", "") if catalog_info else "",
                "chat_format": catalog_info.get("chat_format", None) if catalog_info else None,
            })
        return models

    def _find_catalog_entry(self, filename: str) -> Optional[Dict]:
        """Find a catalog entry matching a filename."""
        for key, entry in config.MODEL_CATALOG.items():
            if entry["filename"] == filename:
                return entry
        return None

    def get_model_path(self, name_or_filename: str) -> Optional[str]:
        """
        Get the full path to a model.
        Accepts either a catalog key (e.g., 'phi-3-mini') or a filename.
        """
        # Check if it's a catalog key
        if name_or_filename in config.MODEL_CATALOG:
            filename = config.MODEL_CATALOG[name_or_filename]["filename"]
        else:
            filename = name_or_filename

        path = self.models_dir / filename
        if path.exists():
            # Basic integrity check: size
            if name_or_filename in config.MODEL_CATALOG:
                expected_gb = config.MODEL_CATALOG[name_or_filename]["size_gb"]
                actual_gb = path.stat().st_size / (1024**3)
                # If size differs by more than 20% (generous tolerance for quantization variances), warn
                if abs(actual_gb - expected_gb) > (expected_gb * 0.2):
                    logger.warning(f"Model size mismatch: {filename} (Expected ~{expected_gb}GB, Found {actual_gb:.2f}GB). File may be corrupted.")
            return str(path)

        return None

    def get_chat_format(self, name_or_filename: str) -> Optional[str]:
        """Get the chat format for a model."""
        if name_or_filename in config.MODEL_CATALOG:
            return config.MODEL_CATALOG[name_or_filename].get("chat_format")

        # Try matching by filename
        for key, entry in config.MODEL_CATALOG.items():
            if entry["filename"] == name_or_filename:
                return entry.get("chat_format")

        return None

    def download_model(self, catalog_key: str) -> Optional[str]:
        """
        Download a model from HuggingFace Hub.

        Args:
            catalog_key: Key from MODEL_CATALOG (e.g., 'phi-3-mini')

        Returns:
            Path to the downloaded model file, or None if failed.
        """
        if catalog_key not in config.MODEL_CATALOG:
            print(f"❌ Unknown model: {catalog_key}")
            print(f"   Available: {', '.join(config.MODEL_CATALOG.keys())}")
            return None

        entry = config.MODEL_CATALOG[catalog_key]
        dest = self.models_dir / entry["filename"]

        if dest.exists():
            print(f"✅ Model already downloaded: {entry['filename']}")
            return str(dest)

        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            print("\n❌ huggingface-hub is not installed.")
            print("   Install it with: pip install huggingface-hub")
            return None

        print(f"\n📥 Downloading: {entry['name']}")
        print(f"   From: huggingface.co/{entry['repo_id']}")
        print(f"   Size: ~{entry['size_gb']} GB")
        print(f"   This is a one-time download. After this, everything is offline.\n")

        try:
            downloaded_path = hf_hub_download(
                repo_id=entry["repo_id"],
                filename=entry["filename"],
                local_dir=str(self.models_dir),
                local_dir_use_symlinks=False,
            )
            print(f"\n✅ Downloaded to: {dest}")
            return str(dest)

        except Exception as e:
            logger.error(f"Download failed: {e}")
            print(f"\n❌ Download failed: {e}")
            return None

    def download_model_interactive(self):
        """Interactive model download — shows the catalog and lets user pick."""
        print("\n" + "=" * 65)
        print("📦  Available Models to Download")
        print("=" * 65)

        keys = list(config.MODEL_CATALOG.keys())
        for i, key in enumerate(keys, 1):
            entry = config.MODEL_CATALOG[key]
            # Check if already downloaded
            downloaded = (self.models_dir / entry["filename"]).exists()
            status = "✅ Downloaded" if downloaded else f"📥 ~{entry['size_gb']} GB"
            print(f"\n  [{i}] {entry['name']}")
            print(f"      {entry['description']}")
            print(f"      Status: {status}")

        print(f"\n  [0] Cancel")
        print("=" * 65)

        while True:
            try:
                choice = input("\nSelect model to download (number): ").strip()
                if choice == "0":
                    print("Cancelled.")
                    return None
                idx = int(choice) - 1
                if 0 <= idx < len(keys):
                    return self.download_model(keys[idx])
                print("Invalid choice. Try again.")
            except (ValueError, KeyboardInterrupt):
                print("\nCancelled.")
                return None

    def get_default_model(self) -> Optional[Dict[str, Any]]:
        """
        Get the best available model.
        Priority: config default > first local model > None
        """
        # Check config default
        if config.DEFAULT_MODEL:
            path = self.get_model_path(config.DEFAULT_MODEL)
            if path:
                return {
                    "path": path,
                    "chat_format": self.get_chat_format(config.DEFAULT_MODEL),
                }

        # Fall back to first available local model
        models = self.list_local_models()
        if models:
            return {
                "path": models[0]["path"],
                "chat_format": models[0].get("chat_format"),
            }

        return None

    def print_status(self):
        """Print the current model status."""
        models = self.list_local_models()
        print(f"\n📁 Models directory: {self.models_dir}")
        if models:
            print(f"📊 {len(models)} model(s) available:\n")
            for m in models:
                print(f"   • {m['filename']} ({m['size_gb']} GB)")
                if m.get("description"):
                    print(f"     {m['description']}")
        else:
            print("⚠️  No models downloaded yet.")
            print("   Run: python -m personal_llm.setup_models")


# ─── CLI Entry Point ──────────────────────────────────────────
def main():
    manager = ModelManager()
    if len(sys.argv) > 1 and sys.argv[1] == "download":
        if len(sys.argv) > 2:
            manager.download_model(sys.argv[2])
        else:
            manager.download_model_interactive()
    else:
        manager.print_status()


if __name__ == "__main__":
    main()
