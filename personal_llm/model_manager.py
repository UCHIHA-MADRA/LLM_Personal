"""
Model Manager — Download, list, and manage GGUF model files.
Downloads from HuggingFace Hub (one-time), then everything is offline.
"""

import os
import sys
import json
import logging
import shutil
import hashlib
import requests
import threading
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable

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

    def download_model_stream(
        self,
        catalog_key: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        cancel_event: Optional[threading.Event] = None
    ) -> Optional[str]:
        """
        Robustly download a model with streaming progress, cancellation, and validation.
        """
        if catalog_key not in config.MODEL_CATALOG:
            if progress_callback:
                progress_callback(0, f"Error: Unknown model {catalog_key}")
            return None

        entry = config.MODEL_CATALOG[catalog_key]
        dest_path = self.models_dir / entry["filename"]
        temp_path = self.models_dir / (entry["filename"] + ".downloading")

        if dest_path.exists():
            if progress_callback:
                progress_callback(1.0, f"Model already downloaded: {entry['filename']}")
            return str(dest_path)

        # 1. Pre-flight check: Disk Space
        expected_bytes = entry.get("size_bytes")
        if expected_bytes:
            free_bytes = shutil.disk_usage(self.models_dir).free
            # Require at least 200MB more than the file size as a safety buffer
            if free_bytes < (expected_bytes + 200 * 1024 * 1024):
                error_msg = f"Error: Not enough disk space. Need {expected_bytes/(1024**3):.2f} GB, but only {free_bytes/(1024**3):.2f} GB free."
                logger.error(error_msg)
                if progress_callback:
                    progress_callback(0, error_msg)
                return None

        url = f"https://huggingface.co/{entry['repo_id']}/resolve/main/{entry['filename']}"
        
        try:
            if progress_callback:
                progress_callback(0, f"Connecting to HuggingFace...")

            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()

            total_size_str = response.headers.get('content-length')
            total_size = int(total_size_str) if total_size_str else expected_bytes or 0
            downloaded_size = 0

            # 2. Download loop with cancellation check
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if cancel_event and cancel_event.is_set():
                        logger.info("Download cancelled by user.")
                        if progress_callback:
                            progress_callback(downloaded_size / total_size if total_size else 0, "Download cancelled.")
                        f.close()
                        # Cleanup temp file on cancel
                        if temp_path.exists():
                            temp_path.unlink()
                        return None
                        
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size and progress_callback:
                            percent = downloaded_size / total_size
                            progress_callback(percent, f"Downloading {entry['name']}... ({downloaded_size/(1024**3):.2f} GB / {total_size/(1024**3):.2f} GB)")

            # 3. Post-flight check: SHA256 Verification
            expected_sha256 = entry.get("sha256")
            if expected_sha256:
                if progress_callback:
                    progress_callback(1.0, "Verifying checksum...")
                
                sha256_hash = hashlib.sha256()
                with open(temp_path, "rb") as f:
                    for byte_block in iter(lambda: f.read(4096 * 1024), b""): # 4MB blocks for hashing
                        sha256_hash.update(byte_block)
                
                actual_sha256 = sha256_hash.hexdigest()
                
                if actual_sha256 != expected_sha256:
                    error_msg = "Error: Checksum mismatch. Download corrupted."
                    logger.error(f"{error_msg} Expected {expected_sha256}, got {actual_sha256}")
                    if progress_callback:
                        progress_callback(1.0, error_msg)
                    temp_path.unlink()
                    return None

            # All good! Rename to final destination
            temp_path.rename(dest_path)
            
            if progress_callback:
                progress_callback(1.0, f"Successfully downloaded {entry['name']}")
                
            return str(dest_path)

        except requests.exceptions.RequestException as e:
            error_msg = f"Network Error: {str(e)}"
            logger.error(error_msg)
            if progress_callback:
                progress_callback(0, error_msg)
            if temp_path.exists():
                temp_path.unlink()
            return None
        except Exception as e:
            error_msg = f"Unexpected Error: {str(e)}"
            logger.error(error_msg)
            if progress_callback:
                progress_callback(0, error_msg)
            if temp_path.exists():
                temp_path.unlink()
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
