"""
Hardware Detection — Detect your PC's specs and recommend models.
Everything runs on YOUR hardware. This module tells you what fits.
"""

import os
import sys
import subprocess
import platform
import logging
from typing import Dict, Any, Optional, List

from . import config

logger = logging.getLogger(__name__)


def detect_hardware() -> Dict[str, Any]:
    """
    Detect your PC's hardware specs.
    Returns CPU, RAM, GPU info — all detected locally.
    """
    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "cpu": _detect_cpu(),
        "ram_gb": _detect_ram(),
        "gpu": _detect_gpu(),
    }
    return info


def _detect_cpu() -> Dict[str, Any]:
    """Detect CPU info."""
    cpu_info = {
        "name": platform.processor() or "Unknown",
        "cores_physical": os.cpu_count() or 1,
    }

    # Try to get a friendlier name on Windows
    if platform.system() == "Windows":
        try:
            result = subprocess.run(
                ["wmic", "cpu", "get", "Name", "/format:list"],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.strip().split("\n"):
                if line.startswith("Name="):
                    cpu_info["name"] = line.split("=", 1)[1].strip()
                    break
        except Exception:
            pass

    return cpu_info


def _detect_ram() -> float:
    """Detect total RAM in GB."""
    try:
        if platform.system() == "Windows":
            result = subprocess.run(
                ["wmic", "memorychip", "get", "Capacity", "/format:list"],
                capture_output=True, text=True, timeout=5
            )
            total = 0
            for line in result.stdout.strip().split("\n"):
                if line.startswith("Capacity="):
                    total += int(line.split("=")[1].strip())
            if total > 0:
                return round(total / (1024**3), 1)
        # Fallback
        import shutil
        total, _, _ = shutil.disk_usage("/")
    except Exception:
        pass

    # Last resort: try psutil-like approach
    try:
        result = subprocess.run(
            ["systeminfo"],
            capture_output=True, text=True, timeout=15
        )
        for line in result.stdout.split("\n"):
            if "Total Physical Memory" in line:
                # Parse "7,599 MB" or "16,384 MB"
                parts = line.split(":")[1].strip()
                number = parts.replace(",", "").replace(".", "").split()[0]
                return round(int(number) / 1024, 1)
    except Exception:
        pass

    return 0.0


def _detect_gpu() -> Dict[str, Any]:
    """Detect NVIDIA GPU info."""
    gpu_info = {
        "available": False,
        "name": "None (CPU-only mode)",
        "vram_mb": 0,
        "driver": "",
        "cuda_available": False,
    }

    # Check nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            if len(parts) >= 3:
                gpu_info["available"] = True
                gpu_info["name"] = parts[0].strip()
                vram_str = parts[1].strip().replace("MiB", "").strip()
                gpu_info["vram_mb"] = int(vram_str)
                gpu_info["driver"] = parts[2].strip()
    except (FileNotFoundError, Exception):
        pass

    # Check for Apple Silicon (M1/M2/M3)
    if platform.system() == "Darwin" and platform.processor() == "arm":
        gpu_info["available"] = True
        gpu_info["name"] = "Apple Silicon (Metal)"
        gpu_info["cuda_available"] = False  # Not CUDA, but Metal
        # VRAM is shared with RAM on Apple Silicon
        gpu_info["vram_mb"] = _detect_ram() * 1024
        return gpu_info

    # Check for AMD/Other via simple presence check (fallback)
    # We can't easily get VRAM for AMD on Windows without 3rd party libs,
    # but we can at least acknowledge it exists.
    if not gpu_info["available"] and platform.system() == "Windows":
        try:
            wmic = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "Name"],
                capture_output=True, text=True
            )
            if "AMD" in wmic.stdout or "Radeon" in wmic.stdout:
                 gpu_info["name"] = "AMD Radeon (OpenCL/Vulkan)"
                 # We assume some capability, but llama.cpp support on Windows AMD is tricky
                 # without specific builds.
        except Exception:
            pass

    return gpu_info


def recommend_models(hardware: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Recommend models based on your hardware.
    Returns a sorted list of models that fit your PC.
    """
    ram_gb = hardware.get("ram_gb", 0)
    vram_mb = hardware.get("gpu", {}).get("vram_mb", 0)
    vram_gb = vram_mb / 1024
    has_gpu = hardware.get("gpu", {}).get("available", False)

    recommendations = []

    for key, entry in config.MODEL_CATALOG.items():
        model_size = entry["size_gb"]
        rec = {
            "key": key,
            "name": entry["name"],
            "size_gb": model_size,
            "description": entry["description"],
            "fits_ram": model_size < (ram_gb * 0.7),  # Leave 30% for OS
            "fits_vram": model_size < vram_gb if has_gpu else False,
            "recommended_mode": "",
            "status": "",
        }

        if rec["fits_vram"] and has_gpu:
            rec["recommended_mode"] = "🟢 GPU (fast)"
            rec["status"] = "✅ FITS YOUR GPU"
        elif rec["fits_ram"]:
            rec["recommended_mode"] = "🟡 CPU (slower but works)"
            rec["status"] = "✅ FITS YOUR RAM"
        else:
            rec["recommended_mode"] = "🔴 Too large"
            rec["status"] = "❌ MAY NOT FIT"

        recommendations.append(rec)

    # Sort: GPU-fit first, then RAM-fit, then too large
    recommendations.sort(key=lambda r: (
        0 if "GPU" in r["recommended_mode"] else
        1 if "CPU" in r["recommended_mode"] else 2,
        r["size_gb"]
    ))

    return recommendations


def print_hardware_report():
    """Print a full hardware report with model recommendations."""
    print()
    print("=" * 65)
    print("  🖥️  YOUR PC — Hardware Report")
    print("=" * 65)

    hw = detect_hardware()

    # System
    print(f"\n  💻 System")
    print(f"     OS:  {hw['os']} {hw['os_version'][:20]}")
    print(f"     CPU: {hw['cpu']['name']}")
    print(f"     Cores: {hw['cpu']['cores_physical']}")

    # RAM
    ram = hw["ram_gb"]
    print(f"\n  🧮 Memory")
    print(f"     RAM: {ram} GB")
    if ram >= 16:
        print(f"     → Excellent! Can run 7B models comfortably")
    elif ram >= 8:
        print(f"     → Good. Best with 3B-4B models, 7B possible with GPU offload")
    else:
        print(f"     → Limited. Stick to small models (3B or less)")

    # GPU
    gpu = hw["gpu"]
    print(f"\n  🎮 GPU")
    if gpu["available"]:
        vram_gb = round(gpu["vram_mb"] / 1024, 1)
        print(f"     GPU:    {gpu['name']}")
        print(f"     VRAM:   {vram_gb} GB")
        print(f"     Driver: {gpu['driver']}")
        print(f"     CUDA:   {'Yes' if gpu['cuda_available'] else 'Install llama-cpp-python with CUDA'}")
        if vram_gb >= 8:
            print(f"     → Great! Can fully offload 7B Q4 models to GPU")
        elif vram_gb >= 4:
            print(f"     → Good. Can partially offload models (split GPU+CPU)")
        else:
            print(f"     → Limited VRAM. Most work will be on CPU")
    else:
        print(f"     No NVIDIA GPU detected — will run on CPU only")
        print(f"     (Slower but still works!)")

    # Model Recommendations
    print(f"\n" + "=" * 65)
    print(f"  📦 Model Recommendations for YOUR Hardware")
    print(f"=" * 65)

    recs = recommend_models(hw)
    for r in recs:
        print(f"\n  {r['status']}")
        print(f"     {r['name']} ({r['size_gb']} GB)")
        print(f"     {r['description']}")
        print(f"     Mode: {r['recommended_mode']}")

    # Best pick
    best = next((r for r in recs if "GPU" in r["recommended_mode"]), None)
    if not best:
        best = next((r for r in recs if "CPU" in r["recommended_mode"]), None)

    if best:
        print(f"\n  ⭐ RECOMMENDED FOR YOU: {best['name']}")
        print(f"     Download: python personal_llm/setup_models.py")

    print()
    print("=" * 65)

    return hw


# CLI entry point
if __name__ == "__main__":
    print_hardware_report()
