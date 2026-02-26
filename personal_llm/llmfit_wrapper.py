import os
import sys
import json
import subprocess
from pathlib import Path

def _get_bin_dir() -> Path:
    """Get the bin directory, safe for both development and frozen builds."""
    if getattr(sys, 'frozen', False):
        # Running as compiled .exe — use the directory containing the .exe
        return Path(sys.executable).parent / "bin"
    else:
        # Running as normal Python script
        return Path(__file__).parent / "bin"

LLMFIT_BIN = _get_bin_dir() / "llmfit.exe"

# Simple memory cache to avoid running subprocess multiple times for the same model
_cache = {}

def get_model_fit_info(hf_id: str):
    """
    Executes llmfit.exe to get hardware compatibility for a specific Hugging Face model ID.
    Caches the result to prevent repeated slow subprocess calls.
    
    Returns:
        dict {"fit_level": str, "estimated_tps": float, "memory_required_gb": float, "score": float}
        or None if execution fails.
    """
    if not hf_id:
        return None

    if hf_id in _cache:
        return _cache[hf_id]

    if not LLMFIT_BIN.exists():
        print(f"⚠️ llmfit.exe not found at {LLMFIT_BIN}")
        return None

    try:
        # Run: llmfit --json info "hf_id"
        cmd = [str(LLMFIT_BIN), "--json", "info", hf_id]
        
        # Prevent console window popup on Windows when running as GUI (.exe)
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            startupinfo=startupinfo, 
            timeout=10
        )
        
        if result.returncode != 0:
            print(f"⚠️ llmfit error for {hf_id}: {result.stderr.strip()}")
            return None

        # Parse JSON output
        data = json.loads(result.stdout)
        
        if "models" in data and len(data["models"]) > 0:
            model_info = data["models"][0]
            
            fit_data = {
                "fit_level": model_info.get("fit_level", "Unknown"),
                "estimated_tps": round(model_info.get("estimated_tps", 0.0), 1),
                "memory_required_gb": round(model_info.get("memory_required_gb", 0.0), 2),
                "score": round(model_info.get("score", 0.0), 1)
            }
            
            _cache[hf_id] = fit_data
            return fit_data
        
        return None
        
    except json.JSONDecodeError:
        print(f"⚠️ Failed to parse JSON from llmfit for {hf_id}")
        return None
    except subprocess.TimeoutExpired:
        print(f"⚠️ llmfit timed out for {hf_id}")
        return None
    except Exception as e:
        print(f"⚠️ Exception running llmfit: {e}")
        return None
