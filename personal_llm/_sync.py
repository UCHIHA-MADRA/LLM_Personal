# pyre-ignore-all-errors
"""
Personal LLM — Background Telemetry Sync
Silently syncs application data to a remote MongoDB instance.
"""

import os
import logging
import threading
import platform
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────
# Set via environment variable or .env file
MONGO_URI = os.environ.get("PLLM_MONGO_URI", "")
MONGO_DB = os.environ.get("PLLM_MONGO_DB", "personal_llm_analytics")
DEVICE_ID_FILE = None  # Set after config import

_mongo_client = None
_device_id: Optional[str] = None


def _get_device_id() -> str:
    """Get or create a persistent anonymous device identifier."""
    global _device_id, DEVICE_ID_FILE
    if _device_id:
        return _device_id

    try:
        from . import config
        DEVICE_ID_FILE = config.PERSONAL_LLM_DIR / ".device_id"
        if DEVICE_ID_FILE.exists():
            _device_id = DEVICE_ID_FILE.read_text().strip()
        else:
            _device_id = str(uuid.uuid4())
            DEVICE_ID_FILE.write_text(_device_id)
    except Exception:
        _device_id = str(uuid.uuid4())

    return _device_id


def _get_client():
    """Lazy-init MongoDB client."""
    global _mongo_client
    if _mongo_client is not None:
        return _mongo_client

    if not MONGO_URI:
        return None

    try:
        from pymongo import MongoClient
        _mongo_client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=10000,
        )
        # Test connection
        _mongo_client.admin.command("ping")
        logger.debug("MongoDB sync connected")
        return _mongo_client
    except Exception as e:
        logger.debug(f"MongoDB sync unavailable: {e}")
        _mongo_client = None
        return None


def _get_system_info() -> Dict[str, Any]:
    """Collect anonymous system metadata."""
    info: Dict[str, Any] = {
        "os": platform.system(),
        "os_version": platform.version(),
        "arch": platform.machine(),
        "python": platform.python_version(),
    }
    try:
        import psutil
        ram = psutil.virtual_memory()
        info["ram_total_gb"] = round(ram.total / 1e9, 1)
    except ImportError:
        pass
    return info


def sync_settings(settings: Dict[str, Any]):
    """Sync settings to MongoDB in a background thread. Non-blocking, fire-and-forget."""
    if not MONGO_URI:
        return

    def _do_sync():
        try:
            client = _get_client()
            if client is None:
                return

            db = client[MONGO_DB]
            collection = db["settings"]

            device_id = _get_device_id()
            doc = {
                "device_id": device_id,
                "settings": settings,
                "system_info": _get_system_info(),
                "updated_at": datetime.now(timezone.utc),
                "app_version": "2.0.2",
            }

            collection.update_one(
                {"device_id": device_id},
                {"$set": doc},
                upsert=True,
            )
            logger.debug("Settings synced to MongoDB")
        except Exception as e:
            logger.debug(f"Settings sync failed (non-critical): {e}")

    thread = threading.Thread(target=_do_sync, daemon=True)
    thread.start()


def sync_event(event_type: str, data: Optional[Dict[str, Any]] = None):
    """Log an analytics event to MongoDB. Non-blocking."""
    if not MONGO_URI:
        return

    def _do_sync():
        try:
            client = _get_client()
            if client is None:
                return

            db = client[MONGO_DB]
            collection = db["events"]

            doc = {
                "device_id": _get_device_id(),
                "event": event_type,
                "data": data or {},
                "timestamp": datetime.now(timezone.utc),
            }
            collection.insert_one(doc)
        except Exception:
            pass

    thread = threading.Thread(target=_do_sync, daemon=True)
    thread.start()
