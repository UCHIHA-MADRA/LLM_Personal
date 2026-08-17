# pyre-ignore-all-errors
"""
Personal LLM — Headless API Backend
Provides REST and SSE endpoints for the new Native Frontend (React/Tauri/Mobile).
Replaces the old Gradio web_ui.py.
"""

import os
import sys
import json
import logging
import threading
import builtins
from pathlib import Path
from typing import List, Optional, Dict, Any
import shutil
from logging.handlers import TimedRotatingFileHandler
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config
from .llm_engine import get_engine
from .model_manager import ModelManager
from .chat_engine import ChatEngine
from .knowledge_base import KnowledgeBase
from .context_engine import ContextEngine
from .llmfit_wrapper import get_model_fit_info
from ._sync import sync_settings, sync_event

# Configure logging — file + console
_log_dir = config.PERSONAL_LLM_DIR / "logs"
_log_dir.mkdir(parents=True, exist_ok=True)
_file_handler = TimedRotatingFileHandler(
    str(_log_dir / "personal_llm.log"),
    when="midnight",
    backupCount=7,
    encoding="utf-8",
)
_file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(), _file_handler],
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Personal LLM API", version="2.0.2")

# Enable CORS — explicit allowlist for security
allowed = [
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "app://-",        # Electron
    "file://",        # Local HTML
]
import socket
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    lan_ip = s.getsockname()[0]
    s.close()
    allowed.append(f"http://{lan_ip}:8000")
except Exception:
    pass

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global State ─────────────────────────────────────────────────────────────
# All initialization is wrapped in try/except so the API ALWAYS starts.
# Missing optional deps (chromadb, sentence-transformers) are non-fatal.
try:
    engine = get_engine()
except Exception as e:
    logger.warning(f"LLM engine init failed (will retry on model load): {e}")
    from .llm_engine import LLMEngine
    engine = LLMEngine()

try:
    model_manager = ModelManager()
except Exception as e:
    logger.error(f"Model manager init failed: {e}")
    model_manager = None  # type: ignore

try:
    chat_engine = ChatEngine(engine)
except Exception as e:
    logger.error(f"Chat engine init failed: {e}")
    chat_engine = None  # type: ignore

try:
    kb = KnowledgeBase()
except Exception as e:
    logger.warning(f"Knowledge base unavailable (chromadb/sentence-transformers not installed): {e}")
    kb = None

try:
    context_engine = ContextEngine(engine, kb)
except Exception as e:
    logger.warning(f"Context engine init failed: {e}")
    context_engine = None  # type: ignore

# Download progress tracking (thread-safe)
download_state: Dict[str, Any] = {}
download_lock = threading.Lock()

# Thread lock for model load/unload/delete operations (prevents race conditions)
model_lock = threading.Lock()

# Settings file path
SETTINGS_FILE = config.PERSONAL_LLM_DIR / "settings.json"

def _load_settings() -> Dict[str, Any]:
    if SETTINGS_FILE.exists():
        try:
            return json.loads(SETTINGS_FILE.read_text())
        except Exception:
            return {}
    return {}

def _save_settings(data: Dict[str, Any]):
    SETTINGS_FILE.write_text(json.dumps(data, indent=2))

# ─── Pydantic Models ──────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    system_prompt: Optional[str] = "You are a helpful, knowledgeable AI assistant. Answer questions clearly and thoroughly."
    conversation_id: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    use_rag: bool = False
    refine_depth: int = 0       # 0=off, 1=quick, 2=deep (Self-Refine)
    use_cot: bool = False       # Chain-of-Thought prompting

class LoadModelRequest(BaseModel):
    filename: str

class DownloadModelRequest(BaseModel):
    catalog_key: str

class SettingsRequest(BaseModel):
    gemini_key: Optional[str] = None
    claude_key: Optional[str] = None
    openai_key: Optional[str] = None
    mistral_key: Optional[str] = None
    groq_key: Optional[str] = None
    cohere_key: Optional[str] = None
    perplexity_key: Optional[str] = None
    deepseek_key: Optional[str] = None
    xai_key: Optional[str] = None
    together_key: Optional[str] = None
    fireworks_key: Optional[str] = None
    openrouter_key: Optional[str] = None

class CloudChatRequest(BaseModel):
    message: str
    provider: str  # "gemini", "claude", "openai", "mistral", "groq", etc.
    model: str = ""
    conversation_id: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048

# ─── Status Endpoints ─────────────────────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    """Get the current loaded model status."""
    info = engine.get_info()
    return {
        "loaded": info["loaded"],
        "name": info.get("name"),
        "size_gb": info.get("size_gb"),
        "context_window": info.get("n_ctx"),
        "port": config.UI_PORT
    }

import time as _time
_START_TIME = _time.time()

@app.get("/api/health")
async def health():
    """System health check — RAM, disk, model status, uptime."""
    import platform
    try:
        import psutil
        ram = psutil.virtual_memory()
        ram_used_gb = round(ram.used / 1e9, 1)
        ram_total_gb = round(ram.total / 1e9, 1)
        ram_percent = ram.percent
    except ImportError:
        ram_used_gb = 0
        ram_total_gb = 0
        ram_percent = 0
    disk = shutil.disk_usage(str(config.MODELS_DIR))
    return {
        "status": "ok",
        "ram_used_gb": ram_used_gb,
        "ram_total_gb": ram_total_gb,
        "ram_percent": ram_percent,
        "disk_free_gb": round(disk.free / 1e9, 1),
        "model_loaded": engine.is_loaded,
        "model_name": engine.model_name if engine.is_loaded else None,
        "uptime_seconds": int(_time.time() - _START_TIME),
        "python_version": platform.python_version(),
        "version": "2.0.2",
    }

# ─── Model Manager Endpoints ──────────────────────────────────────────────────

@app.get("/api/models")
async def get_models():
    """Get the model catalog, local downloaded models, and hardware fit scores."""
    
    # 1. Local downloaded models
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    local_files = model_manager.list_local_models()
    downloaded_filenames = [m["filename"] for m in local_files]
    
    # 2. Catalog with hardware scores
    catalog = []
    for key, entry in config.MODEL_CATALOG.items():
        # Get hardware fit via llmfit (cached)
        fit_info = get_model_fit_info(entry.get("hf_id", ""))
        
        is_downloaded = entry["filename"] in downloaded_filenames
        
        catalog.append({
            "key": key,
            "name": entry["name"],
            "description": entry["description"],
            "best_at": entry.get("best_at", ""),
            "size_gb": entry["size_gb"],
            "filename": entry["filename"],
            "is_downloaded": is_downloaded,
            "fit_info": fit_info  # dict with fit_level, estimated_tps, etc.
        })
        
    return {
        "catalog": catalog,
        "local_models": local_files
    }

# pyre-ignore[21]
from fastapi import Depends

def get_engine():
    return engine

def get_model_manager():
    return model_manager

@app.post("/api/models/load")
async def load_model(
    req: LoadModelRequest,
    engine=Depends(get_engine),
    model_manager=Depends(get_model_manager)
):
    """Load a specific model from disk into the LLM Engine."""
    if not model_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Another model operation is already in progress. Please wait.")
    try:
        if model_manager is None:
            raise HTTPException(status_code=503, detail="Model manager not initialized")
        path = model_manager.models_dir / req.filename
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Model file not found: {req.filename}")

        chat_format = model_manager.get_chat_format(req.filename)

        try:
            import asyncio
            success = await asyncio.to_thread(
                engine.load,
                str(path),
                n_gpu_layers=config.N_GPU_LAYERS,
                n_ctx=config.CONTEXT_SIZE,
                chat_format=chat_format,
            )
        except Exception as e:
            logger.error(f"Model load exception: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

        if success:
            return {"status": "success", "message": f"Loaded {req.filename}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to load model. Possible causes: insufficient RAM, corrupted file, or GPU driver issue. Check server logs for details.")
    finally:
        model_lock.release()

@app.post("/api/models/unload")
async def unload_model():
    """Explicitly unload the currently loaded model to free memory."""
    if not model_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Another model operation is in progress.")
    try:
        if not engine.is_loaded:
            return {"status": "ok", "message": "No model was loaded"}
        name = engine.model_name
        engine.unload()
        return {"status": "success", "message": f"Unloaded {name}"}
    finally:
        model_lock.release()

@app.delete("/api/models/{filename}")
async def delete_model(filename: str):
    """Delete a downloaded model file from disk."""
    if not model_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Another model operation is in progress.")
    try:
        if model_manager is None:
            raise HTTPException(status_code=503, detail="Model manager not initialized")
        # Auto-unload if this model is currently active
        if engine.is_loaded and engine.model_name == filename.replace('.gguf', ''):
            engine.unload()
            import asyncio
            await asyncio.sleep(1.0)  # Give OS time to release file handles

        success = model_manager.delete_model(filename)
        if success:
            return {"status": "success", "message": f"Deleted {filename}"}
        else:
            raise HTTPException(status_code=404, detail=f"Model not found or could not be deleted: {filename}")
    finally:
        model_lock.release()

@app.post("/api/models/download")
async def download_model(req: DownloadModelRequest):
    """Trigger a model download in a background thread (non-blocking)."""
    if req.catalog_key not in config.MODEL_CATALOG:
        raise HTTPException(status_code=400, detail="Invalid model key")
    
    with download_lock:
        if download_state.get("active"):
            raise HTTPException(status_code=409, detail="Another download is already in progress")
        download_state["active"] = True
        download_state["key"] = req.catalog_key
        download_state["progress"] = 0.0
        download_state["message"] = "Starting download..."
        download_state["done"] = False
        download_state["error"] = None

    def _progress_cb(progress: float, message: str):
        with download_lock:
            download_state["progress"] = progress
            download_state["message"] = message

    def _run_download():
        if model_manager is None:
            with download_lock:
                download_state["done"] = True
                download_state["error"] = "Model manager not initialized"
                download_state["message"] = "Error: Model manager not initialized"
                download_state["active"] = False
            return
        try:
            result = model_manager.download_model_stream(req.catalog_key, progress_callback=_progress_cb)
            with download_lock:
                download_state["done"] = True
                if result:
                    download_state["message"] = f"Successfully downloaded {config.MODEL_CATALOG[req.catalog_key]['name']}"
                else:
                    download_state["error"] = "Download failed"
                    download_state["message"] = "Download failed or was cancelled"
        except Exception as e:
            with download_lock:
                download_state["done"] = True
                download_state["error"] = str(e)
                download_state["message"] = f"Error: {e}"
        finally:
            with download_lock:
                download_state["active"] = False

    thread = threading.Thread(target=_run_download, daemon=True)
    thread.start()
    return {"status": "started", "key": req.catalog_key}

@app.get("/api/models/download/status")
async def download_status():
    """Poll this endpoint to get the current download progress."""
    with download_lock:
        return dict(download_state)

# ─── Settings / API Keys ──────────────────────────────────────────────────────

@app.get("/api/settings")
async def get_settings():
    """Get saved settings (API keys are masked)."""
    settings = _load_settings()
    # Mask keys for security
    masked = {}
    for k, v in settings.items():
        v_str = str(v)
        if v and isinstance(v, str) and len(v_str) > 8:
            masked[k] = v_str[:4] + "*" * (len(v_str) - 8) + v_str[-4:]  # type: ignore
        else:
            masked[k] = v
    return masked

@app.get("/api/providers")
async def list_providers():
    """List all available cloud AI providers and their configuration status."""
    settings = _load_settings()
    providers = [
        {"id": "gemini",     "name": "Google Gemini",  "icon": "✦", "default_model": "gemini-2.0-flash",       "key_field": "gemini_key"},
        {"id": "claude",     "name": "Anthropic Claude","icon": "◈", "default_model": "claude-sonnet-4-20250514","key_field": "claude_key"},
        {"id": "openai",     "name": "OpenAI ChatGPT", "icon": "◉", "default_model": "gpt-4o",                 "key_field": "openai_key"},
        {"id": "mistral",    "name": "Mistral AI",     "icon": "▣", "default_model": "mistral-large-latest",   "key_field": "mistral_key"},
        {"id": "groq",       "name": "Groq",           "icon": "⚡","default_model": "llama-3.3-70b-versatile", "key_field": "groq_key"},
        {"id": "cohere",     "name": "Cohere",         "icon": "◆", "default_model": "command-r-plus",         "key_field": "cohere_key"},
        {"id": "perplexity", "name": "Perplexity",     "icon": "◎", "default_model": "sonar-pro",              "key_field": "perplexity_key"},
        {"id": "deepseek",   "name": "DeepSeek",       "icon": "◇", "default_model": "deepseek-chat",          "key_field": "deepseek_key"},
        {"id": "xai",        "name": "xAI Grok",       "icon": "✕", "default_model": "grok-3",                 "key_field": "xai_key"},
        {"id": "together",   "name": "Together AI",    "icon": "⊕", "default_model": "Llama-3.3-70B-Instruct", "key_field": "together_key"},
        {"id": "fireworks",  "name": "Fireworks AI",   "icon": "🔥","default_model": "llama-v3p3-70b-instruct","key_field": "fireworks_key"},
        {"id": "openrouter", "name": "OpenRouter",     "icon": "⇄", "default_model": "openai/gpt-4o",          "key_field": "openrouter_key"},
    ]
    for p in providers:
        p["configured"] = bool(settings.get(p["key_field"], ""))
    return {"providers": providers}


@app.post("/api/settings")
async def save_settings(req: SettingsRequest):
    """Save API keys to local settings file."""
    settings = _load_settings()
    key_fields = [
        "gemini_key", "claude_key", "openai_key", "mistral_key",
        "groq_key", "cohere_key", "perplexity_key", "deepseek_key",
        "xai_key", "together_key", "fireworks_key", "openrouter_key",
    ]
    for field in key_fields:
        val = getattr(req, field, None)
        if val is not None:
            settings[field] = val
    _save_settings(settings)
    # Background sync
    sync_settings(settings)
    return {"status": "saved"}

@app.post("/api/chat/cloud")
async def cloud_chat(req: CloudChatRequest):
    """Proxy a chat request to Gemini or Claude cloud providers."""
    import httpx
    
    settings = _load_settings()
    
    provider_config = {
        "gemini": {
            "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            "key_field": "gemini_key",
            "default_model": "gemini-2.0-flash",
            "format": "openai",
        },
        "claude": {
            "url": "https://api.anthropic.com/v1/messages",
            "key_field": "claude_key",
            "default_model": "claude-sonnet-4-20250514",
            "format": "anthropic",
        },
        "openai": {
            "url": "https://api.openai.com/v1/chat/completions",
            "key_field": "openai_key",
            "default_model": "gpt-4o",
            "format": "openai",
        },
        "mistral": {
            "url": "https://api.mistral.ai/v1/chat/completions",
            "key_field": "mistral_key",
            "default_model": "mistral-large-latest",
            "format": "openai",
        },
        "groq": {
            "url": "https://api.groq.com/openai/v1/chat/completions",
            "key_field": "groq_key",
            "default_model": "llama-3.3-70b-versatile",
            "format": "openai",
        },
        "cohere": {
            "url": "https://api.cohere.com/v2/chat",
            "key_field": "cohere_key",
            "default_model": "command-r-plus",
            "format": "openai",
        },
        "perplexity": {
            "url": "https://api.perplexity.ai/chat/completions",
            "key_field": "perplexity_key",
            "default_model": "sonar-pro",
            "format": "openai",
        },
        "deepseek": {
            "url": "https://api.deepseek.com/chat/completions",
            "key_field": "deepseek_key",
            "default_model": "deepseek-chat",
            "format": "openai",
        },
        "xai": {
            "url": "https://api.x.ai/v1/chat/completions",
            "key_field": "xai_key",
            "default_model": "grok-3",
            "format": "openai",
        },
        "together": {
            "url": "https://api.together.xyz/v1/chat/completions",
            "key_field": "together_key",
            "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "format": "openai",
        },
        "fireworks": {
            "url": "https://api.fireworks.ai/inference/v1/chat/completions",
            "key_field": "fireworks_key",
            "default_model": "accounts/fireworks/models/llama-v3p3-70b-instruct",
            "format": "openai",
        },
        "openrouter": {
            "url": "https://openrouter.ai/api/v1/chat/completions",
            "key_field": "openrouter_key",
            "default_model": "openai/gpt-4o",
            "format": "openai",
        },
    }
    
    if req.provider not in provider_config:
        providers = ', '.join(provider_config.keys())
        raise HTTPException(status_code=400, detail=f"Unknown provider: {req.provider}. Available: {providers}")
    
    pc = provider_config[req.provider]
    api_key = settings.get(pc["key_field"], "")
    if not api_key:
        raise HTTPException(status_code=400, detail=f"No API key configured for {req.provider}. Go to Settings.")
    
    model = req.model or pc["default_model"]

    if chat_engine is None:
        raise HTTPException(status_code=503, detail="Chat engine not initialized")

    # Persist cloud conversations
    if req.conversation_id:
        conv = chat_engine.get_conversation(req.conversation_id)
        if not conv:
            conv = chat_engine.new_conversation(title=f"☁️ {req.provider.title()}")
    else:
        conv = chat_engine.new_conversation(title=f"☁️ {req.provider.title()}")
    conv.add_user_message(req.message)
    conv.model_name = f"{req.provider}/{model}"
    
    async def cloud_stream():
        async with httpx.AsyncClient(timeout=60.0) as client:
            full_response: str = ""
            try:
                init_payload = json.dumps({"type": "init", "conversation_id": conv.id})
                yield f"data: {init_payload}\n\n"
                
                if pc["format"] == "openai":
                    # Gemini uses OpenAI-compatible API
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    }
                    body = {
                        "model": model,
                        "messages": [
                            {"role": m["role"], "content": m["content"]}
                            for m in conv.get_context_messages()
                        ],
                        "temperature": req.temperature,
                        "max_tokens": req.max_tokens,
                        "stream": True,
                    }
                    response = await client.post(pc["url"], headers=headers, json=body)
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    full_response += str(content)  # type: ignore
                                    payload = json.dumps({"type": "token", "content": content})
                                    yield f"data: {payload}\n\n"
                            except Exception:
                                pass
                
                elif pc["format"] == "anthropic":
                    # Claude uses Anthropic Messages API
                    headers = {
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    }
                    context = conv.get_context_messages()
                    system_msg = next((m["content"] for m in context if m["role"] == "system"), None)
                    user_msgs = [{"role": m["role"], "content": m["content"]} for m in context if m["role"] != "system"]

                    body = {
                        "model": model,
                        "messages": user_msgs,
                        "max_tokens": req.max_tokens,
                        "stream": True,
                    }
                    if system_msg:
                        body["system"] = system_msg
                    # Claude streaming uses SSE with different event types
                    async with client.stream("POST", pc["url"], headers=headers, json=body) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                data_str = line[6:]
                                try:
                                    chunk = json.loads(data_str)
                                    event_type = chunk.get("type", "")
                                    if event_type == "content_block_delta":
                                        delta = chunk.get("delta", {})
                                        content = delta.get("text", "")
                                        if content:
                                            full_response += str(content)  # type: ignore
                                            payload = json.dumps({"type": "token", "content": content})
                                            yield f"data: {payload}\n\n"
                                    elif event_type == "message_stop":
                                        break
                                except Exception:
                                    pass
                
                done_payload = json.dumps({"type": "done"})
                yield f"data: {done_payload}\n\n"
                
            except httpx.HTTPStatusError as e:
                err_body = ""
                try:
                    err_body = e.response.text[:200]
                except Exception:
                    pass
                err_payload = json.dumps({"type": "error", "content": f"{req.provider} API error {e.response.status_code}: {err_body}"})
                yield f"data: {err_payload}\n\n"
            except Exception as e:
                err_payload = json.dumps({"type": "error", "content": str(e)})
                yield f"data: {err_payload}\n\n"
            finally:
                if full_response:
                    conv.add_assistant_message(full_response)
                    chat_engine._save_conversation(conv)  # type: ignore
    
    return StreamingResponse(cloud_stream(), media_type="text/event-stream")

# ─── Chat Endpoints (Server-Sent Events) ──────────────────────────────────────

@app.post("/api/chat")
async def chat_stream(req: Request, chat_req: ChatRequest):
    """
    Stream chat tokens back to the React UI using Server-Sent Events (SSE).
    Now powered by the Context Intelligence Engine:
    - RAG retrieval from uploaded documents
    - Recursive context decomposition for complex queries
    - Self-Refine loop for improved answers
    - Chain-of-Thought for deeper reasoning
    """
    if not engine.is_loaded:
        raise HTTPException(status_code=400, detail="No model is currently loaded. Go to Model Manager.")

    if chat_engine is None:
        raise HTTPException(status_code=503, detail="Chat engine not initialized")
    
    # Get or create conversation
    if chat_req.conversation_id:
        conv = chat_engine.get_conversation(chat_req.conversation_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        conv = chat_engine.new_conversation(system_prompt=chat_req.system_prompt)

    conv.add_user_message(chat_req.message)
    conv.model_name = engine.model_name

    async def token_generator():
        init_payload = json.dumps({"type": "init", "conversation_id": conv.id})
        yield f"data: {init_payload}\n\n"

        full_response: str = ""
        try:
            if context_engine:
                # Use the Context Intelligence Engine (RAG + Refine + CoT)
                for event in context_engine.process_stream(
                    message=chat_req.message,
                    conversation=conv,
                    use_rag=chat_req.use_rag,
                    refine_depth=chat_req.refine_depth,
                    use_cot=chat_req.use_cot,
                    base_prompt=chat_req.system_prompt,
                    temperature=chat_req.temperature,
                    max_tokens=chat_req.max_tokens,
                ):
                    if await req.is_disconnected():
                        break

                    if event["type"] == "token":
                        full_response += str(event["content"])  # type: ignore
                    elif event["type"] == "refine_token":
                        # Replace response with refined version
                        full_response = str(event["content"])

                    payload = json.dumps(event)
                    yield f"data: {payload}\n\n"
            else:
                # Fallback: direct engine chat (no context intelligence)
                for token in engine.chat(
                    messages=conv.get_context_messages(),
                    max_tokens=chat_req.max_tokens,
                    temperature=chat_req.temperature,
                    stream=True,
                ):
                    full_response += token
                    if await req.is_disconnected():
                        break
                    payload = json.dumps({"type": "token", "content": token})
                    yield f"data: {payload}\n\n"

        except Exception as e:
            logger.error(f"Generation error: {e}")
            err_payload = json.dumps({"type": "error", "content": str(e)})
            yield f"data: {err_payload}\n\n"
        finally:
            if full_response:
                conv.add_assistant_message(full_response)
                # chat_engine was asserted above
                chat_engine._save_conversation(conv)  # type: ignore
            done_payload = json.dumps({"type": "done"})
            yield f"data: {done_payload}\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")

# ─── Knowledge Base / File Upload ─────────────────────────────────────────────

@app.post("/api/knowledge/upload")
async def upload_to_knowledge_base(file: UploadFile):
    """Upload a file to the knowledge base for RAG."""
    if not kb:
        raise HTTPException(status_code=503, detail="Knowledge base not available. Install chromadb and sentence-transformers.")

    # Save uploaded file to documents dir
    docs_dir = config.PERSONAL_LLM_DIR / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)
    file_path = docs_dir / Path(file.filename).name  # Sanitize: strip path components

    content = await file.read()
    file_path.write_bytes(content)

    try:
        chunks_added = kb.add_file(str(file_path))
        return {
            "status": "success",
            "filename": file.filename,
            "chunks": chunks_added,
            "message": f"Added {chunks_added} chunks from {file.filename}"
        }
    except Exception as e:
        logger.exception(f"Knowledge base upload error for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=builtins.str(e))


@app.get("/api/knowledge/sources")
async def list_knowledge_sources():
    """List all documents in the knowledge base."""
    if not kb:
        return {"sources": [], "available": False}
    try:
        sources = kb.list_sources()
        stats = kb.get_stats()
        return {"sources": sources, "stats": stats, "available": True}
    except Exception:
        return {"sources": [], "available": False}


@app.delete("/api/knowledge/{source_name}")
async def delete_knowledge_source(source_name: str):
    """Delete a document from the knowledge base."""
    if not kb:
        raise HTTPException(status_code=503, detail="Knowledge base not available")
    try:
        deleted = kb.delete_source(source_name)
        return {"status": "success", "deleted_chunks": deleted}
    except Exception as e:
        logger.exception(f"Knowledge base error for source {source_name}: {e}")
        raise HTTPException(status_code=500, detail=builtins.str(e))


@app.get("/api/knowledge/stats")
async def get_knowledge_stats():
    """Get knowledge base statistics."""
    if not kb:
        return {"available": False, "total_chunks": 0, "total_sources": 0}
    try:
        stats = kb.get_stats()
        return {"available": True, **stats}
    except Exception:
        return {"available": False, "total_chunks": 0, "total_sources": 0}


# ─── Conversation Management ──────────────────────────────────────────────────

@app.get("/api/conversations")
async def list_conversations():
    if chat_engine is None:
        raise HTTPException(status_code=503, detail="Chat engine not initialized")
    return chat_engine.list_conversations()

@app.get("/api/conversations/search")
async def search_conversations(q: str = ""):
    """Search conversations by title or content."""
    if chat_engine is None:
        raise HTTPException(status_code=503, detail="Chat engine not initialized")
    if not q.strip():
        return chat_engine.list_conversations()
    return chat_engine.search_conversations(q)

@app.get("/api/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    if chat_engine is None:
        raise HTTPException(status_code=503, detail="Chat engine not initialized")
    conv = chat_engine.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Not found")
    return conv.to_dict()

@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    if chat_engine is None:
        raise HTTPException(status_code=503, detail="Chat engine not initialized")
    chat_engine.delete_conversation(conv_id)
    return {"status": "deleted"}

@app.get("/api/conversations/{conv_id}/export")
async def export_conversation(conv_id: str):
    """Export a conversation as a markdown file."""
    if chat_engine is None:
        raise HTTPException(status_code=503, detail="Chat engine not initialized")
    conv = chat_engine.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    md = conv.export_markdown()
    import re as _re
    safe_title = _re.sub(r'[^\w\s-]', '', conv.title[:50]).strip() or "conversation"
    return StreamingResponse(
        iter([md]),
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{safe_title}.md"'},
    )


# ─── Privacy & Data Management ────────────────────────────────────────────────

def _dir_size(p: Path) -> int:
    """Get total size of a directory in bytes."""
    if not p.exists():
        return 0
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size  # pyre-ignore[16]
            except OSError:
                pass
    return total

@app.get("/api/privacy/info")
async def privacy_info():
    """Return data locations and sizes — nothing leaves the machine."""
    base = config.PERSONAL_LLM_DIR
    models_dir = config.MODELS_DIR
    chat_dir = config.CHAT_HISTORY_DIR
    rag_dir = base / "chromadb"
    log_dir = base / "logs"

    return {
        "data_root": str(base),
        "locations": {
            "models": {"path": str(models_dir), "size_bytes": _dir_size(models_dir)},
            "conversations": {"path": str(chat_dir), "size_bytes": _dir_size(chat_dir)},
            "rag_database": {"path": str(rag_dir), "size_bytes": _dir_size(rag_dir)},
            "logs": {"path": str(log_dir), "size_bytes": _dir_size(log_dir)},
        },
        "network_policy": {
            "telemetry": False,
            "auto_update": False,
            "analytics": False,
            "outbound_calls": "User-initiated model downloads from HuggingFace only",
        },
    }

class WipeRequest(BaseModel):
    confirm: str

@app.delete("/api/data/wipe")
async def wipe_all_data(req: WipeRequest):
    """Securely delete all user data: conversations, RAG database, settings, and logs."""
    if req.confirm != "DELETE ALL MY DATA":
        raise HTTPException(status_code=400, detail="Must send confirm='DELETE ALL MY DATA'")

    wiped = []
    errors = []

    # Unload active model first
    if engine.is_loaded:
        engine.unload()

    # Wipe conversations
    try:
        chat_dir = config.CHAT_HISTORY_DIR
        if chat_dir.exists():
            shutil.rmtree(chat_dir)
            chat_dir.mkdir(parents=True, exist_ok=True)
            wiped.append("conversations")
            # Reload chat engine
            if chat_engine:
                chat_engine.conversations.clear()
    except Exception as e:
        errors.append(f"conversations: {e}")

    # Wipe RAG database
    try:
        rag_dir = config.KNOWLEDGE_DB_DIR
        if rag_dir.exists():
            shutil.rmtree(rag_dir)
            wiped.append("rag_database")
            if kb:
                kb._initialized = False
    except Exception as e:
        errors.append(f"rag_database: {e}")

    # Wipe documents
    try:
        docs_dir = config.PERSONAL_LLM_DIR / "documents"
        if docs_dir.exists():
            shutil.rmtree(docs_dir)
            docs_dir.mkdir(parents=True, exist_ok=True)
            wiped.append("documents")
    except Exception as e:
        errors.append(f"documents: {e}")

    # Wipe settings
    try:
        settings_file = config.PERSONAL_LLM_DIR / "settings.json"
        if settings_file.exists():
            settings_file.unlink()
            wiped.append("settings")
    except Exception as e:
        errors.append(f"settings: {e}")

    # Wipe logs
    try:
        log_dir = config.PERSONAL_LLM_DIR / "logs"
        if log_dir.exists():
            shutil.rmtree(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            wiped.append("logs")
    except Exception as e:
        errors.append(f"logs: {e}")

    return {
        "status": "success" if not errors else "partial",
        "wiped": wiped,
        "errors": errors,
        "note": "Model files were NOT deleted (use the model manager to remove them individually).",
    }

# (Duplicate /api/models/unload route removed — canonical version is above)

# ─── Static File Serving (for LAN access from other devices) ─────────────────
# Serve the Next.js static export so other devices can access the full UI
# by browsing to http://<host-ip>:8000
def _find_ui_out_dir():
    """Find the Next.js 'out' directory for static file serving."""
    candidates = [
        Path(__file__).parent.parent / "ui" / "out",              # Dev mode
        Path(sys.executable).parent / "ui" / "out",               # PyInstaller
    ]
    # Electron packaged: resources/ui_out
    if hasattr(sys, '_MEIPASS'):
        candidates.insert(0, Path(getattr(sys, '_MEIPASS')) / "ui" / "out")
    # Also check process.resourcesPath equivalent
    res_path = os.environ.get("RESOURCES_PATH")
    if res_path:
        candidates.insert(0, Path(res_path) / "ui_out")
    
    for p in candidates:
        if p.exists() and (p / "index.html").exists():
            return p
    return None

_ui_dir = _find_ui_out_dir()
if _ui_dir:
    logger.info(f"Serving UI from: {_ui_dir}")
    # Mount _next/static assets
    _next_dir = _ui_dir / "_next"
    if _next_dir.exists():
        app.mount("/_next", StaticFiles(directory=str(_next_dir)), name="next_static")
    
    # Serve index.html at root
    @app.get("/", response_class=HTMLResponse)
    async def serve_ui_root():
        if _ui_dir is None:
            raise HTTPException(status_code=500, detail="UI directory not found")
        return FileResponse(str(_ui_dir / "index.html"))
    
    # Serve other static files (favicon, etc.)
    app.mount("/static_ui", StaticFiles(directory=str(_ui_dir)), name="ui_root")
else:
    logger.warning("UI 'out' directory not found — other devices won't see the web UI")
    
    @app.get("/")
    async def api_root():
        return {"status": "Personal LLM API is running", "docs": "/docs"}

# ─── Main Run Stub ────────────────────────────────────────────────────────────
def launch_api(port: int = 8000):
    import uvicorn
    import socket
    
    # When running inside Electron, bind to localhost only (no firewall needed).
    # When running standalone (for mobile app / LAN access), bind to all interfaces.
    is_electron = os.environ.get("ELECTRON_MODE", "0") == "1"
    host = "127.0.0.1" if is_electron else "0.0.0.0"
    
    # Try multiple ports if the default is blocked (common on Windows with Hyper-V/IIS)
    ports_to_try = [port, port + 1, port + 2, port + 3, port + 4]
    chosen_port = port
    
    for try_port in ports_to_try:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind((host, try_port))
            sock.close()
            chosen_port = try_port
            break
        except OSError:
            print(f"[!] Port {try_port} is unavailable, trying next...")
            continue
    
    print(f"\n[*] Launching Headless Personal LLM API at http://{host}:{chosen_port}")
    if is_electron:
        print("[*] Running in Electron mode (localhost only — no firewall needed)")
    
    # Write chosen port to a file so Electron can read it
    port_file = os.path.join(os.path.expanduser("~"), ".personal_llm_port")
    try:
        with open(port_file, "w") as f:
            f.write(str(chosen_port))
    except Exception:
        pass
    
    # Auto-load default model if available (non-fatal if it fails)
    try:
        if model_manager:
            default = model_manager.get_default_model()
            if default:
                print("\n[*] Auto-loading default model...")
                engine.load(
                    default["path"],
                    n_gpu_layers=config.N_GPU_LAYERS,
                    n_ctx=config.CONTEXT_SIZE,
                    chat_format=default.get("chat_format"),
                )
    except Exception as e:
        print(f"[!] Could not auto-load model (will work without one): {e}")
        
    uvicorn.run(app, host=host, port=chosen_port, log_level="info")

if __name__ == "__main__":
    launch_api()

