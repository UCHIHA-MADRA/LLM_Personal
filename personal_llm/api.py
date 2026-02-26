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
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from . import config
from .llm_engine import get_engine
from .model_manager import ModelManager
from .chat_engine import ChatEngine
from .knowledge_base import KnowledgeBase
from .llmfit_wrapper import get_model_fit_info

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Personal LLM API", version="2.0")

# Enable CORS for the local React/Tauri frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Localhost dev servers and file:// for Electron
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global State ─────────────────────────────────────────────────────────────
engine = get_engine()
model_manager = ModelManager()
chat_engine = ChatEngine(engine)
kb = KnowledgeBase()

# Download progress tracking (thread-safe)
download_state: Dict[str, Any] = {}
download_lock = threading.Lock()

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

class LoadModelRequest(BaseModel):
    filename: str

class DownloadModelRequest(BaseModel):
    catalog_key: str

class SettingsRequest(BaseModel):
    openai_key: Optional[str] = None
    groq_key: Optional[str] = None
    together_key: Optional[str] = None

class CloudChatRequest(BaseModel):
    message: str
    provider: str  # "openai", "groq", "together"
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

# ─── Model Manager Endpoints ──────────────────────────────────────────────────

@app.get("/api/models")
async def get_models():
    """Get the model catalog, local downloaded models, and hardware fit scores."""
    
    # 1. Local downloaded models
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
            "size_gb": entry["size_gb"],
            "filename": entry["filename"],
            "is_downloaded": is_downloaded,
            "fit_info": fit_info  # dict with fit_level, estimated_tps, etc.
        })
        
    return {
        "catalog": catalog,
        "local_models": local_files
    }

@app.post("/api/models/load")
async def load_model(req: LoadModelRequest):
    """Load a specific model from disk into the LLM Engine."""
    path = model_manager.models_dir / req.filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Model file not found: {req.filename}")

    chat_format = model_manager.get_chat_format(req.filename)
    
    success = engine.load(
        str(path),
        n_gpu_layers=config.N_GPU_LAYERS,
        n_ctx=config.CONTEXT_SIZE,
        chat_format=chat_format,
    )
    
    if success:
        return {"status": "success", "message": f"Loaded {req.filename}"}
    else:
        raise HTTPException(status_code=500, detail="Failed to load model. Check server logs.")

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
        if v and isinstance(v, str) and len(v) > 8:
            masked[k] = v[:4] + "*" * (len(v) - 8) + v[-4:]
        else:
            masked[k] = v
    return masked

@app.post("/api/settings")
async def save_settings(req: SettingsRequest):
    """Save API keys to local settings file."""
    settings = _load_settings()
    if req.openai_key is not None:
        settings["openai_key"] = req.openai_key
    if req.groq_key is not None:
        settings["groq_key"] = req.groq_key
    if req.together_key is not None:
        settings["together_key"] = req.together_key
    _save_settings(settings)
    return {"status": "saved"}

@app.post("/api/chat/cloud")
async def cloud_chat(req: CloudChatRequest):
    """Proxy a chat request to a cloud LLM provider (OpenAI-compatible)."""
    import httpx
    
    settings = _load_settings()
    
    provider_config = {
        "openai": {
            "url": "https://api.openai.com/v1/chat/completions",
            "key_field": "openai_key",
            "default_model": "gpt-3.5-turbo",
        },
        "groq": {
            "url": "https://api.groq.com/openai/v1/chat/completions",
            "key_field": "groq_key",
            "default_model": "llama-3.3-70b-versatile",
        },
        "together": {
            "url": "https://api.together.xyz/v1/chat/completions",
            "key_field": "together_key",
            "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        },
    }
    
    if req.provider not in provider_config:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {req.provider}")
    
    pc = provider_config[req.provider]
    api_key = settings.get(pc["key_field"], "")
    if not api_key:
        raise HTTPException(status_code=400, detail=f"No API key configured for {req.provider}. Go to Settings.")
    
    model = req.model or pc["default_model"]
    
    async def cloud_stream():
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    pc["url"],
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": req.message}],
                        "temperature": req.temperature,
                        "max_tokens": req.max_tokens,
                        "stream": True,
                    },
                )
                response.raise_for_status()
                
                init_payload = json.dumps({"type": "init", "conversation_id": None})
                yield f"data: {init_payload}\n\n"
                
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
                                payload = json.dumps({"type": "token", "content": content})
                                yield f"data: {payload}\n\n"
                        except Exception:
                            pass
                
                done_payload = json.dumps({"type": "done"})
                yield f"data: {done_payload}\n\n"
                
            except httpx.HTTPStatusError as e:
                err_payload = json.dumps({"type": "error", "content": f"{req.provider} API error: {e.response.status_code}"})
                yield f"data: {err_payload}\n\n"
            except Exception as e:
                err_payload = json.dumps({"type": "error", "content": str(e)})
                yield f"data: {err_payload}\n\n"
    
    return StreamingResponse(cloud_stream(), media_type="text/event-stream")

# ─── Chat Endpoints (Server-Sent Events) ──────────────────────────────────────

@app.post("/api/chat")
async def chat_stream(req: Request, chat_req: ChatRequest):
    """
    Stream chat tokens back to the React UI using Server-Sent Events (SSE).
    """
    if not engine.is_loaded:
        raise HTTPException(status_code=400, detail="No model is currently loaded. Go to Model Manager.")

    # Get or create conversation
    if chat_req.conversation_id:
        conv = chat_engine.get_conversation(chat_req.conversation_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        conv = chat_engine.new_conversation(system_prompt=chat_req.system_prompt)
        
    # RAG Context
    rag_context = ""
    if chat_req.use_rag:
        try:
            if kb._collection and kb._collection.count() > 0:
                rag_context = kb.query(chat_req.message)
        except Exception as e:
            logger.warning(f"RAG query failed: {e}")

    # Build context
    system = chat_req.system_prompt
    if rag_context:
        system += f"\n\nUse the following context to answer:\n\n{rag_context}"

    conv.add_user_message(chat_req.message)
    conv.model_name = engine.model_name
    
    messages = conv.get_context_messages()
    if messages and messages[0]["role"] == "system":
        messages[0]["content"] = system

    async def token_generator():
        # SSE format requires `data: {payload}\n\n`
        # We'll send the conversation ID first so the UI can track it
        init_payload = json.dumps({"type": "init", "conversation_id": conv.id})
        yield f"data: {init_payload}\n\n"
        
        full_response = ""
        try:
            # We iterate over the synchronous generator.
            # (Note: In a fully async app, we'd use an async llama-cpp-python wrapper,
            # but standard yield inside FastAPI StreamingResponse handles blocking decently for local apps)
            for token in engine.chat(
                messages=messages,
                max_tokens=chat_req.max_tokens,
                temperature=chat_req.temperature,
                stream=True,
            ):
                full_response += token
                # Check for client disconnect
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
                chat_engine._save_conversation(conv)
            
            done_payload = json.dumps({"type": "done"})
            yield f"data: {done_payload}\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")

# ─── Conversation Management ──────────────────────────────────────────────────

@app.get("/api/conversations")
async def list_conversations():
    return chat_engine.list_conversations()

@app.get("/api/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    conv = chat_engine.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Not found")
    return conv.dict()

@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    chat_engine.delete_conversation(conv_id)
    return {"status": "deleted"}

# ─── Main Run Stub ────────────────────────────────────────────────────────────
def launch_api(port: int = 8000):
    import uvicorn
    print(f"\n[*] Launching Headless Personal LLM API at http://127.0.0.1:{port}")
    
    # Auto-load default model if available
    default = model_manager.get_default_model()
    if default:
        print("\n[*] Auto-loading default model...")
        engine.load(
            default["path"],
            n_gpu_layers=config.N_GPU_LAYERS,
            n_ctx=config.CONTEXT_SIZE,
            chat_format=default.get("chat_format"),
        )
        
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")

if __name__ == "__main__":
    launch_api()
