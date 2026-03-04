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
from pathlib import Path
from typing import List, Optional, Dict, Any
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Personal LLM API", version="2.0.1")

# Enable CORS for the local React/Tauri frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "app://-",       # Electron
        "file://",       # Local file protocol
    ],
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
    model_manager = None

try:
    chat_engine = ChatEngine(engine)
except Exception as e:
    logger.error(f"Chat engine init failed: {e}")
    chat_engine = None

try:
    kb = KnowledgeBase()
except Exception as e:
    logger.warning(f"Knowledge base unavailable (chromadb/sentence-transformers not installed): {e}")
    kb = None

try:
    context_engine = ContextEngine(engine, kb)
except Exception as e:
    logger.warning(f"Context engine init failed: {e}")
    context_engine = None

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
    refine_depth: int = 0       # 0=off, 1=quick, 2=deep (Self-Refine)
    use_cot: bool = False       # Chain-of-Thought prompting

class LoadModelRequest(BaseModel):
    filename: str

class DownloadModelRequest(BaseModel):
    catalog_key: str

class SettingsRequest(BaseModel):
    gemini_key: Optional[str] = None
    claude_key: Optional[str] = None

class CloudChatRequest(BaseModel):
    message: str
    provider: str  # "gemini", "claude"
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
    if req.gemini_key is not None:
        settings["gemini_key"] = req.gemini_key
    if req.claude_key is not None:
        settings["claude_key"] = req.claude_key
    _save_settings(settings)
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
            "format": "openai",  # Gemini supports OpenAI-compatible format
        },
        "claude": {
            "url": "https://api.anthropic.com/v1/messages",
            "key_field": "claude_key",
            "default_model": "claude-3-5-sonnet-20241022",
            "format": "anthropic",  # Claude uses its own format
        },
    }
    
    if req.provider not in provider_config:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {req.provider}. Use 'gemini' or 'claude'.")
    
    pc = provider_config[req.provider]
    api_key = settings.get(pc["key_field"], "")
    if not api_key:
        raise HTTPException(status_code=400, detail=f"No API key configured for {req.provider}. Go to Settings.")
    
    model = req.model or pc["default_model"]

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
            full_response = ""
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
                        "messages": [{"role": "user", "content": req.message}],
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
                                    full_response += content
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
                    body = {
                        "model": model,
                        "messages": [{"role": "user", "content": req.message}],
                        "max_tokens": req.max_tokens,
                        "stream": True,
                    }
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
                                            full_response += content
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
                    chat_engine._save_conversation(conv)
    
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

        full_response = ""
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
                        full_response += event["content"]
                    elif event["type"] == "refine_token":
                        # Replace response with refined version
                        full_response = event["content"]

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
                chat_engine._save_conversation(conv)
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
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


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
    return chat_engine.list_conversations()

@app.get("/api/conversations/search")
async def search_conversations(q: str = ""):
    """Search conversations by title or content."""
    if not q.strip():
        return chat_engine.list_conversations()
    return chat_engine.search_conversations(q)

@app.get("/api/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    conv = chat_engine.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Not found")
    return conv.to_dict()

@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    chat_engine.delete_conversation(conv_id)
    return {"status": "deleted"}

@app.post("/api/models/unload")
async def unload_model():
    """Unload the current model to free GPU/RAM."""
    if not engine.is_loaded:
        return {"status": "no_model_loaded"}
    engine.unload()
    return {"status": "unloaded"}

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
        candidates.insert(0, Path(sys._MEIPASS) / "ui" / "out")
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

