"""
Configuration for Personal LLM system.
All paths are local — nothing leaves your machine.
"""

import os
from pathlib import Path

# ─── Directories ───────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
PERSONAL_LLM_DIR = Path(__file__).parent

# Where downloaded GGUF model files are stored
MODELS_DIR = BASE_DIR / "personal_llm_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Chat history persistence
CHAT_HISTORY_DIR = PERSONAL_LLM_DIR / "chat_history"
CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

# RAG knowledge base storage
KNOWLEDGE_DB_DIR = PERSONAL_LLM_DIR / "knowledge_db"
KNOWLEDGE_DB_DIR.mkdir(parents=True, exist_ok=True)

# Uploaded documents for RAG
DOCUMENTS_DIR = PERSONAL_LLM_DIR / "documents"
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Model Defaults ───────────────────────────────────────────
# Default model to load (filename in MODELS_DIR)
DEFAULT_MODEL = os.getenv("PERSONAL_LLM_MODEL", "")

# GPU layers to offload (-1 = offload ALL layers to GPU for max speed)
N_GPU_LAYERS = int(os.getenv("PERSONAL_LLM_GPU_LAYERS", "-1"))

# Context window size (tokens) — higher = more conversation memory
CONTEXT_SIZE = int(os.getenv("PERSONAL_LLM_CONTEXT", "4096"))

# Maximum tokens to generate per response
MAX_TOKENS = int(os.getenv("PERSONAL_LLM_MAX_TOKENS", "2048"))

# ─── Generation Parameters ────────────────────────────────────
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 40
REPEAT_PENALTY = 1.1

# ─── Chat Memory ──────────────────────────────────────────────
# Max conversation turns to keep in context
MAX_HISTORY_TURNS = 20

# ─── Web UI ───────────────────────────────────────────────────
UI_PORT = int(os.getenv("PERSONAL_LLM_PORT", "7865"))
UI_HOST = "127.0.0.1"  # localhost only — fully private

# ─── RAG Settings ─────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 5

# ─── Model Catalog ────────────────────────────────────────────
# Pre-configured models that can be downloaded from HuggingFace
MODEL_CATALOG = {
    "phi-3-mini": {
        "name": "Phi-3 Mini 3.8B (Q4_K_M)",
        "repo_id": "bartowski/Phi-3.1-mini-4k-instruct-GGUF",
        "filename": "Phi-3.1-mini-4k-instruct-Q4_K_M.gguf",
        "size_gb": 2.4,
        "description": "Microsoft's compact powerhouse. Great for general chat on low-end hardware.",
        "chat_format": "chatml",
    },
    "llama-3.2-3b": {
        "name": "Llama 3.2 3B (Q4_K_M)",
        "repo_id": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "size_gb": 2.0,
        "description": "Meta's latest small model. Fast and capable.",
        "chat_format": "llama-3",
    },
    "mistral-7b": {
        "name": "Mistral 7B Instruct v0.3 (Q4_K_M)",
        "repo_id": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        "filename": "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        "size_gb": 4.4,
        "description": "Mistral's instruction-tuned model. Excellent quality.",
        "chat_format": "mistral-instruct",
    },
    "deepseek-r1-7b": {
        "name": "DeepSeek-R1 Qwen Distill 7B (Q4_K_M)",
        "repo_id": "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF",
        "filename": "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
        "size_gb": 4.7,
        "description": "DeepSeek's reasoning model. Think step-by-step.",
        "chat_format": "chatml",
    },
    "codellama-7b": {
        "name": "CodeLlama 7B Instruct (Q4_K_M)",
        "repo_id": "TheBloke/CodeLlama-7B-Instruct-GGUF",
        "filename": "codellama-7b-instruct.Q4_K_M.gguf",
        "size_gb": 4.1,
        "description": "Meta's code specialist. Writes, debugs, explains code.",
        "chat_format": "llama-2",
    },
}
