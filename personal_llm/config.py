"""
Configuration for Personal LLM system.
All paths are local — nothing leaves your machine.

PyInstaller-safe: detects frozen (compiled) vs development mode
and sets paths accordingly so user data is never lost.
"""

import os
import sys
from pathlib import Path

# ─── PyInstaller-Safe Base Directory ────────────────────────────
# When frozen (compiled to .exe), __file__ points to a temp _MEIPASS folder.
# We need to use the executable's directory instead for persistent data.

def _get_base_dir() -> Path:
    """Get the base directory, safe for both development and frozen builds."""
    if getattr(sys, 'frozen', False):
        # Running as compiled .exe — use the directory containing the .exe
        return Path(sys.executable).parent
    else:
        # Running as normal Python script
        return Path(__file__).parent.parent

def _get_app_dir() -> Path:
    """Get the personal_llm package/app directory."""
    if getattr(sys, 'frozen', False):
        # In frozen mode, user data (chat history, RAG DB) MUST be in a writable location.
        # Program Files is read-only for standard users. We use LocalAppData.
        app_data = os.environ.get('LOCALAPPDATA', os.path.expanduser('~'))
        data_dir = Path(app_data) / "PersonalLLM"
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir
    else:
        return Path(__file__).parent


IS_FROZEN = getattr(sys, 'frozen', False)

BASE_DIR = _get_base_dir()
PERSONAL_LLM_DIR = _get_app_dir()

# ─── Directories ───────────────────────────────────────────────

# Where downloaded GGUF model files are stored
MODELS_DIR = BASE_DIR / "personal_llm_models"
try:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
except PermissionError:
    pass  # In frozen mode, the installer already creates this folder in Program Files

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
UI_PORT = int(os.getenv("PERSONAL_LLM_PORT", "0"))  # 0 = auto-find free port
UI_HOST = "127.0.0.1"  # localhost only — fully private

# ─── RAG Settings ─────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 5

# ─── Model Catalog ────────────────────────────────────────────
# Pre-configured models that can be downloaded from HuggingFace
# Organized by tier: TIER 1 (fully open), TIER 2 (weights + code), TIER 3 (weights, some restrictions)
MODEL_CATALOG = {

    # ══════════════════════════════════════════════════════════════
    # TIER 1 — 100% FULLY OPEN (Code + Weights + Training Data)
    # ══════════════════════════════════════════════════════════════

    "olmo-3-7b": {
        "name": "OLMo 3 7B Instruct (Q4_K_M)",
        "repo_id": "bartowski/OLMo-2-1124-7B-Instruct-GGUF",
        "hf_id": "allenai/OLMo-2-1124-7B-Instruct",
        "filename": "OLMo-2-1124-7B-Instruct-Q4_K_M.gguf",
        "size_gb": 4.4,
        "size_bytes": 4436755456,
        "sha256": "",
        "description": "Allen AI's fully open model. Code, data, AND training pipeline all public. Apache 2.0.",
        "chat_format": "chatml",
        "tier": 1,
        "license": "Apache 2.0",
    },
    "pythia-6.9b": {
        "name": "Pythia 6.9B Deduped (Q4_K_M)",
        "repo_id": "TheBloke/pythia-6.9b-deduped-GGUF",
        "hf_id": "EleutherAI/pythia-6.9b-deduped",
        "filename": "pythia-6.9b-deduped.Q4_K_M.gguf",
        "size_gb": 4.1,
        "size_bytes": 4081344768,
        "sha256": "",
        "description": "EleutherAI's research model. 154 checkpoints released. Ideal for ML research.",
        "chat_format": None,
        "tier": 1,
        "license": "Apache 2.0",
    },
    "gpt-neox-20b": {
        "name": "GPT-NeoX 20B (Q4_K_M)",
        "repo_id": "TheBloke/GPT-NeoX-20B-GGUF",
        "hf_id": "EleutherAI/gpt-neox-20b",
        "filename": "gpt-neox-20b.Q4_K_M.gguf",
        "size_gb": 12.1,
        "size_bytes": 12108906752,
        "sha256": "",
        "description": "EleutherAI's 20B param model. Fully open training code + data. Needs 16GB+ RAM.",
        "chat_format": None,
        "tier": 1,
        "license": "Apache 2.0",
    },
    "cerebras-gpt-6.7b": {
        "name": "Cerebras-GPT 6.7B (Q4_K_M)",
        "repo_id": "TheBloke/Cerebras-GPT-6.7B-GGUF",
        "hf_id": "cerebras/Cerebras-GPT-6.7B",
        "filename": "cerebras-gpt-6.7b.Q4_K_M.gguf",
        "size_gb": 4.0,
        "size_bytes": 3959422976,
        "sha256": "",
        "description": "Cerebras' compute-efficient model. Apache 2.0 with full training recipe.",
        "chat_format": None,
        "tier": 1,
        "license": "Apache 2.0",
    },

    # ══════════════════════════════════════════════════════════════
    # TIER 2 — Weights + Architecture Code (Training data partial)
    # ══════════════════════════════════════════════════════════════

    "deepseek-r1-7b": {
        "name": "DeepSeek-R1 Qwen Distill 7B (Q4_K_M)",
        "repo_id": "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF",
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "filename": "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
        "size_gb": 4.7,
        "size_bytes": 4683073504,
        "sha256": "731ece8d06dc7eda6f6572997feb9ee1258db0784827e642909d9b565641937b",
        "description": "DeepSeek's reasoning model distilled to 7B. Chain-of-thought specialist. MIT license.",
        "chat_format": "chatml",
        "tier": 2,
        "license": "MIT",
    },
    "deepseek-coder-6.7b": {
        "name": "DeepSeek Coder 6.7B Instruct (Q4_K_M)",
        "repo_id": "TheBloke/deepseek-coder-6.7B-instruct-GGUF",
        "hf_id": "deepseek-ai/deepseek-coder-6.7b-instruct",
        "filename": "deepseek-coder-6.7b-instruct.Q4_K_M.gguf",
        "size_gb": 4.0,
        "size_bytes": 3959422976,
        "sha256": "",
        "description": "DeepSeek's dedicated code model. Excels at programming tasks.",
        "chat_format": "chatml",
        "tier": 2,
        "license": "DeepSeek License",
    },
    "mistral-7b": {
        "name": "Mistral 7B Instruct v0.3 (Q4_K_M)",
        "repo_id": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "filename": "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        "size_gb": 4.4,
        "size_bytes": 4372812000,
        "sha256": "1270d22c0fbb3d092fb725d4d96c457b7b687a5f5a715abe1e818da303e562b6",
        "description": "Mistral's flagship 7B. Exceptional quality-to-size ratio. Apache 2.0.",
        "chat_format": "mistral-instruct",
        "tier": 2,
        "license": "Apache 2.0",
    },
    "qwen3-8b": {
        "name": "Qwen3 8B (Q4_K_M)",
        "repo_id": "bartowski/Qwen_Qwen3-8B-GGUF",
        "hf_id": "Qwen/Qwen3-8B",
        "filename": "Qwen_Qwen3-8B-Q4_K_M.gguf",
        "size_gb": 5.0,
        "size_bytes": 5017868544,
        "sha256": "",
        "description": "Alibaba's Qwen3 8B. Strong multilingual + reasoning. Apache 2.0.",
        "chat_format": "chatml",
        "tier": 2,
        "license": "Apache 2.0",
    },
    "qwen3-1.7b": {
        "name": "Qwen3 1.7B (Q4_K_M)",
        "repo_id": "bartowski/Qwen_Qwen3-1.7B-GGUF",
        "hf_id": "Qwen/Qwen3-1.7B",
        "filename": "Qwen_Qwen3-1.7B-Q4_K_M.gguf",
        "size_gb": 1.2,
        "size_bytes": 1200000000,
        "sha256": "",
        "description": "Alibaba's tiny Qwen3. Runs on anything. Great for low-end devices.",
        "chat_format": "chatml",
        "tier": 2,
        "license": "Apache 2.0",
    },
    "falcon-7b": {
        "name": "Falcon 7B Instruct (Q4_K_M)",
        "repo_id": "TheBloke/falcon-7b-instruct-GGUF",
        "hf_id": "tiiuae/falcon-7b-instruct",
        "filename": "falcon-7b-instruct.Q4_K_M.gguf",
        "size_gb": 4.4,
        "size_bytes": 4361840608,
        "sha256": "",
        "description": "TII's Falcon 7B. Strong open model from Abu Dhabi. Apache 2.0.",
        "chat_format": None,
        "tier": 2,
        "license": "Apache 2.0",
    },
    "falcon3-7b": {
        "name": "Falcon3 7B Instruct (Q4_K_M)",
        "repo_id": "bartowski/Falcon3-7B-Instruct-GGUF",
        "hf_id": "tiiuae/Falcon3-7B-Instruct",
        "filename": "Falcon3-7B-Instruct-Q4_K_M.gguf",
        "size_gb": 4.4,
        "size_bytes": 4436755456,
        "sha256": "",
        "description": "TII's latest Falcon3. Improved architecture and training. Apache 2.0.",
        "chat_format": "chatml",
        "tier": 2,
        "license": "Apache 2.0",
    },
    "mpt-7b": {
        "name": "MPT 7B Chat (Q4_K_M)",
        "repo_id": "TheBloke/MPT-7B-Chat-GGUF",
        "hf_id": "mosaicml/mpt-7b-chat",
        "filename": "mpt-7b-chat.Q4_K_M.gguf",
        "size_gb": 4.0,
        "size_bytes": 3959422976,
        "sha256": "",
        "description": "MosaicML's MPT 7B. Commercial-friendly, strong chat model. Apache 2.0.",
        "chat_format": None,
        "tier": 2,
        "license": "Apache 2.0",
    },
    "rwkv-7b": {
        "name": "RWKV-5 World 7B (Q4_K_M)",
        "repo_id": "TheBloke/rwkv-5-world-7B-GGUF",
        "hf_id": "RWKV/rwkv-5-world-7b",
        "filename": "rwkv-5-world-7b.Q4_K_M.gguf",
        "size_gb": 4.4,
        "size_bytes": 4361840608,
        "sha256": "",
        "description": "RWKV: RNN-based, runs with constant memory. No attention = infinite context. Apache 2.0.",
        "chat_format": None,
        "tier": 2,
        "license": "Apache 2.0",
    },
    "starcoder2-7b": {
        "name": "StarCoder2 7B (Q4_K_M)",
        "repo_id": "bartowski/starcoder2-7b-GGUF",
        "hf_id": "bigcode/starcoder2-7b",
        "filename": "starcoder2-7b-Q4_K_M.gguf",
        "size_gb": 4.3,
        "size_bytes": 4298014720,
        "sha256": "",
        "description": "BigCode's code model. Trained on 3.5T tokens from 600+ languages. BigCode OSL.",
        "chat_format": None,
        "tier": 2,
        "license": "BigCode OSL",
    },
    "starcoder2-3b": {
        "name": "StarCoder2 3B (Q4_K_M)",
        "repo_id": "bartowski/starcoder2-3b-GGUF",
        "hf_id": "bigcode/starcoder2-3b",
        "filename": "starcoder2-3b-Q4_K_M.gguf",
        "size_gb": 1.8,
        "size_bytes": 1800000000,
        "sha256": "",
        "description": "Lightweight code model. Fast completions for low-end hardware.",
        "chat_format": None,
        "tier": 2,
        "license": "BigCode OSL",
    },
    "yalm-100b": {
        "name": "YaLM 100B (Q2_K — smallest quant)",
        "repo_id": "TheBloke/YaLM-100B-GGUF",
        "hf_id": "yandex/YaLM-100B",
        "filename": "yalm-100b.Q2_K.gguf",
        "size_gb": 41.0,
        "size_bytes": 41000000000,
        "sha256": "",
        "description": "Yandex's 100B model. Massive. Needs 48GB+ RAM. Apache 2.0.",
        "chat_format": None,
        "tier": 2,
        "license": "Apache 2.0",
    },

    # ══════════════════════════════════════════════════════════════
    # TIER 3 — Weights Open, Some Restrictions
    # ══════════════════════════════════════════════════════════════

    "llama-3.2-3b": {
        "name": "Llama 3.2 3B Instruct (Q4_K_M)",
        "repo_id": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "hf_id": "meta-llama/Llama-3.2-3B-Instruct",
        "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "size_gb": 2.0,
        "size_bytes": 2019377696,
        "sha256": "6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff",
        "description": "Meta's compact Llama 3.2. Fast, smart, and efficient.",
        "chat_format": "llama-3",
        "tier": 3,
        "license": "Llama 3.2 Community",
    },
    "llama-3.2-1b": {
        "name": "Llama 3.2 1B Instruct (Q4_K_M)",
        "repo_id": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "hf_id": "meta-llama/Llama-3.2-1B-Instruct",
        "filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "size_gb": 0.8,
        "size_bytes": 800000000,
        "sha256": "",
        "description": "Meta's tiniest Llama. Runs on absolutely anything, even 2GB RAM.",
        "chat_format": "llama-3",
        "tier": 3,
        "license": "Llama 3.2 Community",
    },
    "llama-3.3-70b": {
        "name": "Llama 3.3 70B Instruct (Q4_K_M)",
        "repo_id": "bartowski/Llama-3.3-70B-Instruct-GGUF",
        "hf_id": "meta-llama/Llama-3.3-70B-Instruct",
        "filename": "Llama-3.3-70B-Instruct-Q4_K_M.gguf",
        "size_gb": 42.0,
        "size_bytes": 42000000000,
        "sha256": "",
        "description": "Meta's flagship 70B. GPT-4 class. Needs 48GB+ RAM or powerful GPU.",
        "chat_format": "llama-3",
        "tier": 3,
        "license": "Llama 3.3 Community",
    },
    "llama-3.1-8b": {
        "name": "Llama 3.1 8B Instruct (Q4_K_M)",
        "repo_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "hf_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "size_gb": 4.9,
        "size_bytes": 4920000000,
        "sha256": "",
        "description": "Meta's workhorse 8B. 128K context window. Excellent all-rounder.",
        "chat_format": "llama-3",
        "tier": 3,
        "license": "Llama 3.1 Community",
    },
    "gemma-3-4b": {
        "name": "Gemma 3 4B Instruct (Q4_K_M)",
        "repo_id": "ggml-org/gemma-3-4b-it-GGUF",
        "hf_id": "google/gemma-3-4b-it",
        "filename": "gemma-3-4b-it-Q4_K_M.gguf",
        "size_gb": 2.8,
        "size_bytes": 2800000000,
        "sha256": "",
        "description": "Google's Gemma 3. Multimodal, 128K context, strong reasoning. Gemma ToS.",
        "chat_format": "gemma",
        "tier": 3,
        "license": "Gemma ToS",
    },
    "gemma-2-9b": {
        "name": "Gemma 2 9B Instruct (Q4_K_M)",
        "repo_id": "bartowski/gemma-2-9b-it-GGUF",
        "hf_id": "google/gemma-2-9b-it",
        "filename": "gemma-2-9b-it-Q4_K_M.gguf",
        "size_gb": 5.8,
        "size_bytes": 5800000000,
        "sha256": "",
        "description": "Google's Gemma 2 9B. Outstanding instruction following. Gemma ToS.",
        "chat_format": "gemma",
        "tier": 3,
        "license": "Gemma ToS",
    },
    "phi-4-mini": {
        "name": "Phi-4 Mini Instruct (Q4_K_M)",
        "repo_id": "unsloth/Phi-4-mini-instruct-GGUF",
        "hf_id": "microsoft/Phi-4-mini-instruct",
        "filename": "Phi-4-mini-instruct-Q4_K_M.gguf",
        "size_gb": 2.4,
        "size_bytes": 2400000000,
        "sha256": "",
        "description": "Microsoft's Phi-4 Mini. Reasoning-dense, compact powerhouse. MIT license.",
        "chat_format": "chatml",
        "tier": 3,
        "license": "MIT",
    },
    "phi-3-mini": {
        "name": "Phi-3 Mini 3.8B (Q4_K_M)",
        "repo_id": "bartowski/Phi-3.1-mini-4k-instruct-GGUF",
        "hf_id": "microsoft/Phi-3-mini-4k-instruct",
        "filename": "Phi-3.1-mini-4k-instruct-Q4_K_M.gguf",
        "size_gb": 2.4,
        "size_bytes": 2393232096,
        "sha256": "d6d25bf078321bea4a079c727b273cb0b5a2e0b4cf3add0f7a2c8e43075c414f",
        "description": "Microsoft's Phi-3. Compact, fast, great for general chat.",
        "chat_format": "chatml",
        "tier": 3,
        "license": "MIT",
    },
    "codellama-7b": {
        "name": "CodeLlama 7B Instruct (Q4_K_M)",
        "repo_id": "TheBloke/CodeLlama-7B-Instruct-GGUF",
        "hf_id": "meta-llama/CodeLlama-7b-Instruct-hf",
        "filename": "codellama-7b-instruct.Q4_K_M.gguf",
        "size_gb": 4.1,
        "size_bytes": 4081095360,
        "sha256": "0701500c591c2c1b910516658e58044cdfa07b2e8b5a2e3b6808d983441daf1a",
        "description": "Meta's code specialist. Writes, debugs, and explains code.",
        "chat_format": "llama-2",
        "tier": 3,
        "license": "Llama 2 Community",
    },
    "falcon-h1-7b": {
        "name": "Falcon H1 7B Instruct (Q4_K_M)",
        "repo_id": "unsloth/Falcon-H1-7B-Instruct-GGUF",
        "hf_id": "tiiuae/Falcon-H1-7B-Instruct",
        "filename": "Falcon-H1-7B-Instruct-Q4_K_M.gguf",
        "size_gb": 4.5,
        "size_bytes": 4500000000,
        "sha256": "",
        "description": "TII's hybrid Falcon H1. Latest architecture with SSMs + attention.",
        "chat_format": "chatml",
        "tier": 2,
        "license": "Apache 2.0",
    },
    "olmo-3-1b": {
        "name": "OLMo 2 1B Instruct (Q4_K_M)",
        "repo_id": "bartowski/OLMo-2-0325-32B-Instruct-GGUF",
        "hf_id": "allenai/OLMo-2-0325-32B-Instruct",
        "filename": "OLMo-2-0325-32B-Instruct-Q4_K_M.gguf",
        "size_gb": 19.5,
        "size_bytes": 19500000000,
        "sha256": "",
        "description": "Allen AI's OLMo 2 32B. Fully open — data, code, eval. Apache 2.0.",
        "chat_format": "chatml",
        "tier": 1,
        "license": "Apache 2.0",
    },
    "qwen2.5-coder-7b": {
        "name": "Qwen2.5 Coder 7B Instruct (Q4_K_M)",
        "repo_id": "bartowski/Qwen2.5.1-Coder-7B-Instruct-GGUF",
        "hf_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "filename": "Qwen2.5.1-Coder-7B-Instruct-Q4_K_M.gguf",
        "size_gb": 4.7,
        "size_bytes": 4700000000,
        "sha256": "",
        "description": "Alibaba's Qwen2.5 Coder. Top-tier code generation. Apache 2.0.",
        "chat_format": "chatml",
        "tier": 2,
        "license": "Apache 2.0",
    },
}
