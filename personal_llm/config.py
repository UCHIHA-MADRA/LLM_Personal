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
    env_dir = os.environ.get('PERSONAL_LLM_DIR')
    if env_dir:
        data_dir = Path(env_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

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
# Uses PERSONAL_LLM_DIR (writable %LOCALAPPDATA% in production) instead of
# BASE_DIR (which is read-only in Electron-packaged builds)
MODELS_DIR = PERSONAL_LLM_DIR / "models"
try:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
except PermissionError:
    pass  # Will be created on first model download

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
UI_HOST = "0.0.0.0"  # Enable local network access

# ─── RAG Settings ─────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 5

# ─── Context Intelligence ─────────────────────────────────────────

# Self-Refine limits
MAX_REFINE_DEPTH = 2

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
        "best_at": "General Chat, Research",
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
        "best_at": "ML Research, Text Generation",
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
        "best_at": "Research, Creative Writing",
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
        "best_at": "Research, Efficiency",
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
        "best_at": "Reasoning, Chain-of-Thought, Math",
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
        "best_at": "Code Generation, Debugging",
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
        "best_at": "General Chat, Instruction Following",
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
        "best_at": "Multilingual, Reasoning, General Chat",
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
        "best_at": "On-Device, Low RAM, Quick Answers",
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
        "best_at": "General Chat, Summarization",
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
        "best_at": "General Chat, Instruction Following",
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
        "best_at": "General Chat, Commercial Use",
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
        "best_at": "Long Context, Low Memory",
        "chat_format": None,
        "tier": 2,
        "license": "Apache 2.0",
    },
    "qwen2.5-coder-7b": {
        "name": "Qwen2.5 Coder 7B Instruct (Q4_K_M)",
        "repo_id": "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
        "hf_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "filename": "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
        "size_gb": 4.7,
        "size_bytes": 4683073504,
        "sha256": "76575bbb1de1647841ff89b72c89a5903873ad523f07d577fbca0e1bfcf6263e",
        "description": "Alibaba's top-tier coding model. Superior at programming, debugging, and logic. Apache 2.0.",
        "best_at": "Code Generation, Debugging, Logic",
        "chat_format": "chatml",
        "tier": 2,
        "license": "Apache 2.0",
    },
    "qwen2.5-coder-3b": {
        "name": "Qwen2.5 Coder 3B Instruct (Q4_K_M)",
        "repo_id": "bartowski/Qwen2.5-Coder-3B-Instruct-GGUF",
        "hf_id": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "filename": "Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf",
        "size_gb": 1.9,
        "size_bytes": 1929903360,
        "sha256": "819e26355d6996f75a334af05f9ff7958fa17f017a5cdfbbe509eab1f37490f6",
        "description": "Lightweight specialist coding model. Extremely fast for local code completion.",
        "best_at": "On-Device Coding, Quick Fixes",
        "chat_format": "chatml",
        "tier": 2,
        "license": "Apache 2.0",
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
        "best_at": "General Knowledge, Multilingual",
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
        "best_at": "General Chat, On-Device",
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
        "best_at": "Ultra Low RAM, Quick Answers",
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
        "best_at": "GPT-4 Class, Reasoning, General",
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
        "best_at": "General Chat, Long Context",
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
        "best_at": "Multimodal, Reasoning, Long Context",
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
        "best_at": "Instruction Following, General Chat",
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
        "best_at": "Reasoning, Math, Compact",
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
        "best_at": "General Chat, Compact",
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
        "best_at": "Code Generation, Code Explanation",
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
        "best_at": "General Chat, Hybrid Architecture",
        "chat_format": "chatml",
        "tier": 2,
        "license": "Apache 2.0",
    },
    "olmo-2-32b": {
        "name": "OLMo 2 32B Instruct (Q4_K_M)",
        "repo_id": "bartowski/OLMo-2-0325-32B-Instruct-GGUF",
        "hf_id": "allenai/OLMo-2-0325-32B-Instruct",
        "filename": "OLMo-2-0325-32B-Instruct-Q4_K_M.gguf",
        "size_gb": 19.5,
        "size_bytes": 19500000000,
        "sha256": "",
        "description": "Allen AI's OLMo 2 32B. Fully open — data, code, eval. Apache 2.0.",
        "best_at": "Research, General Knowledge",
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
        "best_at": "Code Generation, Debugging",
        "chat_format": "chatml",
        "tier": 2,
        "license": "Apache 2.0",
    },

    # ══════════════════════════════════════════════════════════════
    # EXPANDED TOP TIER (> 30B Params)
    # ══════════════════════════════════════════════════════════════

    "nemotron-70b": {
        "name": "Nemotron 70B Instruct (Q4_K_M)",
        "repo_id": "bartowski/Llama-3.1-Nemotron-70B-Instruct-HF-GGUF",
        "hf_id": "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        "filename": "Llama-3.1-Nemotron-70B-Instruct-HF-Q4_K_M.gguf",
        "size_gb": 42.1,
        "size_bytes": 42100000000,
        "sha256": "",
        "description": "NVIDIA's mega-model fine-tuned from Llama 3.1. Exceptional logic and alignment.",
        "best_at": "Logic, Alignment, Reasoning",
        "chat_format": "llama-3",
        "tier": 3,
        "license": "Llama 3.1 Community",
    },
    "yi-1.5-34b": {
        "name": "Yi 1.5 34B Chat (Q4_K_M)",
        "repo_id": "bartowski/Yi-1.5-34B-Chat-GGUF",
        "hf_id": "01-ai/Yi-1.5-34B-Chat",
        "filename": "Yi-1.5-34B-Chat-Q4_K_M.gguf",
        "size_gb": 20.2,
        "size_bytes": 20200000000,
        "sha256": "",
        "description": "01.AI's Yi 1.5 34B model. Incredibly strong bilingual performance.",
        "best_at": "Bilingual (EN/CN), General Knowledge",
        "chat_format": "chatml",
        "tier": 2,
        "license": "Apache 2.0",
    },
    "command-r": {
        "name": "Command-R v01 (Q4_K_M)",
        "repo_id": "pmysl/c4ai-command-r-v01-GGUF",
        "hf_id": "CohereForAI/c4ai-command-r-v01",
        "filename": "c4ai-command-r-v01-Q4_K_M.gguf",
        "size_gb": 21.0,
        "size_bytes": 21000000000,
        "sha256": "",
        "description": "Cohere's open weights model optimized for conversational interaction and long context.",
        "best_at": "Conversation, Long Context, RAG",
        "chat_format": "command-r",
        "tier": 2,
        "license": "CC-BY-NC 4.0",
    },
    "mixtral-8x7b": {
        "name": "Mixtral 8x7B Instruct (Q4_K_M)",
        "repo_id": "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
        "hf_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "filename": "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
        "size_gb": 26.4,
        "size_bytes": 26400000000,
        "sha256": "",
        "description": "Mistral's MoE (Mixture of Experts). Fast generation, top-tier reasoning. Apache 2.0.",
        "best_at": "Reasoning, Fast Generation, MoE",
        "chat_format": "mistral-instruct",
        "tier": 2,
        "license": "Apache 2.0",
    },
    "qwen2.5-32b": {
        "name": "Qwen2.5 32B Instruct (Q4_K_M)",
        "repo_id": "bartowski/Qwen2.5-32B-Instruct-GGUF",
        "hf_id": "Qwen/Qwen2.5-32B-Instruct",
        "filename": "Qwen2.5-32B-Instruct-Q4_K_M.gguf",
        "size_gb": 19.3,
        "size_bytes": 19300000000,
        "sha256": "",
        "description": "Alibaba's Qwen2.5 32B. One of the best open models in its size class. Apache 2.0.",
        "best_at": "General Chat, Reasoning, Multilingual",
        "chat_format": "chatml",
        "tier": 2,
        "license": "Apache 2.0",
    },

    # ══════════════════════════════════════════════════════════════
    # EXPANDED TINY/MICRO TIER (< 3B Params)
    # ══════════════════════════════════════════════════════════════

    "qwen2.5-0.5b": {
        "name": "Qwen2.5 0.5B Instruct (Q4_K_M)",
        "repo_id": "bartowski/Qwen2.5-0.5B-Instruct-GGUF",
        "hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "filename": "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
        "size_gb": 0.4,
        "size_bytes": 400000000,
        "sha256": "",
        "description": "Microscopic 0.5B parameter model. Blazing fast, useful for basic queries.",
        "best_at": "Ultra Fast, Basic Q&A, On-Device",
        "chat_format": "chatml",
        "tier": 2,
        "license": "Apache 2.0",
    },
    "smollm2-1.7b": {
        "name": "SmolLM2 1.7B Instruct (Q4_K_M)",
        "repo_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF",
        "hf_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "filename": "smollm2-1.7b-instruct-q4_k_m.gguf",
        "size_gb": 1.1,
        "size_bytes": 1100000000,
        "sha256": "",
        "description": "HuggingFace's SmolLM2 1.7B. Best in class for on-device applications.",
        "best_at": "On-Device, Low RAM, Quick Chat",
        "chat_format": "chatml",
        "tier": 1,
        "license": "Apache 2.0",
    },
    "smollm2-360m": {
        "name": "SmolLM2 360M Instruct (Q4_K_M)",
        "repo_id": "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF",
        "hf_id": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "filename": "smollm2-360m-instruct-q4_k_m.gguf",
        "size_gb": 0.3,
        "size_bytes": 300000000,
        "sha256": "",
        "description": "Ultra-tiny 360M model. Ideal for simple local classification or low-RAM environments.",
        "best_at": "Classification, Ultra Low RAM",
        "chat_format": "chatml",
        "tier": 1,
        "license": "Apache 2.0",
    },
    "danube3-500m": {
        "name": "h2o-Danube3 500M Chat (Q4_K_M)",
        "repo_id": "bartowski/h2o-danube3-500m-chat-GGUF",
        "hf_id": "h2oai/h2o-danube3-500m-chat",
        "filename": "h2o-danube3-500m-chat-Q4_K_M.gguf",
        "size_gb": 0.4,
        "size_bytes": 400000000,
        "sha256": "",
        "description": "H2O.ai's 500M chat model. Surprisingly coherent conversational agent for its size.",
        "best_at": "Tiny Chat, On-Device",
        "chat_format": "chatml",
        "tier": 1,
        "license": "Apache 2.0",
    },
    "stablelm-2-1.6b": {
        "name": "StableLM 2 Zephyr 1.6B (Q4_K_M)",
        "repo_id": "bartowski/stablelm-2-zephyr-1_6b-GGUF",
        "hf_id": "stabilityai/stablelm-2-zephyr-1_6b",
        "filename": "stablelm-2-zephyr-1_6b-Q4_K_M.gguf",
        "size_gb": 1.1,
        "size_bytes": 1100000000,
        "sha256": "",
        "description": "StabilityAI's 1.6B model fine-tuned using the Zephyr recipe.",
        "best_at": "Compact Chat, Creative Writing",
        "chat_format": "chatml",
        "tier": 2,
        "license": "StabilityAI Non-Commercial",
    },
    "tinyllama-1.1b": {
        "name": "TinyLlama 1.1B Chat (Q4_K_M)",
        "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "hf_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "size_gb": 0.8,
        "size_bytes": 800000000,
        "sha256": "",
        "description": "The classic 1.1B model trained on 3T tokens. MIT License.",
        "best_at": "On-Device, Quick Answers",
        "chat_format": "chatml",
        "tier": 1,
        "license": "MIT",
    },
    "gemma-2-2b": {
        "name": "Gemma 2 2B Instruct (Q4_K_M)",
        "repo_id": "bartowski/gemma-2-2b-it-GGUF",
        "hf_id": "google/gemma-2-2b-it",
        "filename": "gemma-2-2b-it-Q4_K_M.gguf",
        "size_gb": 1.7,
        "size_bytes": 1700000000,
        "sha256": "",
        "description": "Google's ultra-efficient 2B model outperforming much larger legacy models.",
        "best_at": "Efficiency, General Chat",
        "chat_format": "gemma",
        "tier": 3,
        "license": "Gemma ToS",
    },

    # ══════════════════════════════════════════════════════════════
    # EXPANDED SPECIALTY (Coding / Math)
    # ══════════════════════════════════════════════════════════════

    "phind-codellama-34b": {
        "name": "Phind CodeLlama 34B v2 (Q4_K_M)",
        "repo_id": "TheBloke/Phind-CodeLlama-34B-v2-GGUF",
        "hf_id": "Phind/Phind-CodeLlama-34B-v2",
        "filename": "phind-codellama-34b-v2.Q4_K_M.gguf",
        "size_gb": 20.2,
        "size_bytes": 20200000000,
        "sha256": "",
        "description": "Phind's heavy-hitter coding model. Highly ranked for software development tasks.",
        "best_at": "Software Development, Code Review",
        "chat_format": "llama-2",
        "tier": 3,
        "license": "Llama 2 Community",
    },
    "deepseek-math-7b": {
        "name": "DeepSeek Math 7B Instruct (Q4_K_M)",
        "repo_id": "TheBloke/deepseek-math-7B-instruct-GGUF",
        "hf_id": "deepseek-ai/deepseek-math-7b-instruct",
        "filename": "deepseek-math-7b-instruct.Q4_K_M.gguf",
        "size_gb": 4.1,
        "size_bytes": 4100000000,
        "sha256": "",
        "description": "DeepSeek's specialized mathematics and formalized logic model.",
        "best_at": "Mathematics, Formal Logic, Proofs",
        "chat_format": "chatml",
        "tier": 2,
        "license": "DeepSeek License",
    },
}
