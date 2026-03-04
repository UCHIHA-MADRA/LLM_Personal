# 🧠 Personal LLM 2.0 — Complete Project Report

> **Version:** 2.0.2
> **Date:** March 4, 2026 (Updated)
> **Author:** Built for Prabh's personal use
> **Location:** `c:\Users\prabh\Desktop\LLM_Personal\`

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [File-by-File Breakdown](#3-file-by-file-breakdown)
4. [Model Catalog](#4-model-catalog)
5. [Hardware Requirements](#5-hardware-requirements)
6. [Data Flow](#6-data-flow)
7. [Dependencies](#7-dependencies)
8. [Directory Structure](#8-directory-structure)
9. [Setup & Usage](#9-setup--usage)
10. [Security & Privacy](#10-security--privacy)

---

## 1. Project Overview

### What is This?

A **fully offline, private AI assistant** that runs entirely on your personal PC. It loads open-source Large Language Models (LLMs) directly into your computer's memory — no internet required after initial model download, no cloud APIs, no external servers, no Ollama, no Docker.

### Core Philosophy

| Principle                      | Implementation                                                                              |
| ------------------------------ | ------------------------------------------------------------------------------------------- |
| **Zero External Dependencies** | No Ollama, no Docker, no cloud APIs. Python process loads model files directly.             |
| **100% Private**               | All data stays on your disk. Chat history, documents, models — nothing leaves your machine. |
| **Offline-First**              | After one-time model download, zero network calls. Works in airplane mode.                  |
| **Your Hardware**              | Uses YOUR CPU, YOUR GPU, YOUR RAM. You own the entire stack.                                |

### What Can It Do?

- **💬 General Chat** — Multi-turn conversations like ChatGPT, but private
- **🧠 Reasoning** — Step-by-step logical thinking (DeepSeek-R1)
- **💻 Code Generation** — Write, debug, explain code (CodeLlama, Qwen)
- **📄 Document Q&A (RAG)** — Upload PDFs/text files, ask questions about them
- **🌐 Cloud Proxy** — Connect API keys (OpenAI, Groq, Together) to mix local and cloud AI (see [Section 3.4: api.py](#34-apipy) for details)
- **🔄 Multi-Model** — Switch between 28 different AI models on-the-fly
- **📱 Mobile App** — Connect locally from your phone with the React Native Expo app (see [Section 3.12: Mobile App](#312-mobile-app-mobile))
- **💾 Persistent History** — All conversations saved locally as JSON

---

## 2. Architecture

### System Diagram

```
┌───────────────────────────────────────────────────────────────┐
│                    YOUR PERSONAL PC                           │
│                                                               │
│  ┌──────────────────────┐   ┌─────────────────────────────┐   │
│  │   Electron Desktop   │   │    Mobile App (Expo)        │   │
│  │ http://127.0.0.1:PORT│   │ http://<LAN_IP>:8000        │   │
│  └──────────┬───────────┘   └─────────────┬───────────────┘   │
│             │                             │                   │
│  ┌──────────▼─────────────────────────────▼───────────────┐   │
│  │          FastAPI Backend (api.py) — port 8000          │   │
│  │  • REST API for models, status, chat, docs             │   │
│  │  • Cloud inference proxy (OpenAI/Groq/Together)        │   │
│  │  • Background model downloading                        │   │
│  ├────────────────────────────────────────────────────────┤   │
│  │          Gradio Web UI (web_ui.py) — port 7865         │   │
│  │  • Legacy standalone browser UI (localhost only)       │   │
│  └──────┬───────────┬──────────────┬──────────────────────┘   │
│         │           │              │                          │
│  ┌──────▼─────┐ ┌───▼──────┐ ┌────▼───────────────────┐       │
│  │chat_engine │ │model_mgr │ │  knowledge_base.py     │       │
│  │  .py       │ │  .py     │ │  (RAG System)          │       │
│  │            │ │          │ │                        │       │
│  │ • Sessions │ │ • List   │ │ • Ingest documents     │       │
│  │ • History  │ │ • D-load │ │ • Config │ │ • ChromaDB Retrieval   │       │
│  └──────┬─────┘ └──────────┘ └────────────────────────┘       │
│         │                                                     │
│  ┌──────▼──────────────────────────────────────────────┐      │
│  │              llm_engine.py (Core Engine)            │      │
│  │  • llama-cpp-python for GGUF model execution        │      │
│  │  • Full GPU Acceleration (CUDA/Metal)               │      │
│  └──────┬──────────────────────────────────────────────┘      │
│         │                                                     │
│  ┌──────▼──────────────────────────────────────────────┐      │
│  │          GGUF Model Files (on your disk)            │      │
│  │         personal_llm_models/*.gguf                  │      │
│  └─────────────────────────────────────────────────────┘      │
└───────────────────────────────────────────────────────────────┘
```

> **Note on ports:** The system uses multiple ports depending on the interface:
> - **Port 8000** — FastAPI backend (serves both the Electron desktop app and mobile app)
> - **Port 7865** — Gradio standalone web UI (legacy launcher, localhost only)
> - The Electron app dynamically finds an available port for its window, then connects to the FastAPI backend.

### How llama-cpp-python Works

`llama-cpp-python` is not a server — it's a **Python library** that wraps `llama.cpp` (a C/C++ LLM inference engine). When you call `Llama(model_path="model.gguf")`, it:

1. **Reads** the `.gguf` file from disk (just a binary file)
2. **Loads** the model weights into RAM (and optionally GPU VRAM)
3. **Runs inference** using hand-optimized C++ code with SIMD instructions
4. **Returns tokens** directly to your Python code

There is **no server process**, no HTTP calls, no Docker container. Your Python script literally IS the LLM.

---

## 3. File-by-File Breakdown

### 3.1 `__init__.py` (8 lines)

**Purpose:** Package initialization. Makes `personal_llm` importable as a Python package.

**What it contains:**

- Module docstring describing the project
- `__version__ = "2.0.2"` — version identifier

**Key detail:** This file is intentionally minimal. It does NOT auto-import heavy modules like `llama_cpp` or `chromadb` to keep import times fast.

---

### 3.2 `config.py` (489 lines)

**Purpose:** Central configuration file. Every setting the system uses is defined here.

**Sections:**

#### PyInstaller-Safe Path Resolution (Lines 13–43)

The module detects whether the code is running as a normal Python script or as a frozen `.exe` (via PyInstaller/Electron). In frozen mode, user data paths use `%LOCALAPPDATA%\PersonalLLM` instead of the project directory (which may be read-only under `C:\Program Files`).

```
BASE_DIR            → LLM_Personal/                              (project root)
PERSONAL_LLM_DIR    → LLM_Personal/personal_llm/                 (package directory)
MODELS_DIR          → LLM_Personal/personal_llm_models/          (GGUF files stored here)
CHAT_HISTORY_DIR    → LLM_Personal/personal_llm/chat_history/    (JSON conversation files)
KNOWLEDGE_DB_DIR    → LLM_Personal/personal_llm/knowledge_db/    (ChromaDB vector database)
DOCUMENTS_DIR       → LLM_Personal/personal_llm/documents/       (uploaded documents)
```

> [!TIP]
> **Data Persistence:** In development mode, these folders are created inside `personal_llm/`. However, if the app is bundled as a Windows Executable, it uses `%LOCALAPPDATA%\PersonalLLM` instead to prevent Permission Errors when installed in `C:\Program Files`.

#### Model Defaults (Lines 29-40)

| Setting         | Default            | Env Variable              | Purpose                                   |
| --------------- | ------------------ | ------------------------- | ----------------------------------------- |
| `DEFAULT_MODEL` | `""` (auto-detect) | `PERSONAL_LLM_MODEL`      | Which model to load on startup            |
| `N_GPU_LAYERS`  | `-1` (all layers)  | `PERSONAL_LLM_GPU_LAYERS` | How many model layers to put on GPU       |
| `CONTEXT_SIZE`  | `4096` tokens      | `PERSONAL_LLM_CONTEXT`    | How much conversation the model remembers |
| `MAX_TOKENS`    | `2048` tokens      | `PERSONAL_LLM_MAX_TOKENS` | Maximum response length                   |

**Key design:** Every setting can be overridden via environment variables, so you can tune without editing code.

#### Generation Parameters (Lines 42-46)

| Parameter        | Value | Effect                                                                  |
| ---------------- | ----- | ----------------------------------------------------------------------- |
| `TEMPERATURE`    | `0.7` | Controls randomness. Lower = more deterministic, higher = more creative |
| `TOP_P`          | `0.9` | Nucleus sampling — considers top 90% probability tokens                 |
| `TOP_K`          | `40`  | Only considers top 40 tokens at each step                               |
| `REPEAT_PENALTY` | `1.1` | Penalizes repeating the same words (prevents loops)                     |

#### Chat Memory (Line 49-50)

- `MAX_HISTORY_TURNS = 20` — Keeps last 20 turns (40 messages: 20 user + 20 assistant) in context

#### Web UI (Lines 52-54)

- `UI_PORT = 7865` — Gradio server port (legacy standalone mode)
- `UI_HOST = "0.0.0.0"` — Binds to all interfaces (allows LAN access for mobile/other devices)

> **Note:** The FastAPI backend (`api.py`) serves on port **8000** separately. See [Section 3.4](#3.4-api.py).

#### RAG Settings (Lines 94-98)

| Setting                | Value              | Purpose                                        |
| ---------------------- | ------------------ | ---------------------------------------------- |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence-transformers model for embedding text |
| `CHUNK_SIZE`           | `500` characters   | How big each document chunk is                 |
| `CHUNK_OVERLAP`        | `50` characters    | Overlap between chunks to preserve context     |
| `TOP_K_RESULTS`        | `5`                | Number of relevant chunks retrieved per query  |

#### Context Intelligence Settings (Lines 100-104)

| Setting             | Value | Purpose                                        |
| ------------------- | ----- | ---------------------------------------------- |
| `MAX_REFINE_DEPTH`  | `2`   | Maximum Self-Refine iterations per response    |

#### Model Catalog (Lines 105-489)

A dictionary of pre-configured GGUF models that can be downloaded. The catalog started with 5 models in v1.0 and was expanded to **28 models** in v2.0.1 (see [Section 4: Model Catalog](#4-model-catalog) for the full list). Each entry has:

- `name` — Human-readable name
- `repo_id` — HuggingFace repository (e.g., `bartowski/Phi-3.1-mini-4k-instruct-GGUF`)
- `filename` — Exact GGUF filename to download
- `size_gb` — Approximate download size
- `description` — What the model is good at
- `chat_format` — Template format for the model (e.g., `chatml`, `llama-3`, `mistral-instruct`)

---

### 3.3 `llm_engine.py` (375 lines)

**Purpose:** The **core inference engine**. This is where AI actually happens. It wraps `llama-cpp-python` to load GGUF model files and run text generation.

#### Class: `LLMEngine`

**State:**

- `self.model` — The loaded `Llama` object (or `None`)
- `self.model_path` — Path to the loaded `.gguf` file
- `self.model_name` — Filename stem (e.g., `Phi-3.1-mini-4k-instruct-Q4_K_M`)
- `self._is_loaded` — Boolean flag

**Methods:**

| Method                  | Lines   | Purpose                                                                                                                                                               |
| ----------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `check_dependencies()`  | 38-57   | Static method. Checks if `llama-cpp-python` is installed. If not, prints detailed installation instructions for CPU and GPU.                                          |
| `load()`                | 59-128  | Loads a GGUF model from disk. Accepts `model_path`, `n_gpu_layers`, `n_ctx`, `chat_format`. Prints loading progress. Returns `True/False`.                            |
| `unload()`              | 130-141 | Unloads the model, deletes the object, forces garbage collection to free memory.                                                                                      |
| `is_loaded`             | 143-145 | Property. `True` if a model is loaded and ready.                                                                                                                      |
| `generate()`            | 147-185 | Raw text generation from a prompt string. Supports streaming and complete modes. Parameters: `max_tokens`, `temperature`, `top_p`, `top_k`, `repeat_penalty`, `stop`. |
| `_generate_complete()`  | 187-209 | Internal. Generates entire response at once. Calls `self.model()` (llama-cpp-python's `__call__`).                                                                    |
| `_generate_stream()`    | 211-237 | Internal. Yields tokens one at a time. Uses `stream=True` parameter.                                                                                                  |
| `chat()`                | 239-267 | **Multi-turn chat**. Takes a list of `{"role": "user/assistant/system", "content": "..."}` messages. Uses the model's chat template.                                  |
| `_chat_complete()`      | 269-293 | Internal. Chat completion returning full response. Uses `model.create_chat_completion()`. Has **fallback**: if chat template fails, manually formats the prompt.      |
| `_chat_stream()`        | 295-324 | Internal. Streaming chat. Yields delta content from chunks. Same fallback mechanism.                                                                                  |
| `_format_chat_prompt()` | 326-339 | **Fallback prompt formatter**. If the model doesn't support chat templates, manually creates `System: ... User: ... Assistant:` format.                               |
| `get_info()`            | 341-363 | Returns dict with model info: name, path, size, context window.                                                                                                       |

**Singleton pattern:**

- `_engine` — Module-level variable
- `get_engine()` — Returns the global `LLMEngine` instance (creates one if it doesn't exist)

**Key design decisions:**

1. **Graceful degradation**: If `llama-cpp-python` isn't installed, the module still imports — it just can't load models.
2. **Fallback chat format**: If a model's chat template fails, the engine falls back to a simple text-based prompt format.
3. **Stop Token Handling**: Models often have specific "stop words" (e.g., `<|eot_id|>`) to know when to stop generating. `llm_engine` passes these to the inference engine to prevent hallucinating user replies.
4. **Garbage collection on unload**: When switching models, `gc.collect()` and `EmptyWorkingSet` (Windows) are called to aggressively free RAM/VRAM.

---

### 3.4 `api.py`

**Purpose:** FastAPI-based REST backend that serves as the unified API for the Electron desktop app, the mobile app, and cloud inference proxying. Also serves the React UI as static files for LAN browser access.

**Key Endpoints:**

| Endpoint              | Method | Purpose                                                       |
| --------------------- | ------ | ------------------------------------------------------------- |
| `/`                   | GET    | Serves the React UI (static Next.js export) for LAN browsers  |
| `/api/models`         | GET    | List all 28 catalog models with hardware fit info             |
| `/api/models/load`    | POST   | Load a specific model into memory                             |
| `/api/status`         | GET    | Current engine status (loaded model, memory usage)            |
| `/api/chat`           | POST   | Send a message and receive a response (supports streaming)    |
| `/api/knowledge/upload` | POST | Upload a document for RAG indexing                            |
| `/api/knowledge/query`  | POST | Query the knowledge base                                      |
| `/api/chat/cloud`     | POST   | **Cloud Proxy** — Forward requests to OpenAI/Groq/Together AI |
| `/api/models/download`| POST   | Background model downloading from HuggingFace                 |
| `/api/models/unload`  | POST   | Unload the current model to free GPU/RAM                      |
| `/api/conversations`  | GET    | List all conversations                                        |
| `/api/conversations/search` | GET | Search conversations by title or content                  |
| `/api/conversations/{id}` | GET/DELETE | Get or delete a specific conversation              |

**Static File Serving (v2.0.1):** The API now serves the Next.js static export at `/` so other devices on the LAN can access the full React UI by browsing to `http://<host-ip>:8000` — no Electron required. The `_find_ui_out_dir()` function searches for the `out/` directory in multiple candidate locations (dev mode, bundled Electron, PyInstaller).

**Cloud Proxy Feature:** The `/api/chat/cloud` endpoint allows users to configure external API keys (OpenAI, Groq, Together AI) so they can mix local inference with cloud models. The keys are stored locally and never leave the machine except when making the API call to the provider. Cloud conversations are persisted to `chat_history/` just like local chat.

**Runtime:** Runs on `0.0.0.0:8000` (accessible on LAN for mobile app connectivity). The Electron app connects to `127.0.0.1:8000`, while the mobile app and LAN browsers connect via the PC's LAN IP address.

> **Dependency Note:** FastAPI, `uvicorn`, and `python-multipart` are required for this module (see [Section 7: Dependencies](#7-dependencies)).

---

### 3.5 `model_manager.py` (335 lines)

**Purpose:** Manages GGUF model files — listing, downloading, and looking up models.

#### Class: `ModelManager`

**State:**

- `self.models_dir` — Path to the models directory (default: `personal_llm_models/`)

**Methods:**

| Method                         | Lines   | Purpose                                                                                                                                                                  |
| ------------------------------ | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `list_local_models()`          | 25-39   | Scans `models_dir` for `*.gguf` files. Returns list of dicts with `filename`, `path`, `size_gb`, `description`, `chat_format`. Cross-references with `MODEL_CATALOG`.    |
| `_find_catalog_entry()`        | 41-46   | Internal. Matches a filename to a catalog entry.                                                                                                                         |
| `get_model_path()`             | 48-63   | Resolves a catalog key (e.g., `"phi-3-mini"`) or filename to an absolute path. Returns `None` if not found.                                                              |
| `get_chat_format()`            | 65-75   | Looks up the correct chat format (e.g., `"chatml"`, `"llama-3"`) for a model.                                                                                            |
| `download_model()`             | 88-133  | Downloads a model from HuggingFace Hub. Uses `huggingface_hub.hf_hub_download()`. Saves directly to `models_dir`. **One-time operation** — skips if file already exists. |
| `download_model_stream()`      | 135-247 | Robust streaming download with progress callbacks, cancellation via `threading.Event`, SHA256 verification, and disk space checks. Used by the API's background download. |
| `download_model_interactive()` | 249-280 | Interactive CLI menu. Shows all catalog models with download status (`✅ Downloaded` or `📥 ~X GB`). User picks a number.                                                |
| `get_default_model()`          | 282-304 | Returns the best available model. Priority: config default → first local model → `None`.                                                                                 |
| `print_status()`               | 306-318 | Prints current model status to console.                                                                                                                                  |

**CLI Entry Point (Lines 198-211):**

- `python -m personal_llm.model_manager` — Shows status
- `python -m personal_llm.model_manager download` — Interactive download
- `python -m personal_llm.model_manager download phi-3-mini` — Download specific model

**Key design decisions:**

1. **Lazy HuggingFace import**: `huggingface_hub` is only imported when `download_model()` is called, so the module works offline without it.
2. **No symlinks**: `local_dir_use_symlinks=False` ensures the model file is a real file, not a symlink to HuggingFace's cache. This makes the models directory self-contained.

---

### 3.6 `chat_engine.py` (358 lines)

**Purpose:** Multi-turn conversation management with persistent history.

#### Class: `Conversation` (Lines 19-106)

Represents a single chat session.

**State:**

- `self.id` — 8-character UUID (e.g., `"3c499f72"`)
- `self.system_prompt` — The AI's personality/instructions
- `self.title` — Auto-generated from the first user message
- `self.messages` — List of `{"role": "user/assistant", "content": "..."}` dicts
- `self.created_at` / `self.updated_at` — ISO timestamps
- `self.model_name` — Which model was used

**Methods:**

| Method                      | Purpose                                                                                                                             |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `add_user_message()`        | Appends a user message. Auto-sets title from first message (first 60 chars).                                                        |
| `add_assistant_message()`   | Appends an assistant response.                                                                                                      |
| `get_context_messages()`    | Builds the message list for the LLM. Prepends system prompt. Limits to `MAX_HISTORY_TURNS` most recent turns to fit context window. |
| `to_dict()` / `from_dict()` | Serialization to/from JSON-compatible dict.                                                                                         |
| `export_markdown()`         | Exports the conversation as a Markdown string with `🧑 You` and `🤖 AI` headers.                                                    |

#### Class: `ChatEngine` (Lines 109-267)

Manages multiple conversations and interfaces with the LLM engine.

**State:**

- `self.engine` — Reference to `LLMEngine` instance
- `self.conversations` — Dict of `{id: Conversation}`
- `self.active_conversation` — Currently active conversation
- `self.history_dir` — Path to `chat_history/`

**Methods:**

| Method                  | Purpose                                                                                                                            |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `_load_history()`       | On startup, loads all `*.json` files from `chat_history/` into memory.                                                             |
| `_save_conversation()`  | Saves a conversation to `chat_history/{id}.json`.                                                                                  |
| `new_conversation()`    | Creates a new conversation with a system prompt.                                                                                   |
| `get_conversation()`    | Retrieves a conversation by ID.                                                                                                    |
| `list_conversations()`  | Lists all conversations, sorted by most recent.                                                                                    |
| `delete_conversation()` | Deletes a conversation from memory and disk.                                                                                       |
| `send_message()`        | **Main method.** Sends a user message, optionally injects RAG context, gets AI response (streaming or complete), saves to history. |
| `_stream_response()`    | Internal generator. Streams tokens from the engine, accumulates the full response, saves to history when done.                     |

**Key design decisions:**

1. **Dynamic Context Pruning**: `get_context_messages()` uses a token-budget approach — it estimates tokens (approximately 4 chars per token) and greedily fills the context window. It prioritizes System Prompt + RAG Context, then fills remaining space with the most recent history messages. This guarantees we never overflow the context window, preventing crashes even with long conversations.
2. **RAG injection**: When RAG context is available, it's appended to the system prompt.
3. **Stream-then-save**: Tokens are yielded to UI immediately, but saved to history only after completion.

---

### 3.7 `knowledge_base.py` (260 lines)

**Purpose:** RAG (Retrieval-Augmented Generation) system. Upload your documents, the system chunks and embeds them, then retrieves relevant context when you ask questions.

#### How RAG Works in This System

```
1. INGEST: Document → Split into chunks → Embed each chunk → Store in ChromaDB
2. QUERY:  User question → Embed question → Find similar chunks → Inject into LLM prompt
```

#### Helper Functions

| Function                 | Purpose                                                                                      |
| ------------------------ | -------------------------------------------------------------------------------------------- |
| `_get_chromadb()`        | Lazy import of `chromadb`. Only loaded when RAG is first used.                               |
| `_get_embedding_model()` | Lazy load of `sentence-transformers` model (`all-MiniLM-L6-v2`). ~80MB, downloaded once.     |
| `_chunk_text()`          | Splits text into overlapping chunks of `CHUNK_SIZE` characters with `CHUNK_OVERLAP` overlap. |

#### Class: `EmbeddingFunction` (Lines 49-57)

ChromaDB-compatible wrapper around `sentence-transformers`. Implements `__call__()` to convert text strings to float vectors.

#### Class: `KnowledgeBase` (Lines 75-259)

**State:**

- `self.db_dir` — Path to ChromaDB storage
- `self._client` — ChromaDB `PersistentClient`
- `self._collection` — ChromaDB collection named `"personal_knowledge"`
- `self._embedding_fn` — `EmbeddingFunction` instance
- `self._initialized` — Lazy init flag

**Methods:**

| Method                  | Lines   | Purpose                                                                                                                                                                                                                                                                                       |
| ----------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `_ensure_initialized()` | 90-105  | Lazy init. Creates ChromaDB client and collection only when first needed. Uses `cosine` similarity.                                                                                                                                                                                           |
| `add_text()`            | 107-133 | Adds raw text. Chunks it, generates IDs, upserts into ChromaDB.                                                                                                                                                                                                                               |
| `add_file()`            | 135-171 | Adds a file. Supports: `.txt`, `.md`, `.py`, `.js`, `.ts`, `.java`, `.cpp`, `.c`, `.go`, `.rs`, `.rb`, `.php`, `.css`, `.html`, `.xml`, `.yaml`, `.yml`, `.toml`, `.ini`, `.cfg`, `.conf`, `.sh`, `.bat`, `.ps1`, `.sql`, `.json`, `.csv`, `.pdf`. Falls back to text read for unknown types. |
| `_read_pdf()`           | 173-189 | Extracts text from PDF using `PyPDF2.PdfReader`.                                                                                                                                                                                                                                              |
| `query()`               | 191-224 | **Core retrieval method.** Takes a question, queries ChromaDB for similar chunks, returns formatted context string with source attribution.                                                                                                                                                   |
| `get_stats()`           | 226-233 | Returns `{total_chunks, db_path}`.                                                                                                                                                                                                                                                            |
| `clear()`               | 235-245 | Deletes and recreates the ChromaDB collection.                                                                                                                                                                                                                                                |
| `list_sources()`        | 247-259 | Returns list of unique source filenames in the knowledge base.                                                                                                                                                                                                                                |

**Key design decisions:**

1. **Lazy initialization**: Nothing loads until you actually use RAG. This keeps startup fast.
2. **Upsert, not insert**: `add_text()` uses `upsert()`, so re-adding the same document updates it instead of creating duplicates.
3. **Persistent storage**: ChromaDB uses `PersistentClient`, so your knowledge base survives restarts.

---

### 3.8 `hardware.py` (273 lines)

**Purpose:** Detects your PC's hardware specs and recommends which models will run well.

#### Detection Functions

| Function            | Lines  | What It Detects               | How                                                                                 |
| ------------------- | ------ | ----------------------------- | ----------------------------------------------------------------------------------- |
| `detect_hardware()` | 18-30  | Full system specs             | Orchestrator function                                                               |
| `_detect_cpu()`     | 33-54  | CPU name and core count       | `platform.processor()` + `wmic cpu get Name` (Windows) / `/proc/cpuinfo` (Linux)   |
| `_detect_ram()`     | 57-92  | Total RAM in GB               | `wmic memorychip` (Windows) / `/proc/meminfo` (Linux) / `sysctl hw.memsize` (macOS) |
| `_detect_gpu()`     | 95-131 | NVIDIA GPU name, VRAM, driver | `nvidia-smi --query-gpu=...` (cross-platform for NVIDIA GPUs)                       |

> **Cross-platform note:** The detection functions use platform-specific commands with fallbacks. `_detect_cpu()` tries Windows-first commands (`wmic`) and falls back to POSIX methods. `_detect_ram()` has dedicated paths for Windows (`wmic memorychip`), Linux (`/proc/meminfo`), and macOS (`sysctl hw.memsize`). `_detect_gpu()` calls `nvidia-smi`, which is available on all platforms with NVIDIA drivers installed. Apple Silicon (M1/M2/M3) GPU detection is handled implicitly — `llama-cpp-python` auto-uses Metal when available.

---

### 3.8a `llmfit_wrapper.py` (89 lines)

**Purpose:** Wrapper around the `llmfit.exe` binary tool for hardware compatibility scoring. Calculates whether a specific model will fit in your GPU/RAM and estimates tokens-per-second performance.

#### Function: `get_model_fit_info(hf_id: str)`

Executes `llmfit.exe --json info <hf_model_id>` and parses the JSON output. Returns:

| Field | Type | Purpose |
|-------|------|---------|
| `fit_level` | str | "Perfect", "Good", "Tight", "Too Large" |
| `estimated_tps` | float | Predicted tokens per second on your hardware |
| `memory_required_gb` | float | How much RAM/VRAM the model needs |
| `score` | float | Overall compatibility score (0-100) |

**Key design decisions:**

1. **Memory caching**: Results are cached in `_cache` dict to avoid repeated subprocess calls for the same model.
2. **Windows subprocess handling**: Uses `STARTF_USESHOWWINDOW` to prevent console window popups when running as a GUI app.
3. **Graceful failure**: Returns `None` if `llmfit.exe` is not found, times out, or returns invalid JSON.

#### Recommendation Engine

**`recommend_models()` (Lines 134-178):**

For each model in `MODEL_CATALOG`, calculates:

- `fits_vram`: Model size < GPU VRAM → Can run entirely on GPU (fast)
- `fits_ram`: Model size < 70% of RAM → Can run on CPU (leaves 30% for OS)
- `recommended_mode`: 🟢 GPU / 🟡 CPU / 🔴 Too large

Sorts results: GPU-fit models first, then CPU-fit, then too large.

**`print_hardware_report()` (Lines 181-250):**

Prints a formatted report showing:

- System info (OS, CPU, cores)
- RAM with tier assessment (Excellent/Good/Limited)
- GPU details with VRAM assessment
- All models with compatibility status
- Best recommended model

---

### 3.9 `web_ui.py` (796 lines)

**Purpose:** Standalone Gradio web interface. This is the legacy browser-based UI accessible at `http://127.0.0.1:7865`. In v2.0, the primary desktop experience uses Electron + FastAPI instead (see [Section 3.11](#311-electron-desktop-app-ui)), but this Gradio UI remains functional as a standalone launcher.

#### Custom CSS (Lines 29-113)

Implements a **dark glassmorphism** theme:

- Gradient title text (`#667eea → #764ba2`, indigo to purple)
- Rounded borders with subtle glass borders (`rgba(255,255,255,0.08)`)
- Hover animations on buttons (lift + glow shadow)
- Focus glow on input fields
- Inter font family
- Max width of 1200px centered

#### Class: `PersonalLLMUI` (Lines 117-478)

**State:**

- `self.engine` — `LLMEngine` instance
- `self.model_manager` — `ModelManager` instance
- `self.chat_engine` — `ChatEngine` instance
- `self.knowledge_base` — `KnowledgeBase` (lazy loaded)
- `self._current_conversation` — Active `Conversation` object

**UI Methods:**

| Method                    | Purpose                                                                                             |
| ------------------------- | --------------------------------------------------------------------------------------------------- |
| `_get_available_models()` | Lists `.gguf` files in models directory for dropdown                                                |
| `_get_status_text()`      | Builds status bar: `🟢 Model: X | Size: Y GB | Context: Z tokens | 🔒 Fully Offline`               |
| `_load_model()`           | Loads model from dropdown selection. Gets chat format from catalog.                                 |
| `_chat_respond()`         | **Main chat handler.** Streaming response. Handles RAG context injection. Yields updated chat history for real-time display. |
| `_new_chat()`             | Resets chat history, creates new conversation.                                                      |
| `_upload_document()`      | Uploads file to knowledge base via `KnowledgeBase.add_file()`.                                      |
| `_export_chat()`          | Exports current conversation to `personal_llm/exports/chat_{id}.md`.                                |
| `_clear_knowledge_base()` | Clears all RAG documents.                                                                           |
| `_get_kb_stats()`         | Shows knowledge base statistics.                                                                    |

**UI Layout (Lines 305-478):**

```
┌─────────────────────────────────────────────────┐
│              🧠 Personal LLM                     │
│   Your private AI — 100% on your hardware        │
├─────────────────────────────────────────────────┤
│ 🟢 Model: Phi-3.1... | Size: 2.4 GB | 🔒 Offline│
├──────────────────────────┬──────────────────────┤
│                          │ 🤖 Model              │
│                          │ [Dropdown    ▾]       │
│                          │ [Load Model]          │
│      Chat Window         │                      │
│   (streaming responses)  │ 📝 System Prompt      │
│                          │ [Textarea]            │
│                          │                      │
│                          │ ⚙️ Settings           │
│                          │ Temperature: 0.7     │
│                          │ Max Tokens: 2048     │
├──────────────────────────┤                      │
│ [Type message...]  [Send]│ 📄 Knowledge Base     │
│ [🗨️ New Chat] [📤 Export] │ ☐ Use document ctx   │
│                          │ [Upload] [Stats]     │
└──────────────────────────┴──────────────────────┘
```

**Gradio theme:**

- `gr.themes.Soft` with `indigo` primary, `purple` secondary, `slate` neutral
- Font: Inter family
- Dark mode with custom CSS overlay

#### `launch_ui()` Function (Lines 481-509)

Top-level launcher:

1. Creates `PersonalLLMUI` instance
2. Auto-loads default model if available
3. Creates initial conversation
4. Builds the Gradio app
5. Launches at `127.0.0.1:7865` with `inbrowser=True` (auto-opens browser)

---

### 3.10 `setup_models.py` (78 lines)

**Purpose:** Interactive command-line tool to download GGUF models.

**Flow:**

1. Prints banner
2. Shows already-downloaded models (if any)
3. Shows hardware report (CPU, RAM, GPU, model recommendations)
4. Checks if `huggingface-hub` is installed
5. Enters interactive loop:

- Shows catalog with download status
- User picks a model number
- Downloads from HuggingFace Hub
- Asks if user wants another

6. Prints final status and launch command

**Key detail:** This is the **only script that requires internet**. Everything else is offline.

---

### 3.11 Electron Desktop App (`ui/`)

**Purpose:** The primary v2.0 desktop experience. A standalone React + Electron application that bundles a Python interpreter and wraps the FastAPI backend in a native window.

---

### 3.11a `context_engine.py` (498 lines)

**Purpose:** The Context Intelligence Engine — research-backed techniques to improve small LLM response quality. Orchestrates 4 layers of processing that run entirely locally.

#### Layers

| Layer | Technique | Source |
|-------|-----------|--------|
| 1. **RAG Retrieval** | Pull relevant chunks from uploaded documents | Lewis et al., 2020 |
| 2. **Recursive Context** | Decompose complex queries into sub-questions | MIT RLMs, 2025 |
| 3. **Self-Refine** | Auto-critique and improve answers iteratively | Madaan et al., ICLR 2024 |
| 4. **Adaptive Prompting** | Chain-of-Thought and task-specific templates | Wei et al., 2022 |

#### Function: `classify_query(message)` (Lines 102-132)

Classifies user queries as `"simple"`, `"complex"`, or `"code"` using regex patterns. Used to decide which processing layers to activate (e.g., complex queries trigger recursive decomposition).

#### Class: `ContextEngine` (Lines 137-497)

**Methods:**

| Method | Lines | Purpose |
|--------|-------|---------|
| `retrieve_context()` | 157-172 | Query knowledge base for relevant document chunks |
| `recursive_retrieve()` | 176-230 | Decompose complex queries into sub-questions, retrieve context for each independently, then merge results |
| `self_refine()` | 234-301 | Iterative critique → improve loop. Generates answer, critiques it, then produces improved version |
| `build_system_prompt()` | 305-341 | Builds optimized system prompt combining base prompt + RAG context + CoT instructions |
| `process()` | 345-398 | Full pipeline (non-streaming): RAG → Recursive → Generate → Self-Refine |
| `process_stream()` | 400-497 | Streaming pipeline: yields `status`, `context`, `token`, `refine_start`, `refine_token`, and `done` events |

**Key design decisions:**

1. **Adaptive activation**: Only activates layers that will help. A simple "hello" doesn't trigger RAG or CoT.
2. **Graceful degradation**: If knowledge base fails, falls back to direct generation without crashing.
3. **Prompt templates**: All prompts stored in a `PROMPTS` dictionary for easy customization.

---

**How Electron works:**

1. Electron finds or extracts the bundled `python-embed` (embedded Python 3.11 with all dependencies pre-installed).
2. On first run, Python is copied from `resources/python-embed/` to `%LOCALAPPDATA%\PersonalLLM\python-embed\` to avoid Electron's code signing corrupting Python binaries.
3. The Electron main process spawns the Python backend via `-c "from personal_llm.api import launch_api; launch_api()"`.
4. A splash screen displays "Starting AI engine..." while waiting up to 120 seconds for the backend to respond.
5. The React frontend communicates with the FastAPI backend via `http://127.0.0.1:8000` using a dynamic `API_BASE` constant.
6. On window close, the Electron process gracefully shuts down the Python backend.

**Key files in `ui/`:**

| File/Dir | Purpose |
|---|---|
| `app/page.tsx` | Main React UI (chat, model selector, settings, RAG panel) |
| `electron/main.js` | Electron main process (Python lifecycle, splash screen, window management) |
| `python-embed/` | Bundled Python 3.11 with all pip packages pre-installed |
| `python-embed/python311._pth` | Path config — includes `..` to resolve `personal_llm/` from `resources/` |
| `package.json` | Build config with `extraResources` for `personal_llm/` and `python-embed/` |

**Packaging Architecture (v2.0.1):**

```
dist-electron2/win-unpacked/
├── Personal LLM.exe          # Electron executable
├── resources/
│   ├── app.asar              # Bundled React UI + main.js
│   ├── personal_llm/         # Python package (extraResources)
│   │   ├── api.py
│   │   ├── config.py
│   │   └── ...
│   └── python-embed/         # Embedded Python 3.11 (extraResources)
│       ├── python.exe
│       ├── python311._pth    # Includes '..' to reach resources/
│       └── Lib/site-packages/ # fastapi, uvicorn, llama-cpp-python, etc.
```

**Critical Build Notes:**
- The `python311._pth` file must contain `..` so embedded Python can find `personal_llm/` in the parent `resources/` directory.
- `python-multipart` must be installed in `python-embed/` for the file upload endpoint to work.
- The `findPython()` function copies `python-embed` to `%LOCALAPPDATA%` on first run because Electron's code signing process corrupts Python's `.pyd`/`.dll` files.

> **Relationship to `desktop_app.py`:** The older `desktop_app.py` (88 lines) at the project root is a lightweight **pywebview** wrapper that serves the Gradio UI in a native window. It was the v1.0 desktop approach. In v2.0, the Electron app in `ui/` replaces it as the primary desktop experience.

#### `desktop_app.py` (88 lines) — Legacy Desktop Wrapper

**Flow:**

1. Finds an available network port dynamically.
2. Creates an HTML splash screen and displays it using `pywebview`.
3. Starts the Gradio server in a background thread.
4. Polls the Gradio localhost URL until it responds with HTTP 200.
5. Checks if models exist. If none, shows a native Windows error dialog.
6. Replaces the splash screen with the loaded Gradio UI.
7. Gracefully unloads the LLM when the user closes the window.

**CLI Arguments:**

| Flag             | Purpose                           |
| ---------------- | --------------------------------- |
| `--check`        | Only verify setup, don't launch   |
| `--share`        | Create a public Gradio share link |
| `--model <name>` | Load a specific model on startup  |

---

### 3.12 Mobile App (`mobile/`)

**Purpose:** React Native (Expo) Android app that connects to the FastAPI backend over your local network, providing a mobile chat interface.

**Key details:**

- **Framework:** React Native with Expo
- **Connection:** Connects to `http://<YOUR_PC_IP>:8000` over LAN (Wi-Fi). The mobile device must be on the same network as the PC running the backend.
- **Features:** Model switching, multi-turn chat, persistent local settings (stored on-device via AsyncStorage)
- **Routing:** Uses Expo Router with `app/` directory (index screen for chat, settings screen for configuration)
- **Build:** Uses EAS (Expo Application Services) for cloud builds. `eas.json` contains the build configuration. Outputs `.aab` (Android App Bundle) or `.apk`.

> **Important:** The mobile app does NOT run any AI locally on the phone. It sends requests to the FastAPI backend running on your PC. If the PC is off or unreachable, the app cannot function.

---

### 3.13 Launch Website (`website/`)

**Purpose:** A Next.js 16 marketing/launch page for the project. Static site that showcases features and provides download links.

**Key details:**

- **Framework:** Next.js 16 (App Router)
- **Styling:** Tailwind CSS with a dark glassmorphism theme matching the desktop app
- **Animations:** Framer Motion for scroll-triggered animations and transitions
- **Build Output:** Static HTML export (`website/out/`) — can be deployed to any static hosting (GitHub Pages, Vercel, Netlify) for free
- **No backend:** This is a purely static site. It does not connect to or interact with the LLM backend.

---

### 3.14 `launch_personal_llm.py` (127 lines)

**Purpose:** Legacy one-click CLI launcher. The single entry point to start the Gradio web UI.

**Flow:**

1. **Banner** — Prints project title
2. **Check dependencies** — Verifies `llama-cpp-python` and `gradio` are installed
3. **Hardware report** — Calls `hardware.print_hardware_report()` to show PC specs
4. **Check models** — Verifies at least one `.gguf` model exists. If not, offers interactive download.
5. **Launch** — Imports and calls `web_ui.launch_ui()`

> **Note:** In v2.0, the preferred launch method is `npm run electron:dev` (from the `ui/` folder) or `npm run electron:start`. This legacy launcher still works for the standalone Gradio experience.

---

### 3.15 Build & Packaging Files

#### `personal_llm.spec` (PyInstaller Config)

PyInstaller spec file that defines how to compile the Python backend into a standalone Windows `.exe`.

**Key settings:**

- Uses `onedir` mode (not `onefile`) for faster startup without temporary extraction delay
- Bundles `llama_cpp` DLL dependencies for CPU inference
- Configures data paths to use `sys.executable` and `%LOCALAPPDATA%` for user data persistence

#### `installer.iss` (Inno Setup Script)

Inno Setup 6 compiler script that packages the PyInstaller `dist/PersonalLLM` folder into a professional Windows installer (`PersonalLLM_Setup_v1.0.0.exe`, ~300MB).

**Creates:**

- Desktop shortcut
- Start Menu entries
- Uninstaller support
- Standard Windows "Add/Remove Programs" entry

---

### 3.16 `requirements.txt` (19+ lines)

**Dependencies:**

| Package                 | Version | Purpose                                     | Required?            |
| ----------------------- | ------- | ------------------------------------------- | -------------------- |
| `llama-cpp-python`      | ≥0.3.0  | Core LLM engine (loads GGUF files)          | **Yes**              |
| `fastapi`               | ≥0.100  | REST API backend for desktop & mobile apps  | **Yes**              |
| `uvicorn`               | ≥0.20   | ASGI server to run FastAPI                  | **Yes**              |
| `python-multipart`      | ≥0.0.22 | File upload support for FastAPI             | **Yes**              |
| `httpx`                 | ≥0.24.0 | Async HTTP client                           | **Yes**              |
| `requests`              | ≥2.31.0 | Sync HTTP client                            | **Yes**              |
| `pydantic`              | ≥2.0.0  | Data validation                             | **Yes**              |
| `huggingface-hub`       | ≥0.20.0 | Download models (one-time)                  | Yes for setup        |
| `gradio`                | ≥4.0.0  | Legacy web UI framework                     | Only for legacy UI   |
| `chromadb`              | ≥0.5.0  | Vector database for RAG                     | Only for RAG         |
| `sentence-transformers` | ≥3.0.0  | Text embeddings for RAG                     | Only for RAG         |
| `PyPDF2`                | ≥3.0.0  | PDF text extraction                         | Only for PDFs        |
| `pywebview`             | ≥4.0    | Native window for legacy desktop_app.py     | Only for legacy app  |

**Python Version Required:** Python 3.10 or 3.11 recommended. (Tested on 3.10+)

> [!WARNING]
> **Heads Up:** `sentence-transformers` relies on **PyTorch**, which is a heavy download (~2.5 GB). Even though the embedding model itself is small (80 MB), the initial setup will pull this large dependency.

**No cloud dependencies.** Every package runs locally.

---

## 4. Model Catalog

The catalog was massively expanded in v2.0 to include **28 fully open-source and weights-available models**, categorized by tier:

- **Tier 1 (Fully Open):** OLMo 3, Pythia, GPT-NeoX, Cerebras-GPT, OpenCoder
- **Tier 2 (Open Weights/Code):** DeepSeek-R1 (Distill-Qwen), Qwen3, Mistral, Falcon 7B/Falcon3, MPT, RWKV, StarCoder2, YaLM 100B, DeepSeek Coder
- **Tier 3 (Restricted Open):** Llama 3.2 (1B/3B), Llama 3.3 70B, Llama 3.1 8B, Gemma 2/3, Phi-4 Mini, Phi-3, CodeLlama

**All models use Q4_K_M quantization** — a balanced format that compresses 16-bit weights to ~4 bits while preserving quality. This is the sweet spot between size and intelligence.

---

## 5. Hardware Requirements

### Minimum Requirements

| Component   | Minimum                     | Recommended                  |
| ----------- | --------------------------- | ---------------------------- |
| **RAM**     | 4 GB (2B model only)        | 8 GB+                        |
| **Storage** | 2 GB free                   | 10 GB+ (for multiple models) |
| **CPU**     | Any modern x64              | 4+ cores                     |
| **GPU**     | Not required                | NVIDIA with 4GB+ VRAM        |
| **OS**      | Windows 10/11, Linux, macOS | Any                          |

### RAM Usage by Model

| Model              | RAM Needed (CPU mode) | VRAM Needed (GPU mode) |
| ------------------ | --------------------- | ---------------------- |
| Llama 3.2 3B Q4    | ~3 GB                 | ~2 GB                  |
| Phi-3 Mini 3.8B Q4 | ~3.5 GB               | ~2.4 GB                |
| CodeLlama 7B Q4    | ~5.5 GB               | ~4.1 GB                |
| Mistral 7B Q4      | ~6 GB                 | ~4.4 GB                |
| DeepSeek-R1 7B Q4  | ~6.5 GB               | ~4.7 GB                |

### GPU Acceleration

- **NVIDIA**: Best support. Models load into VRAM, inference is **5-10x faster**.
- **Apple Silicon (M1/M2/M3)**: Supported via Metal. `llama-cpp-python` automatically uses the Neural Engine/GPU if installed correctly.
- **AMD GPUs**: Supported via Vulkan/ROCm, but requires specific installation flags (`-DGGML_VULKAN=1`).
- **No GPU**: Models load into RAM, runs on CPU. Slower but functional.

---

## 6. Data Flow

### Chat Flow

```
User types message
    │
    ▼
Electron UI / Mobile App / Gradio UI
    │
    ├── (Electron/Mobile) → POST /chat to FastAPI (api.py:8000)
    │   OR
    ├── (Gradio) → web_ui._chat_respond() directly
    │
    ├── Check if model is loaded
    ├── Create/get conversation
    ├── (Optional) Query RAG knowledge base
    │       └── knowledge_base.query(message)
    │           └── ChromaDB similarity search
    │           └── Returns relevant document chunks
    │
    ├── Build message context:
    │       System prompt (+ RAG context if any)
    │       + Last N turns of conversation
    │
    ▼
chat_engine.send_message()
    │
    ▼
llm_engine.chat(messages, stream=True)
    │
    ├── Try model.create_chat_completion() with chat template
    │   (fallback to manual prompt format if template fails)
    │
    ▼
Tokens yielded one at a time
    │
    ├── Each token → Update UI display (real-time streaming)
    ├── Accumulated into full response
    │
    ▼
Conversation saved to chat_history/{id}.json
```

### Cloud Proxy Flow

```
User selects a cloud model (e.g., GPT-4, Groq Llama)
    │
    ▼
Electron UI / Mobile App
    │
    ├── POST /cloud/chat to FastAPI (api.py:8000)
    │
    ▼
api.py reads locally-stored API key
    │
    ├── Forwards request to external provider (OpenAI/Groq/Together)
    ├── Streams response back to the client
    │
    ▼
Response displayed in UI
    │
    ▼
Conversation saved to chat_history/{id}.json
```

> **Note:** Cloud proxy is the ONLY feature that makes network calls after initial setup. It is entirely opt-in and requires manually configuring API keys.

### RAG (Document Q&A) Flow

```
User uploads document
    │
    ▼
knowledge_base.add_file(path)
    │
    ├── Read file content (PDF via PyPDF2, text files directly)
    ├── Split into 500-char chunks with 50-char overlap
    ├── Embed each chunk using sentence-transformers
    ├── Store in ChromaDB with source metadata
    │
    ▼
User asks a question with RAG enabled
    │
    ▼
knowledge_base.query(question)
    │
    ├── Embed the question
    ├── Find 5 most similar chunks (cosine similarity)
    ├── Format as context with source attribution
    │
    ▼
Context injected into system prompt
    │
    ▼
LLM answers using the retrieved context
```

---

## 7. Dependencies

### Runtime Stack (Core PC App)

```
Your Python Script
    │
    ├── fastapi + uvicorn (REST API server)
    │
    ├── gradio (Legacy Web UI framework — optional in v2.0)
    │
    ├── llama-cpp-python (Python bindings)
    │       │
    │       └── llama.cpp (C++ inference engine, compiled in)
    │               │
    │               ├── CPU: SSE/AVX/AVX2 SIMD instructions
    │               └── GPU: CUDA (NVIDIA) / Metal (Mac) / Vulkan
    │
    ├── chromadb (Vector database — optional, for RAG)
    │       └── SQLite (embedded, local storage)
    │
    ├── sentence-transformers (Text embeddings — optional, for RAG)
    │       └── PyTorch (ML framework)
    │
    ├── huggingface-hub (Model downloader — one-time use)
    │
    └── PyPDF2 (PDF reader — optional)
```

### Desktop App Stack (V2.0)
- **Framework:** Electron + React
- **Backend:** FastAPI (Python) running as a subprocess
- **Communication:** HTTP to `127.0.0.1:8000`

### Mobile App Stack (V2.0)
- **Framework:** React Native (Expo)
- **Networking:** Local LAN connection to the FastAPI backend (`http://YOUR_PC_IP:8000`)
- **Key Features:** Model switching, chat interface, local persistent settings.
- **Build Output:** Standalone Android `.aab` / `.apk`

### Launch Website Stack (V2.0)
- **Framework:** Next.js 16 (App Router)
- **Styling:** Tailwind CSS (Dark Glassmorphism theme to match desktop app)
- **Animations:** Framer Motion
- **Build Output:** Static HTML (`/out`) for free global hosting

### What Does NOT Touch the Internet

| Component             | Network Access                                           |
| --------------------- | -------------------------------------------------------- |
| LLM inference         | ❌ Never                                                 |
| Chat conversations    | ❌ Never                                                 |
| Conversation history  | ❌ Never                                                 |
| RAG document indexing  | ❌ Never                                                 |
| RAG queries           | ❌ Never                                                 |
| Web UI (Gradio)       | ❌ localhost only (127.0.0.1)                            |
| FastAPI backend       | ❌ LAN only (0.0.0.0:8000, not internet-facing)         |
| Cloud Proxy           | ✅ Only when user opts in with API keys                  |
| Model download        | ✅ One-time only, via `setup_models.py`                  |

---

## 8. Directory Structure

```
LLM_Personal/
│
├── personal_llm/                    # Main package (Core LLM Backend/Engine)
│   ├── __init__.py                  # Package init, version string
│   ├── api.py                       # FastAPI REST backend (620+ lines)
│   ├── config.py                    # All configuration + model catalog (489 lines)
│   ├── llm_engine.py                # Core inference engine (375 lines)
│   ├── model_manager.py             # Model listing, downloading, lookup (335 lines)
│   ├── chat_engine.py               # Multi-turn conversation management (358 lines)
│   ├── context_engine.py            # Context Intelligence (RAG/Refine/CoT) (498 lines)
│   ├── knowledge_base.py            # RAG system with ChromaDB (260 lines)
│   ├── hardware.py                  # Hardware detection & model recommendations (273 lines)
│   ├── llmfit_wrapper.py            # Hardware compatibility scoring (89 lines)
│   ├── web_ui.py                    # Gradio web interface (796 lines)
│   ├── setup_models.py              # Interactive model download CLI (96 lines)
│   ├── bin/llmfit.exe               # Hardware scoring binary
│   ├── requirements.txt             # Python dependencies
│   ├── chat_history/                # Saved conversations (JSON files)
│   ├── knowledge_db/                # ChromaDB vector database
│   ├── documents/                   # Uploaded RAG documents
│   └── exports/                     # Exported chat markdown files
│
├── mobile/                          # React Native Mobile App (Expo)
│   ├── app/                         # App routing (index, settings)
│   └── eas.json                     # EAS cloud build config
│
├── website/                         # Next.js Launch Website
│   ├── src/app/                     # Landing page & globals.css
│   └── out/                         # Final static build files
│
├── ui/                              # Electron Desktop App
│   └── (React UI + Electron main process configuration)
│
├── personal_llm_models/             # GGUF model files (auto-created)
│   └── *.gguf                       # Downloaded open-weights models
│
├── .editorconfig                    # Consistent line endings & indentation
├── launch_personal_llm.py           # Legacy CLI Launcher (starts Gradio UI)
├── desktop_app.py                   # Legacy pywebview Desktop Wrapper
├── personal_llm.spec                # PyInstaller build config
├── installer.iss                    # Inno Setup compiler script (Windows installer)
│
└── PROJECT_REPORT.md                # This file
```

**Total codebase: Full-stack monorepo spanning Python, React, React Native, and Next.js.**

---

## 9. Setup & Usage

### 🚀 V2.0.0 Live Assets

- **Mobile App Download:** [🤖 Download Android App (EAS APK)](https://expo.dev/artifacts/eas/bVGpxbF2bXXF9Vn7dfSkv3.apk)
- **Website Source:** See `website/out/` for the deployable Next.js launch site.

### First Time Setup (Backend Engine)

```powershell
# Step 1: Install Python dependencies
pip install -r personal_llm/requirements.txt

# Step 2: (Optional) Enable GPU acceleration
`$env:CMAKE_ARGS="-DGGML_CUDA=on"
pip install llama-cpp-python --force-reinstall --no-cache-dir

# Step 3: Download at least one model
python -m personal_llm.setup_models

# Step 4: Launch the Desktop App (Spins up Backend API + Electron UI)
cd ui
npm install
npm run electron:dev
```

### Daily Usage

```powershell
# In the /ui folder:
npm run electron:start
```

### Alternative: Legacy Gradio Launcher

```powershell
# Launches standalone Gradio web UI at http://127.0.0.1:7865
python launch_personal_llm.py
```

### Building the Electron Desktop EXE (v2.0.1)

The application is compiled into a standalone Windows `.exe` using Electron Builder with a bundled Python environment.

```powershell
# Step 1: Ensure python-embed has all dependencies
ui\python-embed\python.exe -m pip install python-multipart

# Step 2: Build the Next.js frontend and Electron installer
cd ui
npm run electron:build
```

**Output:** `ui/dist-electron2/Personal LLM Setup 2.0.0.exe` (NSIS installer) and `Personal LLM 2.0.0.exe` (portable).

**What gets bundled:**
- `python-embed/` — Full Python 3.11 with `fastapi`, `uvicorn`, `llama-cpp-python`, `python-multipart`, `pydantic`, `httpx`, `requests`, and all other dependencies pre-installed
- `personal_llm/` — The Python backend package (`.py` and `.json` files only)
- The compiled Next.js React UI inside `app.asar`

> [!IMPORTANT]
> **Windows Defender may block `app-builder.exe`** during the build. If you see `ERR_ELECTRON_BUILDER_CANNOT_EXECUTE`, add the project folder as a Windows Defender exclusion:
> ```powershell
> # Run as Administrator:
> Add-MpPreference -ExclusionPath "C:\Users\prabh\Desktop\LLM_Personal"
> ```

### Legacy: PyInstaller Build

```powershell
pyinstaller personal_llm.spec --clean --noconfirm
```

### CLI Options

```powershell
# Check setup without launching
python launch_personal_llm.py --check

# Load a specific model
python launch_personal_llm.py --model phi-3-mini

# Check your hardware
python -c "from personal_llm.hardware import print_hardware_report; print_hardware_report()"

# Check model status
python -m personal_llm.model_manager
```

---

## 10. Security & Privacy

| Aspect           | Detail                                                                            |
| ---------------- | --------------------------------------------------------------------------------- |
| **Network**      | Gradio UI serves on `127.0.0.1` only. FastAPI serves on `0.0.0.0:8000` (LAN-accessible for mobile app — see warning below). |
| **Data Storage** | All data in local directories under `LLM_Personal/` or `%LOCALAPPDATA%\PersonalLLM` |
| **No Telemetry** | Zero analytics, tracking, or phone-home                                           |
| **No Cloud**     | No API keys required. Cloud proxy is opt-in only (user provides their own keys).  |
| **Model Files**  | Binary `.gguf` files on your disk — no DRM, no license servers                    |
| **Chat History** | Plain JSON files you can read, edit, or delete anytime                            |
| **RAG Database** | SQLite-based ChromaDB — a local file, not a server                                |

> [!WARNING]
> **LAN Exposure (FastAPI):** The FastAPI backend binds to `0.0.0.0:8000` so the mobile app can connect over Wi-Fi. This means **any device on your local network** can reach the API. If you're on a shared/public Wi-Fi network, other users could potentially interact with your AI or access uploaded RAG documents. For maximum security, only run the backend on trusted private networks, or use a firewall rule to restrict access to specific IPs.

> [!CRITICAL]
> **External Sharing Warning:** The `--share` flag in the legacy launcher creates a **public link** (e.g., `https://xyz.gradio.live`) to your local machine.
>
> If you run `python launch_personal_llm.py --share`:
>
> 1. Anyone with the link can use your AI.
> 2. Anyone with the link can potentially access your uploaded RAG documents.
> 3. Your IP is not exposed, but your application IS.
>
> **NEVER use `--share` unless you absolutely intend to share access and understand the risks.** Keep it local (`127.0.0.1`) for 100% privacy.

---

## 11. Changelog

### v2.0.1 (March 4, 2026)

**Bug Fixes:**
- **Fixed: `python-multipart` missing** — The file upload endpoint (`/api/knowledge/upload`) crashed the entire FastAPI backend at startup because `python-multipart` was not installed in the bundled `python-embed`. This single missing package prevented ALL API endpoints from working.
- **Fixed: Electron health check URL** — Changed from `http://0.0.0.0:8000` (not connectable on Windows) to `http://127.0.0.1:8000`.
- **Fixed: Frontend fetch URLs** — All 13 `fetch()` calls in `page.tsx` were using raw `window.location.hostname` which resolves to `-` inside Electron's `app://-/` protocol. Replaced with `API_BASE` constant that correctly falls back to `127.0.0.1:8000`.
- **Fixed: `python311._pth` path resolution** — Added `..` to the embedded Python path config so it can find `personal_llm/` in the parent `resources/` directory.
- **Fixed: Python binary corruption** — Electron's code signing process corrupts `.pyd`/`.dll` files in the bundled `python-embed`. Added first-run extraction to `%LOCALAPPDATA%\PersonalLLM\python-embed\`.
- **Fixed: API startup timeout** — Increased from 30 seconds to 120 seconds because heavy ML library imports (`llama_cpp`, `chromadb`, `sentence_transformers`) take 60+ seconds to load.

**New Features:**
- **Static file serving** — `api.py` now serves the Next.js static export at `/`, allowing any device on the LAN to access the full React UI at `http://<host-ip>:8000`.
- **Auto-install dependencies** — `main.js` now includes `python-multipart` in the auto-install dependency list for first-run setup.
- **Model catalog expanded** — From 27 to 28 models (added `qwen2.5-coder-7b`).

### v2.0.2 (March 4, 2026) — Codebase Audit Fixes

**Critical & High Priority Fixes:**
- **Conversation History Fixed** — Resolved `AttributeError` by changing `conv.dict()` to `conv.to_dict()`. Fixed `context_engine.py` bug where conversation history was dropped when RAG/Refine/CoT was enabled.
- **Cloud Chat Persistence** — `/api/chat/cloud` now saves conversations to `chat_history/` just like local chat. Previously, cloud messages vanished on page refresh.
- **Hardware Detection Fixed** — `_detect_ram()` now correctly detects RAM on Linux (`/proc/meminfo`) and macOS (`sysctl hw.memsize`) instead of incorrectly returning disk size.
- **SSE Stream Buffering** — Added partial line buffering in the React UI to prevent tokens from dropping across chunk boundaries during streaming.
- **API Security** — Restricted CORS `allow_origins` to localized ports and Electron protocols. Sanitized file upload names to prevent directory traversal.
- **Missing Dependencies** — Added missing core runtime packages to `requirements.txt` (`fastapi`, `uvicorn`, `python-multipart`, `requests`, `httpx`, `pydantic`).

**New Features & Polish:**
- **New API Endpoints**: Added `POST /api/models/unload` to free up GPU/RAM and `GET /api/conversations/search` to search chat histories.
- **Configuration** — Removed duplicate variable definition for `TOP_K_RESULTS`.
- **Code Quality** — Added `.editorconfig` to enforce LF endings and updated `.gitignore` for settings and local exports.

**Documentation Accuracy Sweep:**
- Fixed all file line counts in the report (e.g., `config.py` 106→489, `model_manager.py` 212→335, `chat_engine.py` 267→358)
- Added full documentation for previously undocumented `context_engine.py` (498 lines) and `llmfit_wrapper.py` (89 lines)
- Fixed endpoint path `/api/cloud/chat` → `/api/chat/cloud`
- Added missing endpoints to the API table (`/api/models/unload`, `/api/conversations/search`, etc.)
- Updated `__init__.py` version reference from `2.0.0` to `2.0.2`
- Corrected `get_context_messages()` documentation (uses token-budget approach, not turn counting)

---

_This report was last updated on March 4, 2026._
