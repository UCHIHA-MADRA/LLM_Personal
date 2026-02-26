# 🧠 Personal LLM 2.0 — Complete Project Report

> **Version:** 2.0.0
> **Date:** February 26, 2026
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
- **🌐 Cloud Proxy** — Connect API keys (OpenAI, Groq, Together) to mix local and cloud AI
- **🔄 Multi-Model** — Switch between 27+ different AI models on-the-fly
- **📱 Mobile App** — Connect locally from your phone with the React Native Expo app
- **💾 Persistent History** — All conversations saved locally as JSON

---

## 2. Architecture

### System Diagram

```
┌───────────────────────────────────────────────────────────────┐
│                    YOUR PERSONAL PC                           │
│                                                               │
│  ┌──────────────────────┐   ┌─────────────────────────────┐   │
│  │     Web Browser      │   │    Mobile App (Expo)        │   │
│  │ http://127.0.0.1:3000│   │ http://10.0.2.2:8000 (LAN)  │   │
│  └──────────┬───────────┘   └─────────────┬───────────────┘   │
│             │                             │                   │
│  ┌──────────▼─────────────────────────────▼───────────────┐   │
│  │          FastAPI Backend (api.py)                      │   │
│  │  • REST API for models, status, chat, docs             │   │
│  │  • Cloud inference fallback (OpenAI/Groq/Together)     │   │
│  │  • Background model downloading                        │   │
│  └──────┬───────────┬──────────────┬──────────────────────┘   │
│         │           │              │                          │
│  ┌──────▼─────┐ ┌───▼──────┐ ┌────▼───────────────────┐       │
│  │chat_engine │ │model_mgr │ │  knowledge_base.py     │       │
│  │  .py       │ │  .py     │ │  (RAG System)          │       │
│  │            │ │          │ │                        │       │
│  │ • Sessions │ │ • List   │ │ • Ingest documents     │       │
│  │ • History  │ │ • D-load │ │ • Chunk & Embed        │       │
│  │ • DB/JSON  │ │ • Config │ │ • ChromaDB Retrieval   │       │
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
- `__version__ = "1.0.0"` — version identifier

**Key detail:** This file is intentionally minimal. It does NOT auto-import heavy modules like `llama_cpp` or `chromadb` to keep import times fast.

---

### 3.2 `config.py` (106 lines)

**Purpose:** Central configuration file. Every setting the system uses is defined here.

**Sections:**

#### Directories (Lines 9-27)

```
BASE_DIR            → LLM_Enhanced/                    (project root)
PERSONAL_LLM_DIR    → LLM_Enhanced/personal_llm/       (package directory)
MODELS_DIR          → LLM_Enhanced/personal_llm_models/ (GGUF files stored here)
CHAT_HISTORY_DIR    → personal_llm/chat_history/        (JSON conversation files)
KNOWLEDGE_DB_DIR    → personal_llm/knowledge_db/        (ChromaDB vector database)
DOCUMENTS_DIR       → personal_llm/documents/           (uploaded documents)
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

- `UI_PORT = 7865` — Gradio server port
- `UI_HOST = "127.0.0.1"` — Localhost only (never exposed to network)

#### RAG Settings (Lines 56-60)

| Setting                | Value              | Purpose                                        |
| ---------------------- | ------------------ | ---------------------------------------------- |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence-transformers model for embedding text |
| `CHUNK_SIZE`           | `500` characters   | How big each document chunk is                 |
| `CHUNK_OVERLAP`        | `50` characters    | Overlap between chunks to preserve context     |
| `TOP_K_RESULTS`        | `5`                | Number of relevant chunks retrieved per query  |

#### Model Catalog (Lines 62-105)

A dictionary of 5 pre-configured GGUF models that can be downloaded. Each entry has:

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

### 3.4 `model_manager.py` (212 lines)

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
| `download_model()`             | 77-124  | Downloads a model from HuggingFace Hub. Uses `huggingface_hub.hf_hub_download()`. Saves directly to `models_dir`. **One-time operation** — skips if file already exists. |
| `download_model_interactive()` | 126-157 | Interactive CLI menu. Shows all catalog models with download status (`✅ Downloaded` or `📥 ~X GB`). User picks a number.                                                |
| `get_default_model()`          | 159-181 | Returns the best available model. Priority: config default → first local model → `None`.                                                                                 |
| `print_status()`               | 183-195 | Prints current model status to console.                                                                                                                                  |

**CLI Entry Point (Lines 198-211):**

- `python -m personal_llm.model_manager` — Shows status
- `python -m personal_llm.model_manager download` — Interactive download
- `python -m personal_llm.model_manager download phi-3-mini` — Download specific model

**Key design decisions:**

1. **Lazy HuggingFace import**: `huggingface_hub` is only imported when `download_model()` is called, so the module works offline without it.
2. **No symlinks**: `local_dir_use_symlinks=False` ensures the model file is a real file, not a symlink to HuggingFace's cache. This makes the models directory self-contained.

---

### 3.5 `chat_engine.py` (267 lines)

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

1. **Dynamic Context Pruning**: `get_context_messages()` calculates exact token usage (approx. 4 chars/token). It prioritizes System Prompt + RAG Context, then fills remaining space with history. This guarantees we never overflow the context window (4096 tokens), preventing crashes even with long conversations.
2. **RAG injection**: When RAG context is available, it's appended to the system prompt.
3. **Stream-then-save**: Tokens are yielded to UI immediately, but saved to history only after completion.

---

### 3.6 `knowledge_base.py` (260 lines)

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

### 3.7 `hardware.py` (256 lines)

**Purpose:** Detects your PC's hardware specs and recommends which models will run well.

#### Detection Functions

| Function            | Lines  | What It Detects               | How                                                       |
| ------------------- | ------ | ----------------------------- | --------------------------------------------------------- |
| `detect_hardware()` | 18-30  | Full system specs             | Orchestrator function                                     |
| `_detect_cpu()`     | 33-54  | CPU name and core count       | `platform.processor()` + `wmic cpu get Name` on Windows   |
| `_detect_ram()`     | 57-92  | Total RAM in GB               | `wmic memorychip get Capacity` → fallback to `systeminfo` |
| `_detect_gpu()`     | 95-131 | NVIDIA GPU name, VRAM, driver | `nvidia-smi --query-gpu=...`                              |

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

### 3.8 `web_ui.py` (510 lines)

**Purpose:** Premium Gradio web interface. This is what you see in the browser.

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

| Method                    | Purpose                                                                                                                      |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ---------- | ----------------- | ----------------- |
| `_get_available_models()` | Lists `.gguf` files in models directory for dropdown                                                                         |
| `_get_status_text()`      | Builds status bar: `🟢 Model: X                                                                                              | Size: Y GB | Context: Z tokens | 🔒 Fully Offline` |
| `_load_model()`           | Loads model from dropdown selection. Gets chat format from catalog.                                                          |
| `_chat_respond()`         | **Main chat handler.** Streaming response. Handles RAG context injection. Yields updated chat history for real-time display. |
| `_new_chat()`             | Resets chat history, creates new conversation.                                                                               |
| `_upload_document()`      | Uploads file to knowledge base via `KnowledgeBase.add_file()`.                                                               |
| `_export_chat()`          | Exports current conversation to `personal_llm/exports/chat_{id}.md`.                                                         |
| `_clear_knowledge_base()` | Clears all RAG documents.                                                                                                    |
| `_get_kb_stats()`         | Shows knowledge base statistics.                                                                                             |

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

### 3.9 `setup_models.py` (78 lines)

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

### 3.10 `launch_personal_llm.py` (127 lines)

**Purpose:** One-click launcher. The single entry point to start everything.

**Flow:**

1. **Banner** — Prints project title
2. **Check dependencies** — Verifies `llama-cpp-python` and `gradio` are installed
3. **Hardware report** — Calls `hardware.print_hardware_report()` to show PC specs
4. **Check models** — Verifies at least one `.gguf` model exists. If not, offers interactive download.
5. **Launch** — Imports and calls `web_ui.launch_ui()`

---

### 3.11 `desktop_app.py` (88 lines)

**Purpose:** The native Windows desktop wrapper.

**Flow:**

1. Finds an available network port dynamically.
2. Creates an HTML splash screen (`Loading Personal LLM...` with a spinner) and displays it using `pywebview`.
3. Starts the Gradio server in a background thread without locking the main thread.
4. Polls the Gradio localhost URL until it responds with HTTP 200.
5. Checks if models exist in the adjacent `personal_llm_models` folder. If none, shows a native Windows error dialog.
6. Replaces the splash screen with the actual loaded Gradio UI.
7. Gracefully unloads the LLM from memory and terminates the server when the user closes the native window.

**CLI Arguments:**

| Flag             | Purpose                           |
| ---------------- | --------------------------------- |
| `--check`        | Only verify setup, don't launch   |
| `--share`        | Create a public Gradio share link |
| `--model <name>` | Load a specific model on startup  |

---

### 3.11 `requirements.txt` (19 lines)

**Dependencies:**

| Package                 | Version | Purpose                            | Required?     |
| ----------------------- | ------- | ---------------------------------- | ------------- |
| `llama-cpp-python`      | ≥0.3.0  | Core LLM engine (loads GGUF files) | **Yes**       |
| `huggingface-hub`       | ≥0.20.0 | Download models (one-time)         | Yes for setup |
| `gradio`                | ≥4.0.0  | Web UI framework                   | **Yes**       |
| `chromadb`              | ≥0.5.0  | Vector database for RAG            | Only for RAG  |
| `sentence-transformers` | ≥3.0.0  | Text embeddings for RAG            | Only for RAG  |
| `PyPDF2`                | ≥3.0.0  | PDF text extraction                | Only for PDFs |

**Python Version Required:** Python 3.10 or 3.11 recommended. (Tested on 3.10+)

> [!WARNING]
> **Heads Up:** `sentence-transformers` relies on **PyTorch**, which is a heavy download (~2.5 GB). Even though the embedding model itself is small (80 MB), the initial setup will pull this large dependency.

**No cloud dependencies.** Every package runs locally.

---

## 4. Model Catalog

The catalog was massively expanded in v2.0 to include **27 fully open-source and weights-available models**, categorized by tier:

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
web_ui._chat_respond()
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
    ├── Each token → Update Gradio chatbot display (real-time)
    ├── Accumulated into full response
    │
    ▼
Conversation saved to chat_history/{id}.json
```

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

### Runtime Stack

```
Your Python Script
    │
    ├── gradio (Web UI framework)
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

### What Does NOT Touch the Internet

| Component             | Network Access                          |
| --------------------- | --------------------------------------- |
| LLM inference         | ❌ Never                                |
| Chat conversations    | ❌ Never                                |
| Conversation history  | ❌ Never                                |
| RAG document indexing | ❌ Never                                |
| RAG queries           | ❌ Never                                |
| Web UI (Gradio)       | ❌ localhost only (127.0.0.1)           |
| Model download        | ✅ One-time only, via `setup_models.py` |

---

## 8. Directory Structure

```
LLM_Enhanced/
│
├── personal_llm/                    # Main package
│   ├── __init__.py                  # Package init (8 lines)
│   ├── config.py                    # All configuration (106 lines)
│   ├── llm_engine.py                # Core inference engine (375 lines)
│   └── ... (other source files)
│
├── personal_llm_models/             # GGUF model files (auto-created)
│   └── *.gguf                      # Downloaded model files
│
├── launch_personal_llm.py           # CLI Launcher
├── desktop_app.py                   # Desktop App Launcher Hook
├── personal_llm.spec                # PyInstaller build config
├── installer.iss                    # Inno Setup compiler script
│
└── PROJECT_REPORT.md                # This file
```

**Total codebase: ~2,418 lines across 11 files**

---

## 9. Setup & Usage

### First Time Setup

```powershell
# Step 1: Install Python dependencies
pip install -r personal_llm/requirements.txt

# Step 2: (Optional) Enable GPU acceleration
$env:CMAKE_ARGS="-DGGML_CUDA=on"
pip install llama-cpp-python --force-reinstall --no-cache-dir

# Step 3: Download a model (requires internet, one-time only)
python personal_llm/setup_models.py

# Step 4: Launch your private AI
python launch_personal_llm.py
```

### Daily Usage

```powershell
# Just run this — everything else is automatic
python launch_personal_llm.py
```

### Building the Desktop Executable

The application can be compiled into a standalone Windows `.exe` using PyInstaller.

```powershell
# Build the standalone application
pyinstaller personal_llm.spec --clean --noconfirm
```

Features of the desktop build:

- Uses `onedir` mode for instant startup (no temporary extraction delay).
- Uses `sys.executable` and `%LOCALAPPDATA%` properly to save user data.
- Automatically bundles the `llama_cpp` DLL dependencies (CPU inference).

### Building the Windows Installer

Once PyInstaller finishes building the `dist/PersonalLLM` folder, you can package it into a professional Setup Wizard.

```powershell
# Requires Inno Setup 6 installed
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer.iss
```

This outputs `PersonalLLM_Setup_v1.0.0.exe` (~300MB), which creates Desktop shortcuts, Start Menu entries, and uninstaller support.

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
| **Network**      | UI serves on `127.0.0.1` only — not accessible from other devices on your network |
| **Data Storage** | All data in local directories under `LLM_Enhanced/`                               |
| **No Telemetry** | Zero analytics, tracking, or phone-home                                           |
| **No Cloud**     | No API keys, no cloud services, no external servers                               |
| **Model Files**  | Binary `.gguf` files on your disk — no DRM, no license servers                    |
| **Chat History** | Plain JSON files you can read, edit, or delete anytime                            |
| **RAG Database** | SQLite-based ChromaDB — a local file, not a server                                |

> [!CRITICAL]
> **External Sharing Warning:** The `--share` flag in the launcher creates a **public link** (e.g., `https://xyz.gradio.live`) to your local machine.
>
> If you run `python launch_personal_llm.py --share`:
>
> 1. Anyone with the link can use your AI.
> 2. Anyone with the link can potentially access your uploaded RAG documents.
> 3. Your IP is not exposed, but your application IS.
>
> **NEVER use `--share` unless you absolutely intend to share access and understand the risks.** Keep it local (`127.0.0.1`) for 100% privacy.

---

_This report was generated on February 18, 2026. All file contents and line counts are accurate as of this date._
