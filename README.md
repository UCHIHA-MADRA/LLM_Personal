# 🧠 Personal LLM — Your Private AI Assistant

> **100% Offline. 100% Private. Zero Monthly Fees.**

Run a powerful AI assistant (like ChatGPT) entirely on your own computer. Your chats, documents, and data never leave your machine. Available as both a **desktop app** and a **web UI**.

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Details |
|---|---|
| **OS** | Windows 10/11 (64-bit) |
| **Python** | 3.10 or higher — [Download](https://www.python.org/downloads/) |
| **Git** | [Download](https://git-scm.com/downloads) |
| **RAM** | 8 GB minimum (16 GB recommended for 7B models) |
| **GPU** | Optional — NVIDIA GPU with CUDA speeds things up dramatically |

### 1. Clone & Install

```powershell
git clone https://github.com/UCHIHA-MADRA/LLM_Personal.git
cd LLM_Personal

# Install all backend dependencies
pip install -r personal_llm/requirements.txt

# Install Electron desktop app dependencies
cd ui
npm install
cd ..
```

> **NVIDIA GPU Users** — Install the CUDA-accelerated LLM engine for 10–50x faster inference:
> ```powershell
> pip install llama-cpp-python --force-reinstall --no-cache-dir --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
> ```
>
> **CPU-Only Users** — Install the standard engine (slower but works on any PC):
> ```powershell
> pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
> ```

### 2. Download a Model

You need at least one AI model file (`.gguf` format). Run the interactive downloader:

```powershell
python personal_llm/setup_models.py
```

This downloads the model once — after that, everything runs offline.

### 3. Launch

You have **two ways** to run Personal LLM:

#### 🖥️ Option A: Desktop App (Recommended)

Opens in a beautiful native desktop window powered by Electron and React.

```powershell
cd ui
npm run electron:dev
```

### Building the Executable

To compile the Electron application into a standalone distribution (e.g., `.exe` for Windows, `.dmg` for macOS, or `.AppImage` for Linux), run the build command on your native operating system:

**For Windows (`.exe`):**
```bash
cd ui
npm run electron:build -- --win
```

**For macOS (`.dmg`):** *(Must be run on a Mac)*
```bash
cd ui
npm run electron:build -- --mac
```

**For Linux (`.AppImage` and `.deb`):**
```bash
cd ui
npm run electron:build -- --linux
```

*Note: The generated setups will be placed in the `ui/dist-electron2/` folder.*

#### 🌐 Option B: Web UI

Opens in your browser at `http://127.0.0.1:7865`.

```powershell
python launch_personal_llm.py
```

---

## 💡 How to Use

### 💬 Chat
- Select a model from the sidebar and click **Load Model**.
- Type your message and press **Enter**.
- The AI remembers your conversation context.

### 📄 Knowledge Base (RAG)
- Expand **"Knowledge Base (RAG)"** in the sidebar.
- Upload a PDF, text file, or code file.
- Check **"Use document context"**.
- Ask questions about your document!

### ⚙️ Management
- **Chat History** — Load or delete past conversations.
- **Search** — Find past chats using the search tool.
- **Manage Docs** — Delete individual uploaded documents from the Knowledge Base.

---

## 📦 Build a Standalone Desktop App (.exe)

Turn Personal LLM into a distributable Windows application that anyone can run — no Python installation required.

### Step 1: Set Up the Build Environment

```powershell
# Install build tools
pip install pyinstaller

# For CPU-only build (smaller, works everywhere):
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# For GPU-accelerated build (requires NVIDIA GPU on target PC):
pip install llama-cpp-python --force-reinstall --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

### Step 2: Build the Executable

```powershell
pyinstaller personal_llm.spec --clean --noconfirm
```

This takes 5–10 minutes. The output will be in `dist/PersonalLLM/`.

### Step 3: Add Your Models

```powershell
# Create the models folder and copy your models into it
mkdir dist\PersonalLLM\personal_llm_models
copy personal_llm_models\*.gguf dist\PersonalLLM\personal_llm_models\
```

### Step 4: Test It

```powershell
.\dist\PersonalLLM\PersonalLLM.exe
```

You should see a splash screen, then the full AI interface loads in a native window.

### What You Get

```
dist/PersonalLLM/
├── PersonalLLM.exe              ← Double-click to launch
├── personal_llm_models/         ← Place .gguf model files here
└── _internal/                   ← Supporting libraries (do not modify)
```

*(User data like chat history and RAG databases will be created automatically in `%LOCALAPPDATA%\PersonalLLM` on first run).*

### Distribution Checklist

When sharing the app with others, include:

| Item | Required? | Notes |
|---|---|---|
| `PersonalLLM.exe` | ✅ Yes | The main application |
| `_internal/` folder | ✅ Yes | All bundled dependencies |
| `personal_llm_models/` folder | ✅ Yes | Must contain at least one `.gguf` model |

> [!IMPORTANT]
> **Model files are large** (2–5 GB each). You'll likely want to distribute models separately
> via a download link rather than bundling them with the app.

### Known Build Notes

- **Windows SmartScreen**: The `.exe` is unsigned, so Windows will show a "Windows protected your PC" warning on first launch. Click **"More info" → "Run anyway"**. To avoid this, you would need a code signing certificate (~$200–400/year).
- **Antivirus**: Some antivirus software may flag PyInstaller executables. This is a known false positive.
- **Build Size**: The `dist/PersonalLLM/` folder is ~1 GB (excluding models). Most of this is PyTorch (used for document embeddings).

---

## ❓ Troubleshooting

**Q: The desktop app shows "No Models Found"**
A: Place `.gguf` model files in `personal_llm_models/` next to the `.exe` (or next to `desktop_app.py` in dev mode).

**Q: It's slow!**
A: If you don't have a GPU, use a smaller model like **Phi-3 Mini** (2.4 GB). It's designed for CPUs.

**Q: "No module named..." error?**
A: Run `pip install -r personal_llm/requirements.txt` and `cd ui && npm install`.

**Q: The port is already in use**
A: The desktop app auto-finds a free port. The web UI uses port 7865 by default — if that's taken, set a custom one:
- **Windows:** `set PERSONAL_LLM_PORT=8080`
- **macOS/Linux:** `export PERSONAL_LLM_PORT=8080`

**Q: Where is my data?**
A: Everything is stored locally:
- **Dev mode**: Inside the `LLM_Personal/personal_llm/` folder
- **Desktop app (.exe)**: In `%LOCALAPPDATA%\PersonalLLM`

---

## 🏗️ Project Structure

```
LLM_Personal/
├── ui/                         ← Electron Desktop App Frontend
├── mobile/                     ← React Native Expo Mobile App
├── website/                    ← Next.js Launch Website
├── launch_personal_llm.py      ← Web UI launcher (browser)
├── setup.py                    ← Cross-platform interactive setup wizard
├── personal_llm.spec           ← PyInstaller build config
├── .editorconfig               ← Editor formatting rules
├── .gitignore                  ← Git exclusion rules
├── personal_llm/
│   ├── api.py                  ← FastAPI REST backend
│   ├── config.py               ← All settings & paths
│   ├── web_ui.py               ← Gradio interface (legacy)
│   ├── llm_engine.py           ← LLM inference engine
│   ├── chat_engine.py          ← Conversation management
│   ├── model_manager.py        ← Model download & selection
│   ├── knowledge_base.py       ← RAG / document search
│   ├── context_engine.py       ← Context Intelligence (RAG + CoT + Self-Refine)
│   ├── llmfit_wrapper.py       ← Hardware compatibility scoring
│   ├── hardware.py             ← GPU/CPU detection
│   └── requirements.txt        ← Python dependencies
└── personal_llm_models/        ← Your downloaded .gguf models
```

## 📚 Advanced Documentation

For technical details, architecture diagrams, and API info, read the [Project Report](PROJECT_REPORT.md).

---

## 📜 Legal

- [MIT License](LICENSE)
- [Privacy Policy](PRIVACY_POLICY.md) — **TL;DR: Your data never leaves your machine.**
- [Terms of Service](TERMS_OF_SERVICE.md)
