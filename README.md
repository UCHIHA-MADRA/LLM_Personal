# 🧠 Personal LLM — Your Private AI Assistant

> **100% Offline. 100% Private. Zero Monthly Fees.**

This project lets you run a powerful AI assistant (like ChatGPT) entirely on your own computer. Your chats, documents, and data never leave your machine.

---

## 🚀 Getting Started (For Beginners)

Follow these steps to get up and running in 5 minutes.

### 1. Requirements
- **Windows PC** (works best with NVIDIA GPU, but runs on CPU too)
- **Python 3.10 or higher** ([Download Here](https://www.python.org/downloads/))
- **Git** ([Download Here](https://git-scm.com/downloads))

### 2. Install
Open `PowerShell` or `Command Prompt` and run:

```powershell
# 1. Clone this repository
git clone https://github.com/YOUR_USERNAME/personal-llm.git
cd personal-llm

# 2. Install required libraries
pip install -r personal_llm/requirements.txt
```

> **Note:** If you have an NVIDIA GPU, run this extra command for speed:
> ```powershell
> pip install llama-cpp-python --force-reinstall --no-cache-dir --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
> ```

### 3. Setup Models
You need to download the AI "brain" files (models). We've made this easy:

```powershell
python personal_llm/setup_models.py
```
*Follow the on-screen prompts to download a model (e.g., Phi-3 Mini or Mistral 7B).*

### 4. Launch!
Start your AI assistant:

```powershell
python launch_personal_llm.py
```
*Click the local URL (e.g., `http://127.0.0.1:7865`) to open the Web UI in your browser.*

---

## 💡 How to Use

### 💬 Chat
- Select a model from the sidebar.
- Type your message and press **Enter**.
- The AI remembers your conversation context.

### 📄 Knowledge Base (RAG)
- Expand the **"Knowledge Base (RAG)"** section in the sidebar.
- Upload a PDF, text file, or code file.
- Check **"Use document context"**.
- Ask questions about your document!

### ⚙️ Management
- **Chat History**: Load or delete past conversations from the sidebar.
- **Search**: Find past chats using the "Search History" tool.
- **Manage Docs**: Delete individual uploaded documents from the Knowledge Base section.

---

## ❓ Troubleshooting

**Q: It's slow!**
A: If you don't have a GPU, switch to a smaller model like **Phi-3 Mini**. It's designed for CPUs.

**Q: "No module named..." error?**
A: Run `pip install -r personal_llm/requirements.txt` again.

**Q: Where is my data?**
A: Everything is stored locally in the `personal_llm/` folder:
- Chats: `personal_llm/chat_history/`
- RAG Database: `personal_llm/knowledge_db/`

---

## 📚 Advanced Documentation
For technical details, architecture diagrams, and API info, read the [Project Report](PROJECT_REPORT.md).
