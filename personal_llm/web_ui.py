# pyre-ignore-all-errors
"""
Personal LLM — Premium Web UI
Beautiful dark glassmorphism Gradio interface for chatting with your local LLM.
"""

import os
import time
import logging
import threading
from pathlib import Path
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    gr = None

from . import config
from .llm_engine import LLMEngine, get_engine
from .model_manager import ModelManager
from .chat_engine import ChatEngine, Conversation
from .knowledge_base import KnowledgeBase
from .llmfit_wrapper import get_model_fit_info


# ─── Custom CSS for premium look ─────────────────────────────
CUSTOM_CSS = """
/* Dark glassmorphism theme */
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

/* Main title */
#title-text {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2em !important;
    font-weight: 800 !important;
    margin-bottom: 0 !important;
}

#subtitle-text {
    text-align: center;
    color: #8b8fa3 !important;
    font-size: 0.95em !important;
    margin-top: 0 !important;
}

/* Chat area */
.chatbot {
    border-radius: 16px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    min-height: 500px !important;
}

/* Buttons */
.primary-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
}

/* Status bar */
#status-bar {
    font-size: 0.85em;
    padding: 8px 16px;
    border-radius: 10px;
    background: rgba(102, 126, 234, 0.1);
    border: 1px solid rgba(102, 126, 234, 0.2);
}

/* Sidebar */
.sidebar-section {
    border-radius: 12px !important;
    padding: 12px !important;
}

/* Input area */
.input-area textarea {
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    transition: border-color 0.3s ease !important;
}

.input-area textarea:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15) !important;
}

/* Model selector */
.model-dropdown {
    border-radius: 10px !important;
}

/* Accordion */
.accordion {
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}
"""


class PersonalLLMUI:
    """Premium Gradio interface for the Personal LLM system."""

    def __init__(self):
        self.engine = get_engine()
        self.model_manager = ModelManager()
        self.chat_engine = ChatEngine(self.engine)
        self.knowledge_base = None  # Lazy init
        self._current_conversation: Optional[Conversation] = None
        self.download_cancel_event = threading.Event()

    def _get_kb(self) -> KnowledgeBase:
        if self.knowledge_base is None:
            self.knowledge_base = KnowledgeBase()
        return self.knowledge_base

    def _get_available_models(self) -> List[str]:
        """Get list of available model filenames."""
        models = self.model_manager.list_local_models()
        return [m["filename"] for m in models] if models else ["No models found"]

    def _get_status_text(self) -> str:
        """Build the status bar text."""
        info = self.engine.get_info()
        if info["loaded"]:
            size = info.get("size_gb", "?")
            ctx = info.get("n_ctx", "?")
            return f"🟢 Model: {info['name']}  |  Size: {size} GB  |  Context: {ctx} tokens  |  🔒 Fully Offline"
        return "🔴 No model loaded — select one above"

    def _load_model(self, model_filename: str) -> str:
        """Load a model by filename."""
        if not model_filename or model_filename == "No models found":
            return "❌ No model selected. Download one first with: python personal_llm/setup_models.py"

        path = self.model_manager.models_dir / model_filename
        if not path.exists():
            return f"❌ Model file not found: {path}"

        chat_format = self.model_manager.get_chat_format(model_filename)

        success = self.engine.load(
            str(path),
            n_gpu_layers=config.N_GPU_LAYERS,
            n_ctx=config.CONTEXT_SIZE,
            chat_format=chat_format,
        )

        if success:
            # Start a new conversation with the new model
            self._current_conversation = self.chat_engine.new_conversation()
            return self._get_status_text()
        return "❌ Failed to load model. Check the console for details."

    def _chat_respond(
        self,
        message: str,
        chat_history: List,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        use_rag: bool,
    ):
        """Process a chat message and stream the response."""
        if not self.engine.is_loaded:
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": "❌ No model loaded. Please select and load a model first."})
            yield chat_history, ""
            return

        # Create conversation if needed
        if self._current_conversation is None:
            self._current_conversation = self.chat_engine.new_conversation(
                system_prompt=system_prompt
            )
        elif system_prompt != self._current_conversation.system_prompt:
            self._current_conversation.system_prompt = system_prompt

        # RAG context
        rag_context = ""
        if use_rag:
            try:
                kb = self._get_kb()
                if kb._collection and kb._collection.count() > 0:
                    rag_context = kb.query(message)
            except Exception as e:
                logger.warning(f"RAG query failed: {e}")

        # Add user message to display
        chat_history.append({"role": "user", "content": message})

        # Stream response
        partial = ""
        try:
            # Add user message to conversation
            self._current_conversation.add_user_message(message)
            self._current_conversation.model_name = self.engine.model_name

            # Build context
            system = self._current_conversation.system_prompt
            if rag_context:
                system += f"\n\nUse the following context to answer:\n\n{rag_context}"

            messages = self._current_conversation.get_context_messages()
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] = system

            for token in self.engine.chat(
                messages=messages,
                max_tokens=int(max_tokens),
                temperature=temperature,
                stream=True,
            ):
                partial += token
                # Update the last assistant message
                display = chat_history + [{"role": "assistant", "content": partial}]
                yield display, ""

            # Finalize
            self._current_conversation.add_assistant_message(partial)
            self.chat_engine._save_conversation(self._current_conversation)

            final_history = chat_history + [{"role": "assistant", "content": partial}]
            yield final_history, ""

        except Exception as e:
            error_msg = f"❌ Generation error: {str(e)}"
            logger.error(error_msg)
            chat_history.append({"role": "assistant", "content": error_msg})
            yield chat_history, ""

    def _new_chat(self, system_prompt: str):
        """Start a new conversation."""
        self._current_conversation = self.chat_engine.new_conversation(
            system_prompt=system_prompt
        )
        return [], "New conversation started ✨"

    def _upload_document(self, file):
        """Upload a document to the knowledge base."""
        if file is None:
            return "No file selected."
        try:
            kb = self._get_kb()
            chunks = kb.add_file(file.name)
            if chunks > 0:
                stats = kb.get_stats()
                return f"✅ Added {chunks} chunks. Total: {stats['total_chunks']} chunks in knowledge base."
            return "⚠️ No content extracted from file."
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def _export_chat(self) -> Optional[str]:
        """Export current conversation as markdown."""
        if self._current_conversation and self._current_conversation.messages:
            md = self._current_conversation.export_markdown()
            # Save to temp file
            export_path = config.PERSONAL_LLM_DIR / "exports"
            export_path.mkdir(exist_ok=True)
            filename = f"chat_{self._current_conversation.id}.md"
            filepath = export_path / filename
            filepath.write_text(md, encoding="utf-8")
            return str(filepath)
        return None

    def _clear_knowledge_base(self):
        """Clear all documents from knowledge base."""
        try:
            kb = self._get_kb()
            kb.clear()
            return "Knowledge base cleared."
        except Exception as e:
            return f"Error: {e}"

    def _get_kb_stats(self) -> str:
        """Get knowledge base statistics."""
        try:
            kb = self._get_kb()
            stats = kb.get_stats()
            sources = kb.list_sources()
            text = f"📊 {stats['total_chunks']} chunks indexed"
            if sources:
                text += f"\n📄 Sources: {', '.join(sources[:10])}"
                if len(sources) > 10:
                    text += f" +{len(sources) - 10} more"
            return text
        except Exception as e:
            return f"Knowledge base not initialized yet."

    # ─── Conversation History Handlers ────────────────

    def _get_conversation_list(self) -> List[str]:
        """Get list of conversation titles for the dropdown."""
        convos = self.chat_engine.list_conversations()
        if not convos:
            return ["No conversations yet"]
        return [f"{c['title']} ({c['message_count']} msgs)" for c in convos]

    def _get_conversation_ids(self) -> List[str]:
        """Get conversation IDs in the same order as the list."""
        convos = self.chat_engine.list_conversations()
        return [c["id"] for c in convos]

    def _load_conversation(self, selection: str):
        """Load a past conversation into the chat display."""
        if not selection or selection == "No conversations yet":
            return [], "No conversation selected."

        convos = self.chat_engine.list_conversations()
        titles = [f"{c['title']} ({c['message_count']} msgs)" for c in convos]

        if selection not in titles:
            return [], "Conversation not found."

        idx = titles.index(selection)
        conv_id = convos[idx]["id"]
        conv = self.chat_engine.get_conversation(conv_id)

        if conv is None:
            return [], "Conversation not found."

        self._current_conversation = conv
        self.chat_engine.active_conversation = conv

        # Convert messages to Gradio chatbot format
        chat_history = []
        for msg in conv.messages:
            if msg["role"] in ("user", "assistant"):
                chat_history.append({"role": msg["role"], "content": msg["content"]})
        return chat_history, f"📂 Loaded: {conv.title}"

    def _delete_selected_conversation(self, selection: str):
        """Delete the selected conversation."""
        if not selection or selection == "No conversations yet":
            return "No conversation selected."

        convos = self.chat_engine.list_conversations()
        titles = [f"{c['title']} ({c['message_count']} msgs)" for c in convos]

        if selection not in titles:
            return "Conversation not found."

        idx = titles.index(selection)
        conv_id = convos[idx]["id"]
        self.chat_engine.delete_conversation(conv_id)
        return f"🗑️ Deleted conversation."

    def _search_conversations(self, query: str) -> str:
        """Search all conversations and return results."""
        if not query or not query.strip():
            return "Enter a search term."

        results = self.chat_engine.search_conversations(query.strip())
        if not results:
            return f"No results for '{query}'."

        text = f"🔍 Found {len(results)} conversation(s):\n\n"
        for r in results[:10]:
            text += f"**{r['title']}** ({r['message_count']} msgs)\n"
            for match in r["matches"][:3]:
                text += f"  {match}\n"
            text += "\n"
        return text

    # ─── RAG Source Management ────────────────────────

    def _get_kb_sources(self) -> List[str]:
        """Get list of document sources in the knowledge base."""
        try:
            kb = self._get_kb()
            sources = kb.list_sources()
            return sources if sources else ["No documents"]
        except Exception:
            return ["Knowledge base not initialized"]

    def _delete_kb_source(self, source_name: str) -> str:
        """Delete a specific document source from the knowledge base."""
        if not source_name or source_name in ("No documents", "Knowledge base not initialized"):
            return "No document selected."
        try:
            kb = self._get_kb()
            deleted = kb.delete_source(source_name)
            if deleted > 0:
                return f"🗑️ Deleted {deleted} chunks from '{source_name}'"
            return f"⚠️ No chunks found for '{source_name}'"
        except Exception as e:
            return f"❌ Error: {e}"

    # ─── Model Management ─────────────────────────────

    def _refresh_models(self):
        """Refresh the model dropdown list."""
        return gr.update(choices=self._get_available_models())

    def _download_model_ui(self, catalog_key: str, progress=gr.Progress()):
        """Download a model with Gradio progress."""
        if getattr(self, "download_cancel_event", None):
            self.download_cancel_event.clear()
        else:
            self.download_cancel_event = threading.Event()
        
        def _progress_cb(percent, msg):
            progress(percent, desc=msg)
            
        result = self.model_manager.download_model_stream(
            catalog_key, 
            progress_callback=_progress_cb,
            cancel_event=self.download_cancel_event
        )
        
        choices = self._get_available_models()
        if result:
            entry = config.MODEL_CATALOG[catalog_key]
            status = self._load_model(entry["filename"])
            return status, gr.update(choices=choices, value=entry["filename"]), "✅ Download complete! Model is ready."
        else:
            return self._get_status_text(), gr.update(choices=choices), "❌ Download failed or cancelled."

    def _cancel_download(self):
        """Cancel an active download."""
        if hasattr(self, 'download_cancel_event'):
            self.download_cancel_event.set()
        return "⚠️ Cancelling download..."

    def build(self) -> gr.Blocks:
        """Build the Gradio UI."""
        if not GRADIO_AVAILABLE:
            raise ImportError("Gradio not installed. Install: pip install gradio")

        with gr.Blocks(title="Personal LLM — Your Private AI") as app:

            # ─── Header ───────────────────────────────
            with gr.Row():
                with gr.Column(scale=4):
                    gr.Markdown("# 🧠 Personal LLM", elem_id="title-text")
                    gr.Markdown(
                        "Your private AI assistant — runs 100% on your hardware, zero cloud dependency",
                        elem_id="subtitle-text",
                    )
                with gr.Column(scale=1, min_width=150):
                    gr.HTML(
                        '<div style="text-align: right; margin-top: 15px;">'
                        '<a href="https://github.com/UCHIHA-MADRA/LLM_Personal" target="_blank" '
                        'style="padding: 10px 15px; background-color: #333; color: white; border-radius: 5px; '
                        'text-decoration: none; font-weight: bold; display: inline-flex; align-items: center; gap: 8px;">'
                        '<svg height="20" width="20" viewBox="0 0 16 16" fill="white">'
                        '<path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>'
                        'View on GitHub</a></div>'
                    )

            # ─── Status Bar ───────────────────────────
            status_bar = gr.Textbox(
                value=self._get_status_text(),
                label="",
                interactive=False,
                elem_id="status-bar",
            )

            with gr.Tabs():
                with gr.Tab("💬 Chat"):
                    with gr.Row():
                        # ─── Main Chat Area ───────────────────
                        with gr.Column(scale=3):
                            chatbot = gr.Chatbot(
                                label="Chat",
                                height=520,
                                placeholder="Load a model and start chatting...",
                            )
        
                            with gr.Row(elem_classes="input-area"):
                                msg_input = gr.Textbox(
                                    placeholder="Type your message... (Enter to send, Shift+Enter for newline)",
                                    label="",
                                    scale=5,
                                    lines=2,
                                    max_lines=6,
                                )
                                send_btn = gr.Button(
                                    "Send ▶",
                                    variant="primary",
                                    scale=1,
                                    elem_classes="primary-btn",
                                )
        
                            with gr.Row():
                                new_chat_btn = gr.Button("🗨️ New Chat", size="sm")
                                export_btn = gr.Button("📤 Export Chat", size="sm")
        
                        # ─── Sidebar ──────────────────────────
                        with gr.Column(scale=1):
                            # Model Selection
                            with gr.Group():
                                gr.Markdown("### 🤖 Model")
                                model_dropdown = gr.Dropdown(
                                    choices=self._get_available_models(),
                                    label="Select Model",
                                    elem_classes="model-dropdown",
                                )
                                with gr.Row():
                                    load_btn = gr.Button(
                                        "Load Model",
                                        variant="primary",
                                        elem_classes="primary-btn",
                                        scale=3,
                                    )
                                    refresh_models_btn = gr.Button("🔄", size="sm", scale=1)
        
                            # Conversation History
                            with gr.Accordion("💬 Chat History", open=False):
                                history_dropdown = gr.Dropdown(
                                    choices=self._get_conversation_list(),
                                    label="Past Conversations",
                                    interactive=True,
                                )
                                with gr.Row():
                                    load_conv_btn = gr.Button("📂 Load", size="sm", scale=2)
                                    delete_conv_btn = gr.Button("🗑️ Delete", size="sm", scale=1)
                                    refresh_conv_btn = gr.Button("🔄", size="sm", scale=1)
                                history_status = gr.Textbox(
                                    label="", interactive=False, lines=1, visible=True,
                                )
        
                            # Conversation Search
                            with gr.Accordion("🔍 Search History", open=False):
                                search_input = gr.Textbox(
                                    placeholder="Search across all chats...",
                                    label="",
                                    lines=1,
                                )
                                search_btn = gr.Button("Search", size="sm")
                                search_results = gr.Textbox(
                                    label="Results", interactive=False, lines=6,
                                )
        
                            # System Prompt
                            with gr.Accordion("📝 System Prompt", open=False):
                                system_prompt = gr.Textbox(
                                    value="You are a helpful, knowledgeable AI assistant. Answer questions clearly and thoroughly.",
                                    label="",
                                    lines=4,
                                    placeholder="Define the AI's personality...",
                                )
        
                            # Generation Settings
                            with gr.Accordion("⚙️ Settings", open=False):
                                temperature = gr.Slider(
                                    minimum=0.0, maximum=2.0, value=0.7,
                                    step=0.1, label="Temperature",
                                )
                                max_tokens = gr.Slider(
                                    minimum=64, maximum=4096, value=2048,
                                    step=64, label="Max Tokens",
                                )
        
                            # RAG / Knowledge Base
                            with gr.Accordion("📄 Knowledge Base (RAG)", open=False):
                                use_rag = gr.Checkbox(
                                    label="Use document context", value=False
                                )
                                file_upload = gr.File(
                                    label="Upload Document",
                                    file_types=[".txt", ".md", ".pdf", ".py", ".json", ".csv"],
                                )
                                upload_status = gr.Textbox(
                                    label="Status", interactive=False, lines=2
                                )
                                with gr.Row():
                                    kb_stats_btn = gr.Button("📊 Stats", size="sm")
                                    clear_kb_btn = gr.Button("🗑️ Clear All", size="sm")
        
                                # Per-document deletion
                                gr.Markdown("**Remove a Document:**")
                                source_dropdown = gr.Dropdown(
                                    choices=self._get_kb_sources(),
                                    label="Select Document",
                                    interactive=True,
                                )
                                with gr.Row():
                                    delete_source_btn = gr.Button("🗑️ Remove Doc", size="sm")
                                    refresh_sources_btn = gr.Button("🔄", size="sm")
        
                            # Export info
                            export_output = gr.Textbox(
                                label="Export Path", interactive=False, visible=False
                            )

                with gr.Tab("📦 Model Manager"):
                    gr.Markdown("### Download AI Models\nSelect a model below to magically download to your device and chat offline.")
                    download_buttons = {}
                    
                    for key, entry in config.MODEL_CATALOG.items():
                        
                        # Hardware check (llmfit)
                        fit_info = get_model_fit_info(entry.get("hf_id", ""))
                        
                        fit_badge = ""
                        speed_estimate = ""
                        if fit_info:
                            level = fit_info["fit_level"]
                            color = "🟢" if level == "Perfect" else "🟡" if level == "Good" or level == "Marginal" else "🔴"
                            fit_badge = f"{color} **Fit:** {level} (Requires {fit_info['memory_required_gb']} GB RAM/VRAM)"
                            if fit_info["estimated_tps"] > 0:
                                speed_estimate = f" | ⚡ **Est. Speed:** ~{int(fit_info['estimated_tps'])} tok/s"
                        
                        with gr.Group():
                            with gr.Row():
                                with gr.Column(scale=4):
                                    gr.Markdown(f"### {entry['name']}\n**Size**: ~{entry['size_gb']} GB\n\n_{entry['description']}_\n\n{fit_badge}{speed_estimate}")
                                with gr.Column(scale=1):
                                    btn = gr.Button("Download ⬇️", variant="primary")
                                    download_buttons[key] = btn
                    
                    with gr.Group():
                        gr.Markdown("### Download Progress")
                        download_status = gr.Textbox(label="Status", interactive=False, lines=1)
                        cancel_dl_btn = gr.Button("Stop Download", variant="stop")

            # ─── Event Handlers ───────────────────────

            # Load model
            load_btn.click(
                fn=self._load_model,
                inputs=[model_dropdown],
                outputs=[status_bar],
            )

            # Refresh model list
            refresh_models_btn.click(
                fn=self._refresh_models,
                outputs=[model_dropdown],
            )
            
            # Model Manager Download Hookups
            def make_dl_handler(k):
                def handler(progress=gr.Progress()):
                    return self._download_model_ui(k, progress=progress)
                return handler

            for key, btn in download_buttons.items():
                btn.click(
                    fn=make_dl_handler(key),
                    outputs=[status_bar, model_dropdown, download_status]
                )

            cancel_dl_btn.click(
                fn=self._cancel_download,
                outputs=[download_status]
            )

            # Send message (button)
            send_btn.click(
                fn=self._chat_respond,
                inputs=[msg_input, chatbot, system_prompt, temperature, max_tokens, use_rag],
                outputs=[chatbot, msg_input],
            )

            # Send message (enter key)
            msg_input.submit(
                fn=self._chat_respond,
                inputs=[msg_input, chatbot, system_prompt, temperature, max_tokens, use_rag],
                outputs=[chatbot, msg_input],
            )

            # New chat
            new_chat_btn.click(
                fn=self._new_chat,
                inputs=[system_prompt],
                outputs=[chatbot, upload_status],
            )

            # Load past conversation
            load_conv_btn.click(
                fn=self._load_conversation,
                inputs=[history_dropdown],
                outputs=[chatbot, history_status],
            )

            # Delete conversation
            delete_conv_btn.click(
                fn=self._delete_selected_conversation,
                inputs=[history_dropdown],
                outputs=[history_status],
            )

            # Refresh conversation list
            refresh_conv_btn.click(
                fn=lambda: gr.update(choices=self._get_conversation_list()),
                outputs=[history_dropdown],
            )

            # Search conversations
            search_btn.click(
                fn=self._search_conversations,
                inputs=[search_input],
                outputs=[search_results],
            )
            search_input.submit(
                fn=self._search_conversations,
                inputs=[search_input],
                outputs=[search_results],
            )

            # Upload document
            file_upload.change(
                fn=self._upload_document,
                inputs=[file_upload],
                outputs=[upload_status],
            )

            # KB stats
            kb_stats_btn.click(
                fn=self._get_kb_stats,
                outputs=[upload_status],
            )

            # Clear KB
            clear_kb_btn.click(
                fn=self._clear_knowledge_base,
                outputs=[upload_status],
            )

            # Delete specific document source
            delete_source_btn.click(
                fn=self._delete_kb_source,
                inputs=[source_dropdown],
                outputs=[upload_status],
            )

            # Refresh source list
            refresh_sources_btn.click(
                fn=lambda: gr.update(choices=self._get_kb_sources()),
                outputs=[source_dropdown],
            )

            # Export
            export_btn.click(
                fn=self._export_chat,
                outputs=[export_output],
            )

        return app



def launch_ui(share: bool = False, prevent_thread_lock: bool = False, inbrowser: bool = True, port_override: Optional[int] = None):
    """Launch the Personal LLM web interface.
    
    Args:
        share: Create a public Gradio share link.
        prevent_thread_lock: If True, don't block the main thread (for desktop app).
        inbrowser: If True, auto-open browser tab.
        port_override: If set, use this port instead of config.UI_PORT.
    """
    ui = PersonalLLMUI()

    # Auto-load default model if available
    default = ui.model_manager.get_default_model()
    if default:
        print("\n🔄 Auto-loading default model...")
        ui.engine.load(
            default["path"],
            n_gpu_layers=config.N_GPU_LAYERS,
            n_ctx=config.CONTEXT_SIZE,
            chat_format=default.get("chat_format"),
        )
        ui._current_conversation = ui.chat_engine.new_conversation()

    app = ui.build()
    
    port = port_override if port_override is not None else config.UI_PORT
    print(f"\n🚀 Launching Personal LLM at http://{config.UI_HOST}:{port}")
    print("   Press Ctrl+C to stop\n")
    return app.launch(
        server_name=config.UI_HOST,
        server_port=port,
        share=share,
        inbrowser=inbrowser,
        prevent_thread_lock=prevent_thread_lock,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="purple",
            neutral_hue="slate",
            font=["Inter", "ui-sans-serif", "system-ui"],
        ),
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    launch_ui()
