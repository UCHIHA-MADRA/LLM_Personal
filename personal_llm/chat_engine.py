"""
Chat Engine — Multi-turn conversation management with persistent history.
Handles system prompts, memory, and session tracking.
"""

import json
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Generator, Any

from . import config
from .llm_engine import LLMEngine, get_engine

logger = logging.getLogger(__name__)


class Conversation:
    """A single conversation session with history."""

    def __init__(
        self,
        conversation_id: Optional[str] = None,
        system_prompt: str = "You are a helpful, knowledgeable AI assistant. Answer questions clearly and thoroughly.",
        title: Optional[str] = None,
    ):
        self.id = conversation_id or str(uuid.uuid4())[:8]
        self.system_prompt = system_prompt
        self.title = title or f"Chat {datetime.now().strftime('%b %d %H:%M')}"
        self.messages: List[Dict[str, str]] = []
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self.model_name = ""

    def add_user_message(self, content: str):
        """Add a user message to the conversation."""
        self.messages.append({"role": "user", "content": content})
        self.updated_at = datetime.now().isoformat()
        # Auto-title from first message
        if not self.title or self.title.startswith("Chat "):
            self.title = content[:60] + ("..." if len(content) > 60 else "")

    def add_assistant_message(self, content: str):
        """Add an assistant response to the conversation."""
        self.messages.append({"role": "assistant", "content": content})
        self.updated_at = datetime.now().isoformat()

    def get_context_messages(
        self,
        max_tokens: int = None,
        reserved_tokens: int = 512,
        rag_context: str = ""
    ) -> List[Dict[str, str]]:
        """
        Get messages formatted for the LLM, dynamically pruning history to fit context.
        
        Args:
            max_tokens: Total context window size (default: config.CONTEXT_SIZE)
            reserved_tokens: Tokens to reserve for the response (default: 512)
            rag_context: RAG context to be included in system prompt
        """
        max_tokens = max_tokens or config.CONTEXT_SIZE
        
        # Estimate token counts (approx 4 chars per token)
        def estimate_tokens(text: str) -> int:
            return len(text) // 4
            
        # 1. Start with system prompt + RAG
        system_content = self.system_prompt
        if rag_context:
            system_content += f"\n\nUse the following context to answer:\n\n{rag_context}"
            
        sys_tokens = estimate_tokens(system_content)
        
        # 2. Calculate remaining budget for history
        available_for_history = max_tokens - sys_tokens - reserved_tokens
        
        if available_for_history < 0:
            logger.warning("System prompt + RAG context exceeds context window!")
            # Truncate RAG if absolutely necessary, but for now just return system
            return [{"role": "system", "content": system_content}]

        # 3. Add history from most recent, until budget full
        history = []
        used_tokens = 0
        
        # Iterate backwards
        for msg in reversed(self.messages):
            msg_tokens = estimate_tokens(msg["content"]) + 10  # +10 for metadata overhead
            if used_tokens + msg_tokens > available_for_history:
                break
            history.insert(0, msg)
            used_tokens += msg_tokens
            
        return [{"role": "system", "content": system_content}] + history

    def to_dict(self) -> Dict[str, Any]:
        """Serialize conversation to dict."""
        return {
            "id": self.id,
            "title": self.title,
            "system_prompt": self.system_prompt,
            "messages": self.messages,
            "model_name": self.model_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Conversation":
        """Deserialize conversation from dict."""
        conv = cls(
            conversation_id=data.get("id"),
            system_prompt=data.get("system_prompt", ""),
            title=data.get("title"),
        )
        conv.messages = data.get("messages", [])
        conv.model_name = data.get("model_name", "")
        conv.created_at = data.get("created_at", "")
        conv.updated_at = data.get("updated_at", "")
        return conv

    def export_markdown(self) -> str:
        """Export the conversation as a markdown string."""
        lines = [f"# {self.title}", ""]
        lines.append(f"**Model:** {self.model_name}")
        lines.append(f"**Date:** {self.created_at[:10]}")
        lines.append(f"**System Prompt:** {self.system_prompt}")
        lines.append("")
        lines.append("---")
        lines.append("")
        for msg in self.messages:
            role = "🧑 You" if msg["role"] == "user" else "🤖 AI"
            lines.append(f"### {role}")
            lines.append("")
            lines.append(msg["content"])
            lines.append("")
        return "\n".join(lines)


class ChatEngine:
    """
    Manages conversations and interfaces with the LLM engine.
    Handles history persistence and streaming responses.
    """

    def __init__(self, engine: Optional[LLMEngine] = None):
        self.engine = engine or get_engine()
        self.conversations: Dict[str, Conversation] = {}
        self.active_conversation: Optional[Conversation] = None
        self.history_dir = config.CHAT_HISTORY_DIR
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self._load_history()

    def _load_history(self):
        """Load conversation history from disk."""
        for f in sorted(self.history_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                conv = Conversation.from_dict(data)
                self.conversations[conv.id] = conv
            except Exception as e:
                logger.warning(f"Failed to load conversation {f.name}: {e}")

    def _save_conversation(self, conv: Conversation):
        """Save a conversation to disk."""
        filepath = self.history_dir / f"{conv.id}.json"
        filepath.write_text(
            json.dumps(conv.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def new_conversation(
        self, system_prompt: str = "", title: str = ""
    ) -> Conversation:
        """Start a new conversation."""
        if not system_prompt:
            system_prompt = "You are a helpful, knowledgeable AI assistant. Answer questions clearly and thoroughly."
        conv = Conversation(system_prompt=system_prompt, title=title)
        conv.model_name = self.engine.model_name
        self.conversations[conv.id] = conv
        self.active_conversation = conv
        return conv

    def get_conversation(self, conv_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        return self.conversations.get(conv_id)

    def list_conversations(self) -> List[Dict[str, Any]]:
        """List all conversations, most recent first."""
        convs = sorted(
            self.conversations.values(),
            key=lambda c: c.updated_at,
            reverse=True,
        )
        return [
            {
                "id": c.id,
                "title": c.title,
                "updated_at": c.updated_at,
                "message_count": len(c.messages),
                "model": c.model_name,
            }
            for c in convs
        ]

    def delete_conversation(self, conv_id: str):
        """Delete a conversation."""
        if conv_id in self.conversations:
            del self.conversations[conv_id]
            filepath = self.history_dir / f"{conv_id}.json"
            if filepath.exists():
                filepath.unlink()
            if self.active_conversation and self.active_conversation.id == conv_id:
                self.active_conversation = None

    def search_conversations(self, query: str) -> List[Dict[str, Any]]:
        """
        Search across all saved conversation titles and messages.

        Args:
            query: Search string (case-insensitive)

        Returns:
            List of dicts with conversation info and matching snippets
        """
        query_lower = query.lower()
        results = []

        for conv in self.conversations.values():
            matches = []

            # Check title
            if conv.title and query_lower in conv.title.lower():
                matches.append(f"📌 Title: {conv.title}")

            # Check messages
            for i, msg in enumerate(conv.messages):
                if query_lower in msg["content"].lower():
                    # Extract snippet around the match
                    content = msg["content"]
                    idx = content.lower().find(query_lower)
                    start = max(0, idx - 40)
                    end = min(len(content), idx + len(query) + 40)
                    snippet = content[start:end]
                    if start > 0:
                        snippet = "..." + snippet
                    if end < len(content):
                        snippet = snippet + "..."
                    role = "🧑" if msg["role"] == "user" else "🤖"
                    matches.append(f"{role} {snippet}")

            if matches:
                results.append({
                    "id": conv.id,
                    "title": conv.title or "Untitled",
                    "updated_at": conv.updated_at,
                    "message_count": len(conv.messages),
                    "matches": matches[:5],  # Limit to 5 snippets per conversation
                })

        # Sort by most recent first
        results.sort(key=lambda x: x["updated_at"], reverse=True)
        return results

    def send_message(
        self,
        message: str,
        conversation: Optional[Conversation] = None,
        rag_context: str = "",
        temperature: float = None,
        max_tokens: int = None,
        stream: bool = True,
    ):
        """
        Send a message and get an AI response.

        Args:
            message: User's message
            conversation: Conversation to use (defaults to active)
            rag_context: Optional RAG context to inject
            temperature: Override default temperature
            max_tokens: Override default max tokens
            stream: If True, yields token strings

        Returns:
            If stream=False: complete response string
            If stream=True: generator yielding token strings
        """
        conv = conversation or self.active_conversation
        if conv is None:
            conv = self.new_conversation()

        # Build system prompt with optional RAG context
        system = conv.system_prompt
        if rag_context:
            system += f"\n\nUse the following context to answer the user's question:\n\n{rag_context}"

        # Add user message
        conv.add_user_message(message)
        conv.model_name = self.engine.model_name

        # Resolve generation parameters FIRST (needed for context budget)
        temp = temperature if temperature is not None else config.TEMPERATURE
        tokens = max_tokens if max_tokens is not None else config.MAX_TOKENS

        # Build context messages with dynamic pruning
        # Use actual model context size when available
        context_budget = config.CONTEXT_SIZE
        if self.engine.is_loaded:
            info = self.engine.get_info()
            if "n_ctx" in info:
                context_budget = info["n_ctx"]

        messages = conv.get_context_messages(
            max_tokens=context_budget,
            reserved_tokens=tokens,  # Reserve space for the response
            rag_context=rag_context,
        )

        if stream:
            return self._stream_response(conv, messages, temp, tokens)
        else:
            response = self.engine.chat(
                messages=messages,
                max_tokens=tokens,
                temperature=temp,
                stream=False,
            )
            conv.add_assistant_message(response)
            self._save_conversation(conv)
            return response

    def _stream_response(
        self,
        conv: Conversation,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Generator[str, None, None]:
        """Stream a response and accumulate it for history."""
        full_response = []

        for token in self.engine.chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        ):
            full_response.append(token)
            yield token

        # Save complete response to history
        complete = "".join(full_response)
        conv.add_assistant_message(complete)
        self._save_conversation(conv)
