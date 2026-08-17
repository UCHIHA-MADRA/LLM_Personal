# pyre-ignore-all-errors
"""Shared pytest fixtures for Personal LLM tests."""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest  # pyre-ignore[21]

# ── Make personal_llm importable ──────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Mock LLMEngine (no real model loading) ────────────────────────
class MockLLMEngine:
    """A fake LLMEngine that returns canned responses without loading a model."""

    def __init__(self):
        self.model = MagicMock()
        self.model_path = "/fake/model.gguf"
        self.model_name = "test-model"
        self._is_loaded = True

    @property
    def is_loaded(self):
        return self._is_loaded

    def load(self, model_path, **kwargs):
        self._is_loaded = True
        return True

    def unload(self):
        self._is_loaded = False

    def generate(self, prompt, stream=False, **kwargs):
        if stream:
            return iter(["Hello", " from", " mock"])
        return "Hello from mock engine"

    def chat(self, messages, stream=False, **kwargs):
        if stream:
            return iter(["Mock", " chat", " response"])
        return "Mock chat response"

    def get_info(self):
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "is_loaded": self._is_loaded,
        }


@pytest.fixture
def mock_engine():
    """Provide a mock LLMEngine instance."""
    return MockLLMEngine()


@pytest.fixture
def temp_dir():
    """Provide a temporary directory that's cleaned up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_conversation_data():
    """Return a dict representing a serialised Conversation."""
    return {
        "id": "abc12345",
        "title": "Test Conversation",
        "system_prompt": "You are a helpful assistant.",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help?"},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
        ],
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:01:00",
        "model_name": "test-model",
    }
