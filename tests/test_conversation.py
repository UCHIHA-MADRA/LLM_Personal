# pyre-ignore-all-errors
"""Tests for Conversation.to_dict / from_dict round-trip and title generation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from personal_llm.chat_engine import Conversation  # pyre-ignore[21]


class TestConversationRoundTrip:
    """Verify serialisation and deserialisation symmetry."""

    def test_to_dict_returns_all_fields(self):
        conv = Conversation()
        conv.add_user_message("Hello world")
        conv.add_assistant_message("Hi there")
        d = conv.to_dict()
        assert "id" in d
        assert "title" in d
        assert "messages" in d
        assert "system_prompt" in d
        assert "created_at" in d
        assert "updated_at" in d

    def test_from_dict_restores_state(self, sample_conversation_data):
        conv = Conversation.from_dict(sample_conversation_data)
        assert conv.id == "abc12345"
        assert conv.title == "Test Conversation"
        assert len(conv.messages) == 4
        assert conv.messages[0]["role"] == "user"
        assert conv.model_name == "test-model"

    def test_round_trip_preserves_data(self):
        original = Conversation(system_prompt="Be concise.")
        original.add_user_message("What is Python?")
        original.add_assistant_message("A programming language.")
        original.add_user_message("Tell me more.")
        original.add_assistant_message("It's versatile and readable.")

        d = original.to_dict()
        restored = Conversation.from_dict(d)

        assert restored.id == original.id
        assert restored.title == original.title
        assert restored.system_prompt == original.system_prompt
        assert len(restored.messages) == len(original.messages)
        for orig_msg, rest_msg in zip(original.messages, restored.messages):
            assert orig_msg["role"] == rest_msg["role"]
            assert orig_msg["content"] == rest_msg["content"]

    def test_empty_conversation_round_trip(self):
        conv = Conversation()
        d = conv.to_dict()
        restored = Conversation.from_dict(d)
        assert restored.id == conv.id
        assert len(restored.messages) == 0


class TestTitleGeneration:
    """Verify auto-title from first user message."""

    def test_title_set_from_first_message(self):
        conv = Conversation()
        conv.add_user_message("How do I cook pasta?")
        assert conv.title == "How do I cook pasta?"

    def test_title_truncated_at_60_chars(self):
        long_msg = "A" * 100
        conv = Conversation()
        conv.add_user_message(long_msg)
        assert len(conv.title) <= 63  # 60 chars + "..."

    def test_title_not_overwritten_by_second_message(self):
        conv = Conversation()
        conv.add_user_message("First question")
        conv.add_user_message("Second question")
        assert conv.title == "First question"

    def test_assistant_message_does_not_set_title(self):
        conv = Conversation()
        conv.add_assistant_message("I'm ready to help")
        assert conv.title == "New Conversation"  # default title


class TestContextMessages:
    """Verify get_context_messages with history limits."""

    def test_includes_system_prompt(self):
        conv = Conversation(system_prompt="You are helpful.")
        conv.add_user_message("Hi")
        msgs = conv.get_context_messages()
        assert msgs[0]["role"] == "system"
        assert "helpful" in msgs[0]["content"]

    def test_limits_history_turns(self):
        conv = Conversation()
        for i in range(50):
            conv.add_user_message(f"Question {i}")
            conv.add_assistant_message(f"Answer {i}")
        msgs = conv.get_context_messages()
        # Should not exceed MAX_HISTORY_TURNS * 2 + 1 (system)
        assert len(msgs) <= 41  # 20 turns * 2 + system

    def test_empty_conversation_returns_system_only(self):
        conv = Conversation(system_prompt="Test prompt")
        msgs = conv.get_context_messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"
