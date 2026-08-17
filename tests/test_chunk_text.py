# pyre-ignore-all-errors
"""Tests for knowledge_base._chunk_text edge cases."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from personal_llm.knowledge_base import _chunk_text  # pyre-ignore[21]


class TestChunkText:
    """Edge-case coverage for the text chunker."""

    def test_empty_string(self):
        chunks = _chunk_text("")
        assert chunks == [] or chunks == [""]

    def test_shorter_than_chunk_size(self):
        text = "Hello world"
        chunks = _chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_exact_chunk_size(self):
        text = "A" * 500
        chunks = _chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) == 1

    def test_multiple_chunks_have_overlap(self):
        text = "A" * 1000
        chunks = _chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) >= 2
        # Verify overlap: end of first chunk should appear at start of second
        if len(chunks) >= 2:
            overlap_region = chunks[0][-50:]
            assert chunks[1].startswith(overlap_region)

    def test_unicode_content(self):
        text = "こんにちは世界" * 100  # Japanese text
        chunks = _chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) >= 1
        # All content should be present across chunks
        full = chunks[0]
        for c in chunks[1:]:
            full += c[10:]  # skip overlap
        # Should contain all the original chars (approximately)
        assert len(full) >= len(text) * 0.9

    def test_single_character_text(self):
        chunks = _chunk_text("X", chunk_size=500, overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == "X"

    def test_zero_overlap(self):
        text = "ABCDEFGHIJ"
        chunks = _chunk_text(text, chunk_size=5, overlap=0)
        assert len(chunks) == 2
        assert chunks[0] == "ABCDE"
        assert chunks[1] == "FGHIJ"
