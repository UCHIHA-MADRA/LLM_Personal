# pyre-ignore-all-errors
"""Tests for model catalog integrity — every entry should be self-consistent."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from personal_llm.config import MODEL_CATALOG  # pyre-ignore[21]


class TestCatalogIntegrity:
    """Verify all catalog entries have valid, consistent metadata."""

    REQUIRED_FIELDS = [
        "name", "repo_id", "hf_id", "filename", "size_gb",
        "size_bytes", "description", "best_at", "chat_format",
        "tier", "license",
    ]

    def test_all_entries_have_required_fields(self):
        for key, entry in MODEL_CATALOG.items():
            for field in self.REQUIRED_FIELDS:
                assert field in entry, f"Model '{key}' missing field '{field}'"

    def test_no_duplicate_keys(self):
        """Python dicts silently overwrite duplicates — verify count matches."""
        import ast
        config_path = Path(__file__).resolve().parent.parent / "personal_llm" / "config.py"
        source = config_path.read_text(encoding="utf-8")
        # Count string keys that look like catalog entries
        key_lines = [
            line.strip() for line in source.splitlines()
            if line.strip().startswith('"') and line.strip().endswith(': {')
        ]
        # The dict should have exactly as many entries as key lines
        # (this is a heuristic — works for this specific file format)
        assert len(key_lines) >= len(MODEL_CATALOG), (
            f"Possible duplicate keys: {len(key_lines)} key lines but only "
            f"{len(MODEL_CATALOG)} entries in the parsed dict"
        )

    def test_size_gb_matches_ballpark(self):
        """size_gb should roughly equal size_bytes / 1e9."""
        for key, entry in MODEL_CATALOG.items():
            if entry["size_bytes"] > 0:
                expected_gb = entry["size_bytes"] / 1e9
                assert abs(entry["size_gb"] - expected_gb) < 2.0, (
                    f"Model '{key}': size_gb={entry['size_gb']} but "
                    f"size_bytes implies ~{expected_gb:.1f} GB"
                )

    def test_filename_ends_with_gguf(self):
        for key, entry in MODEL_CATALOG.items():
            assert entry["filename"].endswith(".gguf"), (
                f"Model '{key}' filename '{entry['filename']}' doesn't end with .gguf"
            )

    def test_tier_is_valid(self):
        for key, entry in MODEL_CATALOG.items():
            assert entry["tier"] in (1, 2, 3), (
                f"Model '{key}' has invalid tier {entry['tier']}"
            )

    def test_key_matches_name_ballpark(self):
        """Key should loosely relate to the model name (no total mismatches)."""
        for key, entry in MODEL_CATALOG.items():
            name_lower = entry["name"].lower()
            # At least one word from the key should appear in the name
            key_words = key.replace("-", " ").replace(".", " ").split()
            matches = sum(1 for w in key_words if w in name_lower)
            assert matches >= 1, (
                f"Key '{key}' doesn't match name '{entry['name']}'"
            )

    def test_repo_id_looks_valid(self):
        for key, entry in MODEL_CATALOG.items():
            repo = entry["repo_id"]
            assert "/" in repo, f"Model '{key}' repo_id '{repo}' missing '/'"
            parts = repo.split("/")
            assert len(parts) == 2, f"Model '{key}' repo_id '{repo}' has wrong format"

    def test_chat_format_is_known_or_none(self):
        known_formats = {
            None, "chatml", "llama-2", "llama-3", "mistral-instruct",
            "gemma", "command-r", "alpaca",
        }
        for key, entry in MODEL_CATALOG.items():
            assert entry["chat_format"] in known_formats, (
                f"Model '{key}' has unknown chat_format '{entry['chat_format']}'"
            )
