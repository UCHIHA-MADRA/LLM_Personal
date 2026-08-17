# pyre-ignore-all-errors
"""Tests for context_engine.classify_query."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from personal_llm.context_engine import classify_query  # pyre-ignore[21]


class TestQueryClassifier:
    """Verify query classification into simple/complex/code."""

    # ── Simple queries ────────────────────────────────────────────
    def test_greeting_is_simple(self):
        assert classify_query("hello") == "simple"

    def test_short_question_is_simple(self):
        assert classify_query("what time is it?") == "simple"

    def test_thanks_is_simple(self):
        assert classify_query("thank you!") == "simple"

    def test_hi_there_is_simple(self):
        assert classify_query("hi there") == "simple"

    # ── Complex queries ───────────────────────────────────────────
    def test_explain_with_examples_is_complex(self):
        result = classify_query("Explain quantum computing with examples and compare it to classical computing")
        assert result == "complex"

    def test_analyze_is_complex(self):
        result = classify_query("Analyze the economic impact of AI on employment")
        assert result == "complex"

    def test_compare_is_complex(self):
        result = classify_query("Compare and contrast React vs Vue vs Angular")
        assert result == "complex"

    # ── Code queries ──────────────────────────────────────────────
    def test_write_function_is_code(self):
        result = classify_query("Write a Python function to sort a list")
        assert result == "code"

    def test_debug_is_code(self):
        result = classify_query("Debug this JavaScript code: const x = null; x.toString()")
        assert result == "code"

    def test_code_snippet_is_code(self):
        result = classify_query("def hello():\n    print('hello')\n\nFix the indentation")
        assert result == "code"

    def test_implement_is_code(self):
        result = classify_query("Implement a binary search algorithm in Java")
        assert result == "code"
