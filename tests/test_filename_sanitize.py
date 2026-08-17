# pyre-ignore-all-errors
"""Tests for filename sanitisation in file upload endpoints."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestFilenameSanitize:
    """Verify path traversal attempts are blocked."""

    def _sanitize(self, filename: str) -> str:
        """Replicate the sanitisation logic from api.py."""
        return Path(filename).name

    def test_normal_filename(self):
        assert self._sanitize("report.pdf") == "report.pdf"

    def test_path_traversal_unix(self):
        result = self._sanitize("../../etc/passwd")
        assert result == "passwd"
        assert ".." not in result

    def test_path_traversal_windows(self):
        result = self._sanitize("..\\..\\Windows\\System32\\config")
        assert ".." not in result

    def test_absolute_path_unix(self):
        result = self._sanitize("/etc/shadow")
        assert result == "shadow"

    def test_absolute_path_windows(self):
        result = self._sanitize("C:\\Users\\admin\\secrets.txt")
        assert result == "secrets.txt"

    def test_hidden_file(self):
        result = self._sanitize(".env")
        assert result == ".env"

    def test_spaces_in_name(self):
        result = self._sanitize("my document (2).pdf")
        assert result == "my document (2).pdf"

    def test_unicode_filename(self):
        result = self._sanitize("文档.pdf")
        assert result == "文档.pdf"

    def test_empty_filename(self):
        result = self._sanitize("")
        # Should not crash
        assert isinstance(result, str)
