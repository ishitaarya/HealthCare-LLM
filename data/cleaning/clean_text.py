"""Deterministic text normalization and basic quality filtering."""

from __future__ import annotations

import re
import unicodedata


_WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
_NEWLINE_RE = re.compile(r"\n{3,}")
_HTML_RE = re.compile(r"<[^>]+>")


def normalize_unicode(text: str) -> str:
    """Normalize compatible Unicode representations to NFC."""
    return unicodedata.normalize("NFC", text)


def remove_html(text: str) -> str:
    """Remove simple HTML/XML tags while preserving their surrounding text."""
    return _HTML_RE.sub(" ", text)


def normalize_whitespace(text: str) -> str:
    """Collapse repeated spaces and excessive blank lines."""
    text = _WHITESPACE_RE.sub(" ", text)
    text = re.sub(r" *\n *", "\n", text)
    return _NEWLINE_RE.sub("\n\n", text).strip()


def clean_text(text: str) -> str:
    """Apply the standard deterministic cleaning sequence."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    text = normalize_unicode(text)
    text = remove_html(text)
    text = normalize_whitespace(text)
    return text


def is_valid_document(text: str, min_chars: int = 20, max_chars: int = 1_000_000) -> bool:
    """Return whether cleaned text falls within basic document-size limits."""
    if min_chars < 0 or max_chars < min_chars:
        raise ValueError("invalid document-size limits")
    length = len(text.strip())
    return min_chars <= length <= max_chars


def clean_document(text: str, min_chars: int = 20, max_chars: int = 1_000_000) -> str | None:
    """Clean one document and return None when it fails basic quality limits."""
    cleaned = clean_text(text)
    if not is_valid_document(cleaned, min_chars=min_chars, max_chars=max_chars):
        return None
    return cleaned
