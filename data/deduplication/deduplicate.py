"""Deterministic duplicate detection for corpus documents."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass


_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class DuplicateResult:
    """Result of deduplicating an ordered collection of documents."""

    unique_documents: list[str]
    duplicate_indices: list[int]
    duplicate_of: dict[int, int]


def normalize_for_dedup(text: str) -> str:
    """Normalize text so formatting-only differences compare as duplicates."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    text = unicodedata.normalize("NFKC", text)
    return _WHITESPACE_RE.sub(" ", text).strip().casefold()


def text_hash(text: str) -> str:
    """Return a stable SHA-256 hash of normalized text."""
    normalized = normalize_for_dedup(text).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def deduplicate_documents(documents: list[str] | tuple[str, ...]) -> DuplicateResult:
    """Keep the first occurrence of each normalized document.

    Duplicate detection is deterministic and preserves the original order of
    unique documents. The returned indices refer to the input collection.
    """
    seen: dict[str, int] = {}
    unique: list[str] = []
    duplicate_indices: list[int] = []
    duplicate_of: dict[int, int] = {}

    for index, document in enumerate(documents):
        key = text_hash(document)
        if key in seen:
            duplicate_indices.append(index)
            duplicate_of[index] = seen[key]
            continue
        seen[key] = index
        unique.append(document)

    return DuplicateResult(unique, duplicate_indices, duplicate_of)


def deduplicate_records(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Deduplicate metadata records containing a ``text`` field.

    Returns ``(unique_records, duplicate_records)``. Duplicate records retain
    their original metadata and receive ``duplicate_of_index``.
    """
    seen: dict[str, int] = {}
    unique_records: list[dict] = []
    duplicate_records: list[dict] = []

    for index, record in enumerate(records):
        if "text" not in record:
            raise KeyError("each record must contain a 'text' field")
        key = text_hash(record["text"])
        if key in seen:
            duplicate = dict(record)
            duplicate["duplicate_of_index"] = seen[key]
            duplicate_records.append(duplicate)
            continue
        seen[key] = index
        unique_records.append(dict(record))

    return unique_records, duplicate_records
