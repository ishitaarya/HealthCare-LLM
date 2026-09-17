"""Conservative healthcare-domain filtering.

This filter is intentionally recall-oriented: domain keywords provide a signal,
not proof that a document is medically relevant. Source-level curation remains
necessary before training.
"""

from __future__ import annotations

import re


MEDICAL_TERMS = frozenset(
    {
        "anatomy", "antibiotic", "blood", "cancer", "cardiac", "clinical",
        "diagnosis", "disease", "doctor", "drug", "health", "hospital",
        "infection", "medicine", "medical", "patient", "pharmacy", "physician",
        "symptom", "therapy", "treatment", "vaccine", "virus", "surgery",
        "laboratory", "diagnostic", "immunization", "pathology", "radiology",
    }
)

_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z]+")


def medical_term_ratio(text: str) -> float:
    """Return the fraction of alphabetic words matching the medical vocabulary."""
    words = [word.lower() for word in _WORD_RE.findall(text)]
    if not words:
        return 0.0
    return sum(word in MEDICAL_TERMS for word in words) / len(words)


def url_ratio(text: str) -> float:
    """Estimate how much of a document consists of URL-like content."""
    if not text.strip():
        return 0.0
    return len(_URL_RE.findall(text)) / max(1, len(text.split()))


def repetition_ratio(text: str) -> float:
    """Measure repeated non-empty lines; high values can indicate scraped noise."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    return 1.0 - (len(set(lines)) / len(lines))


def passes_quality_filter(
    text: str,
    *,
    min_words: int = 10,
    max_url_ratio: float = 0.5,
    max_repetition_ratio: float = 0.5,
) -> bool:
    """Apply generic quality checks without requiring medical keywords."""
    if len(text.split()) < min_words:
        return False
    if url_ratio(text) > max_url_ratio:
        return False
    if repetition_ratio(text) > max_repetition_ratio:
        return False
    return True


def passes_healthcare_filter(
    text: str,
    *,
    min_words: int = 10,
    min_medical_term_ratio: float = 0.01,
) -> bool:
    """Return whether text has enough healthcare signal for downstream review.

    The default threshold is deliberately low so legitimate medical documents
    are not discarded solely because they use varied terminology.
    """
    if not passes_quality_filter(text, min_words=min_words):
        return False
    return medical_term_ratio(text) >= min_medical_term_ratio
