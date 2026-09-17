"""Local-file ingestion with provenance metadata.

This module deliberately does not download external datasets. External sources
must first pass the project's source and licensing review.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ALLOWED_SUFFIXES = {".txt", ".md", ".json", ".jsonl", ".csv"}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ingest_file(
    source_path: str | Path,
    raw_dir: str | Path = "data/raw",
    metadata_dir: str | Path = "data/metadata",
    source_id: str = "unknown",
    license_name: str = "unknown",
    source_url: str | None = None,
) -> dict:
    """Copy one approved local source file into raw storage and record metadata."""
    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError(f"source file not found: {source}")
    if source.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError(f"unsupported file type: {source.suffix}")

    raw_path = Path(raw_dir)
    metadata_path = Path(metadata_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    metadata_path.mkdir(parents=True, exist_ok=True)

    destination = raw_path / source.name
    shutil.copy2(source, destination)
    checksum = sha256_file(destination)

    record = {
        "source_id": source_id,
        "filename": destination.name,
        "path": str(destination).replace("\\", "/"),
        "size_bytes": destination.stat().st_size,
        "sha256": checksum,
        "license": license_name,
        "source_url": source_url,
        "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    manifest = metadata_path / "ingestion_manifest.jsonl"
    with manifest.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record
