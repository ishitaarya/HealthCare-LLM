# Data Pipeline Directory

This directory separates downloaded source material from derived training artifacts.

```text
 data/
 ├── raw/         # Original downloaded source files; never edit in place
 ├── processed/   # Parsed records extracted from raw sources
 ├── cleaned/     # Normalized and filtered text
 ├── tokenized/   # Token ID datasets produced by the tokenizer
 ├── metadata/    # Source, license, provenance, statistics, and processing manifests
 ├── splits/      # Train / validation / test assignments
 └── sources/     # Source-specific configuration and acquisition notes
```

## Rules

- Keep raw files immutable.
- Store provenance and licensing metadata alongside every source.
- Derived directories must be reproducible from earlier stages.
- Do not commit large datasets or restricted clinical data to Git.
- Do not commit PII or secrets.
- Keep train, validation, and test data explicitly separated.

Large datasets should remain outside the Git repository and be referenced through metadata/manifests.

## Pipeline

```text
source
  ↓
raw
  ↓
processed
  ↓
cleaned
  ↓
splits
  ↓
tokenized
  ↓
DataLoader
```
