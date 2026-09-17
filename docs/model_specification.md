# Healthcare LLM — Model Specification

## Purpose

This project is building a healthcare-focused language model for health information and education.

Initial capabilities:

- Explain medical terminology in plain language
- Summarize medical information
- Simplify medical reports
- Answer general health-information questions

The model is not intended to diagnose patients, prescribe treatment, or replace qualified healthcare professionals.

## Architecture

Decoder-only Transformer using causal language modeling.

Core components:

- Token embeddings
- Learned positional embeddings
- Causal multi-head self-attention
- Feed-forward networks
- Layer normalization
- Residual connections
- Language-model output head

## Initial Configuration

| Parameter | Current value | Final target |
|---|---:|---:|
| Vocabulary size | 10,000 | TBD |
| Context length | 256 | 256 initially |
| Embedding dimension | 256 | TBD |
| Transformer layers | 6 | TBD |
| Attention heads | 8 | TBD |
| Dropout | 0.1 | 0.1 initially |
| Batch size | 16 | TBD |
| Learning rate | 3e-4 | TBD |

Final model size will be selected after hardware benchmarking.

## Tokenization

The placeholder tokenizer will be replaced with a real subword tokenizer.

Requirements:

- Efficient handling of medical terminology
- Good handling of uncommon medical terms
- Special tokens for training and inference
- Trainable on the permitted project corpus

## Training

Initial objective:

Causal language modeling / next-token prediction.

Loss:

Cross-entropy.

## Data Strategy

Healthcare data will be separated into:

1. Medical knowledge/pretraining data
2. Instruction and question-answer data
3. Evaluation data

External datasets must be reviewed for:

- Licensing
- Source quality
- Medical relevance
- Duplicates
- Data contamination
- Personally identifiable information
- Preprocessing requirements

## Safety

The model will be evaluated for:

- Medical hallucinations
- Overconfident answers
- Unsupported diagnosis claims
- Unsupported treatment claims
- Failure to communicate uncertainty
- Incorrect medical-report interpretation
- Appropriate escalation to professional care

## Phase 2 Completion

Phase 2 will be considered complete when we have:

- Hardware-informed configuration
- Real tokenizer
- Documented data strategy
- Data ingestion pipeline
- Cleaning and filtering
- Deduplication
- Train/validation/test split
- Tokenized corpus
- DataLoader
- Data-quality tests
- First real-data training run
- Baseline evaluation

## Status

Phase 2 — Step 1: Model specification complete.
