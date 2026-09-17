# Healthcare LLM — Tokenizer Specification

## Decision

The project will use a **BPE-style subword tokenizer** trained on the permitted project corpus.

The tokenizer will be trained on the same cleaned corpus family used for model training, with the final vocabulary size selected after inspecting corpus size and token coverage. The current model configuration uses a provisional vocabulary size of 10,000.

## Why subword tokenization

A word-level tokenizer would create a large vocabulary and would struggle with uncommon medical terms. A character-level tokenizer would represent text compactly at the vocabulary level but would require many tokens to represent ordinary words and medical terminology.

BPE-style subword tokenization provides a middle ground:

- Common words and word pieces can receive compact token sequences.
- Uncommon medical terms can be decomposed into smaller known pieces.
- New or rare words are less likely to become a single unknown token.
- The vocabulary can be controlled for CPU and RAM constraints.

## Healthcare-specific requirements

The tokenizer should handle:

- Medical terminology
- Drug and chemical names
- Anatomical terms
- Abbreviations and acronyms
- Numbers and measurements
- Units such as mg, mL, kg, and mmHg
- Punctuation used in medical reports
- Common spelling variants
- Uncommon or newly encountered terms

The tokenizer must not assume that every unfamiliar medical word is an error.

## Special tokens

The initial tokenizer design will reserve explicit IDs for:

- `<PAD>` — padding when batching requires it
- `<UNK>` — fallback for unsupported input
- `<BOS>` — beginning of sequence when required
- `<EOS>` — end of sequence when required
- `<SEP>` — separator for structured examples when required

Special-token IDs must remain stable after the tokenizer is trained so saved datasets and model checkpoints remain compatible.

## Normalization

Normalization should be conservative. It should preserve medically meaningful distinctions such as:

- Decimal values
- Units
- Hyphens when clinically meaningful
- Case where it affects interpretation
- Punctuation used in reports

Whitespace normalization and removal of accidental control characters are acceptable. Aggressive normalization that changes medical meaning should be avoided.

## Training process

1. Collect legally usable source text.
2. Clean and filter the corpus.
3. Remove or protect personally identifiable information as required by the data policy.
4. Split the corpus into training, validation, and test data before tokenizer training where appropriate to avoid evaluation leakage.
5. Train the tokenizer only on the permitted tokenizer-training portion.
6. Measure token coverage and sequence-length statistics on held-out data.
7. Freeze the tokenizer before the main language-model training run.

## Compatibility

The tokenizer output must be integer token IDs in the range `[0, vocab_size)` and must be compatible with `LanguageModelDataset` and the decoder-only Transformer.

Encoding and decoding must satisfy a practical round-trip property: ordinary supported text should encode and decode without unexpected loss of medically meaningful content.

## Evaluation

Before accepting the tokenizer, test it on examples containing:

- Common English sentences
- Medical terminology
- Long medical terms
- Abbreviations
- Numbers and units
- Punctuation
- Rare or unfamiliar terms
- Short medical-report excerpts

Record:

- Vocabulary size
- Average tokens per word
- Average tokens per character
- Unknown-token rate
- Sequence-length distribution
- Examples of problematic tokenization

## Current status

Phase 2 — Step 5: Tokenizer selection complete.

Next: implement the real tokenizer and train it on the permitted corpus after the data strategy is established.
