# Healthcare LLM — Healthcare Data Strategy

## Purpose

The training corpus will support health-information and education tasks while avoiding use of the model as a diagnostic or prescribing system.

## Dataset layers

### 1. Medical knowledge / pretraining

Use permitted, high-quality medical and health-information text to teach language, terminology, concepts, and factual relationships.

Examples of source categories to evaluate:

- Public-domain government health information
- Openly licensed medical educational material
- Open-access scientific and biomedical literature where the license permits the intended use
- Curated terminology and reference resources with compatible licenses

### 2. Instruction and question-answer data

Use permitted examples covering:

- Plain-language explanations of medical terms
- Health-information questions
- Medical-information summarization
- Simplification of technical text
- Structured explanations of symptoms, tests, and procedures without presenting unsupported diagnoses
- Appropriate uncertainty and professional-care escalation

Instruction data should be clearly separated from raw pretraining text.

### 3. Evaluation data

Keep evaluation examples separate from training data. Include factuality, terminology, summarization, report simplification, uncertainty, and safety cases.

## Data acceptance criteria

Every external source should be reviewed for:

1. License and permitted use
2. Source provenance and authority
3. Medical relevance
4. Recency where medical guidance can change
5. Duplicate or near-duplicate content
6. Personally identifiable information (PII)
7. Formatting and preprocessing requirements
8. Potential benchmark contamination

No source should enter the training corpus solely because it is publicly accessible; public availability does not automatically grant permission for redistribution or model training.

## PII and sensitive information

The pipeline should detect and remove or exclude unnecessary personal information. Patient records, private conversations, credentials, contact details, and other sensitive data should not be included unless there is a documented lawful basis and an explicit project requirement.

## Quality controls

Before training, the pipeline should record source metadata and apply:

- Encoding normalization
- Text cleaning
- Language filtering as required
- Medical-domain relevance filtering
- Deduplication
- Length and corruption checks
- Train/validation/test separation

## Data mixture

The exact mixture will be determined after source discovery and licensing review. We will not assume that a single dataset is sufficient. Pretraining data, instruction data, and evaluation data will remain distinguishable so that experiments and safety evaluations can be reproduced.

## Initial policy

Do not download or train on a dataset until its source, license, intended use, and data quality have been reviewed. Record those decisions in dataset metadata.

## Phase 2 status

Step 7 — Healthcare data strategy documented.
