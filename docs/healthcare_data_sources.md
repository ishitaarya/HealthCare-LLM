# Healthcare LLM — Initial Data Source Candidates

This document records candidate sources for Phase 2 Step 8. A candidate is not automatically approved for training. License and intended-use terms must be checked for the exact material and version before ingestion.

## Candidate A — MedlinePlus public-domain sections

MedlinePlus states that some of its content is public domain, including health-topic summaries, medical-test information, and genetics summaries. Other MedlinePlus content is copyrighted and subject to separate licensing restrictions.

Decision: **Candidate for targeted ingestion of explicitly identified public-domain sections only.** Do not ingest the whole site as if it were public domain.

Source: https://www.medlineplus.gov/about/using/usingcontent/

## Candidate B — PubMed Central Open Access Subset

PMC provides an Open Access Subset containing articles available under Creative Commons or similar licenses that permit more liberal reuse. PMC explicitly states that license terms vary by article and that automated retrieval should use its approved services.

Decision: **Strong candidate for scientific/biomedical text, subject to per-article license filtering.** Store license metadata with each document.

Source: https://pmc.ncbi.nlm.nih.gov/tools/openftlist/

## Candidate C — PubMed metadata and abstracts

NCBI provides PubMed data through FTP and E-utilities. PubMed contains citations and abstracts for biomedical literature. Terms and conditions are provided with the downloadable data.

Decision: **Candidate for metadata/abstract experiments after reviewing the current terms of use.** Full article text should not be assumed to be included or reusable merely because a PubMed record exists.

Source: https://pubmed.ncbi.nlm.nih.gov/download/

## Candidate D — MIMIC-IV

MIMIC-IV provides rich clinical data, but access is credentialed and requires a Data Use Agreement and required training. Its license restricts handling and sharing of the data. Derived datasets/models are also subject to specific guidance.

Decision: **Not part of the initial open corpus.** Consider later only if the project obtains the required access and the intended research use is compatible with the agreement.

Source: https://physionet.org/content/mimiciv/3.1/

## Candidate E — MIMIC-IV-Note

MIMIC-IV-Note contains deidentified free-text clinical notes, but uses the same credentialed-access model and PhysioNet data-use restrictions.

Decision: **Restricted research candidate, not an initial downloadable corpus.**

Source: https://www.physionet.org/content/mimic-iv-note/2.1/

## Initial source priority

For the first ingestion pipeline, prioritize sources that can be reviewed and used without restricted patient-data access:

1. Explicitly public-domain MedlinePlus sections
2. PMC Open Access Subset with compatible licenses
3. Carefully reviewed PubMed metadata/abstract material
4. Restricted clinical datasets only after separate access and legal/usage review

## Important rule

A source being free to read online is not enough. For every document ingested into the training corpus, preserve provenance and licensing metadata so the dataset can be audited later.

## Phase 2 status

Step 8 — Initial healthcare data sources identified and documented.
