# AI incident-report classification: promising safety triage co-pilot, not a replacement for pharmacy review

<!-- SOURCE_METADATA_START -->
> **Source article:** Artificial intelligence-based incident analysis and learning system to enhance patient safety and improve treatment quality
> **Authors:** Abbas J. Jinia, Katherine Chapman, Shi Liu, Cesar Della Biancia, Elizabeth Hipp, Eric Lin, Robin Moulder, Dhwani Parikh, Jason Cordero, Caralaina Pistone, Mary Gil, John Ford, Anyi Li, Jean M. Moran
> **Journal:** npj Digital Medicine
> **Published:** 2026-02-16
> **DOI:** 10.1038/s41746-026-02390-2
<!-- SOURCE_METADATA_END -->

## Quick Take
- A fine-tuned Llama 3.1 8B workflow (AI-ILS) classified radiation‑oncology incident reports using the Human Factors Analysis and Classification System (HFACS) with average AUROC ≈ 0.92, Matthews correlation coefficient 0.72, and overall accuracy ≈ 79% on a 307-case validation set; strict agreement with human review on 350 real incidents was 78.73% (88.45% including “possible agreement”).
- Speed is the operational signal: AI-ILS processed 350 incidents in 29.06 minutes (4.98 seconds/incident) versus human review at 2.42 minutes/incident; practical use cases are backlog reduction, pre-labeling, and trend detection with human adjudication for ambiguous or high‑risk events.

## Why It Matters
- Free-text safety reports are slow to code; manual classification creates backlog across medication‑safety, quality, and risk workflows.
- Inconsistent event coding obscures latent system causes (workflow design, supervision, organizational drivers) that demand different interventions than frontline error fixes.
- Continued manual-only review diverts pharmacist time from corrective actions, education, and prevention toward retrospective sorting.

## What They Did
- Single-center, live institutional deployment at Memorial Sloan Kettering Cancer Center using radiation‑oncology incident narratives from 2019–2023.
- Built an AI workflow to map free-text narratives to four HFACS tiers: unsafe acts, preconditions for unsafe acts, unsafe supervision, and organizational influences.
- Trained on 1,548 expert‑curated mock incidents (1,241 train / 307 validation); tested end‑to‑end on 350 real incidents.
- Pipeline included de-identification, acronym expansion, and concept normalization; compared fine‑tuned LLM to a trained BERT model. Two human reviewers served as the reference; inter-reviewer reliability was moderate (quadratic weighted kappa 0.50).

## What They Found
- Fine-tuning changed performance: base model was near chance; after tuning AUROC rose to ≈0.9 across tiers and overall accuracy exceeded 75%.
- The fine‑tuned LLM outperformed BERT on all tiers. Example: “Preconditions for Unsafe Acts” accuracy 87.5% vs 67.9%; MCC 0.82 vs 0.57.
- On 350 real incidents, strict agreement with human review was 78.73%; agreement rose to 88.45% when “possible agreement” was included. Tier strict agreement ranged 67.84% (Tier 1) to 90.06% (Tier 4).
- Throughput advantage is large (≈29× faster by authors’ timing), but evidence is narrow: single center, one specialty, modest real‑world sample, and a subjective reference standard.
- Key reporting gap: no per‑class false positive/false negative rates or confusion matrices for the 350 real incidents, limiting risk profiling for deployment.

## What This Means for Us
- Immediate utility: a triage co‑pilot for high‑volume safety queues, not an autonomous adjudication tool.
- For pharmacists: shifts routine categorization to the model and reallocates human effort to validating edge cases, investigating root causes, and implementing fixes.
- Ask vendors/internal teams: provide per‑category error rates on your taxonomy, with examples of failures on short, vague, or acronym‑heavy narratives.
- Highest‑oversight scenario: medication events where classification (error vs violation vs supervision vs system design) changes the corrective action.

## Strengths & Limitations
  - **Strengths**
    - Real institutional implementation and measurable throughput versus a clear BERT comparator.
    - End‑to‑end pipeline addressing de‑identification and acronym expansion.
  - **Limitations**
    - Single‑center radiation‑oncology setting reduces generalizability to medication‑use workflows without local validation.
    - Training used mock incidents; human reference agreement was moderate; detailed error profiling on real cases was not reported.

## Bottom Line
Early, credible safety‑operations co‑pilot: useful for incident triage and trend tagging, but not ready for unsupervised pharmacy safety classification.
