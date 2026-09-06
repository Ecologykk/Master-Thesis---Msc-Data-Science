# Legal Summarizer Algorithm (Reproducible Specification)

This document specifies the sentence-ranking summarizer implemented in `src/modeling/llms/summarizer.py` for legal judgment classification prompts.

It is designed to reduce prompt length while preserving decision-relevant legal reasoning.

## Objective

Select a compact subset of sentences from `texto_integral_sem_decisao` that maximizes classification-relevant signal for downstream LLM inference, with explicit safeguards against:

- anchorless drift
- single-category dominance
- token overgrowth

## Pipeline

1. Split document into sentence-like units with a conservative Portuguese legal splitter.

2. Encode each sentence with frozen LegalBERTimbau (`stjiris/bert-large-portuguese-cased-legal-mlm-nli-sts-v1`), mean pooling over token embeddings.

3. Build case-type query embedding (`dv` or `boc`) from 5 templates per case type and average them.

4. Compute document centroid embedding (mean of sentence embeddings).

5. Compute linguistic anchor score per sentence from weighted anchor groups:
   - facts
   - norms
   - proof
   - reasoning
   - jurisprudence
   - conflict

6. Compute final sentence score:

   `score = w_query * cos(sent, query) + w_centroid * cos(sent, centroid) + w_anchor * anchor_score`

7. Apply constrained selection policy:
   - token budget first (`max_summary_tokens`)
   - top-k second (`top_k`)
   - anchor-first pass
   - anchorless fallback only (bounded)
   - optional per-category caps

8. Reconstruct selected sentences in original document order for coherence.

## Default/Current Experimental Configuration

Weights used in current baseline experiments:

- `anchor_weight = 0.45`
- `query_weight = 0.40`
- `centroid_weight = 0.15`

Guardrails used in A/B validation:

- `top_k = 20`
- `max_summary_tokens = 5000`
- `min_anchor_ratio = 0.70`
- `max_anchorless_sentences = 6`
- `max_per_anchor_group = {'facts': 8, 'norms': 8, 'proof': 8, 'reasoning': 8, 'jurisprudence': 8, 'conflict': 8}`

## Anchor Families

The implementation includes:

- `FACTUAL_ANCHORS`
- `LEGAL_NORMS_ANCHORS`
- `PROOF_EVAL_ANCHORS`
- `REASONING_ANCHORS`
- `JURISPRUDENCE_ANCHORS`
- `CONFLICT_ANCHORS`

Notable design choice: conflict/rejection cues are explicitly modeled (for example, "não procede", "não se verifica", "carece de fundamento"), as these are low-frequency but high-signal polarity markers.

## Reproducibility

### Run top-20 qualitative outputs on sample cases

From repository root:

```bash
python -c "from pathlib import Path; import pandas as pd; from src.modeling.llms.summarizer import summarize_from_sentence_embeddings; base=Path('data/processed_data/llm_sentence_embeddings'); df=pd.read_parquet(base/'dv_gold_test.parquet'); r=df.iloc[0]; out=summarize_from_sentence_embeddings(sentences=list(r['sentences']), sentence_embeddings=r['sentence_embeddings'], case_type='dv', top_k=20, query_weight=0.40, centroid_weight=0.15, anchor_weight=0.45, max_summary_tokens=5000, min_anchor_ratio=0.70, max_anchorless_sentences=6, max_per_anchor_group={'facts':8,'norms':8,'proof':8,'reasoning':8,'jurisprudence':8,'conflict':8}); print(len(out.top_indices), len(out.summary_text))"
```

### Existing experiment logs

- Case-level A/B log: `.Agent/tasks/ab_guardrails_top20_log.txt`
- 50-case aggregate mix: `.Agent/tasks/ab_guardrails_mix_50cases.txt`

## Example Output Snippets

Example (DV case `299/23.4SXLSB.L1-5`, selected sentences include):

- "Da sentença recorrida consta a seguinte matéria de facto provada:"
- "Não obstante, é oficioso, pelo tribunal de recurso, o conhecimento dos vícios..."
- "Cumpre decidir."

Example (BOC case `541/21.6T8MGR.L1.S1`, selected sentences include):

- "Assim sendo, mostra-se necessária uma intervenção do Supremo Tribunal de Justiça..."
- "Não estando demonstrado que a transportadora agiu com dolo..."
- "Resulta do exposto, que o Acórdão reclamado optou por seguir a orientação jurisprudencial..."

## Aggregate Distribution (50-case run, guarded top-20)

DV (`n=50`, total selected `943`, mean selected `18.86`):

- facts: `27.25%`
- norms: `15.59%`
- proof: `16.76%`
- reasoning: `11.88%`
- jurisprudence: `22.80%`
- conflict: `4.24%`
- anchorless: `12.62%`

BOC (`n=50`, total selected `944`, mean selected `18.88`):

- facts: `30.19%`
- norms: `18.75%`
- proof: `13.24%`
- reasoning: `11.76%`
- jurisprudence: `18.33%`
- conflict: `3.60%`
- anchorless: `14.30%`

## Token Budget Findings

With `top_k=50` and the same guardrails, `max_summary_tokens=5000` was not binding on the tested 100 gold cases:

- DV: budget binding in `0/50` cases
- BOC: budget binding in `0/50` cases
- selected-token estimate remained below 5000 in all tested cases

This indicates current limits are mainly governed by ranking quality and guardrail constraints, not by token budget saturation.
