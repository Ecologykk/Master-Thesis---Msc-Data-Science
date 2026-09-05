# 09 — Reproducibility Notes

This page consolidates every caveat that affects whether re-running this pipeline reproduces the
numbers in the thesis exactly. Read it before assuming a different result means something is
broken.

## Corpus provenance
The corpus was scraped from two Portuguese court-decision databases:
[dgsi.pt](https://www.dgsi.pt/) (Direção-Geral da Política de Justiça) and
[jurisprudencia.csm.org.pt](https://jurisprudencia.csm.org.pt/) (Conselho Superior da
Magistratura). Both publish court decisions that are already pseudonymised at source by the
publishing courts — no further anonymisation was performed or was necessary for this project.

## Corpus snapshot: November 2025
The corpus used in this dissertation was scraped in **November 2025** and therefore contains
only cases published up to that date. Both source databases are continuously updated, so:
- **Re-scraping today will not reproduce the thesis numbers.** You will get a larger and
  different set of cases (see [02-scraping.md](02-scraping.md)).
- **To reproduce the published results**, use the archived corpus below instead of re-scraping.

## Archived corpus (Zenodo)
The raw and processed corpus, model checkpoints, and prediction outputs used in the thesis are
archived on Zenodo with a permanent DOI:
[10.5281/zenodo.22116675](https://doi.org/10.5281/zenodo.22116675).

The record contains three archives, each with an accompanying SHA-256 checksum file:
- `thesis-raw-data.zip` — raw scraped JSON (unzip into `data/raw_data/`).
- `thesis-processed-data.zip` — cleaned, labelled, and split datasets plus embeddings (unzip into
  `data/processed_data/`).
- `thesis-models.zip` — BERT checkpoints and LLM prediction outputs (unzip into `data/models/`).

The raw corpus is not included in this repository (`data/raw_data/` is gitignored) — only the
small artifacts needed to check reported numbers (gold test sets, the 6 official prediction
outputs, evaluation results with confidence intervals, forest plots) are committed directly.

Once downloaded, unzip each archive under the repo root so its contents land at the paths above —
that's what every doc in this series assumes.

## Hardware nondeterminism
Seeds are fixed and decoding is greedy (`temperature=0.0`, `top_k=1`, `seed=42`) throughout the
BERT and LLM pipelines, so a script is deterministic **on identical hardware**. Results may
still diverge slightly on different hardware: GPU floating-point reductions are non-associative,
and kernel selection, driver version, and tensor-core availability all change the order of
operations. Expect small differences in embeddings and occasional label flips on borderline
cases when re-running on hardware different from the original (an RTX 3050, 6 GB VRAM). This
does not indicate a bug. See the module docstrings on
[`legal_bertimbau_tokenization_embedding.py`](../src/modeling/dl/legal_bertimbau_tokenization_embedding.py),
[`extract_bert_embeddings.py`](../src/modeling/dl/extract_bert_embeddings.py),
[`llms/predict.py`](../src/modeling/llms/predict.py), and
[`llms/summarizer.py`](../src/modeling/llms/summarizer.py) for the same note in context.

## The one gate that must reproduce exactly
Everything upstream of modelling — aggregation, cleaning, and labelling
([03-data-processing.md](03-data-processing.md)) — is pure CPU-bound pandas/regex with no
randomness, so it should reproduce **exactly**, on any hardware, given the same raw corpus:
**1,126 DV cases, 1,802 BoC cases**. `tests/test_reproduction.py` checks this automatically
(`pytest tests/test_reproduction.py -m slow`). If this specific check fails, that is a real
finding worth reporting, not something to route around.

Everything downstream of that (BERT fine-tuning, LLM inference, evaluation metrics) is subject
to the hardware nondeterminism above and should be expected to vary in the fourth-decimal-place
sense, not the "completely different result" sense.

## Known label-heuristic limitation
See [03-data-processing.md](03-data-processing.md#known-limitation-of-the-label-heuristics) —
phrases combining "parcial" and "improcedente" are classified as MANTIDA/DESFAVORÁVEL rather
than as a partial outcome, because the negation check runs before the partial check in both
label functions. This is disclosed, intentional pipeline behaviour discussed in the thesis, and
is pinned down by `tests/unit/test_labels.py` so it cannot silently drift.
