# Predicting Portuguese Court Decisions from Case Text

Code for a Master's thesis in Data Science (Universidade de Lisboa — Faculdade de Ciências,
LASIGE) comparing a fine-tuned BERT classifier against LLM prompting (zero-shot, few-shot,
chain-of-thought) on predicting appellate outcomes from Portuguese court judgment text, across
two case types with different label schemas:

- **DV** (Domestic Violence, appeals) — binary: decision **Kept** or **Altered**.
- **BoC** (Breach of Contract) — ternary: outcome **Favourable**, **Unfavourable**, or
  **Partial**.

## Headline results
- **DeepSeek zero-shot LLM prompting**: MCC 0.592 on DV, 0.468 on BoC — the strongest approach
  tested.
- **Fine-tuned LegalBERTimbau baseline**: collapsed to predicting the majority class on both
  tasks — it did not learn a useful decision boundary on this dataset size (~1,100–1,800
  labelled cases per task). See [`docs/04-bert.md`](docs/04-bert.md) and
  [`docs/06-evaluation.md`](docs/06-evaluation.md) for the full numbers and discussion.
- Full context, limitations, and discussion: [`manuscript/thesis.pdf`](manuscript/thesis.pdf).

## Pipeline
```
  scrape (dgsi.pt, jurisprudencia.csm.org.pt)
          |
          v
  aggregate -> clean -> label -> split          [src/data_processing/]
          |
          |-- 1,126 DV cases, 1,802 BoC cases, 50-case gold test set each
          v
  +-------------------------+     +---------------------------------+
  | fine-tune LegalBERTimbau|     | zero/few-shot/CoT LLM prompting  |
  | src/modeling/dl/        |     | src/modeling/llms/ (via Ollama)  |
  +-------------------------+     +---------------------------------+
          |                                     |
          v                                     v
  bootstrap evaluation + forest plots    [src/evaluation/]
          |
          v
  SHAP explainability                    [src/evaluation/explanation.py]

  (separately) judge annotation study, for label-quality validation
               [judge_labeling_app/]
```

## Start here
Read the docs in this order — each states purpose, prerequisites, exact commands, inputs,
outputs, runtime, and expected result, so no step assumes context you don't already have:

1. [`docs/01-setup.md`](docs/01-setup.md) — environment and dependencies.
2. [`docs/02-scraping.md`](docs/02-scraping.md) — collect raw case documents (or skip this and
   use the archived corpus — see `docs/09`).
3. [`docs/03-data-processing.md`](docs/03-data-processing.md) — clean, label, split. **This is
   the reproducibility gate**: regenerating it from raw data must produce exactly 1,126 DV and
   1,802 BoC cases.
4. [`docs/04-bert.md`](docs/04-bert.md) — fine-tune and evaluate the BERT baseline.
5. [`docs/05-llm.md`](docs/05-llm.md) — zero/few-shot/chain-of-thought LLM inference via Ollama.
6. [`docs/06-evaluation.md`](docs/06-evaluation.md) — bootstrap metrics, confidence intervals,
   forest plots.
7. [`docs/07-explainability.md`](docs/07-explainability.md) — SHAP-based explanation of model
   predictions.
8. [`docs/08-judge-study.md`](docs/08-judge-study.md) — the manual judge-annotation app used to
   validate the label heuristics against real legal expertise.
9. [`docs/09-reproducibility-notes.md`](docs/09-reproducibility-notes.md) — read this if a
   number doesn't match: corpus snapshot date, hardware nondeterminism, and the archived-corpus
   pointer, all in one place.

**New to this codebase?** Point an AI coding assistant (e.g. Claude Code) at this `docs/` folder
and ask it to explain a pipeline stage or trace a specific function before you read the source —
each doc already states purpose, prerequisites, exact commands, and expected output, so the
assistant has real grounding instead of guessing from code alone.

## Repository layout
```
Master-Thesis---Msc-Data-Science/
├── docs/                     # start here — see above
├── manuscript/thesis.pdf     # final compiled thesis
├── src/
│   ├── scrapers/             # dgsi.pt / CSM scrapers
│   ├── data_processing/      # aggregate -> clean -> label -> split (the reproducibility gate)
│   ├── data_processing_exploration/  # the original exploratory notebook + one-off fix scripts
│   ├── eda/                  # exploratory data analysis (plots, UMAP)
│   ├── modeling/dl/          # LegalBERTimbau fine-tuning + inference
│   ├── modeling/llms/        # LLM prompting + inference via Ollama
│   └── evaluation/           # bootstrap evaluation, forest plots, SHAP explainability
├── tests/                    # unit, smoke, and the reproduction gate (pytest)
└── judge_labeling_app/       # Streamlit app used for the judge-validation study
```

## Data and reproducibility
- Small artifacts needed to check reported numbers (gold test sets, prediction outputs,
  evaluation results with confidence intervals, forest plots) are committed directly to this
  repository under `data/`.
- The bulk corpus (raw scrape, full processed sets, model checkpoints) is archived on Zenodo:
  [10.5281/zenodo.22116675](https://doi.org/10.5281/zenodo.22116675) — see
  [`docs/09-reproducibility-notes.md`](docs/09-reproducibility-notes.md) for what's in each
  archive and where to unzip it.
- Cases from both source databases are pseudonymised at source by the publishing courts; no
  further anonymisation was performed or was necessary.

### If you plan to run the scraper
`src/scrapers/` reads from **live, government-run websites** (dgsi.pt and
jurisprudencia.csm.org.pt). Two things to know before you point it at them:

- **Be polite.** The scraper waits **1 second between document requests**
  (`asyncio.sleep(1)` in `scrape_documents`). Please do not remove or shorten that delay, and
  do not run the scraper at scale or in parallel. These are public-service sites for the
  Portuguese courts, not an API.
- **You probably don't need to scrape at all.** The corpus used in this dissertation is a
  **November 2025 snapshot**. Because both databases are updated continuously, re-scraping
  today returns a larger, different set of cases and will **not** reproduce the counts or
  results reported in the thesis. To reproduce published numbers, download the archived corpus
  from the Zenodo record above instead — see
  [`docs/09-reproducibility-notes.md`](docs/09-reproducibility-notes.md).
- The judge-annotation study ([`docs/08-judge-study.md`](docs/08-judge-study.md)) involved
  identifiable legal professionals as study participants. Their responses and identities are
  **not** included in this repository — a separate matter from the case texts themselves. See
  that doc and [`judge_labeling_app/README.md`](judge_labeling_app/README.md) for exactly what
  was collected and what was withheld.

## Licence
[Apache License 2.0](LICENSE).

## Author
**Helton Mendonça** — MSc Data Science, Universidade de Lisboa (Faculdade de Ciências / LASIGE).
