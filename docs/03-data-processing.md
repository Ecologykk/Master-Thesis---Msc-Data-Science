# 03 — Data Processing

## Purpose
Turn the raw scraped JSON into the labelled datasets used for modelling: clean the corpus,
recover missing decisions where possible, apply the binary (DV) / ternary (BoC) label rules,
and split off a fixed 50-case gold test set per case type. This is **the reproducibility gate**
for the whole project — the exact counts below are what the thesis reports, and regenerating
them from raw data is the strongest available check that the rest of the pipeline is intact.

Ported from `raw_processed_data_git/Msc-Data-Science-Dissertation-Data/scripts/`, corrected
against `src/data_processing_exploration/processing_and_eda.ipynb` (the original source of
truth). See the module docstrings in [`src/data_processing/`](../src/data_processing/) for the
exact corrections made and why.

## Prerequisites
- [01-setup.md](01-setup.md) completed.
- Raw data present under `data/raw_data/*.json` — either from your own scrape
  ([02-scraping.md](02-scraping.md)) or the archived corpus from Zenodo (see
  [09-reproducibility-notes.md](09-reproducibility-notes.md)). Only re-scraping will **not**
  reproduce the thesis counts (see the snapshot warning in 02-scraping.md); the Zenodo corpus
  will.

## Exact commands
```bash
python -m src.data_processing.run --case-type dv  --stage all
python -m src.data_processing.run --case-type boc --stage all
```

`--stage` can also be `aggregate`, `clean`, `label`, or `split` to run just one step (each stage
recomputes everything before it in memory — aggregation and cleaning are cheap relative to the
BERT/LLM stages downstream, so this is not wasted work). `--case-type` is `dv` or `boc`.

## Inputs
`data/raw_data/*.json` — filtered by a filename keyword per case type (`violencia_domestica` for
dv, `incumprimento_contratos` for boc; see `aggregate.CASE_TYPE_FILENAME_KEYWORDS`), so any
number of per-court raw files works without editing code.

## Outputs
- `data/processed_data/classification/{binary,ternary}/df_acordaos_{dv,boc}_classification_{binary,ternary}.csv`
  — modelling-ready view (case text + label only).
- `data/processed_data/eda/{binary,ternary}/df_acordaos_{dv,boc}_eda_{binary,ternary}.csv` —
  same rows, plus identifying/provenance columns (`url`, `tribunal`, `n_processo`,
  `juiz_relator`, `data_acordao`) for descriptive analysis. Consumed by
  [`eda.py`](../src/eda/eda.py).
- `data/processed_data/splits/gold_test/{dv,boc}_{gold_test,train_before_cutoff}.csv` —
  intermediate train/test split by publication-date cutoff, before full-text is re-attached.
- `data/processed_data/gold_test/{dv,boc}_gold_test_full.{csv,json}` — the final 50-case gold
  test set per case type, with the full case text (including the decision) re-attached for
  reference. These are the files committed to the repository (see
  [08-judge-study.md](08-judge-study.md) — this is what judges annotated) and the ones a reader
  can check reported numbers against without downloading the full corpus.

## Approximate runtime
A few seconds to low tens of seconds per case type on a laptop — this stage is pure
pandas/regex, no GPU or network involved.

## Expected result
```
DV labelled cases: 1126 (thesis-reported count: 1126)
DV test size: 50, train size: 1064
```
```
BOC labelled cases: 1802 (thesis-reported count: 1802)
BOC test size: 50, train size: 1752
```
**1,126 DV cases and 1,802 BoC cases after cleaning and labelling** — these are the numbers the
thesis reports. If your counts differ, **stop and treat it as a genuine reproducibility
finding** (e.g. a different raw corpus than the archived one, or a code change upstream) —
do not adjust the pipeline to force a match. `tests/test_reproduction.py` makes this check
permanent and machine-checked; see [`../tests/README` (via pytest)](../tests/test_reproduction.py).

To reset and re-run from scratch, just re-run the command — it overwrites its own outputs.
Raw data under `data/raw_data/` is never modified by this stage.

## Known limitation of the label heuristics
Both label functions ([`src/data_processing/labels.py`](../src/data_processing/labels.py))
check negation/unfavourable language *before* partial language. This means phrases like
"parcialmente improcedente" ("partially unsuccessful") are classified as MANTIDA/DESFAVORÁVEL
rather than as a partial outcome — documented in the thesis and pinned down by
`tests/unit/test_labels.py`. This is intentional, disclosed behaviour, not a bug to silently
fix — see the module docstring in `labels.py` for the full explanation.
