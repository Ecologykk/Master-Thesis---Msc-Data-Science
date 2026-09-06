# 04 — BERT Pipeline

## Purpose
Train and evaluate the fine-tuned-BERT classifier baseline: Legal BERTimbau
(`stjiris/bert-large-portuguese-cased-legal-mlm-nli-sts-v1`), used two ways — (a) as a **frozen
encoder** feeding a small MLP head trained on PCA-reduced embeddings ("mlp" mode, CPU-friendly),
and (b) **fully fine-tuned** end-to-end with the top 4 of 24 layers unfrozen ("bert" mode, GPU
required). Both modes classify the same two tasks (DV binary, BoC ternary) and share the same
gold-test evaluation protocol as the LLM pipeline (see [05-llm.md](05-llm.md)), so the two are
directly comparable.

## Prerequisites
- [01-setup.md](01-setup.md) completed, including the optional GPU section if you intend to run
  "bert" mode (fine-tuning requires CUDA — it raises `RuntimeError` on CPU; the frozen "mlp" mode
  works on CPU). Fine-tuning was developed against ~6 GB VRAM.
- [03-data-processing.md](03-data-processing.md) completed — needs
  `data/processed_data/eda/{binary,ternary}/df_acordaos_{dv,boc}_eda_{binary,ternary}.csv` (for
  the embedding-extraction step below) and
  `data/processed_data/splits/gold_test/{dv,boc}_{train_before_cutoff,gold_test}.csv` (for
  training/prediction in both modes).
- For "mlp" mode only: frozen BERT embeddings already extracted (see step 1 below) into
  `data/processed_data/bert_tokens_embedd/df_acordaos_{dv,boc}_eda_{binary,ternary}_legal_bert_embeddings.parquet`.
- All commands below assume the repo root as the working directory but this does not actually
  matter for path resolution — every path in `features.py`/`train.py`/`predict.py` is anchored to
  the repo root via `Path(__file__).resolve().parents[...]`, so they work from any cwd. The
  **exception** is step 1, `legal_bertimbau_tokenization_embedding.py` (see below).

## Exact commands

### Step 1 — Extract frozen BERT embeddings (once, before any training)
[`legal_bertimbau_tokenization_embedding.py`](../src/modeling/dl/legal_bertimbau_tokenization_embedding.py)
is a converted notebook with **no argparse CLI** — it is meant to be run as a plain script, and it
resolves its two input CSVs relative to the *current working directory*
(`ROOT_DIR = Path(".")`), not the repo root. This is an awkward reality of the code as written:
you must either `cd` into a directory containing both
`df_acordaos_dv_eda_binary.csv` and `df_acordaos_boc_eda_ternary.csv` directly (they live in
separate `eda/binary/` and `eda/ternary/` subfolders after step 03, so copy or symlink both into
one flat directory first), or edit `DATASET_PATHS` in the script.

```bash
mkdir -p /tmp/bert_embed_input
cp data/processed_data/eda/binary/df_acordaos_dv_eda_binary.csv /tmp/bert_embed_input/
cp data/processed_data/eda/ternary/df_acordaos_boc_eda_ternary.csv /tmp/bert_embed_input/
cd /tmp/bert_embed_input
python "$OLDPWD/src/modeling/dl/legal_bertimbau_tokenization_embedding.py"
```

This writes `bert_legal_embeddings/*.parquet` under the *current* directory — **not** where
`features.py` expects embeddings to live. Move the two `*_legal_bert_embeddings.parquet` files
into `data/processed_data/bert_tokens_embedd/` (creating it if needed) before running training:

```bash
mkdir -p "$OLDPWD/data/processed_data/bert_tokens_embedd"
cp bert_legal_embeddings/*_legal_bert_embeddings.parquet \
  "$OLDPWD/data/processed_data/bert_tokens_embedd/"
cd "$OLDPWD"
```

### Step 2 — Train
[`train.py`](../src/modeling/dl/train.py):
```bash
# Frozen embeddings + MLP head (CPU ok)
python src/modeling/dl/train.py --case_type dv  --mode mlp --run_name v1
python src/modeling/dl/train.py --case_type boc --mode mlp --run_name v1

# Full BERT fine-tuning (CUDA required)
python src/modeling/dl/train.py --case_type dv  --mode bert --run_name bert_v1
python src/modeling/dl/train.py --case_type boc --mode bert --run_name bert_v1

# Smoke test with synthetic data (no real data or GPU required)
python src/modeling/dl/train.py --smoke_test
```
Other flags: `--n_splits` (TimeSeriesSplit folds, mlp mode, default 5), `--output_dir` (default
`data/models/dl`), `--early_stop_on {ema_loss,loss}` (bert mode, default `ema_loss`).

### Step 3 — Predict / evaluate on the gold test set
[`predict.py`](../src/modeling/dl/predict.py):
```bash
python src/modeling/dl/predict.py --case_type dv  --mode mlp  --run_name v1
python src/modeling/dl/predict.py --case_type dv  --mode bert --run_name bert_v1

# Also score on train/val to diagnose under- vs over-fitting (bert mode only)
python src/modeling/dl/predict.py --case_type dv --mode bert --run_name bert_v1 --eval_train

# SHAP explanations after evaluation — see docs/07-explainability.md
python src/modeling/dl/predict.py --case_type dv --mode bert --run_name bert_v1 --shap --max_shap_docs 10

python src/modeling/dl/predict.py --smoke_test
```

### Step 4 (optional) — Extract fine-tuned embeddings for UMAP visualisation
[`extract_bert_embeddings.py`](../src/modeling/dl/extract_bert_embeddings.py) has a real CLI:
```bash
python src/modeling/dl/extract_bert_embeddings.py --run_name bert_v1 --case_type dv
```

## Inputs
- Step 1: `data/processed_data/eda/{binary,ternary}/df_acordaos_{dv,boc}_eda_{binary,ternary}.csv`.
- Steps 2–3 (mlp mode): `data/processed_data/bert_tokens_embedd/*_legal_bert_embeddings.parquet`
  joined against `data/processed_data/splits/gold_test/{dv,boc}_{train_before_cutoff,gold_test}.csv`
  on `n_processo`.
- Steps 2–3 (bert mode): raw text directly from the same split CSVs (`texto_integral_sem_decisao`
  column) — the parquet embeddings are not needed for bert mode.

## Outputs
- Step 2 (mlp): `data/models/dl/{case_type}/{run_name}_feature_pipeline.joblib`,
  `{run_name}_classifier.pt`.
- Step 2 (bert): `data/models/dl/{case_type}/{run_name}_{case_type}_bert_classifier.pt`, plus a
  loss-curve PNG under `data/models/dl/{case_type}/plots/`.
- Step 3: predictions parquet under `data/models/dl/{case_type}/predictions/`, a forest plot
  (bootstrap-CI metrics) under `data/models/dl/{case_type}/plots/`, and (with `--shap`) SHAP
  outputs under `shap_explanations/{domestic_violence,breach_of_contract}/{run_name}/`.

## Approximate runtime
- Step 1: tens of minutes on GPU for the full corpus (1,126 DV + 1,802 BoC documents, each split
  into overlapping 512-token/128-stride windows through BERT-large); much longer on CPU.
- Step 2 (mlp): a few minutes — a small MLP trained for up to 100 epochs on already-extracted,
  PCA-reduced embeddings.
- Step 2 (bert): up to `BERT_EPOCHS=20` epochs with early stopping (patience 7), one optimizer
  step per document, over roughly 900–1,500 training documents per case type — expect on the
  order of an hour or more on a 6 GB GPU, substantially longer on CPU (and CPU is not actually
  supported — `RuntimeError` is raised).
- Step 3: seconds (mlp) to low minutes (bert, sliding-window inference over 50 gold-test
  documents) without `--shap`; with `--shap`, add ~2–3 minutes per document (`--max_shap_docs 10`
  default), see [07-explainability.md](07-explainability.md).

## Expected result
`predict.py` prints a formatted report (from `run_classification_evaluation`) with Macro-F1, MCC,
and (DV only) TSS point estimates plus 95% bootstrap CIs, per-class precision/recall/F1, and saves
a forest plot. **Headline finding reported in the thesis**: the fine-tuned LegalBERTimbau baseline
(bert mode) collapsed to predicting the majority class — i.e. it did not learn a useful decision
boundary, unlike the DeepSeek zero-shot LLM baseline (MCC 0.592 on DV, 0.468 on BoC — see
[05-llm.md](05-llm.md) and [06-evaluation.md](06-evaluation.md)). If you reproduce training and
get a materially different (non-collapsed) result, treat that as a genuine finding worth
investigating, not a bug to silence.

See [09-reproducibility-notes.md](09-reproducibility-notes.md) for hardware-determinism caveats —
embeddings and fine-tuning results can diverge slightly across GPUs/drivers even with fixed seeds.
