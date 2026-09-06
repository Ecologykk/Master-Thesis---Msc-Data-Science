# 06 — Evaluation Pipeline

## Purpose
Compute the metrics used to compare models: Macro-F1, Matthews Correlation Coefficient (MCC),
True Skill Statistic (TSS, DV only), and per-class precision/recall/F1, each with 95% stratified
bootstrap confidence intervals, plus forest plots visualising them. This is a shared library
([`classification.py`](../src/evaluation/classification.py)) consumed by both the BERT pipeline
([04-bert.md](04-bert.md)) and the LLM pipeline ([05-llm.md](05-llm.md)) — every `predict.py` run
in either pipeline already calls into it automatically. This doc covers that library plus the two
standalone scripts that consolidate results **across** already-completed runs.

## Prerequisites
- At least one prediction parquet already produced by [04-bert.md](04-bert.md)'s `predict.py`
  (`data/models/dl/{case_type}/predictions/*.parquet`) or [05-llm.md](05-llm.md)'s `predict.py`
  (`data/models/llms/{model_short}/*.parquet`) — both schemas include `y_true`, `y_pred`,
  `n_processo`, `run_name`, `case_type`.
- No GPU or network required — this stage is pure numpy/pandas bootstrap resampling.

## Exact commands

### Library usage (no CLI) — already wired into both `predict.py` scripts
[`classification.py`](../src/evaluation/classification.py) exposes
`run_classification_evaluation(y_true, y_pred, case_type, n_bootstraps=1000, split_label=...)`
and `plot_forest(results, case_type, run_name=None, output_dir=None, split_label=...)`. Both
`dl/predict.py`'s `main()`/`main_bert()` and `llms/predict.py`'s `main()` call these directly —
there is no separate command to run for a single model's evaluation; it happens as part of the
`predict.py` runs documented in 04 and 05. Running the module directly only exercises a synthetic
demo, not real data:
```bash
python src/evaluation/classification.py
```

### Consolidate CIs across multiple already-completed runs
[`compute_all_cis.py`](../src/evaluation/compute_all_cis.py) has **no CLI flags** — it is a plain
script with a hardcoded `PREDICTION_FILES` dict pointing at specific run names used in the thesis
(`bert_v3`, DeepSeek zero-shot/few-shot with a `smart9k` suffix). If your own runs used different
`--run_name`/`--model` values, edit `PREDICTION_FILES` in the script before running. Its paths are
`__file__`-anchored to the repo root, so it can be run from any working directory:
```bash
python src/evaluation/compute_all_cis.py
```

### Generate forest plots from the consolidated results
[`generate_forest_plots_from_results.py`](../src/evaluation/generate_forest_plots_from_results.py)
also has no CLI flags, but is likewise runnable from any working directory, after
`compute_all_cis.py`:
```bash
python src/evaluation/generate_forest_plots_from_results.py
```

## Inputs
- `compute_all_cis.py`: the six parquet paths hardcoded in `PREDICTION_FILES` (edit as needed) —
  `data/models/dl/{dv,boc}/predictions/bert_v3_{dv,boc}_predictions.parquet` and
  `data/models/llms/deepseek_r1_8b/{zero_shot,few_shot}_{dv,boc}_predictions*.parquet`.
- `generate_forest_plots_from_results.py`: `data/results/evaluation_results_with_cis.parquet`
  (output of `compute_all_cis.py`).

## Outputs
- `compute_all_cis.py` → `data/results/evaluation_results_with_cis.parquet` — one row per
  model/case-type/run with point estimates and 95% CI bounds for every metric.
- `generate_forest_plots_from_results.py` → `data/results/forest_plots/{prefix}{case_type}_forest_plot.png`
  per row of the consolidated results.
- `data/results/predictions/` — plain copies of the 6 prediction parquets named in
  `PREDICTION_FILES` (the exact ones consolidated above), committed to the repo so a reader can
  inspect raw predictions without the full Zenodo model/data archive. The pipeline itself always
  reads from the original `data/models/dl/...` / `data/models/llms/...` locations; this folder is
  a read-only convenience copy, not a path either script depends on.

## Approximate runtime
Seconds. Bootstrap resampling (`n_bootstraps=1000` by default) over a 50-case gold test set is
cheap pure-numpy work; the whole consolidation over six prediction files completes in well under a
minute.

## Expected result
`compute_all_cis.py` prints, per file, the loaded shape and accuracy, then a preview of all
point-estimate columns in the consolidated DataFrame. `generate_forest_plots_from_results.py`
prints `[OK] {model_name} {case_type.upper()} ({run_name})` per plot saved. **Headline numbers
these scripts should reproduce** (as reported in the thesis): DeepSeek-R1 8B zero-shot MCC 0.592
(DV) / 0.468 (BoC); the fine-tuned LegalBERTimbau baseline (`bert_v3`) collapsed to predicting the
majority class, i.e. MCC near zero and near-zero recall on the minority class(es) — see
[04-bert.md](04-bert.md)'s and [05-llm.md](05-llm.md)'s "Expected result" sections for the
per-pipeline detail.

See [09-reproducibility-notes.md](09-reproducibility-notes.md) for hardware-determinism caveats
affecting the underlying predictions this stage consolidates.

### Quick summary table
[`show_results.py`](../show_results.py) (repo root) prints a compact per-model summary table
plus per-class precision/recall/F1 with CIs, straight from
`data/results/evaluation_results_with_cis.parquet` — useful for a quick look without re-plotting:
```bash
python show_results.py
```
