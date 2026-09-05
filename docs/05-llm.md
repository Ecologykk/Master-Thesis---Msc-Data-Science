# 05 — LLM Pipeline

## Purpose
Run zero-shot, few-shot, chain-of-thought (CoT), and few-shot-CoT prompting against local LLMs
served by [Ollama](https://ollama.ai), on the same two tasks and gold-test split as the BERT
pipeline ([04-bert.md](04-bert.md)), so the two are directly comparable. The pipeline has two
independent stages: [`summarizer.py`](../src/modeling/llms/summarizer.py) does "smart truncation"
— ranking and selecting the most relevant sentences of a long case so it fits the model's context
window without naive head/tail cutting — and
[`predict.py`](../src/modeling/llms/predict.py) does the actual prompting, JSON-label
validation/retry, and bootstrap evaluation.

## Prerequisites
- [01-setup.md](01-setup.md) completed, including the Ollama section: a running Ollama server
  reachable at `OLLAMA_BASE_URL` (defaults to `http://localhost:11434`; override via the
  `OLLAMA_BASE_URL` environment variable — see [`config.py`](../src/modeling/llms/config.py)),
  with at least one of the models in `OLLAMA_MODELS` pulled (`deepseek_r1_8b` →
  `deepseek-r1:8b`, `llama_3_8b` → `llama3.1:8b`, `ministral_3b` → `mistral:7b`).
- [03-data-processing.md](03-data-processing.md) completed —
  `data/processed_data/splits/gold_test/{dv,boc}_gold_test.csv` is loaded directly (raw text, no
  BERT embeddings needed for this pipeline).
- For `--stage few_shot` / `few_shot_cot`: either a precomputed JSON examples mapping
  (`--few_shot_examples_path`), or enriched parquets to build one at runtime (see the "Few-shot
  retrieval" note below).

## Exact commands

### Step 1 (optional) — Smart-truncation sentence embeddings
[`summarizer.py`](../src/modeling/llms/summarizer.py) has an argparse CLI with two modes:
```bash
# Batch mode: expects dv_gold_test.csv, boc_gold_test.csv, dv_train_before_cutoff.csv,
# boc_train_before_cutoff.csv under --input_root (matches data/processed_data/splits/gold_test/)
python src/modeling/llms/summarizer.py --run_train_batch \
  --input_root data/processed_data/splits/gold_test \
  --output_root data/processed_data/llm_summaries

# Single-file mode
python src/modeling/llms/summarizer.py \
  --input_path data/processed_data/splits/gold_test/dv_gold_test.csv \
  --output_path data/processed_data/llm_summaries/dv_gold_test.parquet \
  --case_type dv
```
The CLI's output parquet already includes a `summary_text` column (selected sentences joined with
a single space, alongside the raw `selected_sentences` list and per-sentence scores) — exactly the
`n_processo` + `summary_text` schema `predict.py`'s `--smart_truncated_path` flag (Step 2) expects,
so the two steps chain directly with no bridging needed.

### Step 2 — Inference + gold-test evaluation
[`predict.py`](../src/modeling/llms/predict.py):
```bash
python src/modeling/llms/predict.py \
  --case_type dv --model deepseek_r1_8b --stage zero_shot --run_name deepseek_zs_v1

python src/modeling/llms/predict.py \
  --case_type dv --model deepseek_r1_8b --stage cot --run_name deepseek_cot_v1

python src/modeling/llms/predict.py \
  --case_type dv --model deepseek_r1_8b --stage few_shot --run_name deepseek_fs_v1 \
  --few_shot_query_path data/processed_data/llm_summaries/dv_gold_test_enriched.parquet \
  --few_shot_candidates_path data/processed_data/llm_summaries/dv_train_before_cutoff_enriched.parquet
```
Other useful flags: `--max_cases N` (smoke-test cap), `--smart_truncated_path <parquet>` (use
Step-1 summaries instead of full text), `--num_ctx` (Ollama context window, default 32768),
`--max_prompt_tokens` (default 28000; auto head+tail-truncates the case text if exceeded),
`--request_timeout` (default 1200s), `--no-trace-log` (disable per-case reasoning trace logging),
`--seed`/`--temperature` (default 42 / 0.0 — near-greedy decoding).

If `--few_shot_query_path`/`--few_shot_candidates_path` are omitted, they default to
`data/processed_data/llm_summaries/{case_type}_{split}_enriched.parquet` (see
`_default_enriched_path` in `predict.py`) — exactly what Step 1's `--run_train_batch` writes under
`--output_root data/processed_data/llm_summaries`, so no renaming is needed when both steps use
their defaults.

## Inputs
- `data/processed_data/splits/gold_test/{dv,boc}_gold_test.csv` (`texto_integral_sem_decisao`,
  label column, `n_processo`) — loaded via `features.load_split_text_data`.
- Optional: a `--smart_truncated_path` parquet (`n_processo`, `summary_text`).
- Optional (few-shot stages): a few-shot examples JSON (`--few_shot_examples_path`) or the
  query/candidate enriched parquets described above.

## Outputs
- `data/models/llms/{model_short}/{stage}_{case_type}_predictions.parquet` — one-hot
  `proba_<label>` columns plus `y_true`/`y_pred`/`n_processo`/`run_name`.
- `data/models/llms/{model_short}/plots/{run_name}_{case_type}_forest_plot.png`.
- With `--trace_log` (default on): per-case reasoning traces at
  `data/models/llms/thinking_logs/{model_short}/{stage}_{case_type}[_{run_name}].txt`, including
  every retry attempt's `<think>` block and final JSON output.

## Approximate runtime
The gold test split is 50 cases per case type. Each case is one (or more, on JSON-validation
retry, up to `MAX_RETRIES=3`) Ollama chat call against an 8B reasoning model generating a
`<think>` chain-of-thought before its final JSON label — expect roughly **~60–90 seconds per case**
on consumer GPU hardware, i.e. **~50–75 minutes for a full 50-case gold-test run** per
(case_type, model, stage) combination; CPU-only Ollama backends will be substantially slower.
Few-shot stages add the retrieval-embedding cost (once, amortised) plus longer prompts.

## Expected result
`predict.py` prints the same bootstrap-CI classification report as the BERT pipeline (Macro-F1,
MCC, TSS for DV) and saves a forest plot. **Headline result reported in the thesis**: DeepSeek-R1
8B zero-shot achieved **MCC 0.592 on DV** and **MCC 0.468 on BoC** — clearly better than the
fine-tuned LegalBERTimbau baseline, which collapsed to predicting the majority class (see
[04-bert.md](04-bert.md) and [06-evaluation.md](06-evaluation.md) for the consolidated
comparison).

See [09-reproducibility-notes.md](09-reproducibility-notes.md) for hardware-determinism caveats —
Ollama's `seed`/`temperature=0`/`top_k=1` options make generation *near*-greedy, not bit-exact
across different GPUs/drivers/Ollama versions.
