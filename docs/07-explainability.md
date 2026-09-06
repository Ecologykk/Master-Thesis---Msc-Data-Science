# 07 — Explainability Pipeline (SHAP)

## Purpose
Generate sentence-level SHAP attributions explaining individual predictions, for judge review and
qualitative analysis: which sentences of a case pushed the model toward (red) or away from (blue)
its predicted class. Implemented in
[`explanation.py`](../src/evaluation/explanation.py), which is model-agnostic — it works with
either the BERT pipeline's `predict_fn` ([04-bert.md](04-bert.md),
`make_bert_predict_fn`/`predict_proba`) or the LLM pipeline's ([05-llm.md](05-llm.md),
`make_llm_predict_fn`). SHAP's `Text` masker splits input on the same sentence-boundary regex used
by the smart-truncation summarizer, so masking units line up with how the rest of the pipeline
already treats "a unit of legal reasoning."

## Prerequisites
- The `shap` Python package (installed via [01-setup.md](01-setup.md)'s `requirements.txt`).
- A working `predict_fn` — either:
  - **BERT**: a trained checkpoint from [04-bert.md](04-bert.md) (`train.py --mode bert`, saved
    via `save_bert_model`), loaded with `predict.load_bert_model`; or a frozen-embeddings
    MLP + BERT encoder wired through `predict.make_bert_predict_fn`.
  - **LLM**: a reachable Ollama server ([05-llm.md](05-llm.md)'s prerequisites) plus your own
    `Callable[[str], str]` wrapper around `OllamaClient`/prompt building, passed into
    `explanation.make_llm_predict_fn`.
- Gold-test texts, same source as the other two pipelines:
  `data/processed_data/splits/gold_test/{dv,boc}_gold_test.csv`.

## Exact commands

### Path A — BERT SHAP, wired CLI
Via [`predict.py`](../src/modeling/dl/predict.py) (`--mode bert` only):
```bash
python src/modeling/dl/predict.py --case_type dv --mode bert --run_name bert_v1 \
  --shap --max_shap_docs 10

# Explain specific case IDs instead of the first N (outputs go to a 5_gold_set/ subfolder,
# used for cases also annotated by judges — see docs/08-judge-study.md)
python src/modeling/dl/predict.py --case_type dv --mode bert --run_name bert_v1 \
  --shap --shap_cases "5052/21.7JAPRT-A.P1" "340/21.5TXLSB-E.L1-9"
```

### Path B — LLM SHAP, wired CLI
[`explanation.py`](../src/evaluation/explanation.py) has an argparse CLI for LLM zero-shot SHAP
against a live Ollama server — it reuses the exact same prompt-building
(`prompts.build_zero_shot_prompt`) and deterministic decoding options `llms/predict.py` uses for
its `--stage zero_shot` runs, so attributions stay consistent with how the model is evaluated
elsewhere:
```bash
# First N gold-test documents (default N=3 — see runtime note below)
python src/evaluation/explanation.py --case_type dv --model deepseek_r1_8b --max_docs 3

# Explain specific case IDs instead (outputs still land under the normal model_name folder)
python src/evaluation/explanation.py --case_type dv --model deepseek_r1_8b \
  --shap_cases "5052/21.7JAPRT-A.P1" "340/21.5TXLSB-E.L1-9"

# Synthetic smoke test (no Ollama connection needed) — unchanged, still the default sanity check
python src/evaluation/explanation.py --smoke_test
```
Other useful flags: `--model {deepseek_r1_8b,llama_3_8b,ministral_3b}` (see
`llms.config.OLLAMA_MODELS`), `--split {gold_test,train}`, `--max_evals` (default 50 — see
`llms.config.SHAP_MAX_EVALS_DEFAULT`), `--output_dir` (default `shap_explanations/`),
`--seed`/`--temperature` (default 42 / 0.0, matching `llms/predict.py`), `--request_timeout`
(default 120s).

For BERT's frozen-embeddings + MLP variant, or any other custom `predict_fn` (not the fine-tuned
checkpoint Path A covers), call the library functions directly:
```python
import sys
from pathlib import Path
sys.path.insert(0, "src/evaluation")
sys.path.insert(0, "src/modeling/dl")  # for features.load_split_text_data

from explanation import run_shap_explanation_pipeline, make_llm_predict_fn, LABEL_NAMES_DV
from features import load_split_text_data

texts, y_raw, _, n_processo = load_split_text_data("dv", split="gold_test")

def ollama_label_fn(text: str) -> str:
    # Your own wrapper: build a prompt, call OllamaClient.chat(),
    # parse predicted_label out of the JSON response.
    ...

predict_fn = make_llm_predict_fn(ollama_label_fn, list(LABEL_NAMES_DV.values()))
run_shap_explanation_pipeline(
    texts=texts[:10],
    case_ids=n_processo[:10],
    predict_fn=predict_fn,
    output_names=list(LABEL_NAMES_DV.values()),
    case_type="dv",
    model_name="deepseek_r1_8b_zero_shot",
    output_base_dir=Path("shap_explanations"),
    max_evals=50,
)
```

## Inputs
- Raw case text, sentence-split internally by SHAP's `Text` masker (same regex as
  `summarizer.split_sentences_pt_legal`).
- A trained model/checkpoint (BERT) or a working Ollama connection (LLM).

## Outputs
Under `shap_explanations/{domestic_violence,breach_of_contract}/[5_gold_set/]{model_name}/`:
- `{case_id}.html` — self-contained, colour-highlighted judge-annotation card (UTF-8, no JS).
- `shap_top_terms_{case_id}.png` — top-10 positive/negative sentence bar chart (300 DPI); skipped
  with a warning if every SHAP value is exactly zero (e.g. an overconfident LLM).
- `explanations.json` — all cases in the batch, nested format.
- `explanations.parquet` — long format, one row per (case, sentence), for pandas analysis.

## Approximate runtime
- **BERT** (`dl/predict.py --shap`): the code's own print statement estimates
  `n * 2`–`n * 3` minutes for `n` documents on GPU (its internal SHAP budget is 200
  `predict_fn` evaluations per document). With the default `--max_shap_docs 10`, expect roughly
  **20–30 minutes**.
- **LLM** (`make_llm_predict_fn` path): each SHAP evaluation is one full Ollama chat call, at
  roughly the same ~60–90 seconds/case cost as inference itself
  ([05-llm.md](05-llm.md)). `config.SHAP_MAX_EVALS_DEFAULT = 50` evals/document → **roughly
  50–75 minutes per document**, so batches should be kept small.

## Expected result
Each `{case_id}.html` renders the full case text with sentence-level background-colour shading
(white = neutral, red = pushes toward the predicted class, blue = pushes away), a small metadata
table (model, predicted/true label, class probabilities), and a legend. The SHAP efficiency axiom
is enforced by construction: `base_values[c] + sum(shap_values[:, c]) ≈ predict_fn([text])[0][c]`
for every class `c`. `shap_top_terms_{case_id}.png` shows the same signal as a ranked bar chart —
useful for a quick skim without opening every HTML card.

See [09-reproducibility-notes.md](09-reproducibility-notes.md) for hardware-determinism caveats —
SHAP attributions inherit any nondeterminism from the underlying `predict_fn` (BERT or LLM).
