"""Slice 3 (LLM pipeline): zero-shot inference + evaluation on the gold test split.

Pipeline
--------
1. Load gold-test texts from src/modeling/dl/features.py
2. Build zero-shot prompts (src/modeling/llms/prompts.py)
3. Call Ollama and validate JSON labels (src/modeling/llms/client.py)
4. Save predictions to parquet in data/models/llms/{model}/
5. Run bootstrap evaluation + forest plot (src/evaluation/classification.py)

Example:
-------
python src/modeling/llms/predict.py --case_type dv --model deepseek_r1_8b --stage zero_shot --run_name deepseek_zs_v1

Determinism note
-----------------
Seeds are fixed and decoding is greedy (`temperature=0.0`, `top_k=1`,
`seed=42`), so this script is deterministic on identical hardware. Results
may still diverge slightly on different hardware: GPU floating-point
reductions are non-associative and kernel selection, driver version and
tensor-core availability all change the order of operations. Expect small
differences in embeddings and occasional label flips on borderline cases.
This does not indicate a bug.
"""

from __future__ import annotations

import argparse
import json
import sys
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

_THIS_FILE = Path(__file__).resolve()
_DL_DIR = _THIS_FILE.parents[1] / "dl"
_EVAL_DIR = _THIS_FILE.parents[2] / "evaluation"

for _path in (str(_DL_DIR), str(_EVAL_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from classification import plot_forest, run_classification_evaluation
from features import load_split_text_data

from client import InferenceTraceGateway, OllamaClient
from config import (
    OLLAMA_DETERMINISTIC_OPTIONS,
    OLLAMA_SEED_DEFAULT,
    OLLAMA_TEMPERATURE_DEFAULT,
    LABEL_OUTPUT_NAMES,
    LABEL_OUTPUT_TO_INT,
    LABEL_STR_TO_INT,
    LLM_OUTPUT_DIR,
    OLLAMA_MODELS,
)
from few_shot_retrieval_similarity import (
    build_few_shot_examples_by_case,
    save_few_shot_examples_by_case,
)
from prompts import (
    build_cot_prompt,
    build_few_shot_cot_prompt,
    build_few_shot_prompt,
    build_zero_shot_prompt,
)


def _encode_true_labels(y_raw: np.ndarray, case_type: str) -> np.ndarray:
    """Encode uppercase string labels from split CSV into fixed integer ids."""
    mapping = LABEL_STR_TO_INT[case_type]

    normalized = [str(value).upper().strip() for value in y_raw]
    unknown = sorted(set(normalized) - set(mapping.keys()))
    if unknown:
        raise ValueError(
            f"Found unknown gold labels for case_type='{case_type}': {unknown}"
        )

    return np.array([mapping[value] for value in normalized], dtype=np.int64)


def _get_prompt_builder(stage: str):
    """Return the prompt factory for the chosen stage."""
    if stage == "zero_shot":
        return build_zero_shot_prompt
    if stage == "few_shot":
        return build_few_shot_prompt
    if stage == "cot":
        return build_cot_prompt
    if stage == "few_shot_cot":
        return build_few_shot_cot_prompt

    raise NotImplementedError(
        f"Stage '{stage}' is not implemented yet in predict.py. "
        "Implemented stages: ['zero_shot', 'few_shot', 'cot', 'few_shot_cot']"
    )


def _build_ollama_options(seed: int, temperature: float) -> dict[str, int | float]:
    """Build deterministic Ollama decoding options for classification."""
    if temperature < 0.0:
        raise ValueError("temperature must be >= 0.0")

    options = dict(OLLAMA_DETERMINISTIC_OPTIONS)
    options["seed"] = int(seed)
    options["temperature"] = float(temperature)
    return options


def _estimate_tokens(text: str) -> int:
    """Rough token estimate used only for context-budget guards."""
    return int(ceil(len(text) / 4))


def _estimate_prompt_tokens(messages: list[dict]) -> int:
    """Rough token estimate for a full message list, summing over all message contents."""
    return _estimate_tokens("".join(str(m.get("content", "")) for m in messages))


def _default_enriched_path(case_type: str, split: str) -> Path:
    """Return the default enriched-parquet path for a case type/split combination.

    Matches the `<case_type>_<split>_enriched.parquet` naming `summarizer.py
    --run_train_batch` writes under `--output_root`.
    """
    return (
        _THIS_FILE.parents[3]
        / "data"
        / "processed_data"
        / "llm_summaries"
        / f"{case_type}_{split}_enriched.parquet"
    )


def _load_enriched_df(path: Path) -> pd.DataFrame:
    """Load an enriched parquet file, raising if it does not exist."""
    if not path.exists():
        raise FileNotFoundError(f"Enriched parquet not found: {path}")
    return pd.read_parquet(path)


def _build_few_shot_examples_mapping(
    query_path: Path,
    candidate_path: Path,
    *,
    top_n: int,
    k_per_class: int,
    max_pair_similarity: float,
    max_total_cases: int | None,
    output_path: Path | None,
) -> dict[str, list[dict]]:
    """Load query/candidate enriched parquets and build the few-shot example mapping.

    Thin wrapper around `few_shot_retrieval_similarity.build_few_shot_examples_by_case`
    that also optionally persists the result to `output_path`.
    """
    query_df = _load_enriched_df(query_path)
    candidate_df = _load_enriched_df(candidate_path)
    print(
        f"Building few-shot retrieval map from query={query_path.name} "
        f"(rows={len(query_df)}) and candidates={candidate_path.name} (rows={len(candidate_df)})"
    )
    examples_by_case = build_few_shot_examples_by_case(
        query_df,
        candidate_df,
        top_n=top_n,
        k_per_class=k_per_class,
        max_pair_similarity=max_pair_similarity,
        max_total_cases=max_total_cases,
    )
    if output_path is not None:
        save_few_shot_examples_by_case(examples_by_case, output_path)
        print(f"Few-shot example map saved -> {output_path}")
    return examples_by_case


def _debug_print_prompt_payload(
    *,
    case_id: str,
    stage: str,
    examples: list[dict],
    messages: list[dict],
    prompt_tokens_est: int,
    output_path: Path | None = None,
) -> None:
    """Write debug prompt payload to file or stdout."""
    lines = []
    lines.append("\n" + "=" * 120)
    lines.append(
        f"[DEBUG PROMPT] case_id={case_id} stage={stage} estimated_tokens={prompt_tokens_est}"
    )
    lines.append(f"[DEBUG PROMPT] examples={len(examples)}")
    for idx, ex in enumerate(examples, start=1):
        lines.append(f"[DEBUG PROMPT] example {idx}:")
        lines.append(f"  n_processo={ex.get('n_processo')}")
        lines.append(f"  label={ex.get('label')}")
        lines.append(f"  decision_class={ex.get('decision_class')}")
        lines.append(f"  few_shot_rank={ex.get('few_shot_rank')}")
        lines.append(f"  similarity_score={ex.get('similarity_score')}")
        lines.append(
            f"  selected_sentences_readable={ex.get('selected_sentences_readable')}"
        )
        lines.append(
            f"  selected_sentences_ranked_ordered={ex.get('selected_sentences_ranked_ordered')}"
        )
    lines.append("[DEBUG PROMPT] messages:")
    for msg in messages:
        lines.append(f"\n[{msg.get('role', 'unknown').upper()}]")
        lines.append(msg.get("content", ""))
    lines.append("=" * 120 + "\n")

    output = "\n".join(lines)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            fh.write(output)
        print(f"[DEBUG] Prompt payload written to {output_path}")
    else:
        print(output)


def _truncate_text_to_budget(text: str, text_token_budget: int) -> tuple[str, bool]:
    """Trim long texts to fit context while preserving start+end legal signal."""
    if text_token_budget <= 0:
        raise ValueError("text_token_budget must be > 0")

    max_chars = max(1, int(text_token_budget) * 4)
    if len(text) <= max_chars:
        return text, False

    marker = "\n\n[... TEXTO TRUNCADO AUTOMATICAMENTE PARA CABER NO CONTEXTO DO MODELO ...]\n\n"
    if len(marker) >= max_chars:
        return text[:max_chars], True

    available = max_chars - len(marker)
    head_chars = int(available * 0.55)
    tail_chars = max(0, available - head_chars)
    truncated = text[:head_chars].rstrip() + marker + text[-tail_chars:].lstrip()
    if len(truncated) > max_chars:
        truncated = truncated[:max_chars]
    return truncated, True


def run_gold_test_predictions_llm(
    case_type: str,
    model_short: str,
    stage: str,
    max_cases: int | None = None,
    ollama_options: dict[str, int | float] | None = None,
    trace_gateway: InferenceTraceGateway | None = None,
    max_prompt_tokens: int = 28000,
    smart_truncated_texts: pd.DataFrame | None = None,
    few_shot_examples_by_case: dict[str, list[dict]] | None = None,
    debug_prompt: bool = False,
    debug_prompt_case_id: str | None = None,
    debug_output_path: Path | None = None,
    request_timeout: int = 600,
    run_name: str | None = None,
) -> dict:
    """Run LLM inference on gold-test cases and return predictions/probabilities.

    For every case in the gold-test split, builds a stage-appropriate prompt
    (optionally substituting a smart-truncated summary for the full text and
    injecting retrieved few-shot examples), guards against exceeding
    `max_prompt_tokens` by truncating the case text (head+tail) when needed,
    calls `OllamaClient.chat_with_validation` to get a validated
    `predicted_label`, and accumulates one-hot predictions.

    Args:
        case_type: Either "dv" or "boc".
        model_short: Short model key into `config.OLLAMA_MODELS`.
        stage: One of "zero_shot", "few_shot", "cot", "few_shot_cot"
            (resolved via `_get_prompt_builder`).
        max_cases: Optional cap on the number of gold-test cases to run
            (first N); ignored if `debug_prompt_case_id` is set.
        ollama_options: Optional Ollama decoding options (seed, temperature,
            top_k, num_ctx, ...) passed through to every chat call.
        trace_gateway: Optional `InferenceTraceGateway` used to log each
            attempt's reasoning/output per case.
        max_prompt_tokens: Approximate token budget for the full prompt
            (system+user); if exceeded, the case text is truncated and the
            prompt rebuilt.
        smart_truncated_texts: Optional DataFrame with `n_processo` and
            `summary_text` columns; when a case id matches and its summary is
            non-empty, that summary text replaces the full case text in the
            prompt.
        few_shot_examples_by_case: Required (and validated per-case) when
            `stage` is "few_shot" or "few_shot_cot": mapping from case id to
            its list of few-shot example dicts.
        debug_prompt: If True, prints/writes the assembled prompt payload for
            one case (the first processed, or `debug_prompt_case_id` if set)
            before calling Ollama.
        debug_prompt_case_id: Optional `n_processo` to restrict the run to a
            single case, for debugging.
        debug_output_path: Optional file path to write the debug prompt
            payload to, instead of stdout.
        request_timeout: HTTP read timeout (seconds) for each Ollama chat
            request.
        run_name: Optional run tag recorded in trace log context.

    Returns:
        Dict with keys:
            "y_pred": int64 ndarray of predicted class indices.
            "y_true": int64 ndarray of gold class indices.
            "y_proba": float32 ndarray of shape (n_cases, n_classes), one-hot
                at the predicted class (i.e. not calibrated probabilities —
                Ollama's deterministic JSON output yields a single label per
                case, not a distribution).
            "n_processo": array of case identifiers, in split order.

    Raises:
        ValueError: If `max_prompt_tokens` or `request_timeout` is not
            positive; if `smart_truncated_texts` is missing required columns;
            if `debug_prompt_case_id` does not match any gold-test case; or
            if `max_cases` is not positive when provided.
        RuntimeError: If the configured Ollama model is not available
            locally, or if inference fails for any case (including missing
            few-shot examples for a case when `stage` requires them) — the
            original exception is chained with the failing case id.
    """
    if max_prompt_tokens <= 0:
        raise ValueError("max_prompt_tokens must be a positive integer.")
    if request_timeout <= 0:
        raise ValueError("request_timeout must be a positive integer.")

    debug_prompt_seen = False

    prompt_builder = _get_prompt_builder(stage)

    print("Loading gold test split ...")
    texts, y_raw, _, n_processo = load_split_text_data(case_type, split="gold_test")
    y_true = _encode_true_labels(y_raw, case_type)

    smart_text_map: dict[str, str] = {}
    if smart_truncated_texts is not None:
        required_cols = {"n_processo", "summary_text"}
        missing = required_cols - set(smart_truncated_texts.columns)
        if missing:
            raise ValueError(
                f"smart_truncated_texts is missing required columns: {sorted(missing)}"
            )
        smart_text_map = {
            str(row["n_processo"]): str(row["summary_text"] or "")
            for _, row in smart_truncated_texts.iterrows()
        }
        print(f"Smart-truncated summary map loaded: {len(smart_text_map)} cases")

    if debug_prompt_case_id is not None:
        case_idx = None
        for idx, case_id in enumerate(n_processo):
            if str(case_id) == str(debug_prompt_case_id):
                case_idx = idx
                break
        if case_idx is None:
            raise ValueError(
                f"Debug case '{debug_prompt_case_id}' not found in gold test split. "
                f"Available case IDs: {n_processo[:10].tolist()}..."
            )
        texts = texts[case_idx : case_idx + 1]
        y_true = y_true[case_idx : case_idx + 1]
        n_processo = n_processo[case_idx : case_idx + 1]
        print(f"[DEBUG] Filtering to single case: {debug_prompt_case_id}")
    elif max_cases is not None:
        if max_cases <= 0:
            raise ValueError("max_cases must be a positive integer when provided.")
        texts = texts[:max_cases]
        y_true = y_true[:max_cases]
        n_processo = n_processo[:max_cases]

    print(f"Cases to score: {len(texts)}")

    model_tag = OLLAMA_MODELS[model_short]
    client = OllamaClient()

    if trace_gateway is not None:
        trace_path = trace_gateway.get_log_path(
            model=model_short,
            stage=stage,
            case_type=case_type,
            run_name=run_name,
        )
        print(f"Inference trace log -> {trace_path}")

    if not client.check_model_available(model_tag):
        raise RuntimeError(
            f"Model '{model_tag}' not available in local Ollama. "
            f"Pull it first and retry (short name: '{model_short}')."
        )

    label_to_int = LABEL_OUTPUT_TO_INT[case_type]
    valid_labels = set(label_to_int.keys())
    n_classes = len(LABEL_OUTPUT_NAMES[case_type])

    y_pred = np.empty(len(texts), dtype=np.int64)
    y_proba = np.zeros((len(texts), n_classes), dtype=np.float32)

    print(f"Running {stage} inference with model '{model_short}' ({model_tag}) ...")
    trunc_count = 0
    smart_used_count = 0
    for idx, (case_id, text) in tqdm(
        enumerate(zip(n_processo, texts), start=1),
        desc=f"Classification of {case_type}",
        total=len(texts),
    ):
        try:
            case_id_str = str(case_id)
            text_for_prompt = text
            smart_flag = False
            if smart_text_map:
                candidate = smart_text_map.get(case_id_str, "")
                if candidate.strip():
                    text_for_prompt = candidate
                    smart_flag = True
                    smart_used_count += 1
            examples = (
                few_shot_examples_by_case.get(case_id_str, [])
                if few_shot_examples_by_case
                else []
            )
            if stage in {"few_shot", "few_shot_cot"} and not examples:
                raise ValueError(
                    f"Missing few-shot examples for case '{case_id_str}'. "
                    "Provide a mapping keyed by n_processo."
                )
            if stage in {"few_shot", "few_shot_cot"}:
                messages = prompt_builder(
                    case_type,
                    case_id,
                    text_for_prompt,
                    examples,
                    smart_truncated=smart_flag,
                )
            else:
                messages = prompt_builder(
                    case_type,
                    case_id,
                    text_for_prompt,
                    smart_truncated=smart_flag,
                )
            prompt_tok_est = _estimate_prompt_tokens(messages)
            if debug_prompt and not debug_prompt_seen:
                if debug_prompt_case_id is None or case_id_str == debug_prompt_case_id:
                    _debug_print_prompt_payload(
                        case_id=case_id_str,
                        stage=stage,
                        examples=examples,
                        messages=messages,
                        prompt_tokens_est=prompt_tok_est,
                        output_path=debug_output_path,
                    )
                    debug_prompt_seen = True
            was_truncated = False

            if prompt_tok_est > max_prompt_tokens:
                if stage in {"few_shot", "few_shot_cot"}:
                    fixed_prompt_messages = prompt_builder(
                        case_type,
                        case_id,
                        "",
                        examples,
                        smart_truncated=smart_flag,
                    )
                else:
                    fixed_prompt_messages = prompt_builder(
                        case_type,
                        case_id,
                        "",
                        smart_truncated=smart_flag,
                    )
                fixed_prompt_tok_est = _estimate_prompt_tokens(fixed_prompt_messages)
                text_budget = max(512, max_prompt_tokens - fixed_prompt_tok_est)
                text_for_inference, was_truncated = _truncate_text_to_budget(
                    text_for_prompt,
                    text_budget,
                )
                if stage in {"few_shot", "few_shot_cot"}:
                    messages = prompt_builder(
                        case_type,
                        case_id,
                        text_for_inference,
                        examples,
                        smart_truncated=smart_flag,
                    )
                else:
                    messages = prompt_builder(
                        case_type,
                        case_id,
                        text_for_inference,
                        smart_truncated=smart_flag,
                    )
                prompt_tok_est = _estimate_prompt_tokens(messages)
                trunc_count += 1

                if trunc_count <= 5:
                    print(
                        f"[WARN] Prompt above budget for case '{case_id}' "
                        f"(estimated {prompt_tok_est} tokens after truncation)."
                    )

            parsed = client.chat_with_validation(
                model=model_tag,
                messages=messages,
                valid_labels=valid_labels,
                format="json",
                options=ollama_options,
                trace_gateway=trace_gateway,
                trace_context={
                    "model": model_short,
                    "stage": stage,
                    "case_type": case_type,
                    "case_id": str(case_id),
                    "truncated": "1" if was_truncated else "0",
                    "smart_truncated_input": "1" if smart_flag else "0",
                    "prompt_tokens_est": str(prompt_tok_est),
                    "run_name": run_name or "",
                },
                timeout=request_timeout,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Inference failed for n_processo='{case_id}' "
                f"({idx}/{len(texts)}): {exc}"
            ) from exc

        predicted_label = parsed["predicted_label"]
        pred_int = label_to_int[predicted_label]

        y_pred[idx - 1] = pred_int
        y_proba[idx - 1, pred_int] = 1.0

    if trunc_count:
        print(
            f"[INFO] Context guard truncated {trunc_count}/{len(texts)} cases "
            f"to keep prompts within ~{max_prompt_tokens} tokens."
        )
    if smart_text_map:
        print(
            f"[INFO] Smart-truncated inputs used for {smart_used_count}/{len(texts)} cases."
        )

    return {
        "y_pred": y_pred,
        "y_true": y_true,
        "y_proba": y_proba,
        "n_processo": n_processo,
    }


def save_predictions_parquet_llm(
    results: dict,
    case_type: str,
    model_short: str,
    stage: str,
    run_name: str | None = None,
) -> Path:
    """Save LLM predictions to parquet using the agreed schema.

    Args:
        results: Dict as returned by `run_gold_test_predictions_llm`, with
            keys "n_processo", "y_true", "y_pred", "y_proba".
        case_type: Either "dv" or "boc"; used both for the output subpath
            and to select the class names for `proba_<label>` columns.
        model_short: Short model key; used to name the output subdirectory
            `LLM_OUTPUT_DIR/<model_short>/`.
        stage: Pipeline stage; used in the output filename.
        run_name: Optional run tag stored in the `run_name` column.

    Returns:
        Path the parquet file was written to:
        `LLM_OUTPUT_DIR/<model_short>/<stage>_<case_type>_predictions.parquet`.
    """
    output_dir = LLM_OUTPUT_DIR / model_short
    output_dir.mkdir(parents=True, exist_ok=True)

    output_names = LABEL_OUTPUT_NAMES[case_type]
    y_proba: np.ndarray = results["y_proba"]

    df = pd.DataFrame(
        {
            "n_processo": results["n_processo"],
            "y_true": results["y_true"].astype(np.int32),
            "y_pred": results["y_pred"].astype(np.int32),
            **{
                f"proba_{output_names[i]}": y_proba[:, i]
                for i in range(y_proba.shape[1])
            },
            "run_name": run_name or "",
            "case_type": case_type,
        }
    )

    out_path = output_dir / f"{stage}_{case_type}_predictions.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Predictions saved -> {out_path}")
    return out_path


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the gold-test inference/evaluation run."""
    parser = argparse.ArgumentParser(
        description="Run LLM gold-test inference and evaluation (Slice 3)."
    )
    parser.add_argument(
        "--case_type",
        required=True,
        choices=["dv", "boc"],
        help="Case type: dv (domestic violence) or boc (breach of contract).",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(OLLAMA_MODELS.keys()),
        help="Short model name from config. Example: deepseek_r1_8b.",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=["zero_shot", "few_shot", "cot", "few_shot_cot"],
        help="Prompting stage.",
    )
    parser.add_argument(
        "--few_shot_examples_path",
        default=None,
        help="Optional JSON file with {n_processo: [examples...]} mapping.",
    )
    parser.add_argument(
        "--few_shot_query_path",
        default=None,
        help="Optional enriched gold-test parquet used as retrieval query source.",
    )
    parser.add_argument(
        "--few_shot_candidates_path",
        default=None,
        help="Optional enriched train parquet used as retrieval candidate source.",
    )
    parser.add_argument(
        "--few_shot_examples_output_path",
        default=None,
        help="Optional JSON output path for per-case retrieved few-shot examples.",
    )
    parser.add_argument("--few_shot_top_n", type=int, default=20)
    parser.add_argument("--few_shot_k_per_class", type=int, default=1)
    parser.add_argument("--few_shot_max_pair_similarity", type=float, default=0.9)
    parser.add_argument("--few_shot_max_total_cases", type=int, default=None)
    parser.add_argument(
        "--debug_prompt",
        action="store_true",
        help="Print assembled prompt payload for one case before calling Ollama.",
    )
    parser.add_argument(
        "--debug_prompt_case_id",
        default=None,
        help="Optional n_processo to debug. If specified, only runs inference for that case.",
    )
    parser.add_argument(
        "--debug_output_path",
        default=None,
        help="Optional path to write debug prompt payload to .txt file. If omitted with --debug_prompt, prints to stdout.",
    )
    parser.add_argument(
        "--run_name",
        default="",
        help="Optional run tag stored in parquet and used in forest plot filename.",
    )
    parser.add_argument(
        "--n_bootstraps",
        type=int,
        default=1000,
        help="Number of bootstrap iterations for confidence intervals.",
    )
    parser.add_argument(
        "--max_cases",
        type=int,
        default=None,
        help="Optional cap for quick smoke runs (first N gold-test cases).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=OLLAMA_SEED_DEFAULT,
        help="Deterministic seed for Ollama decoding (Slice 2c).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=OLLAMA_TEMPERATURE_DEFAULT,
        help="Decoding temperature (Slice 2c). Use 0.0 for near-greedy output.",
    )
    parser.add_argument(
        "--num_ctx",
        type=int,
        default=32768,
        help=(
            "Ollama context window passed in options['num_ctx']. "
            "Increase for long legal texts if VRAM allows."
        ),
    )
    parser.add_argument(
        "--max_prompt_tokens",
        type=int,
        default=28000,
        help=(
            "Approximate upper bound for full prompt tokens (system+user). "
            "If exceeded, text is auto-truncated (head+tail) to preserve instructions."
        ),
    )
    parser.add_argument(
        "--trace_log",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable per-case trace logging to txt (reasoning + final output). "
            "Use --no-trace-log to disable."
        ),
    )
    parser.add_argument(
        "--trace_log_dir",
        default=None,
        help="Optional base directory for trace txt logs.",
    )
    parser.add_argument(
        "--smart_truncated_path",
        default=None,
        help=(
            "Optional parquet with precomputed smart summaries (requires columns "
            "'n_processo' and 'summary_text'). If provided, these texts replace "
            "full decision text per matched case id."
        ),
    )
    parser.add_argument(
        "--request_timeout",
        type=int,
        default=1200,
        help=(
            "HTTP read timeout (seconds) for each Ollama chat request. "
            "Increase for slower hardware/models to avoid ReadTimeout."
        ),
    )

    # Reserved for Slice 7 to keep CLI forward-compatible with the plan.
    parser.add_argument(
        "--shap",
        action="store_true",
        help="Reserved for Slice 7 (ignored in Slice 3).",
    )
    parser.add_argument(
        "--shap_cases",
        nargs="*",
        default=None,
        help="Reserved for Slice 7 (ignored in Slice 3).",
    )

    return parser.parse_args()


def main() -> None:
    """CLI entry point: run gold-test LLM inference, save predictions, and evaluate.

    Parses CLI args, builds deterministic Ollama decoding options, optionally
    loads a smart-truncated-text parquet and/or builds or loads a few-shot
    example mapping, runs `run_gold_test_predictions_llm`, saves the
    predictions via `save_predictions_parquet_llm`, then runs bootstrap
    classification evaluation and saves a forest plot.
    """
    args = _parse_args()

    if args.shap or args.shap_cases:
        print(
            "[INFO] --shap and --shap_cases are reserved for Slice 7 and ignored in Slice 3."
        )

    ollama_options = _build_ollama_options(
        seed=args.seed,
        temperature=args.temperature,
    )
    ollama_options["num_ctx"] = int(args.num_ctx)
    print(
        "Slice 2c deterministic options: "
        f"seed={ollama_options['seed']}, "
        f"temperature={ollama_options['temperature']}, "
        f"top_k={ollama_options.get('top_k')}, "
        f"top_p={ollama_options.get('top_p')}, "
        f"repeat_penalty={ollama_options.get('repeat_penalty')}, "
        f"num_ctx={ollama_options.get('num_ctx')}, "
        f"max_prompt_tokens={args.max_prompt_tokens}, "
        f"request_timeout={args.request_timeout}"
    )

    trace_gateway = None
    if args.trace_log:
        trace_gateway = InferenceTraceGateway(base_dir=args.trace_log_dir)

    smart_df = None
    few_shot_examples_by_case = None
    if args.smart_truncated_path:
        smart_path = Path(args.smart_truncated_path)
        if not smart_path.exists():
            raise FileNotFoundError(f"smart_truncated_path not found: {smart_path}")
        smart_df = pd.read_parquet(smart_path)
        print(f"Loaded smart-truncated parquet -> {smart_path} (rows={len(smart_df)})")
    if args.few_shot_examples_path:
        few_shot_path = Path(args.few_shot_examples_path)
        if not few_shot_path.exists():
            raise FileNotFoundError(
                f"few_shot_examples_path not found: {few_shot_path}"
            )
        with few_shot_path.open("r", encoding="utf-8") as fh:
            few_shot_examples_by_case = json.load(fh)
        print(
            f"Loaded few-shot examples mapping -> {few_shot_path} (cases={len(few_shot_examples_by_case)})"
        )
    elif args.stage in {"few_shot", "few_shot_cot"}:
        query_path = (
            Path(args.few_shot_query_path)
            if args.few_shot_query_path
            else _default_enriched_path(args.case_type, "gold_test")
        )
        candidate_path = (
            Path(args.few_shot_candidates_path)
            if args.few_shot_candidates_path
            else _default_enriched_path(args.case_type, "train_before_cutoff")
        )
        output_path = (
            Path(args.few_shot_examples_output_path)
            if args.few_shot_examples_output_path
            else None
        )
        few_shot_examples_by_case = _build_few_shot_examples_mapping(
            query_path,
            candidate_path,
            top_n=args.few_shot_top_n,
            k_per_class=args.few_shot_k_per_class,
            max_pair_similarity=args.few_shot_max_pair_similarity,
            max_total_cases=args.few_shot_max_total_cases,
            output_path=output_path,
        )

    debug_output_path = None
    if args.debug_output_path:
        debug_output_path = Path(args.debug_output_path)
    elif args.debug_prompt_case_id and args.debug_prompt:
        debug_dir = LLM_OUTPUT_DIR / args.model / "debug"
        debug_output_path = (
            debug_dir / f"debug_prompt_{args.debug_prompt_case_id}_{args.stage}.txt"
        )

    results = run_gold_test_predictions_llm(
        case_type=args.case_type,
        model_short=args.model,
        stage=args.stage,
        max_cases=args.max_cases,
        ollama_options=ollama_options,
        trace_gateway=trace_gateway,
        max_prompt_tokens=args.max_prompt_tokens,
        smart_truncated_texts=smart_df,
        few_shot_examples_by_case=few_shot_examples_by_case,
        debug_prompt=args.debug_prompt,
        debug_prompt_case_id=args.debug_prompt_case_id,
        debug_output_path=debug_output_path,
        request_timeout=args.request_timeout,
        run_name=(args.run_name or None),
    )

    save_predictions_parquet_llm(
        results=results,
        case_type=args.case_type,
        model_short=args.model,
        stage=args.stage,
        run_name=args.run_name,
    )

    print("Running classification evaluation ...")
    eval_results = run_classification_evaluation(
        y_true=results["y_true"],
        y_pred=results["y_pred"],
        case_type=args.case_type,
        n_bootstraps=args.n_bootstraps,
    )

    plot_dir = LLM_OUTPUT_DIR / args.model / "plots"
    plot_forest(
        results=eval_results,
        case_type=args.case_type,
        run_name=(args.run_name or None),
        output_dir=plot_dir,
    )


if __name__ == "__main__":
    main()
