"""Compute bootstrap confidence intervals for all model predictions.

Loads all prediction parquets (BERT v3 + LLM runs), computes CIs for:
  - MCC, Macro-F1, TSS (DV only)
  - Per-class precision, recall, F1

Outputs a consolidated parquet with all results + CIs.
"""

from pathlib import Path

import pandas as pd
from classification import (
    BINARY_LABELS,
    LABEL_NAMES,
    TERNARY_LABELS,
    bootstrap_evaluation,
    compute_confidence_intervals,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Model prediction file paths (repo-root-anchored, so this script works from any CWD)
PREDICTION_FILES = {
    "bert_v3_dv": _REPO_ROOT
    / "data/models/dl/dv/predictions/bert_v3_dv_predictions.parquet",
    "bert_v3_boc": _REPO_ROOT
    / "data/models/dl/boc/predictions/bert_v3_boc_predictions.parquet",
    "llm_zs_dv": _REPO_ROOT
    / "data/models/llms/deepseek_r1_8b/zero_shot_dv_predictions_smart9k.parquet",
    "llm_zs_boc": _REPO_ROOT
    / "data/models/llms/deepseek_r1_8b/zero_shot_boc_predictions_smart9k.parquet",
    "llm_fs_dv": _REPO_ROOT
    / "data/models/llms/deepseek_r1_8b/few_shot_dv_predictions.parquet",
    "llm_fs_boc": _REPO_ROOT
    / "data/models/llms/deepseek_r1_8b/few_shot_boc_predictions.parquet",
}

OUTPUT_PATH = _REPO_ROOT / "data/results/evaluation_results_with_cis.parquet"


def extract_metrics_from_ci_dict(ci_dict, labels, model_name, case_type, run_name):
    """Flatten CI dict into a row for the output dataframe."""
    row = {
        "model_name": model_name,
        "case_type": case_type,
        "run_name": run_name,
    }

    # Scalar metrics
    for metric in ["macro_f1", "mcc", "tss"]:
        if metric in ci_dict:
            row[f"{metric}_ci_lower"] = ci_dict[metric]["ci_lower"]
            row[f"{metric}_ci_upper"] = ci_dict[metric]["ci_upper"]

    # Per-class metrics
    for metric in ["precision", "recall", "f1"]:
        if metric in ci_dict:
            class_values = ci_dict[metric]
            for i, class_name in enumerate(LABEL_NAMES[case_type].values()):
                row[f"{metric}_{class_name}_ci_lower"] = class_values[i]["ci_lower"]
                row[f"{metric}_{class_name}_ci_upper"] = class_values[i]["ci_upper"]

    return row


def process_prediction_file(file_path, case_type, model_name, run_name):
    """Load predictions, compute bootstrap CIs, return row dict."""
    print(f"\n  Loading {file_path}...")
    df = pd.read_parquet(file_path)

    y_true = df["y_true"].values
    y_pred = df["y_pred"].values

    labels = BINARY_LABELS if case_type == "dv" else TERNARY_LABELS
    compute_tss = case_type == "dv"

    print(f"    y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
    print(f"    Computing {1000} bootstrap samples (this may take a moment)...")

    # Run bootstrap evaluation
    metrics_list = bootstrap_evaluation(
        y_true, y_pred, labels=labels, compute_tss=compute_tss, n_bootstraps=1000
    )

    # Compute CIs from bootstrap distribution
    ci_dict = compute_confidence_intervals(metrics_list, alpha=0.05)

    # Extract point estimates (from original prediction set)
    from classification import _evaluate_classification

    point_estimates = _evaluate_classification(
        y_true, y_pred, labels=labels, compute_tss=compute_tss
    )

    # Build output row
    row = extract_metrics_from_ci_dict(ci_dict, labels, model_name, case_type, run_name)

    # Add point estimates
    for metric in ["macro_f1", "mcc", "tss"]:
        if metric in point_estimates:
            row[f"{metric}_point"] = point_estimates[metric]

    for metric in ["precision", "recall", "f1"]:
        if metric in point_estimates:
            for i, class_name in enumerate(LABEL_NAMES[case_type].values()):
                row[f"{metric}_{class_name}_point"] = point_estimates[metric][i]

    # Add counts
    row["n_samples"] = len(y_true)
    row["n_correct"] = (y_true == y_pred).sum()
    row["accuracy"] = (y_true == y_pred).mean()

    print(f"    ✓ Done. Accuracy: {row['accuracy']:.3f}")
    return row


def main():
    """Process all prediction files and write consolidated results."""
    output_path = OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("Computing Bootstrap CIs for All Models")
    print(f"{'='*70}")

    results = []

    for key, rel_path in PREDICTION_FILES.items():
        file_path = Path(rel_path)
        if not file_path.exists():
            print(f"\n  ⚠ {key}: {file_path} does not exist — skipping")
            continue

        # Parse key to extract metadata
        parts = key.split("_")
        if key.startswith("bert"):
            model_name = "BERT v3"
            case_type = parts[2]  # "dv" or "boc"
            run_name = "bert_v3"
        else:  # llm
            model_name = "DeepSeek-R1-8B"
            strategy = parts[1]  # "zs" or "fs"
            case_type = parts[2]  # "dv" or "boc"
            run_name = (
                f"{'zero-shot' if strategy == 'zs' else 'few-shot'}_{case_type}_smart9k"
            )

        print(f"\n{key.upper()}")
        print(f"  Model: {model_name}, Case Type: {case_type}, Run: {run_name}")

        row = process_prediction_file(file_path, case_type, model_name, run_name)
        results.append(row)

    # Write consolidated results
    results_df = pd.DataFrame(results)
    results_df.to_parquet(output_path, index=False)

    print(f"\n{'='*70}")
    print(f"Results saved to {output_path}")
    print(f"Shape: {results_df.shape}")
    print(f"{'='*70}\n")

    print("Results Preview:")
    print(
        results_df[
            [col for col in results_df.columns if "point" in col or "model_name" in col]
        ].to_string()
    )


if __name__ == "__main__":
    main()
