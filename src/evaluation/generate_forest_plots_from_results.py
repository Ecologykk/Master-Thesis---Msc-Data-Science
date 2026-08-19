"""
Generate forest plots from evaluation_results_with_cis.parquet using plot_forest().
Converts consolidated results into per-run format for visualization.
"""

from pathlib import Path
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent))
from classification import plot_forest, BINARY_LABELS, TERNARY_LABELS, LABEL_NAMES

results_df = pd.read_parquet(Path("data/results/evaluation_results_with_cis.parquet"))
output_dir = Path("data/results/forest_plots")
output_dir.mkdir(parents=True, exist_ok=True)

for idx, row in results_df.iterrows():
    model_name = row['model_name']
    case_type = row['case_type']
    run_name = row['run_name']

    labels = BINARY_LABELS if case_type == "dv" else TERNARY_LABELS

    results = {
        "point_estimates": {
            "macro_f1": row['macro_f1_point'],
            "mcc": row['mcc_point'],
        },
        "ci": {
            "macro_f1": {
                "ci_lower": row['macro_f1_ci_lower'],
                "ci_upper": row['macro_f1_ci_upper'],
            },
            "mcc": {
                "ci_lower": row['mcc_ci_lower'],
                "ci_upper": row['mcc_ci_upper'],
            },
        },
        "labels": labels,
    }

    if case_type == "dv":
        results["point_estimates"]["tss"] = row['tss_point']
        results["ci"]["tss"] = {
            "ci_lower": row['tss_ci_lower'],
            "ci_upper": row['tss_ci_upper'],
        }

    class_names_dict = LABEL_NAMES[case_type]

    for metric in ["f1", "precision", "recall"]:
        results["point_estimates"][metric] = []
        results["ci"][metric] = []

        for i in labels:
            class_name = class_names_dict[i]
            point_col = f"{metric}_{class_name}_point"
            lower_col = f"{metric}_{class_name}_ci_lower"
            upper_col = f"{metric}_{class_name}_ci_upper"

            results["point_estimates"][metric].append(row[point_col])
            results["ci"][metric].append({
                "ci_lower": row[lower_col],
                "ci_upper": row[upper_col],
            })

    outpath = plot_forest(results, case_type, run_name=run_name, output_dir=output_dir)
    if outpath:
        print(f"[OK] {model_name} {case_type.upper()} ({run_name})")

print(f"\n[OK] All forest plots saved to {output_dir}")
