import pandas as pd

df = pd.read_parquet("data/results/evaluation_results_with_cis.parquet")

print("=" * 120)
print("SUMMARY TABLE — ALL MODELS (for memory update)")
print("=" * 120)
print()

for _, row in df.iterrows():
    print(
        f"{row['model_name']:20} | {row['case_type'].upper():3} | {row['run_name']:30} | Acc {row['accuracy']:.1%} | F1 {row['macro_f1_point']:.3f} [{row['macro_f1_ci_lower']:.3f}–{row['macro_f1_ci_upper']:.3f}] | MCC {row['mcc_point']:.3f} [{row['mcc_ci_lower']:.3f}–{row['mcc_ci_upper']:.3f}]"
    )

print()
print("Per-class metrics for DV (binary):")
print()
dv_df = df[df["case_type"] == "dv"]
for _, row in dv_df.iterrows():
    print(f"{row['run_name']}:")
    print(
        f"  MANTIDA: P {row['precision_MANTIDA_point']:.3f} [{row['precision_MANTIDA_ci_lower']:.3f}–{row['precision_MANTIDA_ci_upper']:.3f}] | R {row['recall_MANTIDA_point']:.3f} [{row['recall_MANTIDA_ci_lower']:.3f}–{row['recall_MANTIDA_ci_upper']:.3f}] | F1 {row['f1_MANTIDA_point']:.3f} [{row['f1_MANTIDA_ci_lower']:.3f}–{row['f1_MANTIDA_ci_upper']:.3f}]"
    )
    print(
        f"  ALTERADA: P {row['precision_ALTERADA_point']:.3f} [{row['precision_ALTERADA_ci_lower']:.3f}–{row['precision_ALTERADA_ci_upper']:.3f}] | R {row['recall_ALTERADA_point']:.3f} [{row['recall_ALTERADA_ci_lower']:.3f}–{row['recall_ALTERADA_ci_upper']:.3f}] | F1 {row['f1_ALTERADA_point']:.3f} [{row['f1_ALTERADA_ci_lower']:.3f}–{row['f1_ALTERADA_ci_upper']:.3f}]"
    )
    print()

print("Per-class metrics for BOC (ternary):")
print()
boc_df = df[df["case_type"] == "boc"]
for _, row in boc_df.iterrows():
    print(f"{row['run_name']}:")
    print(
        f"  DESFAVORAVEL: P {row['precision_DESFAVORAVEL_point']:.3f} [{row['precision_DESFAVORAVEL_ci_lower']:.3f}–{row['precision_DESFAVORAVEL_ci_upper']:.3f}] | R {row['recall_DESFAVORAVEL_point']:.3f} [{row['recall_DESFAVORAVEL_ci_lower']:.3f}–{row['recall_DESFAVORAVEL_ci_upper']:.3f}] | F1 {row['f1_DESFAVORAVEL_point']:.3f} [{row['f1_DESFAVORAVEL_ci_lower']:.3f}–{row['f1_DESFAVORAVEL_ci_upper']:.3f}]"
    )
    print(
        f"  PARCIAL: P {row['precision_PARCIAL_point']:.3f} [{row['precision_PARCIAL_ci_lower']:.3f}–{row['precision_PARCIAL_ci_upper']:.3f}] | R {row['recall_PARCIAL_point']:.3f} [{row['recall_PARCIAL_ci_lower']:.3f}–{row['recall_PARCIAL_ci_upper']:.3f}] | F1 {row['f1_PARCIAL_point']:.3f} [{row['f1_PARCIAL_ci_lower']:.3f}–{row['f1_PARCIAL_ci_upper']:.3f}]"
    )
    print(
        f"  FAVORAVEL: P {row['precision_FAVORAVEL_point']:.3f} [{row['precision_FAVORAVEL_ci_lower']:.3f}–{row['precision_FAVORAVEL_ci_upper']:.3f}] | R {row['recall_FAVORAVEL_point']:.3f} [{row['recall_FAVORAVEL_ci_lower']:.3f}–{row['recall_FAVORAVEL_ci_upper']:.3f}] | F1 {row['f1_FAVORAVEL_point']:.3f} [{row['f1_FAVORAVEL_ci_lower']:.3f}–{row['f1_FAVORAVEL_ci_upper']:.3f}]"
    )
    print()
