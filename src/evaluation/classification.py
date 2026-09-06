"""Classification evaluation metrics: point estimates and bootstrap confidence intervals.

## 5. Classification Evaluation Metrics

Primary metrics:

- **Macro F1**
- **Matthews Correlation Coefficient (MCC)**

Secondary metric:

- **True Skill Statistic (TSS)**
  (only for Domestic Violence)

Also compute:

- Per-class precision
- Per-class recall
- Per-class F1

---

## 6. Bootstrap Confidence Intervals

Model evaluation uses **stratified bootstrap resampling**.

Procedure:

- Bootstrap predictions from each fold
- Compute metric distributions
- Estimate **95% confidence intervals**

Store:

- point estimate
- lower CI bound
- upper CI bound

---


---

## 9. Gold Test Evaluation

Evaluate final model on a **held-out temporal test set**.

Use the same evaluation protocol:

- Stratified bootstrap
- Macro-F1
- MCC
- TSS (DV only)

Report:

- point estimates
- 95% confidence intervals
"""

# Need to do the following in a separte preprocessing step to ensure consistent label encoding across all bootstrap samples and folds.
# Fixed class-to-integer mappings — must be used consistently when encoding labels.
# Binary (Domestic Violence):   0 = kept (decisao_mantida), 1 = altered (decisao_alterada)
# Ternary (Breach of Contract): 0 = unfavorable (decisao_desfavoravel),
#                                1 = partial (decisao_parcial),
#                                2 = favorable (decisao_favoravel)
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

BINARY_LABELS = [0, 1]
TERNARY_LABELS = [0, 1, 2]
SEED = 42  # For reproducibility of bootstrap sampling

LABELS = {"dv": BINARY_LABELS, "boc": TERNARY_LABELS}
LABEL_NAMES = {
    "dv": {0: "MANTIDA", 1: "ALTERADA"},
    "boc": {0: "DESFAVORAVEL", 1: "PARCIAL", 2: "FAVORAVEL"},
}

# Colour palettes — mirror eda.py
CASE_COLOURS: dict[str, str] = {
    "dv": "#E53935",  # red
    "boc": "#1E88E5",  # blue
}

CLASS_COLOURS: dict[str, dict[int, str]] = {
    "dv": {
        0: "#7E57C2",  # MANTIDA — violet
        1: "#FB8C00",  # ALTERADA — orange
    },
    "boc": {
        0: "#D81B60",  # DESFAVORAVEL — magenta
        1: "#26C6DA",  # PARCIAL — turquoise
        2: "#7CB342",  # FAVORAVEL — green-lime
    },
}

CLASS_DISPLAY_NAMES: dict[str, dict[int, str]] = {
    "dv": {0: "Kept", 1: "Changed"},
    "boc": {0: "Unfavourable", 1: "Partial", 2: "Favourable"},
}


def _tss(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=BINARY_LABELS)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return sensitivity + specificity - 1


def _evaluate_classification(y_true, y_pred, labels, compute_tss=False):
    """Compute classification metrics with a fixed class order.

    Parameters
    ----------
    y_true, y_pred : array-like of int
        Ground-truth and predicted integer labels.
    labels : list of int
        Ordered class integers that define the metric array positions.
        Use BINARY_LABELS for DV tasks or TERNARY_LABELS for BoC tasks.
    compute_tss : bool
        Whether to compute TSS (binary DV task only).
    """
    # Passing `labels` guarantees per-class arrays are always aligned to the
    # same positions, even when a class is absent from a bootstrap sample.
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    results = {
        "labels": labels,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
        "mcc": mcc,
        "confusion_matrix": cm,
    }

    if compute_tss:
        results["tss"] = _tss(y_true, y_pred)

    return results


def _stratified_bootstrap_indices(y_true, labels, rng):
    """Return one set of bootstrap indices sampled with replacement within each class.

    Sampling within each class separately guarantees the bootstrap sample
    preserves the original class distribution — critical for small, imbalanced datasets.
    """
    indices = []
    for label in labels:
        class_idx = np.where(y_true == label)[0]
        indices.append(rng.choice(class_idx, size=len(class_idx), replace=True))
    return np.concatenate(indices)


def bootstrap_evaluation(
    y_true, y_pred, labels, compute_tss=False, n_bootstraps=1000, random_state=SEED
):
    """Perform stratified bootstrap evaluation of classification metrics.

    Parameters
    ----------
    y_true, y_pred : array-like of int
        Ground-truth and predicted integer labels for the entire dataset.
    labels : list of int
        Ordered class integers that define the metric array positions.
        Use BINARY_LABELS for DV tasks or TERNARY_LABELS for BoC tasks.
    compute_tss : bool
        Whether to compute TSS (binary DV task only).
    n_bootstraps : int
        Number of bootstrap samples to generate.
    random_state : int or None
        Random seed for reproducibility.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    rng = np.random.default_rng(seed=random_state)
    metrics_list = []

    for _ in range(n_bootstraps):
        idx = _stratified_bootstrap_indices(y_true, labels, rng)
        metrics = _evaluate_classification(
            y_true[idx], y_pred[idx], labels=labels, compute_tss=compute_tss
        )
        metrics_list.append(metrics)

    return metrics_list


def compute_confidence_intervals(metrics_list, alpha=0.05):
    """Compute percentile confidence intervals from bootstrap metric dicts.

    Parameters
    ----------
    metrics_list : list of dict
        Output of bootstrap_evaluation — one metric dict per bootstrap iteration.
    alpha : float
        Significance level. Default 0.05 → 95% CI.

    Returns:
    -------
    dict
        Scalar metrics: {metric: {'ci_lower': float, 'ci_upper': float}}
        Array metrics:  {metric: [{'ci_lower': float, 'ci_upper': float}, ...]}  # one entry per class
    """
    lower_p = 100 * (alpha / 2)
    upper_p = 100 * (1 - alpha / 2)

    scalar_keys = ["macro_f1", "mcc", "tss"]
    array_keys = ["precision", "recall", "f1"]

    ci = {}

    for key in scalar_keys:
        values = [m[key] for m in metrics_list if key in m]
        if values:
            ci[key] = {
                "ci_lower": float(np.percentile(values, lower_p)),
                "ci_upper": float(np.percentile(values, upper_p)),
            }

    for key in array_keys:
        arrays = [m[key] for m in metrics_list if key in m]
        if arrays:
            stacked = np.array(arrays)  # shape (n_bootstraps, n_classes)
            ci[key] = [
                {
                    "ci_lower": float(np.percentile(stacked[:, i], lower_p)),
                    "ci_upper": float(np.percentile(stacked[:, i], upper_p)),
                }
                for i in range(stacked.shape[1])
            ]

    return ci


def run_classification_evaluation(
    y_true,
    y_pred,
    case_type: str,
    n_bootstraps: int = 1000,
    split_label: str = "Gold test",
) -> dict:
    """Compute point estimates + bootstrap CIs and print a formatted report.

    Parameters
    ----------
    y_true, y_pred : array-like of int
    case_type : str — "dv" or "boc"
        Determines the label set, class names, and whether TSS is computed.
    n_bootstraps : int
        Number of bootstrap iterations for CI estimation.
    split_label : str
        Name of the split being scored, used only in the printed report
        (e.g. "Gold test", "Train", "Validation").

    Returns:
    -------
    dict with keys:
        point_estimates : dict — output of _evaluate_classification
        ci              : dict — output of compute_confidence_intervals
        labels          : list[int]
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    labels = LABELS[case_type]
    names = LABEL_NAMES[case_type]
    compute_tss = case_type == "dv"

    point = _evaluate_classification(
        y_true, y_pred, labels=labels, compute_tss=compute_tss
    )
    metrics_list = bootstrap_evaluation(
        y_true,
        y_pred,
        labels=labels,
        compute_tss=compute_tss,
        n_bootstraps=n_bootstraps,
    )
    ci = compute_confidence_intervals(metrics_list)

    def _ci_str(key):
        if key not in ci:
            return ""
        return f"  [{ci[key]['ci_lower']:.4f}, {ci[key]['ci_upper']:.4f}]"

    print(f"\n  {split_label} n  : {len(y_true)}")
    print(f"  Macro F1     : {point['macro_f1']:.4f}{_ci_str('macro_f1')}")
    print(f"  MCC          : {point['mcc']:.4f}{_ci_str('mcc')}")
    if compute_tss:
        print(f"  TSS          : {point['tss']:.4f}{_ci_str('tss')}")

    print("\n  Per-class metrics (95% bootstrap CI):")
    print(f"  {'Class':<16} {'Precision':>24} {'Recall':>24} {'F1':>24}")
    for i, label in enumerate(labels):

        def _vcl(metric_key):
            v = float(point[metric_key][i])
            lo = ci[metric_key][i]["ci_lower"]
            hi = ci[metric_key][i]["ci_upper"]
            return f"{v:.3f} [{lo:.3f}, {hi:.3f}]"

        print(
            f"  {names[label]:<16} {_vcl('precision'):>24} {_vcl('recall'):>24} {_vcl('f1'):>24}"
        )

    return {"point_estimates": point, "ci": ci, "labels": labels}


def plot_forest(
    results: dict,
    case_type: str,
    run_name: str | None = None,
    output_dir: "Path | None" = None,
    split_label: str = "gold test",
) -> "Path | None":
    """Forest plot of classification metrics with 95% bootstrap CIs.

    Two sections:
      - Summary metrics: Macro F1, MCC, TSS (DV only)  — case colour, ◆ marker
      - Per-class metrics: F1, Precision, Recall        — class colours, ● marker

    Parameters
    ----------
    results : dict
        Output of run_classification_evaluation — keys: point_estimates, ci, labels.
    case_type : str — "dv" or "boc"
    run_name : str or None — prefix for the output filename.
    output_dir : Path or None
        If given, saves to ``{output_dir}/{prefix}{case_type}_forest_plot.png``.
        Non-gold splits get a ``_{split}`` suffix so they never overwrite the
        gold-test plot.  If None, calls plt.show() instead.
    split_label : str
        Name of the split being plotted, shown in the title and used as the
        filename suffix (e.g. "gold test", "train", "validation").

    Returns:
    -------
    Path to the saved PNG, or None if output_dir is None.
    """
    from pathlib import Path as _Path

    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    point = results["point_estimates"]
    ci = results["ci"]
    labels = results["labels"]

    case_colour = CASE_COLOURS[case_type]
    class_colours = CLASS_COLOURS[case_type]
    display_names = CLASS_DISPLAY_NAMES[case_type]

    # --- summary rows ---
    summary_rows = []
    for key in ["macro_f1", "mcc"] + (["tss"] if case_type == "dv" else []):
        if key not in ci:
            continue
        summary_rows.append(
            {
                "label": key.replace("_", " ").upper(),
                "point": float(point[key]),
                "ci_lower": ci[key]["ci_lower"],
                "ci_upper": ci[key]["ci_upper"],
                "colour": case_colour,
                "marker": "D",
                "ms": 9,
            }
        )

    # --- per-class rows ---
    class_rows = []
    for metric_key in ["f1", "precision", "recall"]:
        for i, lbl in enumerate(labels):
            class_rows.append(
                {
                    "label": f"{metric_key.upper()}  {display_names[lbl]}",
                    "point": float(point[metric_key][i]),
                    "ci_lower": ci[metric_key][i]["ci_lower"],
                    "ci_upper": ci[metric_key][i]["ci_upper"],
                    "colour": class_colours[lbl],
                    "marker": "o",
                    "ms": 7,
                }
            )

    all_rows = summary_rows + class_rows
    n_rows = len(all_rows)
    y_pos = list(range(n_rows - 1, -1, -1))  # top-to-bottom

    fig, ax = plt.subplots(figsize=(9, n_rows * 0.55 + 2))

    # Divider line between summary and per-class sections
    if summary_rows and class_rows:
        divider_y = y_pos[len(summary_rows) - 1] - 0.5
        ax.axhline(divider_y, color="#cccccc", linewidth=0.8)

    for pos, row in zip(y_pos, all_rows):
        xerr_low = row["point"] - row["ci_lower"]
        xerr_high = row["ci_upper"] - row["point"]
        ax.errorbar(
            row["point"],
            pos,
            xerr=[[xerr_low], [xerr_high]],
            fmt=row["marker"],
            color=row["colour"],
            markersize=row["ms"],
            capsize=4,
            linewidth=1.5,
            elinewidth=1.5,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r["label"] for r in all_rows], fontsize=10)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("Metric Value", fontsize=11)
    # Derive n from the confusion matrix so the title can never go stale.
    n_samples = int(np.sum(point["confusion_matrix"]))
    ax.set_title(
        f"{case_type.upper()} — Classification Metrics ({run_name or 'model'})\n"
        f"95% Bootstrap CI  (n={n_samples} {split_label})",
        fontsize=12,
    )
    ax.set_xlim(-0.25, 1.1)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [mpatches.Patch(color=case_colour, label="Summary metric (◆)")]
    for lbl in labels:
        legend_handles.append(
            mpatches.Patch(color=class_colours[lbl], label=f"{display_names[lbl]} (●)")
        )
    ax.legend(handles=legend_handles, fontsize=9, loc="lower right")

    plt.tight_layout()

    if output_dir is not None:
        output_dir = _Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{run_name}_" if run_name else ""
        # Gold test keeps the original filename; other splits are suffixed.
        suffix = (
            "" if split_label == "gold test" else f"_{split_label.replace(' ', '_')}"
        )
        save_path = output_dir / f"{prefix}{case_type}{suffix}_forest_plot.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Forest plot -> {save_path}")
        return save_path

    plt.show()
    plt.close(fig)
    return None


# Smoke example test with dummy data and visualization of metric distributions and confidence intervals would go here.:

if __name__ == "__main__":
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    # Dummy data — ternary task with class imbalance
    y_true = [0, 1, 1, 1, 1, 1, 2, 2] * 10
    y_pred = [0, 1, 1, 1, 0, 2, 2, 1] * 10
    labels = TERNARY_LABELS
    class_names = {0: "Unfavorable", 1: "Partial", 2: "Favorable"}

    # Point estimates on the full data
    point_est = _evaluate_classification(y_true, y_pred, labels=labels)

    # Bootstrap distribution → CIs
    metrics_list = bootstrap_evaluation(
        y_true, y_pred, labels=labels, n_bootstraps=1000
    )
    ci = compute_confidence_intervals(metrics_list)

    # Build forest-plot rows (top = summary metrics, bottom = per-class)
    rows = []
    for key in ["macro_f1", "mcc"]:
        rows.append(
            {
                "label": key.replace("_", " ").upper(),
                "point": point_est[key],
                "ci_lower": ci[key]["ci_lower"],
                "ci_upper": ci[key]["ci_upper"],
                "summary": True,
            }
        )
    for metric_key in ["f1", "precision", "recall"]:
        for i, cls in enumerate(labels):
            rows.append(
                {
                    "label": f"{metric_key.upper()}  {class_names[cls]}",
                    "point": float(point_est[metric_key][i]),
                    "ci_lower": ci[metric_key][i]["ci_lower"],
                    "ci_upper": ci[metric_key][i]["ci_upper"],
                    "summary": False,
                }
            )

    # Forest plot
    fig, ax = plt.subplots(figsize=(9, len(rows) * 0.6 + 2))
    y_pos = list(range(len(rows) - 1, -1, -1))  # top-to-bottom

    for pos, row in zip(y_pos, rows):
        color = "#2c7bb6" if row["summary"] else "#333333"
        marker = "D" if row["summary"] else "o"
        ms = 9 if row["summary"] else 7

        xerr_low = row["point"] - row["ci_lower"]
        xerr_high = row["ci_upper"] - row["point"]

        ax.errorbar(
            row["point"],
            pos,
            xerr=[[xerr_low], [xerr_high]],
            fmt=marker,
            color=color,
            markersize=ms,
            capsize=4,
            linewidth=1.5,
            elinewidth=1.5,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=10)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("Metric Value", fontsize=11)
    ax.set_title(
        "Classification Metrics — Forest Plot with 95% Bootstrap CI\n"
        "(smoke test · ternary task · n=80)",
        fontsize=12,
    )
    ax.set_xlim(-0.15, 1.05)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    summary_patch = mpatches.Patch(color="#2c7bb6", label="Summary metric (◆)")
    perclass_patch = mpatches.Patch(color="#333333", label="Per-class metric (●)")
    ax.legend(handles=[summary_patch, perclass_patch], fontsize=9, loc="lower right")

    plt.tight_layout()
    # plt.savefig('smoke_forest_plot.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved → smoke_forest_plot.png")
