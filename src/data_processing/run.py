"""CLI entry point for the data-processing pipeline.

Usage:
    python -m src.data_processing.run --case-type {dv,boc} --stage {aggregate,clean,label,split,all}

Each stage recomputes everything before it in memory (aggregation and cleaning are cheap
relative to the LLM/BERT stages downstream), so any stage can be run independently. Only the
`label` and `split` stages persist output files, matching what the notebook actually wrote to
disk under `data/processed_data/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data_processing import aggregate, clean, labels, splits

_REPO_ROOT = Path(__file__).resolve().parents[2]

# case_type -> label schema name, used in output paths and filenames.
_SCHEMA_NAMES: dict[str, str] = {"dv": "binary", "boc": "ternary"}

CLASSIFICATION_DIR: Path = _REPO_ROOT / "data" / "processed_data" / "classification"
EDA_DIR: Path = _REPO_ROOT / "data" / "processed_data" / "eda"


def run_aggregate(case_type: str) -> pd.DataFrame:
    """Run the aggregate stage: load and concatenate raw JSON for one case type.

    Args:
        case_type: Either "dv" or "boc".

    Returns:
        Aggregated raw DataFrame.
    """
    return aggregate.aggregate_case_type(case_type)


def run_clean(case_type: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the aggregate + clean stages.

    Args:
        case_type: Either "dv" or "boc".

    Returns:
        Tuple of (df_raw_all, df_classification, df_eda).
    """
    df_raw_all = run_aggregate(case_type)
    df_classification, df_eda = clean.clean(df_raw_all, case_type)
    return df_raw_all, df_classification, df_eda


def run_label(
    case_type: str, persist: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the aggregate + clean + label stages, and persist the labelled CSVs.

    Args:
        case_type: Either "dv" or "boc".
        persist: If True, write the labelled classification/EDA CSVs to
            `data/processed_data/{classification,eda}/`.

    Returns:
        Tuple of (df_raw_all, df_classification_labelled, df_eda_labelled).
    """
    df_raw_all, df_classification, df_eda = run_clean(case_type)

    df_classification_labelled = labels.apply_labels(df_classification, case_type)
    df_eda_labelled = labels.apply_labels(df_eda, case_type)

    print(
        f"{case_type.upper()} labelled cases: {len(df_eda_labelled)} "
        f"(thesis-reported count: {1126 if case_type == 'dv' else 1802})"
    )

    if persist:
        schema = _SCHEMA_NAMES[case_type]

        class_dir = CLASSIFICATION_DIR / schema
        class_dir.mkdir(parents=True, exist_ok=True)
        df_classification_labelled.to_csv(
            class_dir / f"df_acordaos_{case_type}_classification_{schema}.csv",
            index=False,
        )

        eda_dir = EDA_DIR / schema
        eda_dir.mkdir(parents=True, exist_ok=True)
        df_eda_labelled.to_csv(
            eda_dir / f"df_acordaos_{case_type}_eda_{schema}.csv", index=False
        )

    return df_raw_all, df_classification_labelled, df_eda_labelled


def run_split(case_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full pipeline through the split stage, persisting all intermediate outputs.

    Args:
        case_type: Either "dv" or "boc".

    Returns:
        Tuple of (gold_test_df, train_df).
    """
    df_raw_all, _, df_eda_labelled = run_label(case_type, persist=True)
    return splits.build_gold_test_set(case_type, df_eda_labelled, df_raw_all)


_STAGE_RUNNERS = {
    "aggregate": run_aggregate,
    "clean": run_clean,
    "label": run_label,
    "split": run_split,
    "all": run_split,
}


def main() -> None:
    """Parse CLI arguments and run the requested pipeline stage."""
    parser = argparse.ArgumentParser(
        description="Run the case data-processing pipeline."
    )
    parser.add_argument("--case-type", choices=["dv", "boc"], required=True)
    parser.add_argument(
        "--stage",
        choices=["aggregate", "clean", "label", "split", "all"],
        required=True,
    )
    args = parser.parse_args()

    _STAGE_RUNNERS[args.stage](args.case_type)


if __name__ == "__main__":
    main()
