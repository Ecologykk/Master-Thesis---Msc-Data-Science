"""Split labelled cases into a gold test set and a training pool, by publication-date cutoff.

Ported from the notebook (`src/data_processing_exploration/processing_and_eda.ipynb`,
cells 42 and 46-50) — this stage has no equivalent in the companion data repo's scripts.

The cutoffs and test-set size are fixed constants, chosen to match the gold sets already
committed under `data/processed_data/gold_test/`:
- DV: 50 most recent cases on/after 2025-05-01.
- BoC: 50 most recent cases on/after 2023-10-01.
Cases with an unparseable date are kept out of the test pool and folded into training.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]

DV_CUTOFF = "2025-05-01"
BOC_CUTOFF = "2023-10-01"
N_TEST_CASES = 50

CUTOFFS: dict[str, str] = {"dv": DV_CUTOFF, "boc": BOC_CUTOFF}

# Columns dropped from the gold test set after the full text is re-attached: identifying/
# provenance columns not needed downstream, plus the internal `_acordao_date` helper column.
_GOLD_TEST_DROP_COLUMNS = [
    "tribunal",
    "juiz_relator",
    "data_acordao",
    "descritores",
    "decisao",
    "_acordao_date",
]

SPLITS_DIR: Path = _REPO_ROOT / "data" / "processed_data" / "splits" / "gold_test"
GOLD_TEST_DIR: Path = _REPO_ROOT / "data" / "processed_data" / "gold_test"


def split_by_cutoff(
    df: pd.DataFrame,
    date_col: str,
    cutoff_date: str,
    n: int = N_TEST_CASES,
    include_missing_in_train: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into a fixed-size recent-cases test set and an earlier-cases train pool.

    Args:
        df: Input DataFrame with a date column.
        date_col: Name of the column holding the case date (day-first format).
        cutoff_date: ISO date string. Only cases on/after this date are eligible for the test set.
        n: Number of most-recent eligible cases to place in the test set.
        include_missing_in_train: If True, rows with an unparseable date are added to the
            training pool rather than dropped.

    Returns:
        Tuple of (test_df, train_df). `test_df` is the `n` most recent cases on/after
        `cutoff_date`; `train_df` is every case strictly before it (plus, optionally, cases
        with a missing date).

    Raises:
        ValueError: If fewer than `n` cases are available on/after `cutoff_date`.
    """
    df = df.copy()
    df["_acordao_date"] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")

    cutoff = pd.to_datetime(cutoff_date)
    df_valid = df[df["_acordao_date"].notna()]
    df_missing = df[df["_acordao_date"].isna()]

    test_pool = df_valid[df_valid["_acordao_date"] >= cutoff].sort_values(
        "_acordao_date", ascending=False
    )
    if len(test_pool) < n:
        raise ValueError(
            f"Not enough cases on/after cutoff {cutoff_date}. Needed {n}, found {len(test_pool)}."
        )
    test_df = test_pool.head(n)

    train_df = df_valid[df_valid["_acordao_date"] < cutoff]
    if include_missing_in_train:
        train_df = pd.concat([train_df, df_missing], ignore_index=False)

    return test_df, train_df


def attach_full_text(
    test_df: pd.DataFrame,
    df_all: pd.DataFrame,
    name: str,
    url_col: str = "url",
    text_col: str = "texto_integral_completo",
) -> pd.DataFrame:
    """Merge the full case text (with decision) back onto a test split, matched by URL.

    The EDA view drops `texto_integral_completo` (it only keeps the leakage-safe
    `texto_integral_sem_decisao`), so this re-attaches it from the original aggregated
    DataFrame for the gold test set, where the complete text is wanted for reference.

    Args:
        test_df: Test split (output of `split_by_cutoff`), containing `url_col`.
        df_all: The full aggregated DataFrame for this case type, containing both `url_col`
            and `text_col`.
        name: Human-readable case-type name, used only in error messages.
        url_col: Column to join on.
        text_col: Column to pull from `df_all`.

    Returns:
        `test_df` with `text_col` merged in.

    Raises:
        KeyError: If a required column is missing from either input.
        ValueError: If `df_all` has duplicate URLs, or any test-split row fails to find a match.
    """
    for col in [url_col, text_col]:
        if col not in df_all.columns:
            raise KeyError(f"{name}: column '{col}' not found in df_all")
    if url_col not in test_df.columns:
        raise KeyError(f"{name}: column '{url_col}' not found in test_df")

    dup_count = df_all[url_col].duplicated().sum()
    if dup_count:
        raise ValueError(
            f"{name}: {dup_count} duplicate urls in df_all; cannot safely merge"
        )

    merged = test_df.merge(
        df_all[[url_col, text_col]], on=url_col, how="left", validate="one_to_one"
    )

    missing = merged[text_col].isna().sum()
    if missing:
        examples = merged.loc[merged[text_col].isna(), url_col].head(5).tolist()
        raise ValueError(
            f"{name}: {missing} cases missing full text. Example URLs: {examples}"
        )

    return merged


def build_gold_test_set(
    case_type: str,
    df_eda_labelled: pd.DataFrame,
    df_raw_all: pd.DataFrame,
    splits_dir: Path = SPLITS_DIR,
    gold_test_dir: Path = GOLD_TEST_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split, attach full text, and persist the gold test set and training pool for one case type.

    Args:
        case_type: Either "dv" or "boc".
        df_eda_labelled: EDA-view DataFrame after `labels.apply_labels` (has `url`,
            `data_acordao`, and the label column).
        df_raw_all: Aggregated raw DataFrame for the same case type (output of
            `aggregate.aggregate_case_type`), used to recover the full text with decision.
        splits_dir: Output directory for the intermediate train/test split (before full-text
            attachment).
        gold_test_dir: Output directory for the final gold test set (with full text attached).

    Returns:
        Tuple of (gold_test_df, train_df).

    Raises:
        ValueError: If `case_type` is not "dv" or "boc".
    """
    if case_type not in CUTOFFS:
        raise ValueError(f"Unknown case_type '{case_type}'. Expected 'dv' or 'boc'.")

    test_df, train_df = split_by_cutoff(
        df_eda_labelled, "data_acordao", CUTOFFS[case_type], n=N_TEST_CASES
    )
    print(f"{case_type.upper()} test size: {len(test_df)}, train size: {len(train_df)}")

    splits_dir.mkdir(parents=True, exist_ok=True)
    test_df.drop(columns=["_acordao_date"]).to_csv(
        splits_dir / f"{case_type}_gold_test.csv", index=False
    )
    train_df.drop(columns=["_acordao_date"]).to_csv(
        splits_dir / f"{case_type}_train_before_cutoff.csv", index=False
    )

    test_full = attach_full_text(test_df, df_raw_all, name=case_type.upper())
    test_full = test_full.drop(columns=_GOLD_TEST_DROP_COLUMNS, errors="ignore")

    gold_test_dir.mkdir(parents=True, exist_ok=True)
    test_full.to_csv(gold_test_dir / f"{case_type}_gold_test_full.csv", index=False)
    test_full.to_json(
        gold_test_dir / f"{case_type}_gold_test_full.json",
        orient="table",
        force_ascii=False,
        indent=4,
    )

    return test_full, train_df
