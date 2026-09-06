"""The reproduction gate: regenerating the dataset must match the thesis-reported counts exactly.

Marked @pytest.mark.slow (excluded from the default `pytest` run) because it processes the full
raw corpus, not a fixture slice. Requires `data/raw_data/*.json` to be present — either the
author's local copy, or the archived corpus downloaded from the Zenodo record (see
docs/09-reproducibility-notes.md). Skipped automatically if that directory is empty or missing,
since a fresh clone without the corpus should not fail this test, only report that it could not
run.

This test makes permanent, machine-checked what Slice 3 of the E32 refactor originally verified
by hand: aggregating, cleaning, and labelling the raw corpus produces exactly 1,126 DV cases and
1,802 BoC cases — the counts the thesis reports — and the gold test sets committed under
data/processed_data/gold_test/ still match the freshly regenerated ones, row for row.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data_processing import splits
from src.data_processing.aggregate import RAW_DATA_DIR
from src.data_processing.run import run_label

EXPECTED_COUNTS = {"dv": 1126, "boc": 1802}

GOLD_TEST_DIR = (
    Path(__file__).resolve().parents[1] / "data" / "processed_data" / "gold_test"
)

pytestmark = pytest.mark.slow

_raw_data_available = RAW_DATA_DIR.exists() and any(RAW_DATA_DIR.glob("*.json"))


@pytest.mark.skipif(
    not _raw_data_available,
    reason=(
        f"No raw data found under {RAW_DATA_DIR}. Download the archived corpus from the "
        "Zenodo record (see docs/09-reproducibility-notes.md) to run this test."
    ),
)
@pytest.mark.parametrize("case_type", ["dv", "boc"])
def test_labelled_case_count_matches_thesis(case_type):
    _, _, df_eda_labelled = run_label(case_type, persist=False)
    assert len(df_eda_labelled) == EXPECTED_COUNTS[case_type], (
        f"Reproduction drift for {case_type}: got {len(df_eda_labelled)} cases, "
        f"thesis reports {EXPECTED_COUNTS[case_type]}. This is a reproducibility finding, "
        "not something to silently tune the pipeline to match."
    )


@pytest.mark.skipif(
    not _raw_data_available,
    reason=(
        f"No raw data found under {RAW_DATA_DIR}. Download the archived corpus from the "
        "Zenodo record (see docs/09-reproducibility-notes.md) to run this test."
    ),
)
@pytest.mark.parametrize("case_type", ["dv", "boc"])
def test_gold_test_set_matches_committed_file_row_for_row(case_type, tmp_path):
    committed_path = GOLD_TEST_DIR / f"{case_type}_gold_test_full.csv"
    if not committed_path.exists():
        pytest.skip(
            f"No committed gold test file at {committed_path} to compare against."
        )

    df_raw_all, _, df_eda_labelled = run_label(case_type, persist=False)
    regenerated, _ = splits.build_gold_test_set(
        case_type,
        df_eda_labelled,
        df_raw_all,
        splits_dir=tmp_path / "splits",
        gold_test_dir=tmp_path / "gold_test",
    )

    committed = pd.read_csv(committed_path)
    assert committed.equals(regenerated), (
        f"Regenerated {case_type} gold test set no longer matches the committed file row for "
        "row. This is a reproducibility finding, not something to silently tune the pipeline "
        "to match."
    )
