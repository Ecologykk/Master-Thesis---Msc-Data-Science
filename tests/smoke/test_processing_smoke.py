"""Smoke test: run the full data-processing pipeline on a small real-case slice.

Uses tests/fixtures/sample_{dv,boc}_cases.json — 20 real cases per type, with long text fields
truncated to keep the fixture small (see conftest.py). Full-size runs against real raw data are
covered by tests/test_reproduction.py; this test just confirms the pipeline runs end to end and
produces sane output, fast enough to run on every commit.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_processing import aggregate, clean, labels

EXPECTED_LABEL_COLUMN = {"dv": "decisao_binaria", "boc": "decisao_ternaria"}


@pytest.mark.parametrize(
    "case_type,fixture_name",
    [("dv", "sample_dv_cases.json"), ("boc", "sample_boc_cases.json")],
)
def test_pipeline_runs_end_to_end_on_small_real_slice(
    case_type, fixture_name, fixtures_dir
):
    fixture_path = fixtures_dir / fixture_name
    assert fixture_path.exists(), f"missing fixture: {fixture_path}"

    df_raw = aggregate.load_case_dataframes([fixture_path])[0]
    assert len(df_raw) == 20

    df_classification, df_eda = clean.clean(df_raw, case_type)
    assert len(df_classification) <= len(df_raw)
    assert len(df_classification) == len(df_eda)

    df_eda_labelled = labels.apply_labels(df_eda, case_type)
    label_col = EXPECTED_LABEL_COLUMN[case_type]
    assert label_col in df_eda_labelled.columns
    assert (
        len(df_eda_labelled) > 0
    ), "expected at least one labellable case in the fixture"
    assert df_eda_labelled[label_col].isna().sum() == 0

    # texto_integral_completo (with decision) must never reach the labelled output — only the
    # leakage-safe texto_integral_sem_decisao should survive.
    assert "texto_integral_completo" not in df_eda_labelled.columns
    assert "texto_integral_sem_decisao" in df_eda_labelled.columns
