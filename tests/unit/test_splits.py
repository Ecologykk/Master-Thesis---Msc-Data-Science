"""Unit tests for src.data_processing.splits — cutoff/split arithmetic on synthetic frames.

No real data or network access is used.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_processing.splits import attach_full_text, split_by_cutoff


def _dated_df(dates: list[str | None], n: int | None = None) -> pd.DataFrame:
    """Build a minimal DataFrame with a `data_acordao` column (day-first date strings)."""
    return pd.DataFrame(
        {
            "url": [f"http://example.com/{i}" for i in range(len(dates))],
            "data_acordao": dates,
        }
    )


class TestSplitByCutoff:
    def test_test_set_contains_only_cases_on_or_after_cutoff(self):
        dates = ["01-06-2025", "15-05-2025", "30-04-2025", "01-01-2020"]
        df = _dated_df(dates)
        test_df, train_df = split_by_cutoff(df, "data_acordao", "2025-05-01", n=2)
        assert len(test_df) == 2
        assert set(test_df["data_acordao"]) == {"01-06-2025", "15-05-2025"}

    def test_test_set_is_the_n_most_recent_cases(self):
        dates = ["01-06-2025", "15-05-2025", "02-05-2025"]
        df = _dated_df(dates)
        test_df, _ = split_by_cutoff(df, "data_acordao", "2025-05-01", n=1)
        assert list(test_df["data_acordao"]) == ["01-06-2025"]

    def test_train_set_excludes_test_cases(self):
        dates = ["01-06-2025", "15-05-2025", "30-04-2025", "01-01-2020"]
        df = _dated_df(dates)
        test_df, train_df = split_by_cutoff(df, "data_acordao", "2025-05-01", n=2)
        assert set(test_df["url"]).isdisjoint(set(train_df["url"]))
        assert set(train_df["data_acordao"]) == {"30-04-2025", "01-01-2020"}

    def test_missing_dates_included_in_train_by_default(self):
        dates = ["01-06-2025", "15-05-2025", None]
        df = _dated_df(dates)
        test_df, train_df = split_by_cutoff(
            df, "data_acordao", "2025-05-01", n=1, include_missing_in_train=True
        )
        # Both dated cases are >= cutoff (in the test set or excluded), so train_df here is
        # just the single missing-date row folded in.
        assert len(train_df) == 1
        assert train_df["data_acordao"].isna().sum() == 1

    def test_missing_dates_excluded_when_requested(self):
        dates = ["01-06-2025", "15-05-2025", None]
        df = _dated_df(dates)
        test_df, train_df = split_by_cutoff(
            df, "data_acordao", "2025-05-01", n=1, include_missing_in_train=False
        )
        assert train_df["data_acordao"].isna().sum() == 0

    def test_raises_when_fewer_than_n_cases_available(self):
        dates = ["01-06-2025"]
        df = _dated_df(dates)
        with pytest.raises(ValueError, match="Not enough cases"):
            split_by_cutoff(df, "data_acordao", "2025-05-01", n=5)

    def test_internal_helper_column_present_before_caller_drops_it(self):
        """`_acordao_date` is an internal helper the caller is expected to drop before saving;
        confirm it exists so callers relying on it (e.g. splits.build_gold_test_set) don't break.
        """
        dates = ["01-06-2025", "01-01-2020"]
        df = _dated_df(dates)
        test_df, train_df = split_by_cutoff(df, "data_acordao", "2025-05-01", n=1)
        assert "_acordao_date" in test_df.columns
        assert "_acordao_date" in train_df.columns


class TestAttachFullText:
    def test_merges_text_column_by_url(self):
        test_df = pd.DataFrame({"url": ["a", "b"]})
        df_all = pd.DataFrame(
            {
                "url": ["a", "b", "c"],
                "texto_integral_completo": ["texto a", "texto b", "texto c"],
            }
        )
        result = attach_full_text(test_df, df_all, name="TEST")
        assert list(result["texto_integral_completo"]) == ["texto a", "texto b"]

    def test_raises_on_duplicate_urls_in_source(self):
        test_df = pd.DataFrame({"url": ["a"]})
        df_all = pd.DataFrame(
            {"url": ["a", "a"], "texto_integral_completo": ["texto 1", "texto 2"]}
        )
        with pytest.raises(ValueError, match="duplicate urls"):
            attach_full_text(test_df, df_all, name="TEST")

    def test_raises_when_a_test_row_has_no_match(self):
        test_df = pd.DataFrame({"url": ["a", "missing"]})
        df_all = pd.DataFrame({"url": ["a"], "texto_integral_completo": ["texto a"]})
        with pytest.raises(ValueError, match="missing full text"):
            attach_full_text(test_df, df_all, name="TEST")

    def test_raises_on_missing_column(self):
        test_df = pd.DataFrame({"url": ["a"]})
        df_all = pd.DataFrame({"url": ["a"]})  # no texto_integral_completo column
        with pytest.raises(KeyError):
            attach_full_text(test_df, df_all, name="TEST")
