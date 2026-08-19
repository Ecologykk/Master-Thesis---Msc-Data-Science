"""Unit tests for src.data_processing.clean — cleaning predicates on synthetic frames.

No real data or network access is used; each test builds a tiny hand-written DataFrame with
just the columns the function under test needs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_processing.clean import (
    _drop_cases_with_no_full_text,
    _drop_empty_cases_from_non_peace_courts,
    _drop_missing_extracted_decisions_and_duplicates,
    _extract_decision_summary,
    build_classification_and_eda_frames,
)


class TestDropEmptyCasesFromNonPeaceCourts:
    def test_non_jp_row_missing_field_is_dropped(self):
        df = pd.DataFrame(
            {
                "tribunal": ["TRP", "TRL"],
                "texto_integral_disponivel": [np.nan, "S"],
            }
        )
        result = _drop_empty_cases_from_non_peace_courts(df)
        assert list(result["tribunal"]) == ["TRL"]

    def test_jp_row_missing_field_is_kept(self):
        df = pd.DataFrame(
            {
                "tribunal": ["JP_LISBOA", "TRP"],
                "texto_integral_disponivel": [np.nan, np.nan],
            }
        )
        result = _drop_empty_cases_from_non_peace_courts(df)
        assert list(result["tribunal"]) == ["JP_LISBOA"]

    def test_original_row_order_is_preserved(self):
        df = pd.DataFrame(
            {
                "tribunal": ["TRP", "JP_LISBOA", "TRL"],
                "texto_integral_disponivel": ["S", np.nan, "S"],
            }
        )
        result = _drop_empty_cases_from_non_peace_courts(df)
        assert list(result["tribunal"]) == ["TRP", "JP_LISBOA", "TRL"]


class TestDropCasesWithNoFullText:
    def test_rows_flagged_n_are_dropped(self):
        df = pd.DataFrame({"texto_integral_disponivel": ["S", "N", np.nan]})
        result = _drop_cases_with_no_full_text(df)
        # "N" is dropped; NaN is kept, since NaN never compares equal to "N".
        assert len(result) == 2
        assert "N" not in result["texto_integral_disponivel"].values

    def test_missing_value_is_kept(self):
        """Peace-court rows carved out earlier may have a missing flag; they must survive here."""
        df = pd.DataFrame({"texto_integral_disponivel": [np.nan]})
        result = _drop_cases_with_no_full_text(df)
        assert len(result) == 1


class TestDropMissingExtractedDecisionsAndDuplicates:
    def test_drops_missing_extracted_decision(self):
        df = pd.DataFrame(
            {
                "decisao_extraida_do_texto_integral": ["texto", np.nan],
                "n_processo": ["123/20", "456/20"],
            }
        )
        result = _drop_missing_extracted_decisions_and_duplicates(df)
        assert list(result["n_processo"]) == ["123/20"]

    def test_drops_duplicate_n_processo(self):
        df = pd.DataFrame(
            {
                "decisao_extraida_do_texto_integral": ["a", "b"],
                "n_processo": ["123/20", "123/20"],
            }
        )
        result = _drop_missing_extracted_decisions_and_duplicates(df)
        assert len(result) == 1


class TestExtractDecisionSummary:
    def test_none_input_returns_none(self):
        assert _extract_decision_summary(None) is None

    def test_empty_string_returns_none(self):
        assert _extract_decision_summary("") is None

    def test_partial_pattern_wins_over_favorable_pattern(self):
        """This is the exact bug the notebook documents against the old
        `extract_decision_binary_from_summary`: partial language must be checked before
        favorable language, so "provido parcialmente" is not misread as a full "provido".
        """
        assert (
            _extract_decision_summary("Recurso provido parcialmente.")
            == "PARCIALMENTE PROCEDENTE"
        )

    def test_favorable_pattern(self):
        assert _extract_decision_summary("Recurso julgado procedente.") == "REVOGADA"

    def test_unfavorable_pattern(self):
        # Note: "provimento" alone matches a favorable pattern regardless of "negado" preceding
        # it (this mirrors the notebook's ported logic exactly), so this case must avoid the
        # word "provimento" to actually exercise the unfavorable branch.
        assert _extract_decision_summary("Recurso negado.") == "NEGADO PROVIMENTO"

    def test_unmatched_text_returns_none(self):
        assert _extract_decision_summary("O tribunal reuniu na terça-feira.") is None


class TestBuildClassificationAndEdaFrames:
    def _sample_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "url": ["http://example.com/1"],
                "tribunal": ["TRP"],
                "n_processo": ["123/20"],
                "juiz_relator": ["Juiz A"],
                "data_acordao": ["01-01-2020"],
                "tipo_direito": ["Penal"],
                "tipo_caso": ["DV"],
                "descritores": [["a", "b"]],
                "votacao": ["unanimidade"],
                "meio_processual": ["recurso"],
                "decisao": ["DECISÃO MANTIDA"],
                "sumario": ["resumo"],
                "texto_integral_disponivel": ["S"],
                "texto_integral_completo": ["texto completo"],
                "decisao_extraida_do_texto_integral": ["negado provimento"],
                "texto_integral_sem_decisao": ["texto sem decisao"],
                "metadata_decisao_extraction_method": ["regex"],
                "decisao_resumo_extraida": ["NEGADO PROVIMENTO"],
            }
        )

    def test_classification_frame_drops_identifying_columns(self):
        df_classification, _ = build_classification_and_eda_frames(self._sample_df())
        for col in ["url", "tribunal", "n_processo", "juiz_relator", "data_acordao"]:
            assert col not in df_classification.columns

    def test_classification_frame_keeps_modelling_columns(self):
        df_classification, _ = build_classification_and_eda_frames(self._sample_df())
        for col in ["descritores", "decisao", "texto_integral_sem_decisao"]:
            assert col in df_classification.columns

    def test_eda_frame_keeps_identifying_columns(self):
        _, df_eda = build_classification_and_eda_frames(self._sample_df())
        for col in ["url", "tribunal", "n_processo", "juiz_relator", "data_acordao"]:
            assert col in df_eda.columns

    def test_both_frames_drop_full_text_with_decision(self):
        """texto_integral_completo (contains the decision) must not leak into either view;
        only the leakage-safe texto_integral_sem_decisao survives."""
        df_classification, df_eda = build_classification_and_eda_frames(
            self._sample_df()
        )
        assert "texto_integral_completo" not in df_classification.columns
        assert "texto_integral_completo" not in df_eda.columns
