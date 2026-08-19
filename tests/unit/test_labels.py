"""Unit tests for src.data_processing.labels — the decision-label classifiers.

Covers both label functions against hand-written Portuguese decision strings, and pins down the
"parcialmente improcedente" edge case disclosed in the thesis: because the negation/unfavorable
pattern check runs *before* the partial-decision check in both functions, a phrase combining
"parcial" and "improcedente" is classified as MANTIDA / DESFAVORÁVEL, not as a partial outcome.
This is documented, intentional pipeline behaviour — see the module docstring in
src/data_processing/labels.py — and this test exists so that behaviour cannot silently drift
without a manuscript update to match.
"""

from __future__ import annotations

import pytest

from src.data_processing.labels import (
    extract_decision_binary_from_summary_dv,
    extract_decision_ternary_from_summary_boc,
)


class TestExtractDecisionBinaryFromSummaryDV:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Recurso improcedente.", "DECISÃO MANTIDA"),
            ("Negado provimento ao recurso.", "DECISÃO MANTIDA"),
            ("Apelação improcedente.", "DECISÃO MANTIDA"),
            ("Não provido.", "DECISÃO MANTIDA"),
            ("Desprovido.", "DECISÃO MANTIDA"),
        ],
    )
    def test_kept_via_explicit_negation(self, text, expected):
        assert extract_decision_binary_from_summary_dv(text) == expected

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Recurso provido.", "DECISÃO ALTERADA"),
            ("Decisão alterada.", "DECISÃO ALTERADA"),
            ("Sentença revogada.", "DECISÃO ALTERADA"),
            ("Procedente.", "DECISÃO ALTERADA"),
            ("Provido parcialmente.", "DECISÃO ALTERADA"),
            ("Procedente em parte.", "DECISÃO ALTERADA"),
        ],
    )
    def test_altered_including_partials(self, text, expected):
        assert extract_decision_binary_from_summary_dv(text) == expected

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Decisão mantida.", "DECISÃO MANTIDA"),
            ("Sentença confirmada.", "DECISÃO MANTIDA"),
        ],
    )
    def test_kept_via_explicit_maintenance(self, text, expected):
        assert extract_decision_binary_from_summary_dv(text) == expected

    def test_parcialmente_improcedente_is_classified_as_kept(self):
        """Disclosed edge case: negation check runs before partial, so this reads as MANTIDA."""
        assert (
            extract_decision_binary_from_summary_dv("Parcialmente improcedente.")
            == "DECISÃO MANTIDA"
        )

    @pytest.mark.parametrize("value", [None, "", "   ", float("nan")])
    def test_empty_or_missing_input_returns_none(self, value):
        assert extract_decision_binary_from_summary_dv(value) is None

    def test_unrecognised_text_returns_none(self):
        assert (
            extract_decision_binary_from_summary_dv("O tribunal reuniu na terça-feira.")
            is None
        )


class TestExtractDecisionTernaryFromSummaryBOC:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Recurso improcedente.", "DESFAVORÁVEL"),
            ("Não procede.", "DESFAVORÁVEL"),
            ("Negado provimento.", "DESFAVORÁVEL"),
        ],
    )
    def test_unfavourable_via_explicit_negation(self, text, expected):
        assert extract_decision_ternary_from_summary_boc(text) == expected

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Parcialmente procedente.", "PARCIAL"),
            ("Procedente em parte.", "PARCIAL"),
            ("Provido parcialmente.", "PARCIAL"),
        ],
    )
    def test_partial(self, text, expected):
        assert extract_decision_ternary_from_summary_boc(text) == expected

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Totalmente procedente.", "FAVORÁVEL"),
            ("Recurso provido.", "FAVORÁVEL"),
            ("Sentença revogada.", "FAVORÁVEL"),
        ],
    )
    def test_favourable(self, text, expected):
        assert extract_decision_ternary_from_summary_boc(text) == expected

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Decisão mantida.", "DESFAVORÁVEL"),
            ("Sentença confirmada.", "DESFAVORÁVEL"),
        ],
    )
    def test_unfavourable_via_explicit_maintenance(self, text, expected):
        assert extract_decision_ternary_from_summary_boc(text) == expected

    def test_parcialmente_improcedente_is_classified_as_unfavourable(self):
        """The thesis names this exact case: negation (PRIORITY 0) runs before the partial
        check (PRIORITY 1), so 'parcialmente improcedente' reads as DESFAVORÁVEL, not PARCIAL.
        """
        assert (
            extract_decision_ternary_from_summary_boc("Parcialmente improcedente.")
            == "DESFAVORÁVEL"
        )

    @pytest.mark.parametrize("value", [None, "", "   ", float("nan")])
    def test_empty_or_missing_input_returns_none(self, value):
        assert extract_decision_ternary_from_summary_boc(value) is None

    def test_unrecognised_text_returns_none(self):
        assert (
            extract_decision_ternary_from_summary_boc(
                "O tribunal reuniu na terça-feira."
            )
            is None
        )
