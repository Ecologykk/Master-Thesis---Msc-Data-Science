"""Extract decision labels from case-summary text.

This module replaces `class_re_engineering.py` from the companion data repo entirely, rather
than porting it. That script implemented `_extract_decision_binary_from_summary`: the notebook
(`src/data_processing_exploration/processing_and_eda.ipynb`, cell 22) documents that this
function "had a bug, where it misclassified some partial decisions as favorable, since it
prioritized favorable pattern search, which in turn created mismatches with the ternary
dataset." It also emitted `FAVORÁVEL`/`DESFAVORÁVEL` for both DV and BoC cases, predating the
DV -> MANTIDA/ALTERADA redefinition used in the thesis. Porting it would silently reproduce
different numbers than the thesis reports.

The two functions below (`extract_decision_binary_from_summary_dv` and
`extract_decision_ternary_from_summary_boc`, notebook cells 23-24) are the corrected replacements
and the only label functions used anywhere in this pipeline. The fix: check for partial-decision
language *first*, before checking for favorable/altered language, so a partial win is never
misread as a full one.

Known edge case (disclosed in the thesis, section 3.x / Limitations): pattern order matters for
phrases like "parcialmente improcedente" ("partially unsuccessful"). In both functions below, the
negation/unfavorable check (PRIORITY 0) runs *before* the partial check (PRIORITY 1), so a phrase
combining "parcial" and "improcedente" matches the negation pattern first and is classified as
MANTIDA / DESFAVORÁVEL, even though a human reader would likely call it a partial outcome. The
thesis names this exact case for `extract_decision_ternary_from_summary_boc`: "an expression such
as *parcialmente improcedente* matches the negation pattern first and is classified as UNFAVORABLE
rather than PARTIAL." This is intentional, documented pipeline behaviour, not a bug to fix here —
the manuscript and this code must not disagree about it.
"""

from __future__ import annotations

import re

import pandas as pd


def extract_decision_binary_from_summary_dv(summary_decision: str | None) -> str | None:
    """Classify a Domestic Violence (DV) case summary into a binary outcome label.

    Args:
        summary_decision: Short decision summary text (the `decisao` column).

    Returns:
        "DECISÃO ALTERADA" if the prior decision was changed (including partial changes),
        "DECISÃO MANTIDA" if it was upheld, or None if no pattern matched.
    """
    if not summary_decision or pd.isna(summary_decision):
        return None

    text = str(summary_decision).lower().strip()
    text_clean = re.sub(r"[^\w\s]", " ", text)

    # PRIORITY 0: negated alteration (must come first)
    negated_alteration_patterns = [
        r"\bnegado\s+provimento\b",
        r"\bn[aã]o\s+provido\b",
        r"\bdesprovido\b",
        r"\bnega[rd]\s+provimento\b",
        r"\brecurso\s+improcedente\b",
        r"\bapela[cç][aã]o\s+improcedente\b",
        r"\bimprocedente\b",
        r"\bimprocedência\b",
    ]
    for pattern in negated_alteration_patterns:
        if re.search(pattern, text_clean):
            return "DECISÃO MANTIDA"

    # PRIORITY 1: affirmative alteration (including partials)
    altered_patterns = [
        r"\bparcialmente\s+provido\b",
        r"\bprovido\s+parcialmente\b",
        r"\bprocedente\s+em\s+parte\b",
        r"\bparcial\b",
        r"\bparcialmente\b",
        r"\bem\s+parte\b",
        r"\bprovido\b",
        r"\bprovimento\b",
        r"\bprocedente\b",
        r"\bconcedid[oa]\b",
        r"\balterad[oa]\b",
        r"\balterar\b",
        r"\brevogad[oa]\b",
        r"\brevoga[rd]\b",
        r"\breformad[oa]\b",
    ]
    for pattern in altered_patterns:
        if re.search(pattern, text_clean):
            return "DECISÃO ALTERADA"

    # PRIORITY 2: explicit maintenance
    kept_patterns = [
        r"\bmantid[oa]\b",
        r"\bconfirmad[oa]\b",
        r"\bconfirma[rd]\b",
        r"\bconfirma[çc][aã]o\b",
    ]
    for pattern in kept_patterns:
        if re.search(pattern, text_clean):
            return "DECISÃO MANTIDA"

    return None


def extract_decision_ternary_from_summary_boc(
    summary_decision: str | None,
) -> str | None:
    """Classify a Breach of Contract (BoC) case summary into a ternary outcome label.

    Args:
        summary_decision: Short decision summary text (the `decisao` column).

    Returns:
        "FAVORÁVEL", "DESFAVORÁVEL", "PARCIAL", or None if no pattern matched.
    """
    if not summary_decision or pd.isna(summary_decision):
        return None

    text = str(summary_decision).lower().strip()
    text_clean = re.sub(r"[^\w\s]", " ", text)

    # PRIORITY 0: explicit negation of relief/procedence
    negated_favorable_patterns = [
        r"\bn[aã]o\s+procede\b",
        r"\bn[aã]o\s+procedente\b",
        r"\bnega[rd]\s+provimento\b",
        r"\bnegado\s+provimento\b",
        r"\bn[aã]o\s+provido\b",
        r"\bdesprovido\b",
        r"\brecurso\s+improcedente\b",
        r"\bapela[cç][aã]o\s+improcedente\b",
        r"\bimprocedente\b",
        r"\bimprocedência\b",
    ]
    for pattern in negated_favorable_patterns:
        if re.search(pattern, text_clean):
            return "DESFAVORÁVEL"

    # PRIORITY 1: partial
    partial_patterns = [
        r"\bparcial\b",
        r"\bparcialmente\b",
        r"\bem\s+parte\b",
        r"\bprocedente\s+em\s+parte\b",
        r"\bprovido\s+parcialmente\b",
        r"\bconcedid[oa]\s+parcialmente\b",
        r"\brevogad[oa]\s+parcialmente\b",
        r"\bconfirmad[oa]\s+em\s+parte\b",
    ]
    for pattern in partial_patterns:
        if re.search(pattern, text_clean):
            return "PARCIAL"

    # PRIORITY 2: favorable (affirmative)
    favorable_patterns = [
        r"\btotalmente\s+procedente\b",
        r"\bprocedente\b",
        r"\bprovido\b",
        r"\bprovimento\b",
        r"\bconcedid[oa]\b",
        r"\brevogad[oa]\b",
        r"\brevog[aou]?\b",
    ]
    for pattern in favorable_patterns:
        if re.search(pattern, text_clean):
            return "FAVORÁVEL"

    # PRIORITY 3: explicit unfavorable
    unfavorable_patterns = [
        r"\bmantid[oa]\b",
        r"\bconfirmad[oa]\b",
        r"\bconfirma[çc][aã]o\b",
        r"\bnega[rd]\b",
        r"\bnegad[oa]\b",
    ]
    for pattern in unfavorable_patterns:
        if re.search(pattern, text_clean):
            return "DESFAVORÁVEL"

    return None


# case_type -> (output column name, label function)
_LABEL_FUNCTIONS = {
    "dv": ("decisao_binaria", extract_decision_binary_from_summary_dv),
    "boc": ("decisao_ternaria", extract_decision_ternary_from_summary_boc),
}


def apply_labels(
    df: pd.DataFrame, case_type: str, decision_col: str = "decisao"
) -> pd.DataFrame:
    """Add the appropriate decision-label column and drop rows that could not be labelled.

    Args:
        df: Cleaned DataFrame with a `decisao` column (from `clean.clean`).
        case_type: Either "dv" (uses the binary DV labeller) or "boc" (uses the ternary
            BoC labeller).
        decision_col: Name of the column containing the decision summary text to classify.

    Returns:
        Copy of `df` with a new label column (`decisao_binaria` for dv, `decisao_ternaria`
        for boc) added, and rows where no label could be inferred dropped.

    Raises:
        ValueError: If `case_type` is not "dv" or "boc".
    """
    if case_type not in _LABEL_FUNCTIONS:
        raise ValueError(f"Unknown case_type '{case_type}'. Expected 'dv' or 'boc'.")

    label_col, label_fn = _LABEL_FUNCTIONS[case_type]
    df = df.copy()
    df[label_col] = df[decision_col].apply(label_fn)
    return df.dropna(subset=[label_col])
