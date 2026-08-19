"""Clean an aggregated case DataFrame and split it into classification/EDA-ready views.

Ported from `raw_processed_data_git/Msc-Data-Science-Dissertation-Data/scripts/cleaning.py` and
`data_information.py`, then corrected against the notebook
(`src/data_processing_exploration/processing_and_eda.ipynb`, cells 7-19), which is the current
source of truth. Two corrections relative to the old script:

1. The module-level `type_case = "DV"` global is replaced by a `case_type` function argument.
2. `_align_indexes_by_url`'s `df.set_index(..., inplace=True)` bug (which silently set `df` to
   `None`, since `inplace=True` returns `None`) is not ported. The notebook's own row-alignment
   step (`backfill_missing_decisions` below) does the equivalent `set_index`/`update`/`reset_index`
   sequence correctly, as a statement rather than a reassignment.
"""

from __future__ import annotations

import re

import pandas as pd

# Columns dropped after class re-engineering, shared by both output views.
_COMMON_DROP_COLUMNS = [
    "tipo_direito",
    "tipo_caso",
    "sumario",
    "votacao",
    "meio_processual",
    "texto_integral_disponivel",
    "texto_integral_completo",
    "decisao_extraida_do_texto_integral",
    "metadata_decisao_extraction_method",
    "metadata_decisao_confidence",
    "metadata_decisao_keyword_found",
    "metadata_decisao_requires_manual_review",
    "metadata_decisao_review_reason",
    "metadata_decisao_document_position",
    "decisao_resumo_extraida",
]

# The classification view additionally drops identifying/provenance columns not needed for
# modelling. The EDA view keeps them for descriptive analysis.
_CLASSIFICATION_ONLY_DROP_COLUMNS = [
    "url",
    "tribunal",
    "n_processo",
    "juiz_relator",
    "data_acordao",
]


def _drop_empty_cases_from_non_peace_courts(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows missing `texto_integral_disponivel`, except for peace courts (`JP_*`).

    Peace courts (Julgados de Paz) are kept intact even when this field is missing, since
    imputing it is not worth the effort for the small number of cases affected.

    Args:
        df: Input DataFrame containing case data, with a `tribunal` column.

    Returns:
        Filtered DataFrame with the specified rows dropped, index restored to original order.
    """
    initial_count = len(df)

    df_jp = df[df["tribunal"].str.startswith("JP_", na=False)].copy()
    df_non_jp = df[~df["tribunal"].str.startswith("JP_", na=False)].copy()

    df_non_jp = df_non_jp.dropna(subset=["texto_integral_disponivel"], how="all")

    df = pd.concat([df_non_jp, df_jp], axis=0).sort_index()

    dropped_count = initial_count - len(df)
    print(
        f"Dropped {dropped_count} cases from non-peace courts due to missing "
        "'texto_integral_disponivel'."
    )
    return df


def _extract_decision_summary(verbose_decision: str | None) -> str | None:
    """Infer a short decision summary from a verbose decision text.

    Used to backfill the `decisao` column when it is empty but a longer decision text
    (`decisao_extraida_do_texto_integral`) is available. Partial-decision language is checked
    first so that e.g. "provido parcialmente" is not misread as a full "provido".

    Args:
        verbose_decision: Free-text decision extracted from the full case text, or None.

    Returns:
        One of "PARCIALMENTE PROCEDENTE", "REVOGADA", "NEGADO PROVIMENTO", or None if no
        pattern matched.
    """
    if not verbose_decision:
        return None

    text = verbose_decision.lower()
    text_clean = re.sub(r"[^\w\s]", " ", text)

    partial_patterns = [
        r"\bparcial",
        r"\bparcialmente",
        r"\bem\s+parte",
        r"\brevogad[oa]\s+parcial",
        r"\bprocedente\s+parcial",
        r"\bprovido\s+parcial",
        r"\bconcedid[oa]\s+parcial",
    ]
    for pattern in partial_patterns:
        if re.search(pattern, text_clean):
            return "PARCIALMENTE PROCEDENTE"

    favorable_patterns = [
        r"\brevogad[oa](?!\s*parcial)",
        r"\brevoga[rd](?!\s*parcial)",
        r"\bprocedente(?!\s*(?:parcial|em\s*parte))",
        r"\btotalmente\s+procedente",
        r"\bprovido(?!\s*parcial)",
        r"\bprovimento(?!\s*parcial)",
        r"\bconcedid[oa](?!\s*parcial)",
        r"\balterad[oa]",
        r"\balterar",
        r"\bcondenad[oa]",
    ]
    for pattern in favorable_patterns:
        if re.search(pattern, text_clean):
            return "REVOGADA"

    unfavorable_patterns = [
        r"\bnegad[oa](?!\s*parcial)",
        r"\bnega[rd](?!\s*parcial)",
        r"\bconfirmad[oa](?!\s*parcial)",
        r"\bconfirma[rd](?!\s*parcial)",
        r"\bconfirma[çc][aã]o(?!\s*parcial)",
        r"\bmantid[oa](?!\s*parcial)",
        r"\bimprocedente(?!\s*(?:parcial|em\s*parte))",
        r"\bimprocedência",
        r"\bapela[cç][aã]o\s+improcedente",
        r"\bnegado\s+provimento(?!\s*parcial)",
        r"\brecurso.*improcedente",
    ]
    for pattern in unfavorable_patterns:
        if re.search(pattern, text_clean):
            return "NEGADO PROVIMENTO"

    return None


def backfill_missing_decisions(df: pd.DataFrame) -> pd.DataFrame:
    """Recover missing `decisao` values from `decisao_extraida_do_texto_integral` where possible.

    Aligns two views of the same rows by `url` (the full set, and the subset with a non-missing
    `decisao_extraida_do_texto_integral`), then applies `_extract_decision_summary` to any row
    still missing `decisao`, writing the inferred value into both `decisao` and
    `decisao_resumo_extraida`.

    Args:
        df: Input DataFrame, after the peace-court carve-out has been applied.

    Returns:
        DataFrame with `decisao` backfilled where a pattern could be inferred.
    """
    df_with_extracted_text = df.dropna(
        subset=["decisao_extraida_do_texto_integral"], how="all"
    )

    df = df.copy()
    df.set_index("url", inplace=True)
    df_with_extracted_text = df_with_extracted_text.set_index("url")
    df.update(df_with_extracted_text)
    df.reset_index(inplace=True)

    mask = df["decisao"].isnull() | df["decisao"].astype(str).str.strip().eq("")
    for index in df[mask].index:
        verbose_decision = df.at[index, "decisao_extraida_do_texto_integral"]
        summary_decision = _extract_decision_summary(verbose_decision)
        df.at[index, "decisao"] = summary_decision
        df.at[index, "decisao_resumo_extraida"] = summary_decision

    return df


def _drop_cases_with_no_full_text(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where `texto_integral_disponivel` is exactly "N" (no full text available).

    Args:
        df: Input DataFrame containing case data.

    Returns:
        Filtered DataFrame with those rows dropped. Rows with a missing (NaN) value are kept,
        since NaN never compares equal to the literal string "N".
    """
    return df[df["texto_integral_disponivel"] != "N"]


def _drop_missing_extracted_decisions_and_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows still missing `decisao_extraida_do_texto_integral`, then duplicate cases.

    Duplicates are identified by `n_processo` rather than `url`, since some cases were scraped
    from more than one source database under different URLs.

    Args:
        df: Input DataFrame containing case data.

    Returns:
        Filtered DataFrame with the specified rows dropped.
    """
    df = df.dropna(subset=["decisao_extraida_do_texto_integral"], how="all")
    return df.drop_duplicates(subset=["n_processo"])


def build_classification_and_eda_frames(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a cleaned DataFrame into a classification-ready view and an EDA-ready view.

    The classification view keeps only the columns needed for modelling
    (`descritores`, `decisao`, `texto_integral_sem_decisao`). The EDA view additionally keeps
    identifying/provenance columns (`url`, `tribunal`, `n_processo`, `juiz_relator`,
    `data_acordao`) for descriptive analysis and later gold-test-set construction.

    Args:
        df: Cleaned DataFrame, after `backfill_missing_decisions` and duplicate removal.

    Returns:
        Tuple of (df_classification, df_eda).
    """
    df_classification = df.drop(
        columns=_COMMON_DROP_COLUMNS + _CLASSIFICATION_ONLY_DROP_COLUMNS,
        errors="ignore",
    )
    df_eda = df.drop(columns=_COMMON_DROP_COLUMNS, errors="ignore")
    return df_classification, df_eda


def clean(df: pd.DataFrame, case_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full cleaning pipeline on an aggregated raw DataFrame.

    Args:
        df: Aggregated raw DataFrame for one case type (output of `aggregate.aggregate_case_type`).
        case_type: Either "dv" or "boc". Used only for logging.

    Returns:
        Tuple of (df_classification, df_eda), both still missing the final decision label
        column, which `labels.apply_labels` adds.
    """
    print(f"Starting data cleaning for case_type='{case_type}'...")

    df = _drop_empty_cases_from_non_peace_courts(df)
    df = backfill_missing_decisions(df)
    df = _drop_cases_with_no_full_text(df)
    df = _drop_missing_extracted_decisions_and_duplicates(df)

    df_classification, df_eda = build_classification_and_eda_frames(df)

    print(f"Data cleaning completed: {len(df_eda)} cases remain.")
    return df_classification, df_eda
