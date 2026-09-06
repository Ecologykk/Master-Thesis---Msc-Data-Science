"""Few-shot example retrieval via embedding similarity.

Given a query case's sentence embeddings and a pool of labelled candidate
cases (grouped by decision class), scores and ranks candidates by
set-to-set embedding similarity, optionally diversifies the top picks to
avoid near-duplicate examples, and assembles a decision-class-balanced set
of few-shot examples for prompt construction. Also provides helpers to
group an enriched dataframe into candidate pools and to serialize the
resulting per-case example sets to JSON.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from config import canonicalize_label


@dataclass(frozen=True)
class CandidatePool:
    """Grouped candidate cases for one (case_type, decision_class) pair.

    Attributes:
        case_type: Case type ("dv" or "boc") the candidates belong to.
        decision_class: Decision class label shared by all candidates in
            this pool.
        candidate_ids: Candidate case identifiers (`n_processo`), in the
            pool's order.
        candidate_scores: Per-candidate scores aligned with `candidate_ids`.
            When built via `build_candidate_pools`, these are 0.0
            placeholders, since scoring against a query happens later.
        candidate_ranks: 1-based rank per candidate, aligned with
            `candidate_ids`.
    """

    case_type: str
    decision_class: str
    candidate_ids: list[str]
    candidate_scores: list[float]
    candidate_ranks: list[int]


def _ensure_2d_embeddings(value) -> np.ndarray:
    """Coerce a stored embeddings value (string repr, list, or ndarray) into a 2D float32 array."""
    if value is None:
        return np.zeros((0, 0), dtype=np.float32)
    if isinstance(value, str):
        value = ast.literal_eval(value)
    arr = np.asarray(
        value,
        dtype=object if isinstance(value, (list, tuple, np.ndarray)) else np.float32,
    )
    if isinstance(value, np.ndarray) and value.dtype != object:
        arr = np.asarray(value, dtype=np.float32)
        return arr.reshape(1, -1) if arr.ndim == 1 else arr
    items = value.tolist() if isinstance(value, np.ndarray) else list(value)
    if not items:
        return np.zeros((0, 0), dtype=np.float32)
    stacked = np.stack([np.asarray(v, dtype=np.float32) for v in items], axis=0)
    return stacked.astype(np.float32, copy=False)


def _ensure_seq(value) -> list:
    """Coerce a value (None, list, ndarray-like, or other iterable) into a plain list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        out = value.tolist()
        return out if isinstance(out, list) else [out]
    return list(value)


def _json_safe(value):
    """Convert numpy/pandas objects into JSON-serializable Python values."""
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def _rank_weights(n: int) -> np.ndarray:
    """Return normalized reciprocal-rank weights [1/1, 1/2, ..., 1/n], summing to 1."""
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)
    w = 1.0 / (np.arange(n, dtype=np.float32) + 1.0)
    return (w / w.sum()).astype(np.float32, copy=False)


def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the pairwise cosine similarity matrix between two sets of row vectors."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-9)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-9)
    return (a / a_norm) @ (b / b_norm).T


def set_to_set_similarity(
    query_embeddings: np.ndarray | list,
    candidate_embeddings: np.ndarray | list,
    *,
    query_ranks: Iterable[int] | None = None,
    candidate_ranks: Iterable[int] | None = None,
) -> float:
    """Compute a reciprocal-rank-weighted cosine similarity between two sentence-embedding sets.

    Weights each query sentence's row and each candidate sentence's row by
    its reciprocal rank (`1/(row_index + 1)`, normalized to sum to 1) via
    `_rank_weights`, then aggregates the pairwise cosine similarity matrix
    through those weights into a single score.

    Args:
        query_embeddings: Query case's sentence embeddings; string repr,
            list, or ndarray coercible to shape (n_query, hidden_size) via
            `_ensure_2d_embeddings`.
        candidate_embeddings: Candidate case's sentence embeddings, same
            accepted formats.
        query_ranks: Currently unused by this function — accepted for API
            symmetry with callers that track explicit sentence ranks, but
            the weighting below is derived from row position in
            `query_embeddings`, not from this argument.
        candidate_ranks: Currently unused (see `query_ranks`).

    Returns:
        Reciprocal-rank-weighted cosine similarity between the two
        embedding sets, in [-1, 1]. Returns 0.0 if either set is empty.
    """
    q = _ensure_2d_embeddings(query_embeddings)
    c = _ensure_2d_embeddings(candidate_embeddings)
    if q.size == 0 or c.size == 0:
        return 0.0

    sim = _cosine_similarity_matrix(q, c)
    qw = _rank_weights(q.shape[0])  # Using query and candidate rows for weights
    cw = _rank_weights(c.shape[0])
    score = float(qw @ sim @ cw)
    return score


def build_candidate_pools(
    df: pd.DataFrame,
    *,
    id_column: str = "n_processo",
    case_type_column: str = "case_type",
    decision_class_column: str = "decision_class",
    embeddings_column: str = "selected_sentence_embeddings",
    max_candidates_per_class: int | None = None,
) -> list[CandidatePool]:
    """Group candidate cases into one `CandidatePool` per (case_type, decision_class) pair.

    Args:
        df: DataFrame with one row per candidate case, containing at least
            `id_column`, `case_type_column`, `decision_class_column`, and
            `embeddings_column` (the embeddings column is only checked for
            presence here — its values are not read; scoring against a
            specific query happens later, in
            `score_candidates_against_query`).
        id_column: Column holding the candidate case identifier; rows within
            each group are sorted by this column.
        case_type_column: Column used, together with `decision_class_column`,
            to group candidates into pools.
        decision_class_column: Column used to group candidates by outcome
            class.
        embeddings_column: Name of the sentence-embeddings column; required
            to be present in `df` but not otherwise used by this function.
        max_candidates_per_class: Optional cap on candidates kept per
            (case_type, decision_class) group, applied after sorting by
            `id_column`.

    Returns:
        One `CandidatePool` per (case_type, decision_class) group found in
        `df`, with placeholder `candidate_scores` of 0.0 (populated by later
        scoring) and 1-based `candidate_ranks` in `id_column`-sorted order.

    Raises:
        ValueError: If `df` is missing any of the required columns.
    """
    required = {id_column, case_type_column, decision_class_column, embeddings_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    pools: list[CandidatePool] = []
    for (case_type, decision_class), group in df.groupby(
        [case_type_column, decision_class_column], dropna=False
    ):
        rows = group.sort_values(id_column).copy()
        if max_candidates_per_class is not None:
            rows = rows.head(max_candidates_per_class)
        pools.append(
            CandidatePool(
                case_type=str(case_type),
                decision_class=str(decision_class),
                candidate_ids=[str(v) for v in rows[id_column].tolist()],
                candidate_scores=[0.0 for _ in range(len(rows))],
                candidate_ranks=list(range(1, len(rows) + 1)),
            )
        )
    return pools


def filter_relevant_candidates(
    scored_candidates: pd.DataFrame,
    *,
    top_n: int = 20,
    id_column: str = "n_processo",
    similarity_column: str = "similarity_score",
) -> pd.DataFrame:
    """Keep the top-N candidates by similarity score and assign 1-based ranks.

    Args:
        scored_candidates: DataFrame with at least `id_column` and
            `similarity_column` (typically the output of
            `score_candidates_against_query`).
        top_n: Maximum number of candidates to keep.
        id_column: Candidate case identifier column (only checked for
            presence; not otherwise used).
        similarity_column: Column to sort by, descending.

    Returns:
        Copy of `scored_candidates` sorted by `similarity_column`
        descending, truncated to `top_n` rows, with a new 1-based
        `candidate_rank` column and a reset index.

    Raises:
        ValueError: If `top_n < 1`, or if `scored_candidates` is missing
            `id_column` or `similarity_column`.
    """
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    required = {id_column, similarity_column}
    missing = required - set(scored_candidates.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = (
        scored_candidates.sort_values(similarity_column, ascending=False)
        .head(top_n)
        .copy()
    )
    out["candidate_rank"] = range(1, len(out) + 1)
    return out.reset_index(drop=True)


def relevance_filter_for_query(
    query_row: pd.Series,
    candidate_df: pd.DataFrame,
    *,
    top_n: int = 20,
    id_column: str = "n_processo",
    case_type_column: str = "case_type",
    decision_class_column: str = "decision_class",
    embeddings_column: str = "selected_sentence_embeddings",
) -> pd.DataFrame:
    """Score candidates against a query case, then keep the top-N most similar.

    Args:
        query_row: Series for the query case, with an embeddings value under
            `embeddings_column` (or a `"sentence_embeddings"` fallback).
        candidate_df: DataFrame of candidate cases to score against;
            restricted to the query's case type inside
            `score_candidates_against_query`.
        top_n: Maximum number of top-similarity candidates to keep.
        id_column: Candidate case identifier column.
        case_type_column: Column used to restrict candidates to the query's
            case type.
        decision_class_column: Column identifying each candidate's decision
            class.
        embeddings_column: Candidate embeddings column name.

    Returns:
        Top-`top_n` rows of `candidate_df` (restricted to the query's case
        type), sorted by descending similarity to `query_row`, with
        `similarity_score` and 1-based `candidate_rank` columns added.

    Raises:
        ValueError: If required columns are missing from `candidate_df`
            (raised inside `score_candidates_against_query` or
            `filter_relevant_candidates`).
    """
    scored = score_candidates_against_query(
        query_row,
        candidate_df,
        id_column=id_column,
        case_type_column=case_type_column,
        decision_class_column=decision_class_column,
        embeddings_column=embeddings_column,
    )
    return filter_relevant_candidates(scored, top_n=top_n, id_column=id_column)


def select_diverse_candidates(
    candidate_df: pd.DataFrame,
    *,
    k: int = 2,
    id_column: str = "n_processo",
    similarity_column: str = "similarity_score",
    embeddings_column: str = "selected_sentence_embeddings",
    max_pair_similarity: float | None = None,
) -> pd.DataFrame:
    """Greedily select up to k candidates, skipping near-duplicates of already-picked ones.

    Always keeps the single most-similar candidate first, then walks the
    remaining candidates in descending similarity order, adding each one
    whose `set_to_set_similarity` to every already-chosen candidate is below
    `max_pair_similarity`, until `k` candidates are chosen or the candidates
    are exhausted.

    Args:
        candidate_df: DataFrame with `id_column`, `similarity_column`, and
            `embeddings_column` (typically the output of
            `relevance_filter_for_query`).
        k: Maximum number of candidates to select.
        id_column: Candidate case identifier column (only checked for
            presence).
        similarity_column: Column to rank candidates by, descending.
        embeddings_column: Column holding each candidate's sentence
            embeddings, used for the pairwise diversity check.
        max_pair_similarity: If given, a candidate whose
            `set_to_set_similarity` to any already-chosen candidate is `>=`
            this threshold is skipped. If None, no diversity filtering is
            applied and the top `k` by `similarity_column` are returned.

    Returns:
        Up to `k` rows of `candidate_df`, in descending-similarity selection
        order, with a new 1-based `diversity_rank` column and a reset index.
        Returns `candidate_df` unchanged (empty) if it is empty.

    Raises:
        ValueError: If `k < 1`, or if `candidate_df` is missing `id_column`,
            `similarity_column`, or `embeddings_column`.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    required = {id_column, similarity_column, embeddings_column}
    missing = required - set(candidate_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    ranked = candidate_df.sort_values(similarity_column, ascending=False).reset_index(
        drop=True
    )
    if ranked.empty:
        return ranked

    chosen: list[int] = [0]

    def _is_too_similar(idx: int) -> bool:
        if max_pair_similarity is None:
            return False
        for picked in chosen:
            sim = set_to_set_similarity(
                ranked.loc[idx, embeddings_column],
                ranked.loc[picked, embeddings_column],
                query_ranks=_ensure_seq(ranked.loc[idx].get("sentence_ranks", [])),
                candidate_ranks=_ensure_seq(
                    ranked.loc[picked].get("sentence_ranks", [])
                ),
            )
            if sim >= max_pair_similarity:
                return True
        return False

    for idx in range(1, len(ranked)):
        if len(chosen) >= k:
            break
        if _is_too_similar(idx):
            continue
        chosen.append(idx)

    out = ranked.iloc[chosen].copy()
    out["diversity_rank"] = range(1, len(out) + 1)
    return out.reset_index(drop=True)


def diversify_relevant_candidates(
    query_row: pd.Series,
    candidate_df: pd.DataFrame,
    *,
    top_n: int = 20,
    k: int = 2,
    max_pair_similarity: float | None = 0.9,
    id_column: str = "n_processo",
    case_type_column: str = "case_type",
    decision_class_column: str = "decision_class",
    embeddings_column: str = "selected_sentence_embeddings",
) -> pd.DataFrame:
    """Score+shortlist candidates against a query, then diversify the top picks.

    Combines `relevance_filter_for_query` (score all same-case-type
    candidates and keep the top `top_n`) with `select_diverse_candidates`
    (greedily pick up to `k` of those, skipping near-duplicates).

    Args:
        query_row: Series for the query case.
        candidate_df: DataFrame of candidate cases to score and select from.
        top_n: Shortlist size passed to `relevance_filter_for_query`.
        k: Maximum number of diverse candidates to return.
        max_pair_similarity: Diversity threshold passed to
            `select_diverse_candidates`; candidates too similar to an
            already-chosen one are skipped.
        id_column: Candidate case identifier column.
        case_type_column: Column used to restrict candidates to the query's
            case type.
        decision_class_column: Column identifying each candidate's decision
            class.
        embeddings_column: Candidate embeddings column name.

    Returns:
        Up to `k` diverse, high-similarity candidate rows with
        `similarity_score`, `candidate_rank`, and `diversity_rank` columns
        added.
    """
    relevant = relevance_filter_for_query(
        query_row,
        candidate_df,
        top_n=top_n,
        id_column=id_column,
        case_type_column=case_type_column,
        decision_class_column=decision_class_column,
        embeddings_column=embeddings_column,
    )
    return select_diverse_candidates(
        relevant,
        k=k,
        id_column=id_column,
        similarity_column="similarity_score",
        embeddings_column=embeddings_column,
        max_pair_similarity=max_pair_similarity,
    )


def build_few_shot_set(
    query_row: pd.Series,
    candidate_df: pd.DataFrame,
    *,
    # top_n and max_pair_similarity support a diverse, multi-example-per-class
    # few-shot set (top_n shortlists candidates, max_pair_similarity dedupes
    # near-identical picks). Intended to run with k_per_class > 1, but every
    # production run used k_per_class=1 due to context window / OOM
    # constraints on available hardware, so this machinery is present but
    # effectively inert. Left in place as a reminder for future work with
    # more hardware headroom.
    top_n: int = 20,
    k_per_class: int = 1,
    max_pair_similarity: float | None = 0.9,
    optional_if_context_full: bool = True,
    max_total_cases: int | None = None,
    balanced_classes: bool = True,
    id_column: str = "n_processo",
    case_type_column: str = "case_type",
    decision_class_column: str = "decision_class",
    embeddings_column: str = "selected_sentence_embeddings",
) -> pd.DataFrame:
    """Build a decision-class-balanced few-shot example set for one query case.

    For each decision class present in `candidate_df`, retrieves up to
    `k_per_class` diverse, high-similarity candidates via
    `diversify_relevant_candidates`, then (if `balanced_classes`) truncates
    every class's picks down to the smallest class's count so each class
    contributes equally, concatenates all classes, sorts by
    (`decision_class`, `similarity_score` descending), and optionally caps
    the total at `max_total_cases`.

    Note: `top_n` and `max_pair_similarity` exist to support a diverse,
    multi-example-per-class set when `k_per_class > 1`, but every production
    run used `k_per_class=1`, so in practice at most one candidate is
    retrieved per class and this diversification machinery is effectively
    inert — kept in place for future work with more hardware headroom.

    Args:
        query_row: Series for the query case.
        candidate_df: DataFrame of candidate cases across decision classes.
        top_n: Shortlist size per class before diversification (see note
            above).
        k_per_class: Maximum candidates to keep per decision class.
        max_pair_similarity: Diversity threshold per class (see note above).
        optional_if_context_full: Accepted for API compatibility but
            currently has no effect on the returned value — both internal
            branches return the same result.
        max_total_cases: Optional cap on the total number of examples
            returned, applied after class balancing.
        balanced_classes: If True, truncates every class's picks to the
            smallest class's count so the returned set is balanced across
            classes.
        id_column: Candidate case identifier column.
        case_type_column: Column used to restrict candidates to the query's
            case type.
        decision_class_column: Column used to group candidates into classes.
        embeddings_column: Candidate embeddings column name.

    Returns:
        DataFrame of selected few-shot examples with a `few_shot_rank`
        column (re-numbered after class balancing / total-cases capping),
        sorted by `decision_class` then descending `similarity_score`.
        Empty (with `candidate_df`'s columns plus `few_shot_rank`) if no
        class yielded any candidates.

    Raises:
        ValueError: If `candidate_df` is missing `id_column`,
            `case_type_column`, or `decision_class_column`.
    """
    required = {id_column, case_type_column, decision_class_column}
    missing = required - set(candidate_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    grouped: list[pd.DataFrame] = []
    for _, group in candidate_df.groupby(decision_class_column, dropna=False):
        selected = diversify_relevant_candidates(
            query_row,
            group,
            top_n=top_n,
            k=k_per_class,
            max_pair_similarity=max_pair_similarity,
            id_column=id_column,
            case_type_column=case_type_column,
            decision_class_column=decision_class_column,
            embeddings_column=embeddings_column,
        )
        if not selected.empty:
            selected = selected.copy()
            selected["few_shot_rank"] = range(1, len(selected) + 1)
            grouped.append(selected)

    if not grouped:
        return pd.DataFrame(columns=list(candidate_df.columns) + ["few_shot_rank"])

    if balanced_classes:
        target_per_class = min([len(df) for df in grouped] + [k_per_class])
        grouped = [df.head(target_per_class).copy() for df in grouped]
        for df in grouped:
            df["few_shot_rank"] = range(1, len(df) + 1)

    out = pd.concat(grouped, ignore_index=True)
    out = out.sort_values(
        ["decision_class", "similarity_score"], ascending=[True, False]
    ).reset_index(drop=True)

    if max_total_cases is not None and len(out) > max_total_cases:
        out = out.head(max_total_cases).copy()
        out["few_shot_rank"] = range(1, len(out) + 1)

    if optional_if_context_full and len(out) == 0:
        return out
    return out


def build_few_shot_examples_by_case(
    query_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    *,
    query_id_column: str = "n_processo",
    top_n: int = 20,
    k_per_class: int = 1,
    max_pair_similarity: float | None = 0.9,
    max_total_cases: int | None = None,
    balanced_classes: bool = True,
    id_column: str = "n_processo",
    case_type_column: str = "case_type",
    decision_class_column: str = "decision_class",
    embeddings_column: str = "selected_sentence_embeddings",
) -> dict[str, list[dict]]:
    """Build the per-query-case few-shot example mapping used by `prompts.py`.

    For every row in `query_df`, calls `build_few_shot_set` against
    `candidate_df` and converts the resulting candidates into plain-dict
    example records (canonicalized label, readable/rank-ordered selected
    sentences, rank, similarity score), keyed by the query's case id — the
    format consumed by `prompts.build_few_shot_prompt`/
    `build_few_shot_cot_prompt`.

    Args:
        query_df: DataFrame of query cases, one row per case, needing
            `query_id_column`, `case_type_column`, `decision_class_column`,
            and either `embeddings_column` or a `"sentence_embeddings"`
            fallback column.
        candidate_df: DataFrame of candidate cases to draw few-shot examples
            from (forwarded to `build_few_shot_set`).
        query_id_column: Column identifying each query case.
        top_n: Forwarded to `build_few_shot_set`.
        k_per_class: Forwarded to `build_few_shot_set`.
        max_pair_similarity: Forwarded to `build_few_shot_set`.
        max_total_cases: Forwarded to `build_few_shot_set`.
        balanced_classes: Forwarded to `build_few_shot_set`.
        id_column: Candidate case identifier column.
        case_type_column: Column used to restrict candidates to each
            query's case type.
        decision_class_column: Column used to group candidates, and (via
            `row.get(decision_class_column, row.get("label", ""))`) as the
            fallback source of each example's raw label before
            canonicalization.
        embeddings_column: Candidate/query embeddings column name.

    Returns:
        Dict mapping each query's `n_processo` (string) to a list of example
        dicts with keys `"n_processo"`, `"label"`, `"text"`,
        `"selected_sentences_readable"`, `"selected_sentences_ranked_ordered"`,
        `"few_shot_rank"`, `"decision_class"`, and `"similarity_score"`. A
        query with no retrieved candidates maps to an empty list.

    Raises:
        ValueError: If `query_df` is missing `query_id_column`,
            `case_type_column`, `decision_class_column`, or (when no
            `"sentence_embeddings"` fallback column is present)
            `embeddings_column`.
    """
    required = {query_id_column, case_type_column, decision_class_column}
    has_embeddings = (
        embeddings_column in query_df.columns
        or "sentence_embeddings" in query_df.columns
    )
    if not has_embeddings:
        required.add(embeddings_column)
    missing = required - set(query_df.columns)
    if missing:
        raise ValueError(f"Query dataframe missing required columns: {sorted(missing)}")

    examples_by_case: dict[str, list[dict]] = {}
    for _, query_row in query_df.iterrows():
        query_id = str(query_row[query_id_column])
        query_case_type = str(query_row.get(case_type_column, ""))
        few_shot_df = build_few_shot_set(
            query_row,
            candidate_df,
            top_n=top_n,
            k_per_class=k_per_class,
            max_pair_similarity=max_pair_similarity,
            max_total_cases=max_total_cases,
            balanced_classes=balanced_classes,
            id_column=id_column,
            case_type_column=case_type_column,
            decision_class_column=decision_class_column,
            embeddings_column=embeddings_column,
        )
        if few_shot_df.empty:
            examples_by_case[query_id] = []
            continue

        records: list[dict] = []
        for _, row in few_shot_df.iterrows():
            raw_label = row.get(decision_class_column, row.get("label", ""))
            canonical_label = canonicalize_label(str(raw_label), query_case_type)
            selected_readable = _json_safe(row.get("selected_sentences_readable", []))
            selected_ranked = _json_safe(
                row.get("selected_sentences_ranked_ordered", [])
            )
            text_value = (
                selected_readable
                if selected_readable
                else _json_safe(row.get("text", ""))
            )
            rank_value = row.get("few_shot_rank", 0)
            similarity_value = row.get("similarity_score", 0.0)
            records.append(
                {
                    "n_processo": str(row[id_column]),
                    "label": canonical_label,
                    "text": text_value,
                    "selected_sentences_readable": selected_readable,
                    "selected_sentences_ranked_ordered": selected_ranked,
                    "few_shot_rank": int(rank_value) if rank_value is not None else 0,
                    "decision_class": canonical_label,
                    "similarity_score": (
                        float(similarity_value) if similarity_value is not None else 0.0
                    ),
                }
            )
        examples_by_case[query_id] = records

    return examples_by_case


def save_few_shot_examples_by_case(
    examples_by_case: dict[str, list[dict]], output_path: Path
) -> None:
    """Serialize a per-case few-shot example mapping to a JSON file.

    Args:
        examples_by_case: Mapping from query case id to its list of example
            dicts, as produced by `build_few_shot_examples_by_case`.
        output_path: Destination JSON file path; parent directories are
            created if needed.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = _json_safe(examples_by_case)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(safe_payload, fh, ensure_ascii=False, indent=2)


def score_candidates_against_query(
    query_row: pd.Series,
    candidate_df: pd.DataFrame,
    *,
    id_column: str = "n_processo",
    case_type_column: str = "case_type",
    decision_class_column: str = "decision_class",
    embeddings_column: str = "selected_sentence_embeddings",
) -> pd.DataFrame:
    """Score every same-case-type candidate's similarity to a query case.

    Args:
        query_row: Series for the query case; embeddings are read from
            `embeddings_column`, falling back to `"sentence_embeddings"`.
        candidate_df: DataFrame of candidate cases; embeddings are read
            per-row from `embeddings_column` if that column is present in
            `candidate_df`, otherwise from `"sentence_embeddings"`.
        id_column: Candidate case identifier column (only checked for
            presence).
        case_type_column: Column used to filter `candidate_df` down to rows
            matching `query_row[case_type_column]`.
        decision_class_column: Column identifying each candidate's decision
            class (only checked for presence; not used in scoring itself).
        embeddings_column: Preferred embeddings column name, checked on both
            `query_row` and `candidate_df`.

    Returns:
        Copy of the same-case-type subset of `candidate_df`, with a new
        `similarity_score` column (via `set_to_set_similarity` against
        `query_row`) and sorted by that score descending. An empty frame
        (with an empty `similarity_score` column) if no candidates share the
        query's case type.

    Raises:
        ValueError: If `candidate_df` is missing `id_column`,
            `case_type_column`, `decision_class_column`, or a usable
            embeddings column (neither `embeddings_column` nor
            `"sentence_embeddings"` present).
    """
    embeddings_col = (
        embeddings_column
        if embeddings_column in candidate_df.columns
        else "sentence_embeddings"
    )
    required = {id_column, case_type_column, decision_class_column, embeddings_col}
    missing = required - set(candidate_df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. "
            "Use few_shot_metadata output with selected_sentence_embeddings."
        )

    q_emb = _ensure_2d_embeddings(
        query_row.get(embeddings_column, query_row.get("sentence_embeddings"))
    )
    q_ranks = _ensure_seq(query_row.get("sentence_ranks", []))

    same_case_type = candidate_df[
        candidate_df[case_type_column].astype(str) == str(query_row[case_type_column])
    ].copy()
    if same_case_type.empty:
        return same_case_type.assign(similarity_score=pd.Series(dtype=np.float32))

    scores: list[float] = []
    for _, cand in same_case_type.iterrows():
        score = set_to_set_similarity(
            q_emb,
            cand[embeddings_col],
            query_ranks=q_ranks,
            candidate_ranks=_ensure_seq(cand.get("sentence_ranks", [])),
        )
        scores.append(score)

    same_case_type["similarity_score"] = np.asarray(scores, dtype=np.float32)
    return same_case_type.sort_values("similarity_score", ascending=False).reset_index(
        drop=True
    )
