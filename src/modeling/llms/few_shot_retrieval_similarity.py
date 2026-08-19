from __future__ import annotations

from dataclasses import dataclass
import ast
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from config import canonicalize_label


@dataclass(frozen=True)
class CandidatePool:
    case_type: str
    decision_class: str
    candidate_ids: list[str]
    candidate_scores: list[float]
    candidate_ranks: list[int]


def _ensure_2d_embeddings(value) -> np.ndarray:
    if value is None:
        return np.zeros((0, 0), dtype=np.float32)
    if isinstance(value, str):
        value = ast.literal_eval(value)
    arr = np.asarray(value, dtype=object if isinstance(value, (list, tuple, np.ndarray)) else np.float32)
    if isinstance(value, np.ndarray) and value.dtype != object:
        arr = np.asarray(value, dtype=np.float32)
        return arr.reshape(1, -1) if arr.ndim == 1 else arr
    items = value.tolist() if isinstance(value, np.ndarray) else list(value)
    if not items:
        return np.zeros((0, 0), dtype=np.float32)
    stacked = np.stack([np.asarray(v, dtype=np.float32) for v in items], axis=0)
    return stacked.astype(np.float32, copy=False)


def _ensure_seq(value) -> list:
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
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)
    w = 1.0 / (np.arange(n, dtype=np.float32) + 1.0)
    return (w / w.sum()).astype(np.float32, copy=False)


def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
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
    q = _ensure_2d_embeddings(query_embeddings)
    c = _ensure_2d_embeddings(candidate_embeddings)
    if q.size == 0 or c.size == 0:
        return 0.0

    sim = _cosine_similarity_matrix(q, c)
    qw = _rank_weights(q.shape[0]) # Using query and candidate rows for weights
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
    required = {id_column, case_type_column, decision_class_column, embeddings_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    pools: list[CandidatePool] = []
    for (case_type, decision_class), group in df.groupby([case_type_column, decision_class_column], dropna=False):
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
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    required = {id_column, similarity_column}
    missing = required - set(scored_candidates.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = scored_candidates.sort_values(similarity_column, ascending=False).head(top_n).copy()
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
    if k < 1:
        raise ValueError("k must be >= 1")
    required = {id_column, similarity_column, embeddings_column}
    missing = required - set(candidate_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    ranked = candidate_df.sort_values(similarity_column, ascending=False).reset_index(drop=True)
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
                candidate_ranks=_ensure_seq(ranked.loc[picked].get("sentence_ranks", [])),
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
    out = out.sort_values(["decision_class", "similarity_score"], ascending=[True, False]).reset_index(drop=True)

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
    required = {query_id_column, case_type_column, decision_class_column}
    has_embeddings = embeddings_column in query_df.columns or "sentence_embeddings" in query_df.columns
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
            selected_ranked = _json_safe(row.get("selected_sentences_ranked_ordered", []))
            text_value = selected_readable if selected_readable else _json_safe(row.get("text", ""))
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
                    "similarity_score": float(similarity_value) if similarity_value is not None else 0.0,
                }
            )
        examples_by_case[query_id] = records

    return examples_by_case


def save_few_shot_examples_by_case(examples_by_case: dict[str, list[dict]], output_path: Path) -> None:
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
    embeddings_col = embeddings_column if embeddings_column in candidate_df.columns else "sentence_embeddings"
    required = {id_column, case_type_column, decision_class_column, embeddings_col}
    missing = required - set(candidate_df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. "
            "Use few_shot_metadata output with selected_sentence_embeddings."
        )

    q_emb = _ensure_2d_embeddings(query_row.get(embeddings_column, query_row.get("sentence_embeddings")))
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
    return same_case_type.sort_values("similarity_score", ascending=False).reset_index(drop=True)
