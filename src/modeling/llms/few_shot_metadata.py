from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pandas as pd
import numpy as np

from src.modeling.llms.summarizer import (
    LEGAL_BERTIMBAU_MODEL,
    summarize_from_sentence_embeddings,
)

MAX_TOKEN_COUNT = 9000 # to ensure consistency with previous version of truncation.
SPLIT_ROOT = Path("data/processed_data/splits/gold_test")

def _ensure_seq(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        out = value.tolist()
        return out if isinstance(out, list) else [out]
    return list(value)


def _normalize_embeddings(value) -> np.ndarray:
    if value is None:
        return np.zeros((0, 0), dtype=np.float32)
    if isinstance(value, str):
        value = ast.literal_eval(value)

    if isinstance(value, np.ndarray):
        if value.size == 0:
            return np.zeros((0, 0), dtype=np.float32)
        if value.dtype == object:
            try:
                return np.stack([np.asarray(v, dtype=np.float32) for v in value], axis=0)
            except Exception as exc:
                raise ValueError(f"Failed to stack embeddings array: {exc}") from exc
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        if arr.ndim == 2:
            return arr
        raise ValueError("Embeddings must be a 1D or 2D sequence of vectors.")
    else:
        items = list(value)

    if len(items) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    try:
        return np.stack([np.asarray(v, dtype=np.float32) for v in items], axis=0)
    except Exception:
        arr = np.asarray(items, dtype=np.float32)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        if arr.ndim == 2:
            return arr
    raise ValueError("Embeddings must be a 1D or 2D sequence of vectors.")


def _load_embeddings_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format '{path.suffix}'. Use .parquet or .csv")


def _load_class_lookup(case_type: str, split_kind: str) -> pd.DataFrame:
    class_col = "decisao_ternaria" if case_type == "boc" else "decisao_binaria"
    path = SPLIT_ROOT / f"{case_type}_{split_kind}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Class source CSV not found: {path}")
    df = pd.read_csv(path, usecols=["n_processo", class_col])
    return df.rename(columns={class_col: "decision_class"})


def enrich_sentence_metadata(
    df: pd.DataFrame,
    *,
    id_column: str = "n_processo",
    case_type: str,
    sentences_column: str = "sentences",
    embeddings_column: str = "sentence_embeddings",
    summary_top_k: int | None = None,
    max_summary_tokens: int | None = None,
    class_lookup: pd.DataFrame | None = None,
    model_name: str = LEGAL_BERTIMBAU_MODEL,
) -> pd.DataFrame:
    if embeddings_column not in df.columns and "sentences_embeddings" in df.columns:
        embeddings_column = "sentences_embeddings"

    missing = {id_column, sentences_column, embeddings_column} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows: list[dict] = []
    for row in df.to_dict(orient="records"):
        sentences = [str(s) for s in _ensure_seq(row[sentences_column])]
        embeddings = _normalize_embeddings(row[embeddings_column])
        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2D after normalization for row {row[id_column]}")
        decision_class = None
        if class_lookup is not None:
            match = class_lookup.loc[class_lookup[id_column] == row[id_column], "decision_class"]
            if match.empty:
                raise ValueError(f"Missing decision class for row {row[id_column]}")
            decision_class = match.iloc[0]

        scored = summarize_from_sentence_embeddings(
            sentences=sentences,
            sentence_embeddings=embeddings,
            case_type=case_type,
            top_k=summary_top_k,
            max_summary_tokens=max_summary_tokens,
            model_name=model_name,
        )
        sentence_scores = [float(v) for v in scored.scores.tolist()]
        sentence_rank_order = [int(i) for i in np.argsort(scored.scores)[::-1]]
        sentence_ranks = [0] * len(sentence_scores)
        for rank, idx in enumerate(sentence_rank_order, start=1):
            sentence_ranks[idx] = rank

        selected = [idx for idx in sentence_rank_order if idx in set(scored.top_indices)]
        selected_readable = sorted(selected, reverse=False)
        selected_scores = [sentence_scores[i] for i in selected]
        selected_ranks = [sentence_ranks[i] for i in selected]
        selected_embeddings = [embeddings[i].tolist() for i in selected]
        rows.append(
            {
                id_column: row[id_column],
                "case_type": case_type,
                "decision_class": decision_class,
                "full_sentence_count_before": len(sentences),
                "selected_sentence_count_after": len(selected),
                "sentence_retention_pct": (100.0 * len(selected) / len(sentences)) if sentences else 0.0,
                "sentence_reduction_abs": len(sentences) - len(selected),
                "sentence_reduction_pct": (100.0 * (len(sentences) - len(selected)) / len(sentences)) if sentences else 0.0,
                "selected_sentence_indices": selected,
                "selected_sentence_indices_readable": selected_readable,
                "selected_sentence_ranks": selected_ranks,
                "selected_sentence_scores": selected_scores,
                "selected_sentences_ranked_ordered": [sentences[i] for i in selected],
                "selected_sentences_readable": [sentences[i] for i in selected_readable],
                "selected_sentence_embeddings": selected_embeddings,
                "sentence_scores": sentence_scores,
                "sentence_ranks": sentence_ranks,
                "sentence_rank_order": sentence_rank_order,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich retained sentence metadata from embeddings parquet.")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--id_column", default="n_processo")
    parser.add_argument("--sentences_column", default="sentences")
    parser.add_argument("--embeddings_column", default="sentence_embeddings")
    parser.add_argument("--summary_top_k", type=int, default=None)
    parser.add_argument("--max_summary_tokens", type=int, default=MAX_TOKEN_COUNT)
    args = parser.parse_args()

    in_path = Path(args.input_path)
    out_path = Path(args.output_path)
    stem = in_path.stem.lower()
    if stem.startswith("dv"):
        case_type = "dv"
    elif stem.startswith("boc"):
        case_type = "boc"
    else:
        raise ValueError(f"Cannot infer case type from input file name: {in_path.name}")
    split_kind = "gold_test" if "gold_test" in stem else "train_before_cutoff"
    df = _load_embeddings_frame(in_path)
    class_lookup = _load_class_lookup(case_type, split_kind)
    enriched = enrich_sentence_metadata(
        df,
        id_column=args.id_column,
        case_type=case_type,
        sentences_column=args.sentences_column,
        embeddings_column=args.embeddings_column,
        summary_top_k=args.summary_top_k,
        max_summary_tokens=args.max_summary_tokens,
        class_lookup=class_lookup,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(out_path, index=False)
    print(f"Saved enriched metadata -> {out_path}")


if __name__ == "__main__":
    main()
