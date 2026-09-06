"""features.py: preprocessing pipeline for Legal BERTimbau frozen embeddings.

Provides:
  - load_split_data()        -- load embeddings + labels from parquet + split CSV
  - load_split_text_data()   -- load raw texts + labels from split CSV (for BERT fine-tuning)
  - encode_labels()          -- map string labels to integers
  - build_feature_pipeline() -- sklearn Pipeline: StandardScaler → PCA(0.95)

Imported by train.py and predict.py; contains no persistence logic.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "stjiris/bert-large-portuguese-cased-legal-mlm-nli-sts-v1"
EMBED_DIM = 1024
SEED = 42

_REPO_ROOT = Path(__file__).resolve().parents[3]  # src/modeling/dl → repo root

EMBEDDINGS_PATHS: dict[str, Path] = {
    "dv": _REPO_ROOT
    / "data/processed_data/bert_tokens_embedd"
    / "df_acordaos_dv_eda_binary_legal_bert_embeddings.parquet",
    "boc": _REPO_ROOT
    / "data/processed_data/bert_tokens_embedd"
    / "df_acordaos_boc_eda_ternary_legal_bert_embeddings.parquet",
}

SPLIT_PATHS: dict[str, dict[str, Path]] = {
    "dv": {
        "train": _REPO_ROOT
        / "data/processed_data/splits/gold_test/dv_train_before_cutoff.csv",
        "gold_test": _REPO_ROOT
        / "data/processed_data/splits/gold_test/dv_gold_test.csv",
    },
    "boc": {
        "train": _REPO_ROOT
        / "data/processed_data/splits/gold_test/boc_train_before_cutoff.csv",
        "gold_test": _REPO_ROOT
        / "data/processed_data/splits/gold_test/boc_gold_test.csv",
    },
}

# Label encodings — must match classification.py and explanation.py
# Binary  (DV):  0 = DECISÃO MANTIDA,      1 = DECISÃO ALTERADA
# Ternary (BoC): 0 = DESFAVORÁVEL, 1 = PARCIAL, 2 = FAVORÁVEL
LABEL_MAPPINGS: dict[str, dict[str, int]] = {
    "dv": {
        "DECISÃO MANTIDA": 0,
        "DECISÃO ALTERADA": 1,
    },
    "boc": {
        "DESFAVORÁVEL": 0,
        "PARCIAL": 1,
        "FAVORÁVEL": 2,
    },
}

N_CLASSES: dict[str, int] = {"dv": 2, "boc": 3}

# True  → BCEWithLogitsLoss (sigmoid + BCE, single output neuron)
# False → CrossEntropyLoss  (log-softmax, n_classes output neurons)
IS_BINARY: dict[str, bool] = {"dv": True, "boc": False}

# Label columns in the split CSVs (used by load_split_text_data)
LABEL_COLUMNS: dict[str, str] = {"dv": "decisao_binaria", "boc": "decisao_ternaria"}

# Raw text column name (same in both split CSVs and embedding parquets)
TEXT_COLUMN = "texto_integral_sem_decisao"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_split_data(
    case_type: str,
    split: str = "train",
) -> tuple[np.ndarray, np.ndarray, pd.Series, list[str]]:
    """Load pre-extracted frozen BERT embeddings for a given case_type and split.

    Joins the parquet embedding file with the split CSV on n_processo to
    ensure only cases that (a) have embeddings and (b) belong to the
    requested temporal split are returned.

    Args:
        case_type: "dv" (Domestic Violence) or "boc" (Breach of Contract).
        split: "train" or "gold_test". Gold test must only be used for final
            evaluation.

    Returns:
        Tuple of (X, y_raw, dates, n_processo):
            X: np.ndarray, shape (n_samples, EMBED_DIM=1024), dtype float32.
            y_raw: np.ndarray of str, shape (n_samples,) — un-encoded string
                labels, e.g. "decisao_mantida".
            dates: pd.Series of datetime, shape (n_samples,) — decision
                dates, used for temporal ordering in TimeSeriesSplit.
            n_processo: list[str] — case identifiers aligned with rows of
                X / y_raw.

    Raises:
        ValueError: If ``case_type`` or ``split`` is not a recognised value.
        RuntimeError: If no n_processo overlaps between the parquet embeddings
            and the split CSV.
    """
    # TODO: Remmeber to enforce consisten data types in the incoming parquet and csv,
    # Because as of now embeddigns came as strings and could not be loaded as float
    # E.g., ValueError: could not convert string to float: np.str_....
    case_type = case_type.lower()
    if case_type not in EMBEDDINGS_PATHS:
        raise ValueError(f"Unknown case_type '{case_type}'. Choose 'dv' or 'boc'.")
    if split not in ("train", "gold_test"):
        raise ValueError(f"Unknown split '{split}'. Choose 'train' or 'gold_test'.")

    emb_df = pd.read_parquet(EMBEDDINGS_PATHS[case_type])
    split_df = pd.read_csv(SPLIT_PATHS[case_type][split], usecols=["n_processo"])

    # Inner join — keeps only cases present in both files
    merged = emb_df.merge(split_df[["n_processo"]], on="n_processo", how="inner")

    if merged.empty:
        raise RuntimeError(
            f"No overlapping n_processo between parquet and {split} CSV "
            f"for case_type='{case_type}'. Check that embeddings were "
            f"generated from the same dataset."
        )

    # Embeddings may be stored as JSON/Python-list strings — parse if needed
    sample = merged["embedding"].iloc[0]
    if isinstance(sample, (str, np.str_)):
        import json

        merged["embedding"] = merged["embedding"].apply(
            lambda s: json.loads(s) if isinstance(s, (str, np.str_)) else s
        )

    X = np.stack(merged["embedding"].values).astype(np.float32)  # (n, 1024)
    y_raw = merged["class_label"].values.astype(str)
    dates = pd.to_datetime(merged["data_acordao"], format="%d/%m/%Y", errors="coerce")
    n_processo = merged["n_processo"].tolist()

    return X, y_raw, dates, n_processo


def load_split_text_data(
    case_type: str,
    split: str = "train",
) -> tuple[list[str], np.ndarray, pd.Series, list[str]]:
    """Load raw texts + labels from the split CSV (for BERT fine-tuning).

    Reads directly from the split CSV — does NOT touch the parquet.
    The split CSVs contain texto_integral_sem_decisao, data_acordao, and the
    task-specific label column (decisao_binaria / decisao_ternaria).

    Args:
        case_type: "dv" (Domestic Violence) or "boc" (Breach of Contract).
        split: "train" or "gold_test". Gold test must only be used for final
            evaluation.

    Returns:
        Tuple of (texts, y_raw, dates, n_processo):
            texts: list[str] — raw document texts.
            y_raw: np.ndarray of str — un-encoded string labels (uppercase,
                stripped).
            dates: pd.Series of datetime — decision dates for temporal
                ordering.
            n_processo: list[str] — case identifiers aligned with rows.

    Raises:
        ValueError: If ``case_type`` or ``split`` is not a recognised value.
    """
    case_type = case_type.lower()
    if case_type not in SPLIT_PATHS:
        raise ValueError(f"Unknown case_type '{case_type}'. Choose 'dv' or 'boc'.")
    if split not in ("train", "gold_test"):
        raise ValueError(f"Unknown split '{split}'. Choose 'train' or 'gold_test'.")

    label_col = LABEL_COLUMNS[case_type]
    df = pd.read_csv(
        SPLIT_PATHS[case_type][split],
        usecols=["n_processo", TEXT_COLUMN, "data_acordao", label_col],
    )

    # Normalize label values to match LABEL_MAPPINGS keys (uppercase, stripped)
    df[label_col] = df[label_col].astype(str).str.upper().str.strip()

    texts = df[TEXT_COLUMN].fillna("").tolist()
    y_raw = df[label_col].values
    dates = pd.to_datetime(df["data_acordao"], format="%d/%m/%Y", errors="coerce")
    n_processo = df["n_processo"].astype(str).tolist()

    return texts, y_raw, dates, n_processo


def load_bert_train_val_split(
    case_type: str,
    val_ratio: float = 0.15,
) -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
    """Load training data and return a temporal 85/15 train/val split.

    Sorts the training set by data_acordao and reserves the last val_ratio
    fraction as a temporal validation set for BERT early stopping.
    Labels are already integer-encoded (via encode_labels).

    Args:
        case_type: "dv" or "boc".
        val_ratio: Fraction of training data to use as validation
            (default 0.15).

    Returns:
        Tuple of (texts_tr, y_tr, texts_val, y_val):
            texts_tr: list[str] — training document texts.
            y_tr: np.ndarray of int — training labels.
            texts_val: list[str] — validation document texts.
            y_val: np.ndarray of int — validation labels.
    """
    texts_tr, y_tr, _, texts_val, y_val, _ = load_bert_train_val_split_with_ids(
        case_type, val_ratio=val_ratio
    )
    return texts_tr, y_tr, texts_val, y_val


def load_bert_train_val_split_with_ids(
    case_type: str,
    val_ratio: float = 0.15,
) -> tuple[list[str], np.ndarray, list[str], list[str], np.ndarray, list[str]]:
    """Temporal 85/15 train/val split that also returns case identifiers.

    Identical split logic to ``load_bert_train_val_split`` (which delegates
    here), but additionally returns the ``n_processo`` identifier for every
    row so predictions can be traced back to individual cases.  Used when
    scoring the model on the data it was fitted on, to diagnose whether a
    collapsed model is underfitting or overfitting.

    Args:
        case_type: "dv" or "boc".
        val_ratio: Fraction of training data to use as validation.

    Returns:
        Tuple of (texts_tr, y_tr, ids_tr, texts_val, y_val, ids_val):
            texts_tr: list[str] — training document texts.
            y_tr: np.ndarray of int — training labels.
            ids_tr: list[str] — training case identifiers.
            texts_val: list[str] — validation document texts.
            y_val: np.ndarray of int — validation labels.
            ids_val: list[str] — validation case identifiers.
    """
    texts, y_raw, dates, n_processo = load_split_text_data(case_type, split="train")
    y = encode_labels(y_raw, case_type)

    sort_idx = np.argsort(dates.values)
    texts_sorted = [texts[i] for i in sort_idx]
    ids_sorted = [n_processo[i] for i in sort_idx]
    y_sorted = y[sort_idx]

    cutoff = int(len(texts_sorted) * (1.0 - val_ratio))
    return (
        texts_sorted[:cutoff],
        y_sorted[:cutoff],
        ids_sorted[:cutoff],
        texts_sorted[cutoff:],
        y_sorted[cutoff:],
        ids_sorted[cutoff:],
    )


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------


def encode_labels(y_raw: np.ndarray, case_type: str) -> np.ndarray:
    """Map string labels to integers using the fixed LABEL_MAPPINGS.

    Args:
        y_raw: Array-like of string labels.
        case_type: "dv" or "boc".

    Returns:
        np.ndarray of int64, shape (n_samples,).

    Raises:
        ValueError: If any label in ``y_raw`` is not in the mapping for
            ``case_type``.
    """
    mapping = LABEL_MAPPINGS[case_type.lower()]
    try:
        return np.array([mapping[label] for label in y_raw], dtype=np.int64)
    except KeyError as exc:
        unknown = set(y_raw) - set(mapping)
        raise ValueError(
            f"Unknown label(s) {unknown} for case_type='{case_type}'. "
            f"Expected one of {list(mapping)}."
        ) from exc


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------


def build_feature_pipeline() -> Pipeline:
    """Return an *unfitted* sklearn preprocessing pipeline.

    Steps
    -----
    1. StandardScaler  -- zero-mean, unit-variance per embedding dimension
    2. PCA(0.95)       -- retain 95% explained variance; reduces 1024-dim to
                         ~80–200 components (exact count determined at fit time)

    Notes:
    -----
    svd_solver='full' is required when n_components is a float (variance ratio).
    Must be fitted ONLY on training data to prevent data leakage into validation
    or test sets. Use sklearn.base.clone() to get a fresh copy per CV fold.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95, svd_solver="full", random_state=SEED)),
        ]
    )


# ---------------------------------------------------------------------------
# Smoke test (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running features.py smoke test...\n")
    for ct in ("dv", "boc"):
        print(f"--- case_type={ct} ---")

        # ── Frozen embeddings path ────────────────────────────────────────
        X, y_raw, dates, n_proc = load_split_data(ct, split="train")
        print(f"  X.shape          : {X.shape}")
        print(f"  Unique labels    : {sorted(set(y_raw))}")
        print(f"  Date range       : {dates.min().date()} → {dates.max().date()}")
        print(f"  n_processo sample: {n_proc[:3]}")

        y = encode_labels(y_raw, ct)
        unique, counts = np.unique(y, return_counts=True)
        print(f"  y encoded        : {dict(zip(unique.tolist(), counts.tolist()))}")

        pipeline = build_feature_pipeline()
        X_reduced = pipeline.fit_transform(X)
        pca_dim = pipeline.named_steps["pca"].n_components_
        print(f"  PCA output shape : {X_reduced.shape}  (pca_dim={pca_dim})")
        assert X_reduced.shape[1] < EMBED_DIM, "PCA did not reduce dimensionality."

        # ── Raw text path (for BERT fine-tuning) ─────────────────────────
        texts, y_raw_text, dates_text, n_proc_text = load_split_text_data(
            ct, split="train"
        )
        print(f"  texts count      : {len(texts)}")
        print(f"  text sample      : {texts[0][:80]!r}...")
        print(f"  Unique labels    : {sorted(set(y_raw_text))}")
        assert (
            len(texts) == len(y_raw_text) == len(dates_text) == len(n_proc_text)
        ), "load_split_text_data: length mismatch across returned arrays"
        y_text_encoded = encode_labels(y_raw_text, ct)
        unique_t, counts_t = np.unique(y_text_encoded, return_counts=True)
        print(
            f"  y encoded        : {dict(zip(unique_t.tolist(), counts_t.tolist()))}\n"
        )

    print("Smoke test passed.")
