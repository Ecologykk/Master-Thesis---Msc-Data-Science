#!/usr/bin/env python
# coding: utf-8

"""Tokenize legal case documents and extract frozen Legal BERTimbau embeddings.

Converted from a Jupyter notebook (retains the ``# In[...]:`` cell markers):
loads the cleaned DV (binary) and BoC (ternary) datasets from CSV, tokenizes
each document into overlapping sliding windows with Legal BERTimbau, mean-pools
the per-window token embeddings into a single frozen document embedding, and
writes the resulting embedding and tokenization-summary tables to parquet.

Selecting a CUDA device via ``nvidia-smi`` runs unconditionally at import
time (cheap). The expensive work — reading the two full dataset CSVs, the
in-process smoke test, downloading/loading the Legal BERTimbau model, and
tokenizing/encoding the full corpus — only runs under
``if __name__ == "__main__":``, so importing this module for its constants
or helper functions (e.g. ``predict.py``'s ``MAX_LENGTH``/``STRIDE``/
``WINDOW_BATCH_SIZE``/``mean_pool`` fallback import) is safe and side-effect
free. Run it directly to (re)generate the frozen embeddings:
``python src/modeling/dl/legal_bertimbau_tokenization_embedding.py``.

Seeds are fixed and decoding is greedy (`temperature=0.0`, `top_k=1`,
`seed=42`), so this script is deterministic on identical hardware. Results may
still diverge slightly on different hardware: GPU floating-point reductions
are non-associative and kernel selection, driver version and tensor-core
availability all change the order of operations. Expect small differences in
embeddings and occasional label flips on borderline cases. This does not
indicate a bug.
"""

# # Legal BERTimbau Tokenization and Frozen Embeddings
#
# This notebook loads the cleaned legal datasets from `new_clean_data/`, tokenizes documents with Legal BERTimbau, extracts frozen document embeddings with batching and sliding windows, and saves the final embedding tables in parquet format.

# In[1]:


# The following is to be uncommented when I can only use a few cuda devices at a time
# import os

# # Set this before importing torch/transformers if you want the notebook to see only these GPUs.
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5"
# print("CUDA_VISIBLE_DEVICES=", os.environ["CUDA_VISIBLE_DEVICES"])


# In[ ]:


import os
import subprocess


def get_all_cuda_device_ids():
    """Return all visible GPU indices as strings, or an empty list if unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except Exception:
        return []


ALL_CUDA_DEVICE_IDS = get_all_cuda_device_ids()
if ALL_CUDA_DEVICE_IDS:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(ALL_CUDA_DEVICE_IDS)
    print("CUDA_VISIBLE_DEVICES=", os.environ["CUDA_VISIBLE_DEVICES"])
else:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    print("No CUDA devices detected. CUDA_VISIBLE_DEVICES was left unset.")


# In[1]:


from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = _REPO_ROOT / "data" / "processed_data" / "eda"
DATASET_PATHS = [
    DATA_DIR / "binary" / "df_acordaos_dv_eda_binary.csv",
    DATA_DIR / "ternary" / "df_acordaos_boc_eda_ternary.csv",
]

if __name__ == "__main__":
    schema_summary = {}
    for path in DATASET_PATHS:
        df_preview = pd.read_csv(path, nrows=2)
        schema_summary[path.name] = {
            "columns": list(df_preview.columns),
            "dtypes": {
                column: str(dtype) for column, dtype in df_preview.dtypes.items()
            },
        }
    print(schema_summary)


# ## Pipeline Configuration
#
# This section defines the Legal BERTimbau model, the dataset-specific class columns, batching, and the output locations for parquet exports.

# In[2]:


from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer
import csv

MODEL_NAME = "stjiris/bert-large-portuguese-cased-legal-mlm-nli-sts-v1"
TEXT_COLUMN = "texto_integral_sem_decisao"
ID_COLUMN = "n_processo"
OUTPUT_DIR = _REPO_ROOT / "data" / "processed_data" / "bert_tokens_embedd"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_CONFIGS = [
    {
        "name": "dv_binary",
        "path": DATASET_PATHS[0],
        "class_column": "decisao_binaria",
        "embedding_output": OUTPUT_DIR
        / "df_acordaos_dv_eda_binary_legal_bert_embeddings.parquet",
        "tokenization_output": OUTPUT_DIR
        / "df_acordaos_dv_eda_binary_tokenization_summary.parquet",
    },
    {
        "name": "boc_ternary",
        "path": DATASET_PATHS[1],
        "class_column": "decisao_ternaria",
        "embedding_output": OUTPUT_DIR
        / "df_acordaos_boc_eda_ternary_legal_bert_embeddings.parquet",
        "tokenization_output": OUTPUT_DIR
        / "df_acordaos_boc_eda_ternary_tokenization_summary.parquet",
    },
]

METADATA_COLUMNS = [
    "url",
    "tribunal",
    ID_COLUMN,
    "juiz_relator",
    "data_acordao",
    "descritores",
    "decisao",
]

MAX_LENGTH = 512
STRIDE = 128
WINDOW_BATCH_SIZE = 64
MIN_WINDOW_BATCH_SIZE = 1
USE_FP16_INFERENCE = True


def select_best_device():
    """Pick the CUDA device with most free memory; fallback to CPU."""
    if not torch.cuda.is_available():
        return torch.device("cpu")

    best_idx = 0
    best_free = -1
    for idx in range(torch.cuda.device_count()):
        try:
            with torch.cuda.device(idx):
                free_mem, _ = torch.cuda.mem_get_info()
            if free_mem > best_free:
                best_free = free_mem
                best_idx = idx
        except Exception:
            continue

    return torch.device(f"cuda:{best_idx}")


# DEVICE = 0 select_best_device()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    selected_device_idx = DEVICE.index if DEVICE.index is not None else 0
    gpu_name = torch.cuda.get_device_name(selected_device_idx)
    gpu_memory = torch.cuda.get_device_properties(selected_device_idx).total_memory / (
        1024**3
    )
    free_mem, total_mem = torch.cuda.mem_get_info(selected_device_idx)
    print(
        {
            "model_name": MODEL_NAME,
            "device": str(DEVICE),
            "gpu_name": gpu_name,
            "gpu_memory_gb": f"{gpu_memory:.1f}",
            "gpu_free_memory_gb": f"{free_mem / (1024**3):.2f}",
            "gpu_total_memory_gb": f"{total_mem / (1024**3):.2f}",
            "window_batch_size": WINDOW_BATCH_SIZE,
            "min_window_batch_size": MIN_WINDOW_BATCH_SIZE,
            "fp16_inference": USE_FP16_INFERENCE,
            "max_length": MAX_LENGTH,
            "stride": STRIDE,
            "output_dir": str(OUTPUT_DIR),
        }
    )
else:
    print(
        {
            "model_name": MODEL_NAME,
            "device": str(DEVICE),
            "window_batch_size": WINDOW_BATCH_SIZE,
            "min_window_batch_size": MIN_WINDOW_BATCH_SIZE,
            "fp16_inference": USE_FP16_INFERENCE,
            "max_length": MAX_LENGTH,
            "stride": STRIDE,
            "output_dir": str(OUTPUT_DIR),
        }
    )


# In[3]:


def normalize_text(value):
    """Collapse a raw cell value into whitespace-normalized text.

    Args:
        value: Raw cell value from the source CSV (may be NaN).

    Returns:
        str: Empty string if ``value`` is NaN, otherwise the string form of
        ``value`` with runs of whitespace collapsed to single spaces and
        leading/trailing whitespace stripped.
    """
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def validate_input_frame(df, class_column):
    """Validate that a raw dataset frame has the columns and id integrity required downstream.

    Args:
        df: Raw dataset DataFrame as loaded from CSV.
        class_column: Name of the dataset-specific label column to require
            (e.g. "decisao_binaria" or "decisao_ternaria").

    Raises:
        KeyError: If any required column (metadata, text, or class_column) is
            missing from ``df``.
        ValueError: If ``ID_COLUMN`` contains null values or is not unique
            (duplicate document ids).
    """
    required_columns = set(METADATA_COLUMNS + [TEXT_COLUMN, class_column])
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    if df[ID_COLUMN].isna().any():
        raise ValueError(f"Column '{ID_COLUMN}' contains null document ids.")

    duplicated_ids = df[ID_COLUMN][df[ID_COLUMN].duplicated()].astype(str).tolist()
    if duplicated_ids:
        preview = duplicated_ids[:5]
        raise ValueError(
            f"Column '{ID_COLUMN}' must uniquely identify documents. Duplicate values found: {preview}"
        )


def load_dataset(config):
    """Load, validate, and clean one dataset CSV for tokenization.

    Reads the CSV named in ``config["path"]``, validates its schema via
    ``validate_input_frame``, drops rows with a missing text or label value,
    normalizes the text column, and drops rows left with empty text after
    normalization.

    Args:
        config: One entry of ``DATASET_CONFIGS`` — dict with keys "name",
            "path", and "class_column".

    Returns:
        pd.DataFrame: Cleaned dataset with added ``dataset_name`` (from
        ``config["name"]``) and ``class_label`` (string copy of
        ``config["class_column"]``) columns, and ``ID_COLUMN`` cast to str.
    """
    df = pd.read_csv(config["path"], quoting=csv.QUOTE_ALL)
    validate_input_frame(df, config["class_column"])

    df = df.dropna(subset=[TEXT_COLUMN, config["class_column"]]).copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].map(normalize_text)
    df = df[df[TEXT_COLUMN].astype(bool)].copy()

    df["dataset_name"] = config["name"]
    df["class_label"] = df[config["class_column"]].astype(str)
    df[ID_COLUMN] = df[ID_COLUMN].astype(str)
    return df


def mean_pool(last_hidden_state, attention_mask):
    """Mean-pool per-token hidden states into one embedding, ignoring padding.

    Args:
        last_hidden_state: Token-level hidden states from the encoder, shape
            (batch, seq_len, hidden_size).
        attention_mask: Attention mask, shape (batch, seq_len), with 1 for
            real tokens and 0 for padding.

    Returns:
        torch.Tensor, shape (batch, hidden_size) — the attention-mask-weighted
        mean of the token embeddings for each sequence in the batch.
    """
    expanded_mask = (
        attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    )
    masked_embeddings = last_hidden_state * expanded_mask
    summed_embeddings = masked_embeddings.sum(dim=1)
    token_counts = expanded_mask.sum(dim=1).clamp(min=1e-9)
    return summed_embeddings / token_counts


def tokenize_documents(df, tokenizer):
    """Tokenize every document into overlapping sliding windows.

    Each document is tokenized with truncation and ``return_overflowing_tokens``
    so that documents longer than ``MAX_LENGTH`` are split into multiple
    overlapping windows (stride ``STRIDE``) instead of being truncated to the
    first window only.

    Args:
        df: Cleaned dataset DataFrame (as returned by ``load_dataset``), must
            include ``dataset_name``, ``ID_COLUMN``, ``class_label``, and
            ``TEXT_COLUMN``.
        tokenizer: HuggingFace tokenizer for the Legal BERTimbau model.

    Returns:
        tuple: ``(window_records, tokenization_df)`` where ``window_records``
        is a list of dicts (one per window) with keys "doc_key", "input_ids",
        "attention_mask", "window_index"; and ``tokenization_df`` is a
        per-document pd.DataFrame summarizing window counts and a token
        preview, for QA purposes.
    """
    window_records = []
    tokenization_rows = []

    for row in tqdm(
        df.to_dict(orient="records"), total=len(df), desc="Tokenizing documents"
    ):
        text = row[TEXT_COLUMN]
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            stride=STRIDE,
            return_overflowing_tokens=True,
            return_attention_mask=True,
        )

        doc_key = (row["dataset_name"], row[ID_COLUMN])
        input_windows = tokenized["input_ids"]
        attention_windows = tokenized["attention_mask"]

        for window_index, (input_ids, attention_mask) in enumerate(
            zip(input_windows, attention_windows)
        ):
            window_records.append(
                {
                    "doc_key": doc_key,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "window_index": window_index,
                }
            )

        tokenization_rows.append(
            {
                "dataset_name": row["dataset_name"],
                ID_COLUMN: row[ID_COLUMN],
                "class_label": row["class_label"],
                "window_count": len(input_windows),
                "first_window_token_count": (
                    int(sum(attention_windows[0])) if attention_windows else 0
                ),
                "first_window_tokens_preview": (
                    tokenizer.convert_ids_to_tokens(input_windows[0][:24])
                    if input_windows
                    else []
                ),
            }
        )

    return window_records, pd.DataFrame(tokenization_rows)


def encode_windows(window_records, tokenizer, model):
    """Encode windows with adaptive batch size to survive CUDA OOM."""
    doc_embedding_sums = defaultdict(lambda: None)
    doc_weight_sums = defaultdict(float)

    if not window_records:
        return {}

    use_fp16 = bool(
        torch.cuda.is_available() and DEVICE.type == "cuda" and USE_FP16_INFERENCE
    )
    batch_size = WINDOW_BATCH_SIZE
    start = 0
    progress = tqdm(total=len(window_records), desc="Encoding windows")

    while start < len(window_records):
        batch_records = window_records[start : start + batch_size]
        batch_inputs = tokenizer.pad(
            [
                {
                    "input_ids": record["input_ids"],
                    "attention_mask": record["attention_mask"],
                }
                for record in batch_records
            ],
            padding=True,
            return_tensors="pt",
        )
        batch_inputs = {key: value.to(DEVICE) for key, value in batch_inputs.items()}

        try:
            with torch.inference_mode():
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=use_fp16
                ):
                    outputs = model(**batch_inputs)
                    pooled = mean_pool(
                        outputs.last_hidden_state, batch_inputs["attention_mask"]
                    )
                batch_embeddings = pooled.float().cpu().numpy()

            batch_weights = (
                batch_inputs["attention_mask"].sum(dim=1).cpu().numpy().astype(float)
            )

            for record, embedding, weight in zip(
                batch_records, batch_embeddings, batch_weights
            ):
                doc_key = record["doc_key"]
                weighted_embedding = embedding * weight
                if doc_embedding_sums[doc_key] is None:
                    doc_embedding_sums[doc_key] = weighted_embedding
                else:
                    doc_embedding_sums[doc_key] += weighted_embedding
                doc_weight_sums[doc_key] += weight

            start += len(batch_records)
            progress.update(len(batch_records))

        except torch.cuda.OutOfMemoryError:
            if DEVICE.type != "cuda":
                raise

            if batch_size <= MIN_WINDOW_BATCH_SIZE:
                raise RuntimeError(
                    "CUDA OOM even with minimum batch size. "
                    "Close other GPU jobs or run on CPU / smaller model."
                )

            previous_batch_size = batch_size
            batch_size = max(MIN_WINDOW_BATCH_SIZE, batch_size // 2)
            print(
                f"[WARN] CUDA OOM at window index {start} with batch_size={previous_batch_size}. "
                f"Retrying with batch_size={batch_size}."
            )
            torch.cuda.empty_cache()

        finally:
            del batch_inputs

    progress.close()

    return {
        doc_key: (doc_embedding_sums[doc_key] / doc_weight_sums[doc_key]).tolist()
        for doc_key in doc_embedding_sums
    }


def build_embedding_frame(df, tokenization_df, embedding_lookup):
    """Assemble the final per-document embedding table.

    Joins document metadata from ``df`` with the window count from
    ``tokenization_df`` and the pooled document embedding from
    ``embedding_lookup``, keyed by ``(dataset_name, ID_COLUMN)``.

    Args:
        df: Cleaned dataset DataFrame (as returned by ``load_dataset``).
        tokenization_df: Per-document tokenization summary, as returned by
            ``tokenize_documents``.
        embedding_lookup: Mapping of ``(dataset_name, ID_COLUMN)`` to the
            pooled document embedding (list[float]), as returned by
            ``encode_windows``.

    Returns:
        pd.DataFrame: One row per document with metadata columns, class
        label, text, window count, and the ``embedding`` column.
    """
    tokenization_lookup = tokenization_df.set_index(["dataset_name", ID_COLUMN])[
        "window_count"
    ].to_dict()

    embeddings_df = df[
        [
            "dataset_name",
            *METADATA_COLUMNS,
            "class_label",
            TEXT_COLUMN,
        ]
    ].copy()
    embeddings_df["doc_key"] = list(
        zip(embeddings_df["dataset_name"], embeddings_df[ID_COLUMN])
    )
    embeddings_df["window_count"] = embeddings_df["doc_key"].map(tokenization_lookup)
    embeddings_df["embedding"] = embeddings_df["doc_key"].map(embedding_lookup)
    embeddings_df = embeddings_df.drop(columns=["doc_key"])
    return embeddings_df


def process_dataset(config, tokenizer, model):
    """Run the full tokenize → encode → assemble pipeline for one dataset.

    Args:
        config: One entry of ``DATASET_CONFIGS`` — dict with keys "name",
            "path", "class_column", "embedding_output", "tokenization_output".
        tokenizer: HuggingFace tokenizer for the Legal BERTimbau model.
        model: Loaded Legal BERTimbau encoder (``AutoModel``), on ``DEVICE``.

    Returns:
        tuple: ``(df, tokenization_df, embeddings_df)`` — the cleaned input
        DataFrame, the per-document tokenization summary, and the final
        per-document embedding table (see ``build_embedding_frame``).
    """
    df = load_dataset(config)
    window_records, tokenization_df = tokenize_documents(df, tokenizer)
    embedding_lookup = encode_windows(window_records, tokenizer, model)
    embeddings_df = build_embedding_frame(df, tokenization_df, embedding_lookup)
    return df, tokenization_df, embeddings_df


# In[4]:


def run_pipeline_smoke_test():
    """Fast preflight test for tokenization + embedding + parquet roundtrip."""
    import tempfile

    class MockTokenizer:
        pad_token_id = 0

        def __call__(
            self,
            text,
            truncation=True,
            max_length=512,
            stride=128,
            return_overflowing_tokens=True,
            return_attention_mask=True,
        ):
            words = str(text).split()
            if not words:
                return {"input_ids": [], "attention_mask": []}

            # Deterministic pseudo token ids in a valid range.
            token_ids = [101] + [(len(word) % 200) + 100 for word in words] + [102]
            window_size = max_length
            step = max(window_size - stride, 1)

            input_windows = []
            attention_windows = []
            for start in range(0, len(token_ids), step):
                window_ids = token_ids[start : start + window_size]
                if not window_ids:
                    break
                input_windows.append(window_ids)
                attention_windows.append([1] * len(window_ids))
                if len(window_ids) < window_size:
                    break

            return {"input_ids": input_windows, "attention_mask": attention_windows}

        def pad(self, encoded_inputs, padding=True, return_tensors="pt"):
            max_len = max(len(item["input_ids"]) for item in encoded_inputs)
            batch_input_ids = []
            batch_attention_mask = []

            for item in encoded_inputs:
                ids = item["input_ids"]
                mask = item["attention_mask"]
                pad_len = max_len - len(ids)
                batch_input_ids.append(ids + [self.pad_token_id] * pad_len)
                batch_attention_mask.append(mask + [0] * pad_len)

            return {
                "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            }

        def convert_ids_to_tokens(self, ids):
            return [f"tok_{token_id}" for token_id in ids]

    class MockModel(torch.nn.Module):
        def __init__(self, hidden_size=16):
            super().__init__()
            self.hidden_size = hidden_size

        def forward(self, input_ids, attention_mask):
            # Deterministic embedding tensor derived from token ids.
            base = input_ids.unsqueeze(-1).float()
            scale = torch.arange(
                1, self.hidden_size + 1, device=input_ids.device
            ).float()
            last_hidden_state = (
                torch.sin(base / scale) * attention_mask.unsqueeze(-1).float()
            )
            return type("MockOutput", (), {"last_hidden_state": last_hidden_state})

    smoke_df = pd.DataFrame(
        {
            "dataset_name": ["smoke", "smoke"],
            "url": ["http://example.com/1", "http://example.com/2"],
            "tribunal": ["trib_a", "trib_b"],
            ID_COLUMN: ["smk-001", "smk-002"],
            "juiz_relator": ["judge_a", "judge_b"],
            "data_acordao": ["2024-01-01", "2024-01-02"],
            "descritores": ["teste", "pipeline"],
            "decisao": ["A", "B"],
            TEXT_COLUMN: [
                "texto juridico para testar o pipeline de tokenizacao e embedding",
                "outro texto curto para validar a serializacao em parquet",
            ],
            "class_label": ["1", "0"],
        }
    )

    smoke_tokenizer = MockTokenizer()
    smoke_model = MockModel(hidden_size=16).to(DEVICE)
    smoke_model.eval()

    window_records, tokenization_df = tokenize_documents(smoke_df, smoke_tokenizer)
    assert window_records, "Smoke test failed: no tokenization windows generated."
    assert len(tokenization_df) == len(
        smoke_df
    ), "Smoke test failed: tokenization row mismatch."

    embedding_lookup = encode_windows(window_records, smoke_tokenizer, smoke_model)
    assert len(embedding_lookup) == len(
        smoke_df
    ), "Smoke test failed: missing document embeddings."

    embeddings_df = build_embedding_frame(smoke_df, tokenization_df, embedding_lookup)
    assert len(embeddings_df) == len(
        smoke_df
    ), "Smoke test failed: embedding frame row mismatch."
    assert (
        embeddings_df["embedding"].notna().all()
    ), "Smoke test failed: found empty embeddings."

    embedding_dims = {len(vec) for vec in embeddings_df["embedding"]}
    assert embedding_dims == {
        16
    }, f"Smoke test failed: unexpected embedding dimensions {embedding_dims}."

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        smoke_tokenization_path = tmp_dir_path / "smoke_tokenization.parquet"
        smoke_embedding_path = tmp_dir_path / "smoke_embeddings.parquet"

        tokenization_df.to_parquet(
            smoke_tokenization_path, index=False, engine="fastparquet"
        )
        embeddings_df.to_parquet(
            smoke_embedding_path, index=False, engine="fastparquet"
        )

        tokenization_roundtrip = pd.read_parquet(
            smoke_tokenization_path, engine="fastparquet"
        )
        embeddings_roundtrip = pd.read_parquet(
            smoke_embedding_path, engine="fastparquet"
        )

        assert len(tokenization_roundtrip) == len(
            tokenization_df
        ), "Smoke test failed: tokenization parquet roundtrip mismatch."
        assert len(embeddings_roundtrip) == len(
            embeddings_df
        ), "Smoke test failed: embeddings parquet roundtrip mismatch."

    print(
        "Smoke test passed: tokenization, embedding, and parquet roundtrip are working."
    )
    return True


if __name__ == "__main__":
    smoke_test_ok = run_pipeline_smoke_test()
    assert (
        smoke_test_ok
    ), "Smoke test failed. Fix issues before running full embeddings."

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"Loading {MODEL_NAME} ...")
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()

    if DEVICE.type == "cuda" and USE_FP16_INFERENCE:
        model.half()
        print("Model converted to float16 for inference.")

    print(f"Model loaded on {DEVICE}")
    torch.cuda.empty_cache()

    results = {}
    for config in DATASET_CONFIGS:
        print(f"\nProcessing {config['name']} from {config['path']}")
        dataset_df, tokenization_df, embeddings_df = process_dataset(
            config, tokenizer, model
        )

        tokenization_df.to_parquet(
            config["tokenization_output"], index=False, engine="fastparquet"
        )
        embeddings_df.to_parquet(
            config["embedding_output"], index=False, engine="fastparquet"
        )

        results[config["name"]] = {
            "documents": len(dataset_df),
            "tokenization_output": str(config["tokenization_output"]),
            "embedding_output": str(config["embedding_output"]),
            "embedding_dimension": (
                len(embeddings_df.iloc[0]["embedding"])
                if not embeddings_df.empty
                else 0
            ),
            "average_window_count": (
                float(tokenization_df["window_count"].mean())
                if not tokenization_df.empty
                else 0.0
            ),
        }
        print(results[config["name"]])

results
