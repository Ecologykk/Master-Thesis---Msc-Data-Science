"""extract_bert_embeddings.py: extract fine-tuned document embeddings for visualisation.

Extract document embeddings from a fine-tuned LegalBertForClassification model
for UMAP visualisation and class-overlap analysis.

For each document the sliding-window mean-pooled [CLS] representation is
computed **before** the classification head — this is the 1024-dim vector the
model uses as its final document representation.

Outputs a parquet compatible with ``eda.plotly_umap_3d()``:
    n_processo                — case identifier
    class_label               — raw string decision label (e.g. "FAVORÁVEL")
    texto_integral_sem_decisao — full document text (truncated for display)
    embedding                 — list[float], shape (1024,)
    split                     — "train" or "gold_test"

Usage
-----
# Both case types, bert_v4 model
python extract_bert_embeddings.py --run_name bert_v4

# Single case type
python extract_bert_embeddings.py --case_type dv --run_name bert_v4

Seeds are fixed and decoding is greedy (`temperature=0.0`, `top_k=1`,
`seed=42`), so this script is deterministic on identical hardware. Results may
still diverge slightly on different hardware: GPU floating-point reductions
are non-associative and kernel selection, driver version and tensor-core
availability all change the order of operations. Expect small differences in
embeddings and occasional label flips on borderline cases. This does not
indicate a bug.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from features import load_split_text_data
from train import (
    BERT_MAX_WINDOWS,
    BERT_WINDOW_BATCH,
    DEFAULT_OUTPUT_DIR,
    LegalBertForClassification,
    _tokenize_sliding_window,
)
from predict import load_bert_model

# Fine-tuned embeddings are saved alongside the frozen-BERT embeddings parquets
_REPO_ROOT = _HERE.parents[2]
_EMB_DIR = _REPO_ROOT / "data" / "processed_data" / "bert_tokens_embedd"


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------


def _get_doc_embedding(
    model: LegalBertForClassification,
    text: str,
    tokenizer,
    max_windows: int,
    window_mbatch: int,
    device: torch.device,
) -> np.ndarray:
    """Return mean-pooled CLS embedding for one document (before classifier head).

    Args:
        model: Fine-tuned ``LegalBertForClassification``, in eval mode.
        text: Raw document text.
        tokenizer: HuggingFace tokenizer matching ``model``.
        max_windows: Cap on sliding windows per document.
        window_mbatch: Window mini-batch size through the encoder.
        device: Torch device to run inference on.

    Returns:
        np.ndarray, shape (hidden_size,), dtype float32 — 1024-dim for BERT-large.
    """
    input_ids, attention_mask = _tokenize_sliding_window(
        text, tokenizer, max_windows, device
    )
    all_cls: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, input_ids.shape[0], window_mbatch):
            ids_mb = input_ids[start : start + window_mbatch]
            mask_mb = attention_mask[start : start + window_mbatch]
            cls = model.get_cls_repr(ids_mb, mask_mb)
            all_cls.append(cls.cpu().float())
    return torch.cat(all_cls, dim=0).mean(dim=0).numpy()


def extract_embeddings_for_split(
    case_type: str,
    split: str,
    model: LegalBertForClassification,
    tokenizer,
    device: torch.device,
    max_windows: int = BERT_MAX_WINDOWS,
    window_mbatch: int = BERT_WINDOW_BATCH,
) -> pd.DataFrame:
    """Extract embeddings for all documents in one split.

    Args:
        case_type: "dv" or "boc".
        split: "train" or "gold_test".
        model: Fine-tuned ``LegalBertForClassification``, in eval mode.
        tokenizer: HuggingFace tokenizer matching ``model``.
        device: Torch device to run inference on.
        max_windows: Cap on sliding windows per document.
        window_mbatch: Window mini-batch size through the encoder.

    Returns:
        pd.DataFrame with columns: n_processo, class_label,
        texto_integral_sem_decisao, embedding, split.
    """
    texts, y_raw, _, n_processo = load_split_text_data(case_type, split=split)

    embeddings: list[list[float]] = []
    for text in tqdm(texts, desc=f"  {case_type} {split}", unit="doc", leave=False):
        emb = _get_doc_embedding(
            model, text, tokenizer, max_windows, window_mbatch, device
        )
        embeddings.append(emb.tolist())

    return pd.DataFrame(
        {
            "n_processo": n_processo,
            "class_label": y_raw,
            "texto_integral_sem_decisao": texts,
            "embedding": embeddings,
            "split": split,
        }
    )


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------


def extract_and_save(
    case_type: str,
    run_name: str,
    model_dir: Path = DEFAULT_OUTPUT_DIR,
    output_dir: Path = _EMB_DIR,
    device: torch.device | None = None,
) -> Path:
    """Load fine-tuned model, extract embeddings for all splits, save parquet.

    Args:
        case_type: "dv" or "boc".
        run_name: Model version tag, e.g. "bert_v4".
        model_dir: Root model artefact directory.
        output_dir: Directory for the output parquet.
        device: Torch device (auto-selected if None).

    Returns:
        Path to the saved parquet file.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[{case_type.upper()}] Loading fine-tuned model ({run_name}) ...")
    model = load_bert_model(
        case_type, model_dir=model_dir, device=device, run_name=run_name
    )
    model.eval()

    from transformers import AutoTokenizer

    model_name = getattr(
        model, "model_name", "stjiris/bert-large-portuguese-cased-legal-mlm-nli-sts-v1"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    max_windows = getattr(model, "max_windows", BERT_MAX_WINDOWS)
    window_mbatch = getattr(model, "window_mbatch", BERT_WINDOW_BATCH)

    dfs: list[pd.DataFrame] = []
    for split in ("train", "gold_test"):
        print(f"  Extracting {split} embeddings ...")
        df_split = extract_embeddings_for_split(
            case_type, split, model, tokenizer, device, max_windows, window_mbatch
        )
        dfs.append(df_split)
        print(f"    → {len(df_split)} documents")

    df_all = pd.concat(dfs, ignore_index=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{run_name}_{case_type}_finetuned_embeddings.parquet"
    df_all.to_parquet(out_path, index=False, engine="pyarrow")

    print(f"  Saved → {out_path}  ({len(df_all)} total documents)")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the embedding-extraction script."""
    parser = argparse.ArgumentParser(
        description="Extract fine-tuned BERT embeddings for UMAP visualisation."
    )
    parser.add_argument(
        "--case_type",
        choices=["dv", "boc"],
        default=None,
        help="Case type to process. Omit to process both.",
    )
    parser.add_argument(
        "--run_name",
        required=True,
        help="Model version tag used when saving (e.g. bert_v4).",
    )
    parser.add_argument(
        "--model_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Root model artefact directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    case_types = [args.case_type] if args.case_type else ["dv", "boc"]
    for ct in case_types:
        extract_and_save(
            case_type=ct,
            run_name=args.run_name,
            model_dir=Path(args.model_dir),
        )
    print("\nDone.")
