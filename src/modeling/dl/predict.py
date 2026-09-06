"""predict.py: inference and gold-test evaluation for the Legal BERTimbau classification head.

Provides:
  BERTConfig                    — tokenisation constants for the frozen encoder

  load_model(case_type, model_dir, device)
      → (fitted_pipeline, LegalBertClassifier)

  predict(X, fitted_pipeline, model, case_type, device)
      → np.ndarray[int64]  — class predictions from pre-extracted embeddings

  predict_proba(X, fitted_pipeline, model, case_type, device)
      → np.ndarray[float32]  — shape (n, n_classes)

  make_bert_predict_fn(fitted_pipeline, model, bert_model, tokenizer, case_type,
                       bert_config, device)
      → Callable[[list[str]], np.ndarray]
        SHAP-compatible: list[str] → (n, n_classes) float32.
        Pass directly to explanation.compute_shap_explanation or
        explanation.run_shap_explanation_pipeline.

  run_gold_test_predictions(case_type, fitted_pipeline, model, model_dir, device)
      → dict: y_pred, y_true, y_proba, n_processo
        Model-agnostic: accepts any (pipeline, model) pair or loads from disk.

Gold test split is accessed ONLY inside run_gold_test_predictions.
Never call load_split_data(split='gold_test') elsewhere.
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
import torch

# classification.py lives in src/evaluation/ — two levels up from src/modeling/dl/
_eval_dir = str(Path(__file__).resolve().parents[2] / "evaluation")
if _eval_dir not in sys.path:
    sys.path.insert(0, _eval_dir)
from classification import LABEL_NAMES, plot_forest, run_classification_evaluation
from explanation import LABEL_NAMES_DV, LABEL_NAMES_BOC, run_shap_explanation_pipeline

from features import (
    IS_BINARY,
    N_CLASSES,
    encode_labels,
    load_bert_train_val_split_with_ids,
    load_split_data,
    load_split_text_data,
)
from train import (
    BERT_DROPOUT,
    DEFAULT_OUTPUT_DIR,
    DROPOUT_PROB,
    HIDDEN_DIM,
    LegalBertClassifier,
    LegalBertForClassification,
    predict_from_logits,
    proba_from_logits,
)

# ---------------------------------------------------------------------------
# Conditional import from the embedding module.
#
# legal_bertimbau_tokenization_embedding.py guards its expensive top-level
# work (CSV loads, smoke test, model download/tokenization/encoding) behind
# `if __name__ == "__main__":`, so importing it here is safe. The try/except
# is kept only as a defensive fallback in case that module is ever moved,
# renamed, or its dependencies (e.g. the HF model download) are unavailable.
# ---------------------------------------------------------------------------
try:
    from legal_bertimbau_tokenization_embedding import (  # type: ignore
        MAX_LENGTH as _MAX_LENGTH,
        STRIDE as _STRIDE,
        WINDOW_BATCH_SIZE as _WINDOW_BATCH_SIZE,
        mean_pool as _mean_pool,
    )
except Exception:
    _MAX_LENGTH = 512
    _STRIDE = 128
    _WINDOW_BATCH_SIZE = 64

    def _mean_pool(
        last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Fallback mean pool (mirrors legal_bertimbau_tokenization_embedding.mean_pool)."""
        expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked = last_hidden_state * expanded
        return masked.sum(dim=1) / expanded.sum(dim=1).clamp(min=1e-9)


# ---------------------------------------------------------------------------
# BERTConfig — tokenisation constants for the frozen encoder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BERTConfig:
    """Tokenisation and batching parameters for the Legal BERTimbau encoder.

    Defaults mirror the values used when the parquet embeddings were generated
    (legal_bertimbau_tokenization_embedding.py).  Pass a custom BERTConfig
    only if you regenerated the embeddings with different settings.

    Attributes:
        model_name: HuggingFace model identifier.
        max_length: Maximum tokens per window (BERT positional limit = 512).
        stride: Overlap between consecutive windows.
        window_batch_size: GPU micro-batch size for window encoding.
    """

    model_name: str = "stjiris/bert-large-portuguese-cased-legal-mlm-nli-sts-v1"
    max_length: int = field(default_factory=lambda: _MAX_LENGTH)
    stride: int = field(default_factory=lambda: _STRIDE)
    window_batch_size: int = field(default_factory=lambda: _WINDOW_BATCH_SIZE)


# Module-level default — used when callers omit bert_config
DEFAULT_BERT_CONFIG = BERTConfig()


# ---------------------------------------------------------------------------
# Model I/O
# ---------------------------------------------------------------------------


def load_model(
    case_type: str,
    model_dir: Path = DEFAULT_OUTPUT_DIR,
    device: torch.device | None = None,
    run_name: str | None = None,
) -> tuple:
    """Load fitted sklearn pipeline and PyTorch classifier from disk.

    Artefacts expected (written by train.save_model):
        {model_dir}/{case_type}/{run_name}_feature_pipeline.joblib
        {model_dir}/{case_type}/{run_name}_classifier.pt

    If run_name is None, falls back to the legacy bare filenames
    (feature_pipeline.joblib / classifier.pt).

    Args:
        case_type: "dv" or "boc".
        model_dir: Root model directory (parent of the case_type subdirectory).
        device: Torch device to load the classifier onto (auto-selected if None).
        run_name: Prefix used when saving (e.g. "v1"). None falls back to the
            legacy bare filenames.

    Returns:
        Tuple of (fitted_pipeline, model): a fitted
        ``sklearn.pipeline.Pipeline`` and a ``LegalBertClassifier`` in eval
        mode on ``device``.

    Raises:
        FileNotFoundError: If the pipeline or classifier artefact is missing.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    artifact_dir = Path(model_dir) / case_type
    prefix = f"{run_name}_" if run_name else ""
    pipeline_path = artifact_dir / f"{prefix}feature_pipeline.joblib"
    classifier_path = artifact_dir / f"{prefix}classifier.pt"

    if not pipeline_path.exists():
        raise FileNotFoundError(f"Pipeline not found: {pipeline_path}")
    if not classifier_path.exists():
        raise FileNotFoundError(f"Classifier not found: {classifier_path}")

    fitted_pipeline = joblib.load(pipeline_path)

    # weights_only=False: checkpoint contains non-tensor metadata (str, int, float)
    checkpoint = torch.load(classifier_path, map_location=device, weights_only=False)
    model = LegalBertClassifier(
        pca_dim=checkpoint["pca_dim"],
        output_dim=checkpoint["output_dim"],
        hidden_dim=checkpoint.get("hidden_dim", HIDDEN_DIM),
        dropout_prob=checkpoint.get("dropout_prob", DROPOUT_PROB),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return fitted_pipeline, model


def load_bert_model(
    case_type: str,
    model_dir: Path = DEFAULT_OUTPUT_DIR,
    device: torch.device | None = None,
    run_name: str | None = None,
) -> LegalBertForClassification:
    """Load a fine-tuned LegalBertForClassification from disk.

    Artefact expected (written by train.save_bert_model):
        {model_dir}/{case_type}/{run_name}_bert_classifier.pt

    Args:
        case_type: "dv" or "boc".
        model_dir: Root model directory (parent of the case_type subdirectory).
        device: Torch device to load the model onto (auto-selected if None).
        run_name: Prefix used when saving (e.g. "bert_v1").

    Returns:
        The fine-tuned ``LegalBertForClassification`` in eval mode on
        ``device``. The returned model carries a ``model_name`` attribute
        (the HuggingFace identifier stored in the checkpoint) for use by
        ``predict_bert``.

    Raises:
        FileNotFoundError: If the classifier artefact is missing.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    artifact_dir = Path(model_dir) / case_type
    prefix = f"{run_name}_" if run_name else ""
    classifier_path = artifact_dir / f"{prefix}{case_type}_bert_classifier.pt"

    if not classifier_path.exists():
        raise FileNotFoundError(f"BERT classifier not found: {classifier_path}")

    checkpoint = torch.load(classifier_path, map_location=device, weights_only=False)
    model = LegalBertForClassification(
        n_classes=checkpoint["n_classes"],
        dropout_prob=checkpoint.get("dropout_prob", BERT_DROPOUT),
        n_frozen_layers=checkpoint.get("n_frozen_layers", 0),
    )
    model.load_state_dict(checkpoint["state_dict"])
    # Attach stored attributes for downstream inference
    model.model_name = checkpoint.get("model_name", DEFAULT_BERT_CONFIG.model_name)
    model.max_windows = checkpoint.get("max_windows", DEFAULT_BERT_CONFIG.max_length)
    model.window_mbatch = checkpoint.get(
        "window_mbatch", DEFAULT_BERT_CONFIG.window_batch_size
    )
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inference on pre-extracted embeddings
# ---------------------------------------------------------------------------


def predict(
    X: np.ndarray,
    fitted_pipeline,
    model,
    case_type: str,
    device: torch.device | None = None,
) -> np.ndarray:
    """Predict integer class labels from pre-extracted BERT embeddings.

    Model-agnostic: works with both PyTorch (LegalBertClassifier / nn.Module)
    and sklearn estimators (anything with a .predict() method).

    Args:
        X: Raw frozen BERT embeddings, not yet scaled/PCA-reduced, shape (n, 1024).
        fitted_pipeline: Fitted ``sklearn.pipeline.Pipeline`` (scaler + PCA).
        model: A ``LegalBertClassifier`` (or other ``nn.Module``) or a sklearn
            estimator exposing ``.predict()``.
        case_type: "dv" or "boc".
        device: Torch device for PyTorch models (auto-selected if None,
            ignored for sklearn models).

    Returns:
        np.ndarray of int64, shape (n,) — predicted class labels.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_reduced = fitted_pipeline.transform(X).astype(np.float32)

    if isinstance(model, torch.nn.Module):
        X_t = torch.tensor(X_reduced, dtype=torch.float32, device=device)
        model.eval()
        with torch.no_grad():
            logits = model(X_t)
        return predict_from_logits(logits, IS_BINARY[case_type])
    else:
        return model.predict(X_reduced).astype(np.int64)


def predict_proba(
    X: np.ndarray,
    fitted_pipeline,
    model,
    case_type: str,
    device: torch.device | None = None,
) -> np.ndarray:
    """Return probability estimates from pre-extracted BERT embeddings.

    Args:
        X: Raw frozen BERT embeddings, not yet scaled/PCA-reduced, shape (n, 1024).
        fitted_pipeline: Fitted ``sklearn.pipeline.Pipeline`` (scaler + PCA).
        model: A ``LegalBertClassifier`` (or other ``nn.Module``) or a sklearn
            estimator exposing ``.predict_proba()``.
        case_type: "dv" or "boc".
        device: Torch device for PyTorch models (auto-selected if None,
            ignored for sklearn models).

    Returns:
        np.ndarray of float32, shape (n, n_classes) — rows sum to 1.0.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_reduced = fitted_pipeline.transform(X).astype(np.float32)

    if isinstance(model, torch.nn.Module):
        X_t = torch.tensor(X_reduced, dtype=torch.float32, device=device)
        model.eval()
        with torch.no_grad():
            logits = model(X_t)
        return proba_from_logits(logits, IS_BINARY[case_type])
    else:
        return model.predict_proba(X_reduced).astype(np.float32)


# ---------------------------------------------------------------------------
# Inference on fine-tuned BERT (end-to-end)
# ---------------------------------------------------------------------------


def _run_bert_batched(
    texts: list[str],
    model: LegalBertForClassification,
    tokenizer,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Encode texts with sliding-window aggregation; returns concatenated logits on CPU.

    Each document is tokenized into overlapping 512-token windows (stride 128).
    Windows are capped at ``model.max_windows`` using equally-spaced sampling so
    that documents of any length receive full coverage.  The [CLS] representation
    of each window is mean-pooled into a single document vector before the
    classification head is applied — matching the training procedure exactly.

    ``batch_size`` controls the window mini-batch size (not the document batch
    size); documents are always processed one at a time.
    """
    from train import _tokenize_sliding_window, _forward_sliding_window

    use_amp = device.type == "cuda"
    max_windows = getattr(model, "max_windows", 64)
    window_mbatch = getattr(model, "window_mbatch", batch_size)

    all_logits: list[torch.Tensor] = []
    with torch.no_grad():
        for text in texts:
            input_ids, attention_mask = _tokenize_sliding_window(
                text, tokenizer, max_windows, device
            )
            logits = _forward_sliding_window(
                model, input_ids, attention_mask, window_mbatch, use_amp
            )
            all_logits.append(logits.cpu())
    return torch.cat(all_logits, dim=0)


def predict_bert(
    texts: list[str],
    model: LegalBertForClassification,
    case_type: str,
    device: torch.device | None = None,
    batch_size: int = 8,
) -> np.ndarray:
    """Predict integer class labels from raw texts using a fine-tuned BERT model.

    Args:
        texts: Raw document texts.
        model: Fine-tuned ``LegalBertForClassification``, in eval mode.
        case_type: "dv" or "boc".
        device: Torch device (auto-selected if None).
        batch_size: Window mini-batch size passed to sliding-window inference.

    Returns:
        np.ndarray of int64, shape (n,) — predicted class labels.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import AutoTokenizer

    model_name = getattr(model, "model_name", DEFAULT_BERT_CONFIG.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model.to(device).eval()

    logits = _run_bert_batched(texts, model, tokenizer, device, batch_size)
    return predict_from_logits(logits, IS_BINARY[case_type])


def predict_bert_proba(
    texts: list[str],
    model: LegalBertForClassification,
    case_type: str,
    device: torch.device | None = None,
    batch_size: int = 8,
) -> np.ndarray:
    """Return probability estimates from raw texts using a fine-tuned BERT model.

    Args:
        texts: Raw document texts.
        model: Fine-tuned ``LegalBertForClassification``, in eval mode.
        case_type: "dv" or "boc".
        device: Torch device (auto-selected if None).
        batch_size: Window mini-batch size passed to sliding-window inference.

    Returns:
        np.ndarray of float32, shape (n, n_classes) — rows sum to 1.0.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import AutoTokenizer

    model_name = getattr(model, "model_name", DEFAULT_BERT_CONFIG.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model.to(device).eval()

    logits = _run_bert_batched(texts, model, tokenizer, device, batch_size)
    return proba_from_logits(logits, IS_BINARY[case_type])


# ---------------------------------------------------------------------------
# BERT re-encoding bridge for SHAP
# ---------------------------------------------------------------------------


def _encode_texts_to_embeddings(
    texts: list[str],
    tokenizer,
    bert_model: torch.nn.Module,
    device: torch.device,
    bert_config: BERTConfig = DEFAULT_BERT_CONFIG,
) -> np.ndarray:
    """Encode raw text strings to frozen BERT document embeddings.

    Implements the same sliding-window + weighted-mean-pool strategy as
    legal_bertimbau_tokenization_embedding.py:
      1. Tokenise each text into overlapping windows (max_length, stride).
      2. Mean-pool each window's token embeddings (excluding padding).
      3. Token-count-weighted average across windows → one embed_dim vector.

    encode_windows() from the embedding module is NOT used here because it
    relies on a module-level DEVICE global; this function accepts device as
    a parameter.  _mean_pool is imported from the embedding module (or falls
    back to the local equivalent above) for consistency.

    Parameters
    ----------
    texts : list[str]
    tokenizer : HuggingFace AutoTokenizer
    bert_model : AutoModel, eval mode, already on device
    device : torch.device
    bert_config : BERTConfig — tokenisation parameters (default: DEFAULT_BERT_CONFIG)

    Returns:
    -------
    np.ndarray, shape (n_texts, embed_dim), dtype float32
    """
    all_embeddings: list[np.ndarray] = []

    for text in texts:
        encoded = tokenizer(
            text or " ",  # guard against empty / SHAP-masked strings
            truncation=True,
            max_length=bert_config.max_length,
            stride=bert_config.stride,
            return_overflowing_tokens=True,
            return_attention_mask=True,
        )
        windows = list(zip(encoded["input_ids"], encoded["attention_mask"]))

        win_embeddings: list[np.ndarray] = []
        win_weights: list[np.ndarray] = []

        for batch_start in range(0, len(windows), bert_config.window_batch_size):
            batch_windows = windows[
                batch_start : batch_start + bert_config.window_batch_size
            ]
            padded = tokenizer.pad(
                [
                    {"input_ids": ids, "attention_mask": mask}
                    for ids, mask in batch_windows
                ],
                padding=True,
                return_tensors="pt",
            )
            padded = {k: v.to(device) for k, v in padded.items()}

            with torch.no_grad():
                outputs = bert_model(**padded)
                pooled = _mean_pool(outputs.last_hidden_state, padded["attention_mask"])
                batch_embs = pooled.float().cpu().numpy()  # (batch, embed_dim)
                batch_wts = padded["attention_mask"].sum(dim=1).float().cpu().numpy()

            win_embeddings.append(batch_embs)
            win_weights.append(batch_wts)

        all_embs = np.vstack(win_embeddings)  # (n_windows, embed_dim)
        all_wts = np.concatenate(win_weights)  # (n_windows,)
        doc_emb = (all_embs * all_wts[:, None]).sum(axis=0) / all_wts.sum()
        all_embeddings.append(doc_emb.astype(np.float32))

    return np.stack(all_embeddings)  # (n_texts, embed_dim)


def make_bert_predict_fn(
    fitted_pipeline,
    model,
    bert_model: torch.nn.Module,
    tokenizer,
    case_type: str,
    bert_config: BERTConfig = DEFAULT_BERT_CONFIG,
    device: torch.device | None = None,
) -> Callable[[list[str]], np.ndarray]:
    """Build a SHAP-compatible predict function backed by frozen BERT.

    The returned callable satisfies the contract expected by
    ``explanation.compute_shap_explanation`` and
    ``explanation.run_shap_explanation_pipeline``:

        predict_fn : list[str] → np.ndarray shape (n, n_classes), float32

    SHAP calls predict_fn hundreds of times with masked variants of the input
    text (words replaced by spaces).  Each call:
      1. Re-encodes the masked texts via sliding-window frozen BERT → embed_dim
      2. Applies fitted StandardScaler + PCA pipeline
      3. Runs the classification head → probabilities

    Args:
        fitted_pipeline: Fitted ``sklearn.pipeline.Pipeline`` (scaler + PCA).
        model: A ``LegalBertClassifier`` (or other ``nn.Module``) or a sklearn
            estimator exposing ``.predict_proba()``.
        bert_model: Frozen BERT encoder (transformers ``AutoModel``);
            ``hidden_size`` must match the embed_dim used when the parquet
            embeddings were generated (1024).
        tokenizer: HuggingFace ``AutoTokenizer`` matching ``bert_model``.
        case_type: "dv" or "boc".
        bert_config: Tokenisation parameters (default: ``DEFAULT_BERT_CONFIG``).
        device: Torch device to run ``bert_model``/``model`` on (auto-selected
            if None).

    Returns:
        A callable ``predict_fn(texts) -> np.ndarray`` of shape
        (n, n_classes), float32.

    Example:
    -------
    from transformers import AutoModel, AutoTokenizer
    pipeline, mlp = load_model("dv", model_dir)
    bert = AutoModel.from_pretrained(DEFAULT_BERT_CONFIG.model_name).eval()
    tok  = AutoTokenizer.from_pretrained(DEFAULT_BERT_CONFIG.model_name)
    predict_fn = make_bert_predict_fn(pipeline, mlp, bert, tok, "dv")
    # → pass predict_fn to explanation.run_shap_explanation_pipeline(...)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bert_model = bert_model.to(device).eval()
    if isinstance(model, torch.nn.Module):
        model = model.to(device).eval()

    def predict_fn(texts: list[str]) -> np.ndarray:
        X_raw = _encode_texts_to_embeddings(
            texts, tokenizer, bert_model, device, bert_config
        )
        return predict_proba(X_raw, fitted_pipeline, model, case_type, device)

    return predict_fn


# ---------------------------------------------------------------------------
# Gold-test evaluation — model-agnostic entry point
# ---------------------------------------------------------------------------


def run_gold_test_predictions(
    case_type: str,
    fitted_pipeline=None,
    model=None,
    model_dir: Path | None = None,
    device: torch.device | None = None,
    run_name: str | None = None,
) -> dict:
    """Load a trained model and evaluate on the held-out gold test split.

    Gold test data is NEVER used during CV or final model training.
    This is the ONLY function that calls load_split_data(split='gold_test').

    Model supply — exactly one of:
      a) ``model_dir`` is given  → load (fitted_pipeline, model) from disk via
         load_model().  Overrides any provided fitted_pipeline / model.
      b) Both ``fitted_pipeline`` and ``model`` are given directly → use as-is.
         Accepts any model with a PyTorch or sklearn interface (see predict /
         predict_proba).  Useful for evaluating Raw BERT embeddings + any
         downstream classifier without saving to disk first.

    Args:
        case_type: "dv" or "boc".
        fitted_pipeline: Fitted ``sklearn.pipeline.Pipeline``, or None to load
            from disk via ``model_dir``.
        model: A trained model (PyTorch or sklearn interface), or None to
            load from disk via ``model_dir``.
        model_dir: Root model directory (parent of the case_type
            subdirectory). If given, overrides ``fitted_pipeline`` / ``model``
            with disk-loaded versions.
        device: Torch device (auto-selected if None).
        run_name: Prefix used when saving, forwarded to ``load_model`` when
            ``model_dir`` is given.

    Returns:
        dict with keys:
            y_pred     : np.ndarray[int64],   shape (n,)
            y_true     : np.ndarray[int64],   shape (n,)
            y_proba    : np.ndarray[float32], shape (n, n_classes)
            n_processo : list[str]

    Raises:
        ValueError: If ``model_dir`` is None and either ``fitted_pipeline``
            or ``model`` is also None.

    Example — bootstrap CI
    ----------------------
    from classification import bootstrap_evaluation, compute_confidence_intervals
    from classification import BINARY_LABELS, TERNARY_LABELS
    labels  = BINARY_LABELS if case_type == "dv" else TERNARY_LABELS
    results = run_gold_test_predictions("dv", model_dir=model_dir)
    m_list  = bootstrap_evaluation(results["y_true"], results["y_pred"], labels)
    ci      = compute_confidence_intervals(m_list)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_dir is not None:
        print(f"  Loading model from {Path(model_dir) / case_type} ...")
        fitted_pipeline, model = load_model(case_type, model_dir, device, run_name)
    elif fitted_pipeline is None or model is None:
        raise ValueError("Provide either model_dir or both fitted_pipeline and model.")

    # Gold test split — loaded here and ONLY here
    print("  Loading gold test data ...")
    X, y_raw, _, n_processo = load_split_data(case_type, split="gold_test")
    y_true = encode_labels(y_raw, case_type)

    print(f"  Running predictions on {len(X)} gold test cases ...")
    y_pred = predict(X, fitted_pipeline, model, case_type, device)
    y_proba = predict_proba(X, fitted_pipeline, model, case_type, device)

    return {
        "y_pred": y_pred,
        "y_true": y_true,
        "y_proba": y_proba,
        "n_processo": n_processo,
    }


def run_gold_test_predictions_bert(
    case_type: str,
    model: LegalBertForClassification | None = None,
    model_dir: Path | None = None,
    device: torch.device | None = None,
    run_name: str | None = None,
    batch_size: int = 8,
) -> dict:
    """Evaluate a fine-tuned BERT model on the held-out gold test split.

    Gold test data is NEVER used during CV or final model training.
    This is the ONLY function that calls load_split_text_data(split='gold_test').

    Model supply — exactly one of:
      a) ``model_dir`` is given  → load model from disk via load_bert_model().
      b) ``model`` is given directly → use as-is (skips disk load).

    Args:
        case_type: "dv" or "boc".
        model: Fine-tuned ``LegalBertForClassification``, or None to load
            from disk via ``model_dir``.
        model_dir: Root model directory (parent of the case_type
            subdirectory). If given, loads the model from disk via
            ``load_bert_model``.
        device: Torch device (auto-selected if None).
        run_name: Prefix used when saving (e.g. "bert_v1").
        batch_size: Window mini-batch size passed to sliding-window inference.

    Returns:
        dict with keys:
            y_pred     : np.ndarray[int64],   shape (n,)
            y_true     : np.ndarray[int64],   shape (n,)
            y_proba    : np.ndarray[float32], shape (n, n_classes)
            n_processo : list[str]

    Raises:
        ValueError: If both ``model_dir`` and ``model`` are None.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_dir is not None:
        print(f"  Loading BERT model from {Path(model_dir) / case_type} ...")
        model = load_bert_model(case_type, model_dir, device, run_name)
    elif model is None:
        raise ValueError("Provide either model_dir or model.")

    print("  Loading gold test text data ...")
    texts, y_raw, _, n_processo = load_split_text_data(case_type, split="gold_test")
    y_true = encode_labels(y_raw, case_type)

    print(f"  Running BERT predictions on {len(texts)} gold test cases ...")
    y_pred = predict_bert(texts, model, case_type, device, batch_size)
    y_proba = predict_bert_proba(texts, model, case_type, device, batch_size)

    return {
        "y_pred": y_pred,
        "y_true": y_true,
        "y_proba": y_proba,
        "n_processo": n_processo,
    }


def run_train_val_predictions_bert(
    case_type: str,
    subset: str,
    model: LegalBertForClassification | None = None,
    model_dir: Path | None = None,
    device: torch.device | None = None,
    run_name: str | None = None,
    batch_size: int = 8,
    val_ratio: float = 0.15,
) -> dict:
    """Score a fine-tuned BERT model on the data it was fitted on.

    This is the diagnostic counterpart to ``run_gold_test_predictions_bert``.
    It never touches the gold test split — it reuses the exact temporal 85/15
    train/val split produced by ``load_bert_train_val_split_with_ids``, which
    is the same split the trainer used, so ``subset="train"`` scores the model
    on documents it actually saw during fitting.

    Comparing train against gold test settles whether a collapsed model is
    underfitting or overfitting:
      - poor on train AND gold  → underfitting (the model never fit the data)
      - strong on train, poor on gold → overfitting (memorised, did not generalise)

    Inference uses the same eval-mode sliding-window path as the gold test, so
    the two numbers are directly comparable.

    Args:
        case_type: "dv" or "boc".
        subset: "train" or "val".
        model: Fine-tuned ``LegalBertForClassification``, or None to load
            from disk via ``model_dir``.
        model_dir: Root model directory; used to load the model from disk
            when ``model`` is not given.
        device: Torch device (auto-selected if None).
        run_name: Checkpoint tag used when loading from disk.
        batch_size: Window mini-batch size passed to sliding-window inference.
        val_ratio: Validation fraction of the train split; must match the
            ratio used during training.

    Returns:
        dict with the same keys as ``run_gold_test_predictions_bert``:
        y_pred, y_true, y_proba, n_processo.

    Raises:
        ValueError: If ``subset`` is not "train" or "val", or if both
            ``model_dir`` and ``model`` are None.
    """
    if subset not in ("train", "val"):
        raise ValueError(f"Unknown subset '{subset}'. Choose 'train' or 'val'.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_dir is not None:
        print(f"  Loading BERT model from {Path(model_dir) / case_type} ...")
        model = load_bert_model(case_type, model_dir, device, run_name)
    elif model is None:
        raise ValueError("Provide either model_dir or model.")

    print(f"  Loading {subset} text data ...")
    texts_tr, y_tr, ids_tr, texts_val, y_val, ids_val = (
        load_bert_train_val_split_with_ids(case_type, val_ratio=val_ratio)
    )
    texts, y_true, n_processo = (
        (texts_tr, y_tr, ids_tr) if subset == "train" else (texts_val, y_val, ids_val)
    )

    print(f"  Running BERT predictions on {len(texts)} {subset} cases ...")
    y_pred = predict_bert(texts, model, case_type, device, batch_size)
    y_proba = predict_bert_proba(texts, model, case_type, device, batch_size)

    return {
        "y_pred": y_pred,
        "y_true": y_true,
        "y_proba": y_proba,
        "n_processo": n_processo,
    }


# ---------------------------------------------------------------------------
# Persist predictions to parquet
# ---------------------------------------------------------------------------


def save_predictions_parquet(
    results: dict,
    case_type: str,
    run_name: str | None = None,
    output_dir: Path | None = None,
    split: str = "gold_test",
) -> Path:
    """Serialise prediction results to a parquet file.

    Creates a tidy DataFrame with one row per case:
        n_processo   — case identifier
        y_true       — integer ground-truth label
        y_pred       — integer predicted label
        proba_<NAME> — predicted probability for each class (one column per class)
        run_name     — model version tag (for easy filtering in downstream plots)
        case_type    — "dv" or "boc"
        split        — which split these predictions came from

    Args:
        results: dict returned by ``run_gold_test_predictions``[``_bert``] or
            ``run_train_val_predictions_bert`` — keys: y_pred, y_true,
            y_proba, n_processo.
        case_type: "dv" or "boc".
        run_name: Version tag written into the ``run_name`` column and used
            as the filename prefix (e.g. "bert_v4").
        output_dir: Directory to write into; defaults to
            ``DEFAULT_OUTPUT_DIR / case_type / "predictions"``.
        split: "gold_test", "train" or "val". Gold test keeps the original
            filename for backwards compatibility with existing analysis
            notebooks; other splits are suffixed so nothing is overwritten.

    Returns:
        Path to the saved parquet file.
    """
    if output_dir is None:
        output_dir = Path(DEFAULT_OUTPUT_DIR) / case_type / "predictions"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = LABEL_NAMES[case_type]  # {0: "MANTIDA", 1: "ALTERADA", …}
    y_proba: np.ndarray = results["y_proba"]  # (n, n_classes)

    df = pd.DataFrame(
        {
            "n_processo": results["n_processo"],
            "y_true": results["y_true"].astype(np.int32),
            "y_pred": results["y_pred"].astype(np.int32),
            **{
                f"proba_{class_names[i]}": y_proba[:, i]
                for i in range(y_proba.shape[1])
            },
            "run_name": run_name or "",
            "case_type": case_type,
            "split": split,
        }
    )

    prefix = f"{run_name}_" if run_name else ""
    suffix = "" if split == "gold_test" else f"_{split}"
    out_path = output_dir / f"{prefix}{case_type}{suffix}_predictions.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  Predictions saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# SHAP explanation helper
# ---------------------------------------------------------------------------

_SHAP_OUTPUT_NAMES: dict[str, list[str]] = {
    "dv": list(LABEL_NAMES_DV.values()),
    "boc": list(LABEL_NAMES_BOC.values()),
}

# SHAP outputs follow eda_viz/ convention: shap_explanations/{case_dir}/{model_name}/
_SHAP_BASE_DIR = Path(DEFAULT_OUTPUT_DIR).parents[2] / "shap_explanations"


def _run_shap_bert(
    model: "LegalBertForClassification",
    texts: list[str],
    n_processo: list[str],
    y_true: "np.ndarray",
    case_type: str,
    run_name: str,
    max_shap_docs: int | None = 10,
    batch_size: int = 1,
    shap_case_ids: list[str] | None = None,
) -> None:
    """Run SHAP sentence-level explanations for fine-tuned BERT on gold test documents.

    Parameters
    ----------
    model         : fine-tuned LegalBertForClassification, already on device
    texts         : raw document texts (gold test split)
    n_processo    : case identifiers aligned with texts
    y_true        : integer ground-truth labels aligned with texts
    case_type     : "dv" or "boc"
    run_name      : used as model name in output paths (e.g. "bert_v4")
    max_shap_docs : cap on number of documents to explain (None = all).
                    Ignored when shap_case_ids is provided.
    batch_size    : predict_fn batch size passed to predict_bert_proba
    shap_case_ids : if given, explain only these specific n_processo values
                    (raw, unsanitized). Outputs go to 5_gold_set/ subfolder.
                    Takes priority over max_shap_docs.
    """
    device = next(model.parameters()).device

    def predict_fn(ts: list[str]) -> "np.ndarray":
        return predict_bert_proba(ts, model, case_type, device, batch_size)

    # --- select documents to explain ---
    output_subdir: str | None = None
    if shap_case_ids is not None:
        id_set = set(shap_case_ids)
        indices = [i for i, cid in enumerate(n_processo) if cid in id_set]
        not_found = id_set - {n_processo[i] for i in indices}
        if not_found:
            print(
                f"  WARNING: {len(not_found)} requested case(s) not found in gold test: {sorted(not_found)}"
            )
        if not indices:
            print("  No matching cases found — skipping SHAP.")
            return
        texts = [texts[i] for i in indices]
        n_processo = [n_processo[i] for i in indices]
        y_true = y_true[np.array(indices)]
        output_subdir = "5_gold_set"
    elif max_shap_docs is not None:
        texts = texts[:max_shap_docs]
        n_processo = n_processo[:max_shap_docs]
        y_true = y_true[:max_shap_docs]

    # n_processo values often contain "/" (e.g. "299/23.4SXLSB.L1-5") which
    # would be interpreted as a path separator in filenames — sanitize to "_".
    import re as _re

    safe_ids = [_re.sub(r'[/\\:*?"<>|]', "_", cid) for cid in n_processo]

    # Ensure the output directory tree exists before SHAP starts writing files.
    from explanation import CASE_TYPE_OUTPUT_DIRS

    case_dir = CASE_TYPE_OUTPUT_DIRS.get(case_type, case_type)
    if output_subdir:
        shap_model_dir = (
            _SHAP_BASE_DIR / case_dir / output_subdir / (run_name or "bert")
        )
    else:
        shap_model_dir = _SHAP_BASE_DIR / case_dir / (run_name or "bert")
    shap_model_dir.mkdir(parents=True, exist_ok=True)

    n = len(texts)
    print(f"\n  Running SHAP on {n} document(s) — est. {n * 2}–{n * 3} min on GPU ...")
    run_shap_explanation_pipeline(
        texts=texts,
        case_ids=safe_ids,
        predict_fn=predict_fn,
        output_names=_SHAP_OUTPUT_NAMES[case_type],
        case_type=case_type,
        model_name=run_name or "bert",
        output_base_dir=_SHAP_BASE_DIR,
        true_labels=list(y_true),
        max_evals=200,
        subdir=output_subdir,
    )
    print(f"  SHAP outputs → {shap_model_dir}")


# ---------------------------------------------------------------------------
# CLI main — gold test point metrics
# ---------------------------------------------------------------------------


def main(
    case_type: str,
    model_dir: Path = DEFAULT_OUTPUT_DIR,
    run_name: str | None = None,
) -> dict:
    """Run gold test evaluation (MLP), print metrics with bootstrap CIs, save forest plot.

    Loads the trained frozen-embeddings pipeline/model, predicts on the gold
    test split, evaluates with bootstrap confidence intervals, saves a forest
    plot of the metrics, and persists the predictions to parquet.

    Args:
        case_type: "dv" or "boc".
        model_dir: Root model directory (parent of the case_type subdirectory).
        run_name: Prefix used when the model was saved (e.g. "v1").

    Returns:
        dict returned by ``run_gold_test_predictions`` — keys: y_pred,
        y_true, y_proba, n_processo.
    """
    print(f"\n{'=' * 60}")
    print(f"Legal BERTimbau — Gold Test Evaluation ({case_type.upper()})")
    print(f"{'=' * 60}\n")

    results = run_gold_test_predictions(
        case_type, model_dir=model_dir, run_name=run_name
    )
    eval_results = run_classification_evaluation(
        results["y_true"], results["y_pred"], case_type
    )
    plots_dir = Path(model_dir) / case_type / "plots"
    plot_forest(eval_results, case_type, run_name=run_name, output_dir=plots_dir)
    save_predictions_parquet(results, case_type, run_name=run_name)
    return results


def _print_fit_diagnosis(split_evals: dict, case_type: str) -> None:
    """Print a train / validation / gold-test comparison to diagnose model fit.

    A collapsed classifier can fail in two opposite ways, and the gap between
    train and held-out performance is what tells them apart:

      - poor on train AND gold        → underfitting: the model never fitted
        the data it saw, so there is no gap to attribute to overfitting.
      - strong on train, poor on gold → overfitting: the model memorised the
        training data and failed to generalise.

    The printed verdict is a reading aid; the saved parquet files are the
    evidence.
    """
    names = LABEL_NAMES[case_type]
    order = ["train", "validation", "gold test"]

    print(f"\n{'=' * 74}")
    print(f"FIT DIAGNOSIS — {case_type.upper()}  (one model, three splits)")
    print(f"{'=' * 74}")
    print(
        f"  {'Split':<12} {'n':>6} {'Macro-F1':>10} {'MCC':>8}   Predicted class counts"
    )

    scores: dict[str, tuple[float, float]] = {}
    for label in order:
        if label not in split_evals:
            continue
        res, ev = split_evals[label]
        point = ev["point_estimates"]
        y_pred = np.asarray(res["y_pred"])
        counts = ", ".join(
            f"{names[lbl]}={int((y_pred == lbl).sum())}" for lbl in ev["labels"]
        )
        scores[label] = (float(point["macro_f1"]), float(point["mcc"]))
        print(
            f"  {label:<12} {len(y_pred):>6} {point['macro_f1']:>10.4f} "
            f"{point['mcc']:>8.4f}   {counts}"
        )

    if "train" in scores and "gold test" in scores:
        gap_f1 = scores["train"][0] - scores["gold test"][0]
        gap_mcc = scores["train"][1] - scores["gold test"][1]
        train_mcc = scores["train"][1]
        print(f"\n  Train - gold gap : Macro-F1 {gap_f1:+.4f}   MCC {gap_mcc:+.4f}")

        if abs(train_mcc) < 0.05 and gap_f1 < 0.10:
            verdict = (
                "UNDERFITTING. The model is no better than chance on the data\n"
                "                     it was trained on, so there is no train/held-out\n"
                "                     gap that overfitting could explain."
            )
        elif gap_f1 > 0.20:
            verdict = (
                "OVERFITTING. The model fits the training data substantially\n"
                "                     better than the held-out gold test."
            )
        else:
            verdict = (
                "INCONCLUSIVE from these numbers alone — inspect the per-class\n"
                "                     metrics and the loss curves before concluding."
            )
        print(f"  Reading          : {verdict}")
    print(f"{'=' * 74}\n")


def main_bert(
    case_type: str,
    model_dir: Path = DEFAULT_OUTPUT_DIR,
    run_name: str | None = None,
    batch_size: int = 8,
    run_shap: bool = False,
    max_shap_docs: int | None = 10,
    shap_case_ids: list[str] | None = None,
    eval_train: bool = False,
    val_ratio: float = 0.15,
) -> dict:
    """Run gold test evaluation (fine-tuned BERT), print metrics with bootstrap CIs, save forest plot.

    When ``eval_train`` is set, the same model is additionally scored on the
    train and validation subsets it was fitted on, and a train-vs-gold summary
    is printed to diagnose underfitting versus overfitting.

    Args:
        case_type: "dv" or "boc".
        model_dir: Root model directory (parent of the case_type subdirectory).
        run_name: Prefix used when the model was saved (e.g. "bert_v1").
        batch_size: Window mini-batch size passed to sliding-window inference.
        run_shap: If True, run SHAP sentence-level explanations after
            evaluation.
        max_shap_docs: Cap on number of gold-test documents to explain with
            SHAP (None = all). Ignored when ``shap_case_ids`` is given.
        shap_case_ids: If given, explain only these specific n_processo
            values instead of the first ``max_shap_docs`` gold-test cases.
        eval_train: If True, additionally score the model on the train and
            validation subsets and print a fit diagnosis.
        val_ratio: Validation fraction of the train split; must match the
            ratio used during training.

    Returns:
        dict returned by ``run_gold_test_predictions_bert`` — keys: y_pred,
        y_true, y_proba, n_processo.
    """
    print(f"\n{'=' * 60}")
    print(f"Legal BERTimbau (fine-tuned) — Gold Test Evaluation ({case_type.upper()})")
    print(f"{'=' * 60}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_bert_model(
        case_type, model_dir=model_dir, device=device, run_name=run_name
    )

    results = run_gold_test_predictions_bert(
        case_type, model=model, batch_size=batch_size
    )
    eval_results = run_classification_evaluation(
        results["y_true"], results["y_pred"], case_type
    )
    plots_dir = Path(model_dir) / case_type / "plots"
    plot_forest(eval_results, case_type, run_name=run_name, output_dir=plots_dir)
    save_predictions_parquet(results, case_type, run_name=run_name)

    if eval_train:
        split_evals = {"gold test": (results, eval_results)}
        for subset, label in (("train", "train"), ("val", "validation")):
            print(f"\n{'-' * 60}")
            print(f"Scoring model on the {label} subset ({case_type.upper()})")
            print(f"{'-' * 60}")
            sub_results = run_train_val_predictions_bert(
                case_type,
                subset=subset,
                model=model,
                batch_size=batch_size,
                val_ratio=val_ratio,
            )
            sub_eval = run_classification_evaluation(
                sub_results["y_true"],
                sub_results["y_pred"],
                case_type,
                split_label=label.capitalize(),
            )
            plot_forest(
                sub_eval,
                case_type,
                run_name=run_name,
                output_dir=plots_dir,
                split_label=label,
            )
            save_predictions_parquet(
                sub_results, case_type, run_name=run_name, split=subset
            )
            split_evals[label] = (sub_results, sub_eval)

        _print_fit_diagnosis(split_evals, case_type)

    if run_shap:
        from features import load_split_text_data

        texts, _, _, n_processo = load_split_text_data(case_type, split="gold_test")
        _run_shap_bert(
            model=model,
            texts=texts,
            n_processo=n_processo,
            y_true=results["y_true"],
            case_type=case_type,
            run_name=run_name or "bert",
            max_shap_docs=max_shap_docs,
            batch_size=batch_size,
            shap_case_ids=shap_case_ids,
        )

    return results


# ---------------------------------------------------------------------------
# Smoke test — no real data, no HuggingFace download required
# ---------------------------------------------------------------------------


def _run_smoke_test() -> None:
    """End-to-end smoke test with synthetic data and a mock BERT.

    Tests:
      1. train_final_model + save_model → write PyTorch artifacts to tmp dir
      2. load_model                     → reload from tmp dir
      3. predict / predict_proba        → PyTorch model, check shapes & dtypes
      4. predict / predict_proba        → sklearn LogisticRegression (duck-typing)
      5. make_bert_predict_fn           → SHAP output shape with mock BERT
         (also verifies BERTConfig is threaded through correctly)

    Does NOT test run_gold_test_predictions (requires real parquet/CSV files).
    Does NOT require an internet connection or HuggingFace model download.
    """
    import tempfile

    from sklearn.linear_model import LogisticRegression

    from train import save_model, train_final_model

    # ── Mock BERT: random last_hidden_state (1024-dim) ───────────────────
    class _MockBert(torch.nn.Module):
        def forward(self, input_ids, attention_mask=None, **kwargs):
            b, s = input_ids.shape
            hs = torch.randn(b, s, 1024, device=input_ids.device)
            return type("Out", (), {"last_hidden_state": hs})()

    # ── Mock HuggingFace tokenizer ───────────────────────────────────────
    class _MockTokenizer:
        def __call__(
            self,
            text,
            truncation=True,
            max_length=512,
            stride=128,
            return_overflowing_tokens=True,
            return_attention_mask=True,
        ):
            n_tok = min(max(len(str(text).split()) + 2, 3), max_length)
            return {
                "input_ids": [list(range(n_tok))],
                "attention_mask": [[1] * n_tok],
            }

        def pad(self, encoded_inputs, padding=True, return_tensors="pt"):
            max_len = max(len(item["input_ids"]) for item in encoded_inputs)
            ids_batch, mask_batch = [], []
            for item in encoded_inputs:
                pad_len = max_len - len(item["input_ids"])
                ids_batch.append(item["input_ids"] + [0] * pad_len)
                mask_batch.append(item["attention_mask"] + [0] * pad_len)
            return {
                "input_ids": torch.tensor(ids_batch, dtype=torch.long),
                "attention_mask": torch.tensor(mask_batch, dtype=torch.long),
            }

    print("Running predict.py smoke test...\n")

    rng = np.random.default_rng(42)
    N = 200
    device = torch.device("cpu")
    mock_bert = _MockBert().eval()
    mock_tok = _MockTokenizer()
    # Use a BERTConfig with smaller window_batch_size to exercise the loop
    smoke_cfg = BERTConfig(window_batch_size=2)

    for case_type in ("dv", "boc"):
        print(f"--- case_type={case_type} ---")
        n_classes = N_CLASSES[case_type]

        X = rng.standard_normal((N, 1024)).astype(np.float32)
        y = np.tile(np.arange(n_classes), N // n_classes + 1)[:N].astype(np.int64)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # 1 ── train + save
            print("  [1] train_final_model + save_model...")
            fitted_pipeline, fitted_model = train_final_model(X, y, case_type, device)
            save_model(fitted_pipeline, fitted_model, case_type, output_dir=tmp_path)

            # 2 ── load_model
            print("  [2] load_model() from disk...")
            loaded_pipeline, loaded_model = load_model(case_type, tmp_path, device)

            # 3 ── predict / predict_proba — PyTorch model
            print("  [3] predict / predict_proba — PyTorch LegalBertClassifier...")
            y_pred = predict(X, loaded_pipeline, loaded_model, case_type, device)
            y_proba = predict_proba(X, loaded_pipeline, loaded_model, case_type, device)

            assert y_pred.shape == (N,), f"y_pred shape: {y_pred.shape}"
            assert np.issubdtype(y_pred.dtype, np.integer), f"dtype: {y_pred.dtype}"
            assert y_proba.shape == (N, n_classes), f"y_proba shape: {y_proba.shape}"
            assert np.allclose(y_proba.sum(axis=1), 1.0, atol=1e-4)
            print(f"  OK — y_pred {y_pred.shape}, y_proba {y_proba.shape}")

            # 4 ── predict / predict_proba — sklearn model
            print("  [4] predict / predict_proba — sklearn LogisticRegression...")
            X_reduced = loaded_pipeline.transform(X).astype(np.float32)
            lr = LogisticRegression(max_iter=10, random_state=42).fit(X_reduced, y)
            y_pred_lr = predict(X, loaded_pipeline, lr, case_type)
            y_proba_lr = predict_proba(X, loaded_pipeline, lr, case_type)

            assert y_pred_lr.shape == (N,)
            assert y_proba_lr.shape == (N, n_classes)
            assert np.allclose(y_proba_lr.sum(axis=1), 1.0, atol=1e-4)
            print(f"  OK — LR y_pred {y_pred_lr.shape}, y_proba {y_proba_lr.shape}")

            # 5 ── make_bert_predict_fn with BERTConfig
            print("  [5] make_bert_predict_fn (BERTConfig threaded through)...")
            predict_fn = make_bert_predict_fn(
                loaded_pipeline,
                loaded_model,
                mock_bert,
                mock_tok,
                case_type,
                bert_config=smoke_cfg,
                device=device,
            )
            test_texts = [
                "texto legal de teste para verificar o pipeline",
                "outro documento juridico para validar a saida",
                "",  # edge case: empty string from SHAP masking
            ]
            proba_out = predict_fn(test_texts)

            assert proba_out.shape == (
                len(test_texts),
                n_classes,
            ), f"predict_fn shape: {proba_out.shape}"
            assert np.allclose(proba_out.sum(axis=1), 1.0, atol=1e-4)
            print(
                f"  OK — make_bert_predict_fn shape {proba_out.shape} "
                f"(SHAP-compatible)\n"
            )

    # 6 ── load_bert_model — save/load round-trip with mock HuggingFace model
    print("--- BERT model I/O (monkeypatched AutoModel) ---")
    from train import save_bert_model
    import transformers as _tf

    class _MockHFModel(torch.nn.Module):
        """Minimal AutoModel stand-in — avoids HuggingFace download in tests."""

        def __init__(self):
            super().__init__()
            self.config = type("Cfg", (), {"hidden_size": 1024})()
            self._p = torch.nn.Parameter(
                torch.zeros(1)
            )  # ensure state_dict is non-empty

        def forward(self, input_ids, attention_mask=None, **_):
            b, s = input_ids.shape
            return type("Out", (), {"last_hidden_state": torch.randn(b, s, 1024)})()

    class _MockBatchTokenizer:
        """Tokenizer stand-in for both single-doc sliding-window and batch calls."""

        pad_token_id = 0

        def __call__(
            self,
            texts,
            truncation=True,
            max_length=512,
            stride=128,
            padding=None,
            return_tensors=None,
            return_overflowing_tokens=False,
            return_attention_mask=True,
        ):
            # Single string → sliding-window path (return 2 mock windows as pt tensors)
            if isinstance(texts, str):
                n_windows = 2
                ids = torch.zeros(n_windows, max_length, dtype=torch.long)
                mask = torch.ones(n_windows, max_length, dtype=torch.long)
                return {"input_ids": ids, "attention_mask": mask}
            # List of strings → batch path
            n = len(texts)
            ids = torch.zeros(n, max_length, dtype=torch.long)
            mask = torch.ones(n, max_length, dtype=torch.long)
            return {"input_ids": ids, "attention_mask": mask}

    _orig_automodel = _tf.AutoModel
    _orig_autotokenizer = _tf.AutoTokenizer
    _tf.AutoModel = type(
        "_M", (), {"from_pretrained": staticmethod(lambda *a, **kw: _MockHFModel())}
    )
    _tf.AutoTokenizer = type(
        "_T",
        (),
        {"from_pretrained": staticmethod(lambda *a, **kw: _MockBatchTokenizer())},
    )
    try:
        for case_type in ("dv", "boc"):
            n_classes = 1 if IS_BINARY[case_type] else N_CLASSES[case_type]
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                bert_clf = LegalBertForClassification(n_classes=n_classes).to(device)
                save_bert_model(
                    bert_clf, case_type, output_dir=tmp_path, run_name="smoke_test"
                )

                print(f"  [6] load_bert_model round-trip — case_type={case_type}...")
                loaded = load_bert_model(
                    case_type, tmp_path, device, run_name="smoke_test"
                )
                assert isinstance(loaded, LegalBertForClassification)
                assert loaded.model_name == DEFAULT_BERT_CONFIG.model_name
                print(
                    f"  OK — loaded LegalBertForClassification, n_classes={n_classes}"
                )

                print(
                    f"  [7] predict_bert / predict_bert_proba — case_type={case_type}..."
                )
                test_texts = ["texto juridico de teste", "outro documento legal", ""]
                y_pred_b = predict_bert(
                    test_texts, loaded, case_type, device, batch_size=2
                )
                y_proba_b = predict_bert_proba(
                    test_texts, loaded, case_type, device, batch_size=2
                )

                assert y_pred_b.shape == (len(test_texts),), f"shape: {y_pred_b.shape}"
                assert np.issubdtype(
                    y_pred_b.dtype, np.integer
                ), f"dtype: {y_pred_b.dtype}"
                assert y_proba_b.shape == (len(test_texts), N_CLASSES[case_type])
                assert np.allclose(y_proba_b.sum(axis=1), 1.0, atol=1e-4)
                print(f"  OK — y_pred {y_pred_b.shape}, y_proba {y_proba_b.shape}")
    finally:
        _tf.AutoModel = _orig_automodel
        _tf.AutoTokenizer = _orig_autotokenizer

    print("Smoke test passed.")


# ---------------------------------------------------------------------------
# Argparse entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Predict gold test outcomes with a trained Legal BERTimbau head.\n"
            "Use --smoke_test to validate the pipeline with synthetic data."
        )
    )
    parser.add_argument("--case_type", choices=["dv", "boc"], default=None)
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Run name prefix used when saving (e.g. 'bert_v1').",
    )
    parser.add_argument(
        "--mode",
        choices=["mlp", "bert"],
        default="mlp",
        help="mlp: frozen-embeddings MLP (default). bert: fine-tuned BERT end-to-end.",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run end-to-end smoke test (no real data or HuggingFace model required).",
    )
    parser.add_argument(
        "--shap",
        action="store_true",
        help="Run SHAP sentence-level explanations after evaluation (bert mode only). Slow: ~2-3 min/doc.",
    )
    parser.add_argument(
        "--max_shap_docs",
        type=int,
        default=10,
        help="Max number of gold-test documents to explain with SHAP (default: 10). 0 = all. Ignored when --shap_cases is set.",
    )
    parser.add_argument(
        "--shap_cases",
        nargs="+",
        default=None,
        metavar="N_PROCESSO",
        help=(
            "Explain only these specific case IDs (n_processo). "
            "Outputs are saved under 5_gold_set/ subfolder. "
            'Example: --shap_cases "5052/21.7JAPRT-A.P1" "340/21.5TXLSB-E.L1-9"'
        ),
    )
    parser.add_argument(
        "--eval_train",
        action="store_true",
        help=(
            "Also score the model on the train and validation subsets it was fitted on "
            "(bert mode only), and print a train-vs-gold fit diagnosis to tell "
            "underfitting apart from overfitting. Slow: the train split is ~20-35x "
            "larger than the gold test set."
        ),
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Validation fraction of the train split. Must match the value used during training (default: 0.15).",
    )
    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke_test()
    else:
        if args.case_type is None:
            parser.error("--case_type is required unless --smoke_test is set")
        if args.eval_train and args.mode != "bert":
            parser.error("--eval_train is only supported with --mode bert")
        max_shap = None if args.max_shap_docs == 0 else args.max_shap_docs
        if args.mode == "bert":
            main_bert(
                args.case_type,
                args.model_dir,
                args.run_name,
                run_shap=args.shap,
                max_shap_docs=max_shap,
                shap_case_ids=args.shap_cases,
                eval_train=args.eval_train,
                val_ratio=args.val_ratio,
            )
        else:
            main(args.case_type, args.model_dir, args.run_name)
