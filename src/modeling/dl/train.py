"""train.py: training pipeline for the Legal BERTimbau classification head.

Architecture
------------
  Frozen BERT embeddings (1024-dim, from parquet)
     → StandardScaler → PCA(0.95)  [sklearn Pipeline from features.py]
     → LegalBertClassifier MLP head  [PyTorch nn.Module]

Loss functions
--------------
  Binary  (DV):  BCEWithLogitsLoss  — sigmoid + BCE in one stable op;
                                      pos_weight handles class imbalance.
  Ternary (BoC): CrossEntropyLoss   — log-softmax + NLL;
                                      weight vector handles class imbalance.

CV strategy: TimeSeriesSplit (sklearn) on data sorted by data_acordao date.
"""

import argparse
import copy
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight

from features import (
    IS_BINARY,
    MODEL_NAME,
    N_CLASSES,
    SEED,
    build_feature_pipeline,
    encode_labels,
    load_split_data,
    load_bert_train_val_split,
)

import sys

_eval_dir = str(Path(__file__).resolve().parents[2] / "evaluation")
if _eval_dir not in sys.path:
    sys.path.insert(0, _eval_dir)

from classification import compute_confidence_intervals

# --- Frozen-embeddings MLP ---
N_EPOCHS = 100
LEARNING_RATE = 1e-3
HIDDEN_DIM = 256
DROPOUT_PROB = 0.3
L2_WEIGHT_DECAY = 1e-4
L1_LAMBDA = 0.0
EARLY_STOPPING_PATIENCE = 10

# --- BERT fine-tuning ---
BERT_LR = 5e-6  # reduced from 1e-5; slower learning → wider window before memorization
BERT_EPOCHS = 20  # was 5; allow full learning curve with early stopping
BERT_BATCH_SIZE = 4  # kept for legacy MLP-style loops
BERT_DROPOUT = 0.3  # bumped from 0.2; stronger regularization for small datasets
BERT_WEIGHT_DECAY = (
    0.05  # increased from 0.01; stronger L2 penalty against memorization
)
BERT_LABEL_SMOOTHING = (
    0.1  # prevents overconfident predictions; helps BOC minority classes
)
BERT_VAL_RATIO = (
    0.15  # last 15 % of chronologically sorted train set used for val / early stopping
)
BERT_PATIENCE = 7  # patience epochs for val-loss early stopping
BERT_EMA_ALPHA = 0.3  # EMA smoothing factor for val-loss early stopping signal
BERT_FROZEN_LAYERS = 20  # freeze layers 0-19, fine-tune layers 20-23 (top 4) + head
BERT_MAX_WINDOWS = 16  # cap windows per document; equally-spaced if exceeded
BERT_WINDOW_BATCH = 16  # windows per mini-batch through the unfrozen layers
BERT_WARMUP_RATIO = 0.10  # linear warmup over first 10 % of total training steps
BERT_GRAD_CLIP = 1.0  # max gradient norm; standard for BERT fine-tuning

DEFAULT_N_SPLITS = 5
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "data/models/dl"


class LegalBertClassifier(nn.Module):
    """Small MLP head trained on PCA-reduced frozen BERT embeddings.

    Args:
        pca_dim: Number of PCA components (inferred from the fitted pipeline
            at train time).
        output_dim: 1 for a binary task (BCEWithLogitsLoss), >1 for a
            multi-class task (CrossEntropyLoss).
        hidden_dim: Width of the single hidden layer.
        dropout_prob: Dropout probability applied after the hidden activation.
    """

    def __init__(
        self,
        pca_dim: int,
        output_dim: int,
        hidden_dim: int = HIDDEN_DIM,
        dropout_prob: float = DROPOUT_PROB,
    ):
        """Build the Linear → ReLU → Dropout → Linear head. See class docstring for Args."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pca_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute classification logits for a batch of PCA-reduced embeddings.

        Args:
            x: Input tensor, shape (batch, pca_dim).

        Returns:
            Logits tensor, shape (batch, output_dim).
        """
        return self.net(x)


class LegalBertForClassification(nn.Module):
    """Legal BERTimbau encoder with a linear classification head on [CLS].

    Supports partial layer freezing: the bottom ``n_frozen_layers`` transformer
    layers (plus the embedding layer) are frozen at init — their parameters
    receive no gradients and require no optimizer states.  The top
    ``(total_layers - n_frozen_layers)`` layers and the classification head
    are trainable.

    During the forward pass, ``get_cls_repr`` explicitly runs frozen layers
    inside ``torch.no_grad()`` so their activations are never stored in the
    computational graph, reducing VRAM usage when processing many windows.

    Args:
        n_classes: 1 for a binary task (BCEWithLogitsLoss, single logit), >1
            for a multi-class task (CrossEntropyLoss).
        dropout_prob: Dropout applied between the CLS representation and the
            classifier.
        n_frozen_layers: Number of BERT encoder layers (0-indexed from the
            bottom) to freeze. 0 = train all layers (original behaviour).
            20 = freeze layers 0-19, fine-tune layers 20-23 + head (top-4 plan).
    """

    def __init__(
        self,
        n_classes: int,
        dropout_prob: float = BERT_DROPOUT,
        n_frozen_layers: int = 0,
    ):
        """Load the pretrained encoder, attach the head, and freeze bottom layers. See class docstring for Args."""
        super().__init__()
        from transformers import AutoModel  # lazy import — only needed for BERT mode

        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        hidden_size = self.bert.config.hidden_size  # 1024 for bert-large
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, n_classes)
        self.n_frozen_layers = n_frozen_layers

        if n_frozen_layers > 0:
            # Freeze embedding layer
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            # Freeze bottom n_frozen_layers transformer layers
            for layer in self.bert.encoder.layer[:n_frozen_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

    def get_cls_repr(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return the [CLS] representation for a batch of windows.

        When ``n_frozen_layers > 0``, the frozen layers are run inside
        ``torch.no_grad()`` so their activations are not stored in the
        computational graph.  A ``.detach()`` at the boundary ensures the
        gradient stops there and does not propagate into frozen parameters.

        Args:
            input_ids: Token id tensor, shape (batch, seq_len).
            attention_mask: Attention mask tensor, shape (batch, seq_len).

        Returns:
            torch.Tensor, shape (batch, hidden_size) — the [CLS] token
            representation for each window in the batch.
        """
        if self.n_frozen_layers == 0:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state[:, 0, :]

        extended_mask = self.bert.get_extended_attention_mask(
            attention_mask, input_ids.shape
        )

        # --- Frozen layers: no graph tracking, no activation storage ---
        with torch.no_grad():
            hidden = self.bert.embeddings(input_ids=input_ids)
            for layer in self.bert.encoder.layer[: self.n_frozen_layers]:
                hidden = layer(hidden, attention_mask=extended_mask)[0]
        hidden = hidden.detach()  # explicit graph cut at the frozen/unfrozen boundary

        # --- Unfrozen top layers: full gradient tracking ---
        for layer in self.bert.encoder.layer[self.n_frozen_layers :]:
            hidden = layer(hidden, attention_mask=extended_mask)[0]

        return hidden[:, 0, :]  # [CLS] token

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Single-window forward pass — used by smoke tests and basic inference.

        Production inference instead uses ``get_cls_repr`` directly via the
        sliding-window helpers (``_tokenize_sliding_window`` /
        ``_forward_sliding_window``) to aggregate over multiple windows.

        Args:
            input_ids: Token id tensor, shape (batch, seq_len).
            attention_mask: Attention mask tensor, shape (batch, seq_len).

        Returns:
            Logits tensor, shape (batch, n_classes).
        """
        cls = self.get_cls_repr(input_ids, attention_mask)
        return self.classifier(self.dropout(cls))


def _build_criterion(
    case_type: str,
    y_train: np.ndarray,
    device: torch.device,
    label_smoothing: float = 0.0,
) -> nn.Module:
    """Return the appropriate loss function with class weighting.

    Binary (DV):
        BCEWithLogitsLoss with pos_weight = n_negative / n_positive.

    Ternary (BoC):
        CrossEntropyLoss with balanced class weights + label_smoothing.
        label_smoothing prevents overconfident predictions on minority classes.
    """
    if IS_BINARY[case_type]:
        n_neg = int(np.sum(y_train == 0))
        n_pos = int(np.sum(y_train == 1))
        pos_weight = torch.tensor(
            [n_neg / max(n_pos, 1)], dtype=torch.float32, device=device
        )
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        classes = np.arange(N_CLASSES[case_type])
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        return nn.CrossEntropyLoss(
            weight=weight_tensor, label_smoothing=label_smoothing
        )


def predict_from_logits(logits: torch.Tensor, is_binary: bool) -> np.ndarray:
    """Convert raw model logits to integer class predictions.

    Binary:  sigmoid(logits) >= 0.5  →  0 or 1
    Multi:   argmax(logits)          →  0, 1, … n_classes-1
    """
    if is_binary:
        probs = torch.sigmoid(logits).squeeze(dim=1)
        return (probs >= 0.5).long().cpu().numpy()
    else:
        return logits.argmax(dim=1).cpu().numpy()


def proba_from_logits(logits: torch.Tensor, is_binary: bool) -> np.ndarray:
    """Convert logits to probability arrays of shape (n, n_classes).

    Binary:  sigmoid → [p(class=0), p(class=1)]
    Multi:   softmax → [p(class=0), …, p(class=n-1)]
    """
    if is_binary:
        p = torch.sigmoid(logits).squeeze(dim=1).cpu().numpy()
        return np.column_stack([1.0 - p, p])  # shape (n, 2)
    else:
        return torch.softmax(logits, dim=1).cpu().numpy()


def _train_one_model(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    pca_dim: int,
    case_type: str,
    device: torch.device,
    model: nn.Module | None = None,
    n_epochs: int = N_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
) -> tuple[nn.Module, list[float]]:
    """Train any torch.nn.Module with early stopping on val loss.

    If *model* is None, a default ``LegalBertClassifier`` is built from
    *pca_dim* and *case_type*.  Pass any ``nn.Module`` to use a different
    architecture — it must accept ``(batch, pca_dim)`` float tensors and
    produce logits compatible with the loss inferred from *case_type*.

    Returns:
    -------
    model : nn.Module  (best weights restored)
    val_loss_curve : list[float]  (one entry per completed epoch)
    """
    torch.manual_seed(SEED)

    if model is None:
        output_dim = 1 if IS_BINARY[case_type] else N_CLASSES[case_type]
        model = LegalBertClassifier(pca_dim=pca_dim, output_dim=output_dim)
    model = model.to(device)
    criterion = _build_criterion(case_type, y_tr, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=L2_WEIGHT_DECAY
    )

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)

    if IS_BINARY[case_type]:
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device).unsqueeze(1)
        y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device).unsqueeze(1)
    else:
        y_tr_t = torch.tensor(y_tr, dtype=torch.long, device=device)
        y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    val_loss_curve: list[float] = []

    epoch_bar = tqdm(range(n_epochs), desc="epochs", unit="ep", leave=False)
    for epoch in epoch_bar:
        # Training step
        model.train()
        optimizer.zero_grad()
        logits = model(X_tr_t)
        loss = criterion(logits, y_tr_t)
        if L1_LAMBDA > 0:
            l1_reg = sum(p.abs().sum() for p in model.parameters())
            loss = loss + L1_LAMBDA * l1_reg
        loss.backward()
        optimizer.step()

        # Validation step
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()
        val_loss_curve.append(val_loss)
        epoch_bar.set_postfix(
            train_loss=f"{loss.item():.4f}", val_loss=f"{val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, val_loss_curve


def _tokenize_sliding_window(
    text: str,
    tokenizer,
    max_windows: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize one document into sliding-window tensors.

    Uses a 512-token window with 128-token stride.  If the number of windows
    exceeds ``max_windows``, equally-spaced indices are selected so the entire
    document length is represented (beginning, middle, and end).

    Returns:
    -------
    input_ids     : LongTensor  (n_windows, 512)
    attention_mask: LongTensor  (n_windows, 512)
    """
    enc = tokenizer(
        text or " ",
        truncation=True,
        max_length=512,
        stride=128,
        return_overflowing_tokens=True,
        return_attention_mask=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = enc["input_ids"]  # (n_windows, 512)
    attention_mask = enc["attention_mask"]

    n_windows = input_ids.shape[0]
    if n_windows > max_windows:
        idx = torch.linspace(0, n_windows - 1, max_windows).long()
        input_ids = input_ids[idx]
        attention_mask = attention_mask[idx]

    return input_ids.to(device), attention_mask.to(device)


def _forward_sliding_window(
    model: "LegalBertForClassification",
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    window_mbatch: int,
    use_amp: bool,
) -> torch.Tensor:
    """Run sliding-window forward pass and return mean-pooled document logits.

    Processes windows in mini-batches of ``window_mbatch`` to bound peak VRAM.
    The [CLS] representation of each window is collected, mean-pooled into a
    single document vector, then passed through the classification head.

    Returns:
    -------
    logits : FloatTensor, shape (1, n_classes)
    """
    all_cls: list[torch.Tensor] = []
    n_windows = input_ids.shape[0]

    for start in range(0, n_windows, window_mbatch):
        ids_mb = input_ids[start : start + window_mbatch]
        mask_mb = attention_mask[start : start + window_mbatch]
        if use_amp:
            with torch.amp.autocast("cuda"):
                cls = model.get_cls_repr(ids_mb, mask_mb)
        else:
            cls = model.get_cls_repr(ids_mb, mask_mb)
        all_cls.append(cls)

    doc_repr = torch.cat(all_cls, dim=0).mean(dim=0, keepdim=True)  # (1, hidden)
    return model.classifier(model.dropout(doc_repr))  # (1, n_classes)


def _train_one_bert_model(
    texts_tr: list[str],
    y_tr: np.ndarray,
    texts_val: list[str],
    y_val: np.ndarray,
    case_type: str,
    device: torch.device,
    n_epochs: int = BERT_EPOCHS,
    patience: int = BERT_PATIENCE,
    n_frozen_layers: int = BERT_FROZEN_LAYERS,
    max_windows: int = BERT_MAX_WINDOWS,
    window_mbatch: int = BERT_WINDOW_BATCH,
    lr: float = BERT_LR,
    weight_decay: float = BERT_WEIGHT_DECAY,
    warmup_ratio: float = BERT_WARMUP_RATIO,
    grad_clip: float = BERT_GRAD_CLIP,
    label_smoothing: float = BERT_LABEL_SMOOTHING,
    ema_alpha: float = BERT_EMA_ALPHA,
    early_stop_on: str = "ema_loss",
) -> tuple["LegalBertForClassification", list[float], list[float]]:
    """Fine-tune Legal BERTimbau with sliding-window aggregation.

    Architecture
    ------------
    - Bottom ``n_frozen_layers`` BERT layers are frozen (embeddings + encoder).
    - Top ``(total_layers - n_frozen_layers)`` layers and the classifier head
      are trainable.
    - Each document is encoded by tokenizing into overlapping 512-token windows
      (stride 128), capped at ``max_windows`` equally-spaced windows.
    - Window [CLS] representations are mean-pooled into a document vector that
      is fed to the linear classifier — capturing full document context.

    Training loop
    -------------
    - One optimizer step per document (effective batch size = 1).
    - Documents are shuffled at the start of every epoch.
    - Linear warmup over the first ``warmup_ratio`` of total steps, then linear
      decay to zero.
    - Gradient clipping at ``grad_clip`` prevents exploding gradients.
    - fp16 mixed precision (AMP) when CUDA is available.

    CUDA is required — raises RuntimeError on CPU.

    early_stop_on controls the stopping signal:
        "ema_loss" (default) : EMA-smoothed val loss — ``ema = alpha * val + (1-alpha) * ema``
        "loss"               : raw val loss, no smoothing

    Returns:
    -------
    model          : LegalBertForClassification  (best checkpoint by val loss)
    train_loss_curve : list[float]  (one entry per epoch)
    val_loss_curve   : list[float]  (one entry per epoch)
    """
    if device.type != "cuda":
        raise RuntimeError(
            "BERT fine-tuning requires a CUDA GPU. "
            "torch.cuda.is_available() returned False. "
            "Fix your PyTorch CUDA installation before running --mode bert."
        )

    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda")

    # Pre-tokenize train and val docs once — avoids repeated tokenization per epoch
    _cpu = torch.device("cpu")
    tok_cache_tr: list[tuple[torch.Tensor, torch.Tensor]] = [
        _tokenize_sliding_window(t, tokenizer, max_windows, _cpu) for t in texts_tr
    ]
    tok_cache_val: list[tuple[torch.Tensor, torch.Tensor]] = [
        _tokenize_sliding_window(t, tokenizer, max_windows, _cpu) for t in texts_val
    ]

    torch.manual_seed(SEED)
    n_classes = 1 if IS_BINARY[case_type] else N_CLASSES[case_type]
    model = LegalBertForClassification(
        n_classes=n_classes,
        n_frozen_layers=n_frozen_layers,
    ).to(device)

    criterion = _build_criterion(
        case_type, y_tr, device, label_smoothing=label_smoothing
    )

    # Only optimise trainable parameters (frozen params are excluded)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    total_steps = n_epochs * len(texts_tr)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    train_loss_curve: list[float] = []
    val_loss_curve: list[float] = []
    ema_loss: float | None = None
    n_tr = len(texts_tr)

    epoch_bar = tqdm(range(n_epochs), desc="bert epochs", unit="ep", leave=False)
    for _ in epoch_bar:
        # --- Training pass (shuffle documents each epoch) ---
        model.train()
        perm = torch.randperm(n_tr)
        train_loss_sum = 0.0

        doc_bar = tqdm(perm, desc="  train docs", unit="doc", leave=False)
        doc_count = 0
        for doc_idx in doc_bar:
            doc_idx = doc_idx.item()
            input_ids = tok_cache_tr[doc_idx][0].to(device)
            attention_mask = tok_cache_tr[doc_idx][1].to(device)
            label = y_tr[doc_idx]
            if IS_BINARY[case_type]:
                label_t = torch.tensor([[float(label)]], device=device)
            else:
                label_t = torch.tensor([int(label)], dtype=torch.long, device=device)

            optimizer.zero_grad()
            logits = _forward_sliding_window(
                model, input_ids, attention_mask, window_mbatch, use_amp
            )
            if use_amp:
                with torch.amp.autocast("cuda"):
                    loss = criterion(logits, label_t)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(logits, label_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                optimizer.step()
            scheduler.step()
            train_loss_sum += loss.item()
            doc_count += 1
            doc_bar.set_postfix(
                loss=f"{loss.item():.4f}", avg=f"{train_loss_sum / doc_count:.4f}"
            )

        # --- Validation pass ---
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for val_idx in range(len(texts_val)):
                input_ids = tok_cache_val[val_idx][0].to(device)
                attention_mask = tok_cache_val[val_idx][1].to(device)
                label = y_val[val_idx]
                if IS_BINARY[case_type]:
                    label_t = torch.tensor([[float(label)]], device=device)
                else:
                    label_t = torch.tensor(
                        [int(label)], dtype=torch.long, device=device
                    )
                logits = _forward_sliding_window(
                    model, input_ids, attention_mask, window_mbatch, use_amp
                )
                val_loss_total += criterion(logits, label_t).item()

        val_loss = val_loss_total / len(texts_val)

        epoch_train_avg = train_loss_sum / n_tr
        train_loss_curve.append(epoch_train_avg)
        val_loss_curve.append(val_loss)

        if early_stop_on == "ema_loss":
            ema_loss = (
                val_loss
                if ema_loss is None
                else ema_alpha * val_loss + (1.0 - ema_alpha) * ema_loss
            )
            stop_signal = ema_loss
        else:
            stop_signal = val_loss

        epoch_bar.set_postfix(
            tr=f"{epoch_train_avg:.4f}",
            val=f"{val_loss:.4f}",
            stop=f"{stop_signal:.4f}",
        )

        if stop_signal < best_val_loss:
            best_val_loss = stop_signal
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, train_loss_curve, val_loss_curve


def plot_loss_curves(
    train_losses: list[float],
    val_losses: list[float],
    case_type: str,
    run_name: str | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    val_f1_curve: list[float] | None = None,
) -> Path:
    """Plot training curves and save to disk.

    Produces a two-panel figure when val_f1_curve is provided:
      - Top:    Train loss vs Val loss  (overfitting diagnosis)
      - Bottom: Val macro-F1 per epoch  (generalization quality)

    Best-epoch marker is placed at the highest val F1 when available,
    otherwise at the lowest val loss.

    Saves to {output_dir}/{case_type}/plots/{run_name}_{case_type}_loss_curves.png.

    Args:
        train_losses: Per-epoch training loss values.
        val_losses: Per-epoch validation loss values.
        case_type: "dv" or "boc".
        run_name: Optional run tag used in the filename and plot title.
        output_dir: Root model directory; the plot is saved under
            ``{output_dir}/{case_type}/plots/``.
        val_f1_curve: Optional per-epoch validation macro-F1 values; when
            given, a second panel is added and the best-epoch marker uses F1
            instead of loss.

    Returns:
        Path to the saved PNG file.
    """
    import matplotlib.pyplot as plt

    plots_dir = Path(output_dir) / case_type / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{run_name}_" if run_name else ""
    plot_path = plots_dir / f"{prefix}{case_type}_loss_curves.png"

    epochs = range(1, len(val_losses) + 1)
    best_epoch = (
        (int(np.argmax(val_f1_curve)) + 1)
        if val_f1_curve
        else (int(np.argmin(val_losses)) + 1)
    )
    best_epoch_label = (
        f"Best F1 epoch ({best_epoch})"
        if val_f1_curve
        else f"Best val loss epoch ({best_epoch})"
    )

    n_panels = 2 if val_f1_curve else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 4 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    axes[0].plot(epochs, train_losses, label="Train loss", marker="o", markersize=4)
    axes[0].plot(epochs, val_losses, label="Val loss", marker="s", markersize=4)
    axes[0].axvline(
        best_epoch, color="gray", linestyle="--", alpha=0.7, label=best_epoch_label
    )
    axes[0].set_ylabel("Loss")
    axes[0].set_title(
        f"{case_type.upper()} — Training Curves ({run_name or 'training'})"
    )
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if val_f1_curve:
        axes[1].plot(
            epochs,
            val_f1_curve,
            label="Val macro-F1",
            color="green",
            marker="^",
            markersize=4,
        )
        axes[1].axvline(best_epoch, color="gray", linestyle="--", alpha=0.7)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Macro-F1")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[0].set_xlabel("Epoch")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"  Loss curve -> {plot_path}")
    return plot_path


def run_timeseries_cv(
    X: np.ndarray,
    y: np.ndarray,
    dates: "pd.Series",
    case_type: str,
    n_splits: int = DEFAULT_N_SPLITS,
    device: torch.device | None = None,
) -> list[dict]:
    """Run TimeSeriesSplit CV with a fresh pipeline + model per fold.

    Temporal ordering is enforced by sorting all data by date before splitting.
    Class weighting is computed from each training fold independently.

    Args:
        X: Frozen BERT embeddings, shape (n, EMBED_DIM).
        y: Integer-encoded labels, shape (n,).
        dates: Decision dates aligned with X/y, used to sort chronologically.
        case_type: "dv" or "boc".
        n_splits: Number of TimeSeriesSplit folds.
        device: Torch device (auto-selected if None).

    Returns:
        list of dicts, one per fold:
            {
              'fold'            : int,
              'y_true'          : np.ndarray,
              'y_pred'          : np.ndarray,
              'pca_n_components': int,
              'train_size'      : int,
              'val_size'        : int,
              'val_loss_curve'  : list[float],
            }
    """
    import pandas as pd

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sorting by date, as we use TimeSeriesSplit to prevent temporal leakage between folds.
    sort_idx = np.argsort(dates.values)
    X = X[sort_idx]
    y = y[sort_idx]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        tqdm(tscv.split(X), total=n_splits, desc="CV folds", unit="fold")
    ):
        X_tr_raw, X_val_raw = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        fold_pipeline = clone(build_feature_pipeline())
        X_tr = fold_pipeline.fit_transform(X_tr_raw).astype(np.float32)
        X_val = fold_pipeline.transform(X_val_raw).astype(np.float32)

        pca_dim = fold_pipeline.named_steps["pca"].n_components_

        model, val_loss_curve = _train_one_model(
            X_tr,
            y_tr,
            X_val,
            y_val,
            pca_dim=pca_dim,
            case_type=case_type,
            device=device,
        )

        with torch.no_grad():
            logits = model(torch.tensor(X_val, dtype=torch.float32, device=device))
        y_pred = predict_from_logits(logits, IS_BINARY[case_type])

        fold_results.append(
            {
                "fold": fold_idx,
                "y_true": y_val,
                "y_pred": y_pred,
                "pca_n_components": pca_dim,
                "train_size": len(train_idx),
                "val_size": len(val_idx),
                "val_loss_curve": val_loss_curve,
            }
        )
        print(
            f"  Fold {fold_idx}: train={len(train_idx)}, val={len(val_idx)}, "
            f"pca_dim={pca_dim}, best_val_loss={min(val_loss_curve):.4f}"
        )

    return fold_results


def train_final_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    case_type: str,
    device: torch.device | None = None,
) -> tuple["Pipeline", "LegalBertClassifier"]:
    """Fit the feature pipeline and classifier on the full training set.

    The final model is trained for N_EPOCHS without early stopping, maximising
    use of all available training data. Regularisation (Dropout, L2, L1) guards
    against overfitting.

    Args:
        X_train: Frozen BERT embeddings, shape (n, EMBED_DIM).
        y_train: Integer-encoded labels, shape (n,).
        case_type: "dv" or "boc".
        device: Torch device (auto-selected if None).

    Returns:
        Tuple of (fitted_pipeline, fitted_model): a fitted
        ``sklearn.pipeline.Pipeline`` and a ``LegalBertClassifier`` in eval
        mode.
    """
    from sklearn.pipeline import Pipeline

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_pipeline = build_feature_pipeline()
    X_reduced = feature_pipeline.fit_transform(X_train).astype(np.float32)
    pca_dim = feature_pipeline.named_steps["pca"].n_components_

    output_dim = 1 if IS_BINARY[case_type] else N_CLASSES[case_type]
    torch.manual_seed(SEED)
    model = LegalBertClassifier(pca_dim=pca_dim, output_dim=output_dim).to(device)
    criterion = _build_criterion(case_type, y_train, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=L2_WEIGHT_DECAY
    )

    X_t = torch.tensor(X_reduced, dtype=torch.float32, device=device)
    if IS_BINARY[case_type]:
        y_t = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1)
    else:
        y_t = torch.tensor(y_train, dtype=torch.long, device=device)

    for _ in tqdm(range(N_EPOCHS), desc="final model", unit="ep"):
        model.train()
        optimizer.zero_grad()
        logits = model(X_t)
        loss = criterion(logits, y_t)
        if L1_LAMBDA > 0:
            l1_reg = sum(p.abs().sum() for p in model.parameters())
            loss = loss + L1_LAMBDA * l1_reg
        loss.backward()
        optimizer.step()

    model.eval()
    return feature_pipeline, model


def run_timeseries_cv_bert(
    texts: list[str],
    y: np.ndarray,
    dates: "pd.Series",
    case_type: str,
    n_splits: int = DEFAULT_N_SPLITS,
    device: torch.device | None = None,
) -> list[dict]:
    """TimeSeriesSplit CV using full BERT fine-tuning per fold.

    Requires CUDA — delegates the check to _train_one_bert_model.
    Returns the same fold-result dict structure as run_timeseries_cv
    (pca_n_components is None since no PCA is used here).

    Args:
        texts: Raw document texts, aligned with y and dates.
        y: Integer-encoded labels, shape (n,).
        dates: Decision dates aligned with texts/y, used to sort chronologically.
        case_type: "dv" or "boc".
        n_splits: Number of TimeSeriesSplit folds.
        device: Torch device (auto-selected if None).

    Returns:
        list of per-fold result dicts; see ``run_timeseries_cv`` for the
        dict shape (``pca_n_components`` is always None here).
    """
    import pandas as pd

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sort_idx = np.argsort(dates.values)
    texts_sorted = [texts[i] for i in sort_idx]
    y_sorted = y[sort_idx]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        tqdm(tscv.split(y_sorted), total=n_splits, desc="BERT CV folds", unit="fold")
    ):
        texts_tr = [texts_sorted[i] for i in train_idx]
        texts_val = [texts_sorted[i] for i in val_idx]
        y_tr, y_val = y_sorted[train_idx], y_sorted[val_idx]

        model, _, val_loss_curve = _train_one_bert_model(
            texts_tr,
            y_tr,
            texts_val,
            y_val,
            case_type=case_type,
            device=device,
        )

        # Collect predictions using sliding-window aggregation
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        all_logits = []
        with torch.no_grad():
            for text in texts_val:
                input_ids, attention_mask = _tokenize_sliding_window(
                    text, tokenizer, BERT_MAX_WINDOWS, device
                )
                logits = _forward_sliding_window(
                    model,
                    input_ids,
                    attention_mask,
                    BERT_WINDOW_BATCH,
                    device.type == "cuda",
                )
                all_logits.append(logits.cpu())
        logits_cat = torch.cat(all_logits, dim=0)
        y_pred = predict_from_logits(logits_cat, IS_BINARY[case_type])

        fold_results.append(
            {
                "fold": fold_idx,
                "y_true": y_val,
                "y_pred": y_pred,
                "pca_n_components": None,
                "train_size": len(train_idx),
                "val_size": len(val_idx),
                "val_loss_curve": val_loss_curve,
            }
        )
        print(
            f"  BERT Fold {fold_idx}: train={len(train_idx)}, val={len(val_idx)}, "
            f"best_val_loss={min(val_loss_curve):.4f}"
        )

    return fold_results


def train_final_bert_model(
    texts_train: list[str],
    y_train: np.ndarray,
    case_type: str,
    device: torch.device | None = None,
    n_frozen_layers: int = BERT_FROZEN_LAYERS,
    max_windows: int = BERT_MAX_WINDOWS,
    window_mbatch: int = BERT_WINDOW_BATCH,
    lr: float = BERT_LR,
    warmup_ratio: float = BERT_WARMUP_RATIO,
    grad_clip: float = BERT_GRAD_CLIP,
) -> "LegalBertForClassification":
    """Fine-tune BERT on the full training set (no validation split).

    Uses the same sliding-window aggregation and layer-freezing strategy as
    ``_train_one_bert_model``, but trains for the full ``BERT_EPOCHS`` without
    early stopping, maximising use of all available training data.

    CUDA required — raises RuntimeError on CPU.

    Args:
        texts_train: Raw document texts for the full training set.
        y_train: Integer-encoded labels, shape (n,).
        case_type: "dv" or "boc".
        device: Torch device (auto-selected if None).
        n_frozen_layers: Number of bottom BERT encoder layers to freeze.
        max_windows: Cap on sliding windows per document.
        window_mbatch: Window mini-batch size through the unfrozen layers.
        lr: Learning rate for the AdamW optimizer.
        warmup_ratio: Fraction of total steps used for linear LR warmup.
        grad_clip: Max gradient norm for gradient clipping.

    Returns:
        The fine-tuned ``LegalBertForClassification`` in eval mode.

    Raises:
        RuntimeError: If ``device`` is not a CUDA device.
    """
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        raise RuntimeError(
            "BERT fine-tuning requires a CUDA GPU. "
            "torch.cuda.is_available() returned False. "
            "Fix your PyTorch CUDA installation before running --mode bert."
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    use_amp = True
    scaler = torch.amp.GradScaler("cuda")

    # Pre-tokenize all documents once — avoids re-tokenizing (slow CPU op) every epoch
    print("  Pre-tokenizing documents (one-time cost)...")
    tok_cache: list[tuple[torch.Tensor, torch.Tensor]] = [
        _tokenize_sliding_window(t, tokenizer, max_windows, torch.device("cpu"))
        for t in tqdm(texts_train, desc="  pre-tok", unit="doc", leave=False)
    ]

    torch.manual_seed(SEED)
    n_classes = 1 if IS_BINARY[case_type] else N_CLASSES[case_type]
    model = LegalBertForClassification(
        n_classes=n_classes,
        n_frozen_layers=n_frozen_layers,
    ).to(device)

    criterion = _build_criterion(case_type, y_train, device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    total_steps = BERT_EPOCHS * len(texts_train)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    n_tr = len(texts_train)
    epoch_bar = tqdm(range(BERT_EPOCHS), desc="bert final model", unit="ep")
    for _ in epoch_bar:
        model.train()
        perm = torch.randperm(n_tr)
        loss_sum = 0.0
        doc_bar = tqdm(perm, desc="  docs", unit="doc", leave=False)
        for doc_idx in doc_bar:
            doc_idx = doc_idx.item()
            input_ids = tok_cache[doc_idx][0].to(device)
            attention_mask = tok_cache[doc_idx][1].to(device)
            label = y_train[doc_idx]
            if IS_BINARY[case_type]:
                label_t = torch.tensor([[float(label)]], device=device)
            else:
                label_t = torch.tensor([int(label)], dtype=torch.long, device=device)

            optimizer.zero_grad()
            logits = _forward_sliding_window(
                model, input_ids, attention_mask, window_mbatch, use_amp
            )
            with torch.amp.autocast("cuda"):
                loss = criterion(logits, label_t)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += loss.item()
            doc_bar.set_postfix(loss=f"{loss.item():.4f}")
        epoch_bar.set_postfix(avg_loss=f"{loss_sum / n_tr:.4f}")

    model.eval()
    return model


def save_model(
    fitted_pipeline,
    fitted_model: LegalBertClassifier,
    case_type: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    run_name: str = None,
) -> Path:
    """Save the fitted sklearn pipeline and PyTorch model to disk.

    Files
    -----
    {output_dir}/{case_type}/feature_pipeline.joblib
    {output_dir}/{case_type}/classifier.pt

    Args:
        fitted_pipeline: Fitted ``sklearn.pipeline.Pipeline`` (scaler + PCA).
        fitted_model: Trained ``LegalBertClassifier``.
        case_type: "dv" or "boc".
        output_dir: Root model directory (parent of the case_type
            subdirectory) to write into.
        run_name: Optional prefix for the saved filenames (e.g. "v1").

    Returns:
        The case_type model directory the artefacts were written into.
    """
    model_dir = Path(output_dir) / case_type
    model_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{run_name}_" if run_name else ""
    pipeline_path = model_dir / f"{prefix}feature_pipeline.joblib"
    classifier_path = model_dir / f"{prefix}classifier.pt"

    joblib.dump(fitted_pipeline, pipeline_path)

    pca_dim = fitted_pipeline.named_steps["pca"].n_components_
    output_dim = 1 if IS_BINARY[case_type] else N_CLASSES[case_type]
    torch.save(
        {
            "state_dict": fitted_model.state_dict(),
            "pca_dim": pca_dim,
            "output_dim": output_dim,
            "case_type": case_type,
            "hidden_dim": HIDDEN_DIM,
            "dropout_prob": DROPOUT_PROB,
        },
        classifier_path,
    )

    print(f"  Pipeline   -> {pipeline_path}")
    print(f"  Classifier -> {classifier_path}")
    return model_dir


def save_bert_model(
    fitted_model: "LegalBertForClassification",
    case_type: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    run_name: str = None,
) -> Path:
    """Save a fine-tuned LegalBertForClassification to disk.

    File
    ----
    {output_dir}/{case_type}/{run_name}_bert_classifier.pt

    Args:
        fitted_model: Fine-tuned ``LegalBertForClassification``.
        case_type: "dv" or "boc".
        output_dir: Root model directory (parent of the case_type
            subdirectory) to write into.
        run_name: Optional prefix for the saved filename (e.g. "bert_v1").

    Returns:
        The case_type model directory the artefact was written into.
    """
    model_dir = Path(output_dir) / case_type
    model_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{run_name}_" if run_name else ""
    classifier_path = model_dir / f"{prefix}{case_type}_bert_classifier.pt"
    n_classes = 1 if IS_BINARY[case_type] else N_CLASSES[case_type]
    torch.save(
        {
            "state_dict": fitted_model.state_dict(),
            "n_classes": n_classes,
            "case_type": case_type,
            "dropout_prob": BERT_DROPOUT,
            "model_name": MODEL_NAME,
            "n_frozen_layers": getattr(fitted_model, "n_frozen_layers", 0),
            "max_windows": BERT_MAX_WINDOWS,
            "window_mbatch": BERT_WINDOW_BATCH,
        },
        classifier_path,
    )

    print(f"  BERT Classifier -> {classifier_path}")
    return model_dir


def _run_smoke_test() -> None:
    """End-to-end smoke test using synthetic 1024-dim embeddings.

    Exercises: run_timeseries_cv → train_final_model → save_model for
    both binary (dv) and ternary (boc) tasks without touching real data.
    """
    import tempfile

    import pandas as pd

    print("Running train.py smoke test with synthetic data...\n")

    rng = np.random.default_rng(seed=SEED)
    N = 300  # enough samples for 3-fold TimeSeriesSplit
    # Monotonically increasing dates — required by TimeSeriesSplit sort
    dates = pd.Series(pd.date_range("2015-01-01", periods=N, freq="D"))

    for case_type in ("dv", "boc"):
        print(f"--- case_type={case_type} ---")
        n_classes = N_CLASSES[case_type]

        # Synthetic embeddings: standard Gaussian (mimics post-BERT distribution)
        X = rng.standard_normal((N, 1024)).astype(np.float32)
        # Balanced labels cycling through all classes
        y = np.tile(np.arange(n_classes), N // n_classes + 1)[:N].astype(np.int64)

        unique, counts = np.unique(y, return_counts=True)
        print(
            f"  X.shape: {X.shape}, "
            f"y distribution: {dict(zip(unique.tolist(), counts.tolist()))}"
        )

        # ── Cross-validation ──────────────────────────────────────────────
        print("  Running 3-fold TimeSeriesSplit CV...")
        cv_results = run_timeseries_cv(X, y, dates, case_type=case_type, n_splits=3)
        assert len(cv_results) == 3, "Expected 3 CV folds"
        for r in cv_results:
            assert r["y_pred"].shape == r["y_true"].shape, "Shape mismatch in fold"
            assert np.issubdtype(
                r["y_pred"].dtype, np.integer
            ), f"y_pred must be integer, got {r['y_pred'].dtype}"
        print(f"  CV passed — {len(cv_results)} folds OK")

        # ── Final model ───────────────────────────────────────────────────
        print("  Training final model on full synthetic set...")
        fitted_pipeline, fitted_model = train_final_model(X, y, case_type=case_type)
        pca_dim = fitted_pipeline.named_steps["pca"].n_components_
        assert pca_dim > 0, "PCA returned 0 components"
        print(f"  Final model OK — pca_dim={pca_dim}")

        # ── Save / load round-trip ────────────────────────────────────────
        print("  Testing save_model (temp dir)...")
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = save_model(
                fitted_pipeline,
                fitted_model,
                case_type=case_type,
                output_dir=Path(tmp),
                run_name="smoke_test",
            )
            assert (
                model_dir / "smoke_test_feature_pipeline.joblib"
            ).exists(), "smoke_test_feature_pipeline.joblib not found after save"
            assert (
                model_dir / "smoke_test_classifier.pt"
            ).exists(), "smoke_test_classifier.pt not found after save"
        print("  Save/load round-trip OK\n")

    print("Smoke test passed.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(
    case_type: str,
    mode: str = "mlp",
    n_splits: int = DEFAULT_N_SPLITS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    run_name: str = None,
    early_stop_on: str = "ema_loss",
) -> None:
    """Run the CLI training pipeline: CV, final-model fit, and artefact saving.

    In "mlp" mode, loads frozen embeddings, runs TimeSeriesSplit CV with the
    MLP head, trains the final model on the full training set, and saves the
    pipeline + classifier. In "bert" mode, loads raw texts, fine-tunes BERT
    with a temporal train/val split and early stopping, then saves the model
    and a loss-curve plot.

    Args:
        case_type: "dv" or "boc".
        mode: "mlp" (frozen embeddings + MLP head) or "bert" (full BERT
            fine-tuning; requires CUDA).
        n_splits: Number of TimeSeriesSplit CV folds (mlp mode only).
        output_dir: Root model directory to save artefacts into.
        run_name: Optional prefix for saved filenames.
        early_stop_on: Early stopping signal for BERT mode: "ema_loss"
            (EMA-smoothed val loss) or "loss" (raw val loss).

    Raises:
        ValueError: If ``mode`` is not "mlp" or "bert".
    """
    from sklearn.metrics import f1_score, matthews_corrcoef

    print(f"\n{'=' * 60}")
    print(f"Legal BERTimbau Classifier — Training ({case_type.upper()}, mode={mode})")
    print(f"{'=' * 60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}\n")

    if mode == "mlp":
        print(
            f"  N_EPOCHS={N_EPOCHS}, HIDDEN_DIM={HIDDEN_DIM}, "
            f"DROPOUT_PROB={DROPOUT_PROB}\n"
            f"  LEARNING_RATE={LEARNING_RATE}, L2_WEIGHT_DECAY={L2_WEIGHT_DECAY}, "
            f"L1_LAMBDA={L1_LAMBDA}, PATIENCE={EARLY_STOPPING_PATIENCE}"
        )
        print("\nLoading training data (frozen embeddings)...")
        X, y_raw, dates, _ = load_split_data(case_type, split="train")
        y = encode_labels(y_raw, case_type)
        unique, counts = np.unique(y, return_counts=True)
        print(
            f"  {len(X)} cases  |  class distribution: {dict(zip(unique.tolist(), counts.tolist()))}"
        )

        print(f"\nRunning {n_splits}-fold TimeSeriesSplit CV...")
        cv_results = run_timeseries_cv(
            X, y, dates, case_type=case_type, n_splits=n_splits, device=device
        )

        print("\nCV Summary:")
        for r in cv_results:
            macro_f1 = f1_score(
                r["y_true"], r["y_pred"], average="macro", zero_division=0
            )
            mcc = matthews_corrcoef(r["y_true"], r["y_pred"])
            print(f"  Fold {r['fold']}: macro_F1={macro_f1:.4f}  MCC={mcc:.4f}")

        print("\nTraining final model on full training set...")
        fitted_pipeline, fitted_model = train_final_model(
            X, y, case_type=case_type, device=device
        )

        print("\nSaving model artifacts...")
        save_model(
            fitted_pipeline,
            fitted_model,
            case_type=case_type,
            output_dir=output_dir,
            run_name=run_name,
        )

    elif mode == "bert":
        print(
            f"  BERT_EPOCHS={BERT_EPOCHS}, BERT_LR={BERT_LR}, BERT_PATIENCE={BERT_PATIENCE}\n"
            f"  BERT_DROPOUT={BERT_DROPOUT}, BERT_WEIGHT_DECAY={BERT_WEIGHT_DECAY}, "
            f"BERT_LABEL_SMOOTHING={BERT_LABEL_SMOOTHING}, BERT_VAL_RATIO={BERT_VAL_RATIO}\n"
            f"  Early stopping: {early_stop_on} (EMA alpha={BERT_EMA_ALPHA})"
        )
        print("\nLoading training data (raw text for BERT fine-tuning)...")
        texts_tr, y_tr, texts_val, y_val = load_bert_train_val_split(
            case_type, val_ratio=BERT_VAL_RATIO
        )
        unique_tr, counts_tr = np.unique(y_tr, return_counts=True)
        unique_val, counts_val = np.unique(y_val, return_counts=True)
        print(
            f"  Train: {len(texts_tr)} cases  |  class dist: {dict(zip(unique_tr.tolist(), counts_tr.tolist()))}"
        )
        print(
            f"  Val:   {len(texts_val)} cases  |  class dist: {dict(zip(unique_val.tolist(), counts_val.tolist()))}"
        )

        print("\nTraining BERT model with temporal val split + early stopping...")
        fitted_model, train_loss_curve, val_loss_curve = _train_one_bert_model(
            texts_tr,
            y_tr,
            texts_val,
            y_val,
            case_type=case_type,
            device=device,
            early_stop_on=early_stop_on,
        )

        print("\nSaving model artifacts...")
        save_bert_model(
            fitted_model, case_type=case_type, output_dir=output_dir, run_name=run_name
        )
        plot_loss_curves(
            train_loss_curve,
            val_loss_curve,
            case_type,
            run_name=run_name,
            output_dir=output_dir,
        )

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'mlp' or 'bert'.")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Legal BERTimbau classification head."
    )
    parser.add_argument("--case_type", choices=["dv", "boc"], default=None)
    parser.add_argument(
        "--mode",
        choices=["mlp", "bert"],
        default="mlp",
        help="mlp: frozen embeddings + MLP head (CPU ok).  bert: full BERT fine-tuning (CUDA required).",
    )
    parser.add_argument("--n_splits", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name prefix for saved model files.",
    )
    parser.add_argument(
        "--early_stop_on",
        choices=["ema_loss", "loss"],
        default="ema_loss",
        help="Early stopping signal for BERT: 'ema_loss' (EMA-smoothed val loss) or 'loss' (raw val loss).",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run end-to-end smoke test with synthetic data (no real data required).",
    )
    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke_test()
    else:
        if args.case_type is None:
            parser.error("--case_type is required unless --smoke_test is set")
        main(
            args.case_type,
            args.mode,
            args.n_splits,
            args.output_dir,
            args.run_name,
            args.early_stop_on,
        )
