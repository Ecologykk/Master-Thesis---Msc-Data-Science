"""Slice: smart truncation strategy (steps 1 to 4, partial ranking).

This module implements:
1) Sentence splitting for Portuguese legal decisions.
2) Frozen LegalBERTimbau sentence embeddings (mean pooling).
3) Query embedding for case-type intent (DV/BOC).
4) Document centroid embedding.
5) Linguistic anchor scoring (facts, norms, proof, reasoning, jurisprudence, conflict).
6) Combined query+centroid+anchor sentence scoring.

Determinism note
-----------------
Seeds are fixed and decoding is greedy (`temperature=0.0`, `top_k=1`,
`seed=42`), so this script is deterministic on identical hardware. Results
may still diverge slightly on different hardware: GPU floating-point
reductions are non-associative and kernel selection, driver version and
tensor-core availability all change the order of operations. Expect small
differences in embeddings and occasional label flips on borderline cases.
This does not indicate a bug.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

LEGAL_BERTIMBAU_MODEL = "stjiris/bert-large-portuguese-cased-legal-mlm-nli-sts-v1"
CASE_TYPE_QUERIES: dict[str, tuple[str, ...]] = {
    "dv": (
        "fundamentos jurídicos para manter ou alterar a decisão recorrida",
        "razões do tribunal de recurso para confirmar ou revogar a decisão",
        "factos e normas que justificam provimento ou improvimento do recurso",
        "avaliação da prova e enquadramento legal para manter ou modificar a decisão",
        "análise de fundamentos rejeitados ou aceites que influenciam a manutenção ou alteração da decisão",
    ),
    "boc": (
        "fundamentos jurídicos para procedência, improcedência ou procedência parcial da ação",
        "razões do tribunal para decidir favorável, desfavorável ou parcialmente favorável",
        "factos provados e normas aplicadas que determinam o resultado da ação",
        "apreciação da prova e enquadramento legal para decisão total ou parcial da ação",
        "análise de pedidos aceites ou rejeitados e sua influência no resultado final da ação",
    ),
}

FACTUAL_ANCHORS = (
    "factos provados",
    "matéria de facto",
    "matéria de facto provada",
    "dos factos não provados",
    "com relevância para a decisão",
    "resulta dos autos",
    "ficou demonstrado que",
    "apurou-se que",
)

LEGAL_NORMS_ANCHORS = (
    "nos termos do artigo",
    "de acordo com o disposto",
    "ao abrigo do",
    "nos termos legais",
    "regime jurídico aplicável",
    "conforme previsto",
    "à luz do",
    "dispõe o artigo",
    "nos termos dos artigos",
    "à luz do regime",
)

PROOF_EVAL_ANCHORS = (
    "apreciação da prova",
    "valor probatório",
    "prova testemunhal",
    "prova documental",
    "não se provou",
    "carece de prova",
    "insuficiência de prova",
    "livre apreciação da prova",
    "convicção do tribunal",
    "não ficou demonstrado",
)

REASONING_ANCHORS = (
    "entende o tribunal",
    "considera-se que",
    "conclui-se que",
    "não assiste razão",
    "não procede o argumento",
    "improcedem as alegações",
    "é de concluir",
    "cumpre apreciar",
    "cumpre decidir",
    "impõe-se concluir",
    "assim sendo",
    "neste contexto",
    "face ao exposto",
)

JURISPRUDENCE_ANCHORS = (
    "jurisprudência",
    "acórdão do",
    "segundo entendimento",
    "conforme decidido",
    "em casos semelhantes",
    "orientação jurisprudencial",
    "supremo tribunal de justiça",
    "uniformização de jurisprudência",
)

CONFLICT_ANCHORS = (
    "não se verifica",
    "não assiste razão",
    "carece de fundamento",
)

ANCHOR_GROUPS: dict[str, tuple[str, ...]] = {
    "facts": FACTUAL_ANCHORS,
    "norms": LEGAL_NORMS_ANCHORS,
    "proof": PROOF_EVAL_ANCHORS,
    "reasoning": REASONING_ANCHORS,
    "jurisprudence": JURISPRUDENCE_ANCHORS,
    "conflict": CONFLICT_ANCHORS,
}

ANCHOR_GROUP_WEIGHTS: dict[str, float] = {
    "facts": 1.0,
    "norms": 0.9,
    "proof": 1.1,
    "reasoning": 1.0,
    "jurisprudence": 0.8,
    "conflict": 1.3,
}

# Conservative splitter for legal prose; keeps punctuation attached.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;:])\s+")


@dataclass(frozen=True)
class SentenceEmbeddingBatch:
    """Container for sentence-level embeddings extracted from a document.

    Attributes:
        sentences: Sentence strings, in document order.
        embeddings: float32 ndarray of shape (n_sentences, hidden_size),
            aligned row-for-row with `sentences`.
    """

    sentences: list[str]
    embeddings: np.ndarray  # shape: (n_sentences, hidden_size), dtype float32


@dataclass(frozen=True)
class SentenceScoreBatch:
    """Container for sentence scores and top-k summary selection.

    Attributes:
        scores: float32 ndarray of shape (n_sentences,) with one combined
            score per sentence, in original document order.
        top_indices: Indices (into `scores`/the source sentence list) of the
            sentences selected for the summary, sorted ascending (document
            order).
        summary_text: Selected sentences joined with a single space, in
            document order.
    """

    scores: np.ndarray  # shape: (n_sentences,), dtype float32
    top_indices: list[int]
    summary_text: str


def split_sentences_pt_legal(text: str, *, min_chars: int = 8) -> list[str]:
    """Split legal text into sentence-like units with lightweight normalization.

    Collapses all whitespace, splits on the conservative `[.!?;:]`-followed-
    by-whitespace boundary (keeping punctuation attached to each sentence),
    and drops fragments shorter than `min_chars`.

    Args:
        text: Raw text to split; falsy/None values yield an empty list.
        min_chars: Minimum character length for a split fragment to be kept.

    Returns:
        List of sentence strings, in original order.

    Raises:
        ValueError: If `min_chars < 1`.
    """
    if min_chars < 1:
        raise ValueError("min_chars must be >= 1")

    clean = " ".join(str(text or "").split())
    if not clean:
        return []

    parts = _SENTENCE_SPLIT_RE.split(clean)
    out: list[str] = []
    for part in parts:
        s = part.strip()
        if len(s) >= min_chars:
            out.append(s)
    return out


def _mean_pool(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Mean-pool token embeddings over non-padding positions, per the attention mask."""
    expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * expanded
    return masked.sum(dim=1) / expanded.sum(dim=1).clamp(min=1e-9)


def _best_device() -> torch.device:
    """Return the CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _normalize_case_type(case_type: str) -> str:
    """Normalize and validate a case type string against `CASE_TYPE_QUERIES`."""
    key = str(case_type or "").strip().lower()
    if key not in CASE_TYPE_QUERIES:
        valid = ", ".join(sorted(CASE_TYPE_QUERIES))
        raise ValueError(f"Unknown case_type '{case_type}'. Expected one of: {valid}")
    return key


@lru_cache(maxsize=1)
def _load_encoder(model_name: str = LEGAL_BERTIMBAU_MODEL):
    """Lazy singleton load for tokenizer/model to avoid repeated startup cost."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(_best_device())
    return tokenizer, model


def encode_sentences_legalbert(
    sentences: Iterable[str],
    *,
    model_name: str = LEGAL_BERTIMBAU_MODEL,
    batch_size: int = 16,
    max_length: int = 256,
) -> np.ndarray:
    """Encode sentence list with frozen LegalBERTimbau and mean pooling.

    Empty/whitespace-only sentences are dropped before encoding. The
    tokenizer/model pair is lazily loaded once per `model_name` (via
    `_load_encoder`) and reused across calls; the model is frozen (`eval()`
    mode, no gradient computation).

    Args:
        sentences: Sentences to encode.
        model_name: HuggingFace model id to load/use.
        batch_size: Number of sentences encoded per forward pass.
        max_length: Max token length per sentence; longer sentences are
            truncated.

    Returns:
        float32 ndarray of shape (n_sentences, hidden_size), where
        `n_sentences` counts only the non-empty sentences after stripping.
        Shape is (0, 0) if no non-empty sentences remain.

    Raises:
        ValueError: If `batch_size < 1` or `max_length < 8`.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if max_length < 8:
        raise ValueError("max_length must be >= 8")

    sentence_list = [str(s).strip() for s in sentences if str(s).strip()]
    if not sentence_list:
        return np.zeros((0, 0), dtype=np.float32)

    tokenizer, model = _load_encoder(model_name)
    device = next(model.parameters()).device
    chunks: list[np.ndarray] = []

    with torch.no_grad():
        for i in range(0, len(sentence_list), batch_size):
            batch = sentence_list[i : i + batch_size]
            tok = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = tok["input_ids"].to(device)
            attention_mask = tok["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pooled = _mean_pool(outputs.last_hidden_state, attention_mask)
            chunks.append(pooled.float().cpu().numpy())

    return np.vstack(chunks).astype(np.float32, copy=False)


def _ensure_2d_embeddings(embeddings: np.ndarray | list) -> np.ndarray:
    """Normalize nested embedding inputs to float32 ndarray (n_sentences, hidden_size)."""
    if isinstance(embeddings, np.ndarray):
        arr = embeddings
    else:
        arr = np.asarray(embeddings, dtype=object)

    if arr.ndim == 2 and arr.dtype != object:
        return arr.astype(np.float32, copy=False)

    if arr.ndim == 1:
        if arr.size == 0:
            return np.zeros((0, 0), dtype=np.float32)
        first = arr[0]
        if isinstance(first, np.ndarray):
            return np.vstack([np.asarray(v, dtype=np.float32) for v in arr])
        if isinstance(first, (list, tuple)):
            return np.vstack([np.asarray(v, dtype=np.float32) for v in arr])

    raise ValueError("Embeddings must be a 2D array or a sequence of vectors.")


def _cosine_similarity_matrix_vector(
    matrix: np.ndarray, vector: np.ndarray
) -> np.ndarray:
    """Cosine similarity between each matrix row and a vector."""
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)

    m = matrix.astype(np.float32, copy=False)
    v = vector.astype(np.float32, copy=False)
    m_norm = np.linalg.norm(m, axis=1)
    v_norm = float(np.linalg.norm(v))
    denom = np.clip(m_norm * max(v_norm, 1e-9), 1e-9, None)  # avoid division by zero
    sims = (m @ v) / denom
    return sims.astype(np.float32, copy=False)


@lru_cache(maxsize=32)
def _encode_text_cached(text: str, model_name: str) -> np.ndarray:
    """Encode a single text string with LegalBERTimbau, caching by (text, model_name)."""
    emb = encode_sentences_legalbert(
        [text],
        model_name=model_name,
        batch_size=1,
        max_length=128,
    )
    if emb.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    return emb[0]


@lru_cache(maxsize=8)
def _query_embedding_cached(case_type: str, model_name: str) -> np.ndarray:
    """Compute and cache the mean query embedding over `CASE_TYPE_QUERIES[case_type]`."""
    key = _normalize_case_type(case_type)
    query_texts = CASE_TYPE_QUERIES[key]
    vectors = [_encode_text_cached(q, model_name=model_name) for q in query_texts]
    if not vectors:
        return np.zeros((0,), dtype=np.float32)
    mat = np.vstack(vectors).astype(np.float32, copy=False)
    return mat.mean(axis=0).astype(np.float32, copy=False)


def query_embedding_for_case_type(
    case_type: str,
    *,
    model_name: str = LEGAL_BERTIMBAU_MODEL,
) -> np.ndarray:
    """Return mean query embedding over multiple templates for case type intent.

    Args:
        case_type: Case type key into `CASE_TYPE_QUERIES` ("dv" or "boc";
            validated/normalized by `_normalize_case_type`).
        model_name: LegalBERTimbau model id used to encode the query
            templates.

    Returns:
        float32 ndarray of shape (hidden_size,): the mean of the encoded
        `CASE_TYPE_QUERIES[case_type]` template embeddings. Shape (0,) if
        no query templates are defined for the case type.

    Raises:
        ValueError: If `case_type` is not a recognized key (raised inside
            `_normalize_case_type`, via the cached helper).
    """
    return _query_embedding_cached(case_type=case_type, model_name=model_name)


def anchor_scores_for_sentences(
    sentences: list[str] | np.ndarray,
    *,
    anchor_group_weights: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Compute normalized anchor score per sentence using weighted anchor groups.

    For each sentence, checks membership in each anchor group in
    `ANCHOR_GROUPS` (case-insensitive substring match against any pattern in
    the group), sums the weights of matched groups, and normalizes by the
    total weight of all groups with weight > 0.

    Args:
        sentences: Sentences to score, in any order.
        anchor_group_weights: Optional overrides for `ANCHOR_GROUP_WEIGHTS`,
            merged on top of the defaults (unspecified groups keep their
            default weight).

    Returns:
        float32 ndarray of shape (len(sentences),) in [0, 1], where higher
        means stronger linguistic anchor match. All zeros if `sentences` is
        empty or if every group weight resolves to 0.

    Raises:
        ValueError: If `anchor_group_weights` contains an unknown group name,
            or a negative weight.
    """
    sentence_list = [str(s).lower() for s in list(sentences)]
    if not sentence_list:
        return np.zeros((0,), dtype=np.float32)

    group_weights = dict(ANCHOR_GROUP_WEIGHTS)
    if anchor_group_weights is not None:
        for k, v in anchor_group_weights.items():
            if k not in ANCHOR_GROUPS:
                raise ValueError(
                    f"Unknown anchor group '{k}'. Expected one of: {sorted(ANCHOR_GROUPS)}"
                )
            if v < 0:
                raise ValueError(f"Anchor group weight for '{k}' must be >= 0.")
            group_weights[k] = float(v)

    weighted = np.zeros((len(sentence_list),), dtype=np.float32)
    total = 0.0
    for group, patterns in ANCHOR_GROUPS.items():
        gw = float(group_weights.get(group, 0.0))
        if gw <= 0:
            continue
        hits = np.array(
            [
                1.0 if any(p in sent for p in patterns) else 0.0
                for sent in sentence_list
            ],
            dtype=np.float32,
        )
        weighted += gw * hits
        total += gw

    if total <= 0:
        return np.zeros((len(sentence_list),), dtype=np.float32)
    return (weighted / total).astype(np.float32, copy=False)


def _sentence_anchor_categories(sentence: str) -> set[str]:
    """Return the set of anchor group names (from ANCHOR_GROUPS) matched by a sentence."""
    s = sentence.lower()
    return {
        cat for cat, patterns in ANCHOR_GROUPS.items() if any(p in s for p in patterns)
    }


def _estimate_sentence_tokens(sentence: str) -> int:
    """Cheap token estimate for budgeted selection (roughly 4 chars/token + overhead)."""
    return max(1, int(len(sentence) / 4) + 2)


def document_centroid_embedding(sentence_embeddings: np.ndarray | list) -> np.ndarray:
    """Compute centroid embedding as mean sentence vector.

    Args:
        sentence_embeddings: Per-sentence embeddings; coercible to a 2D
            float32 array via `_ensure_2d_embeddings`.

    Returns:
        float32 ndarray of shape (hidden_size,): the mean over all sentence
        rows. Shape (0,) if `sentence_embeddings` is empty.
    """
    mat = _ensure_2d_embeddings(sentence_embeddings)
    if mat.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return mat.mean(axis=0).astype(np.float32, copy=False)


def score_sentences_query_centroid(
    sentence_embeddings: np.ndarray | list,
    *,
    case_type: str,
    sentences: list[str] | np.ndarray | None = None,
    query_weight: float = 0.7,
    centroid_weight: float = 0.3,
    anchor_weight: float = 0.35,
    anchor_group_weights: Mapping[str, float] | None = None,
    model_name: str = LEGAL_BERTIMBAU_MODEL,
) -> np.ndarray:
    """Score sentences using weighted query similarity + centroid + linguistic anchors.

    Combines three cosine-similarity-based signals per sentence — similarity
    to the case-type query embedding, similarity to the document centroid,
    and the linguistic anchor score — into one weighted score. The three
    weights are renormalized to sum to 1 before combination, so only their
    relative magnitudes matter.

    Args:
        sentence_embeddings: Per-sentence embeddings; coercible to a 2D
            float32 array via `_ensure_2d_embeddings`.
        case_type: Case type used to look up the query embedding (see
            `query_embedding_for_case_type`).
        sentences: Sentence strings aligned with `sentence_embeddings`;
            required when `anchor_weight > 0` (used for anchor matching).
        query_weight: Relative weight for query-similarity.
        centroid_weight: Relative weight for centroid-similarity.
        anchor_weight: Relative weight for the linguistic anchor score.
        anchor_group_weights: Optional overrides forwarded to
            `anchor_scores_for_sentences`.
        model_name: LegalBERTimbau model id used for the query embedding.

    Returns:
        float32 ndarray of shape (n_sentences,) with the combined,
        weight-normalized score per sentence. All zeros if
        `sentence_embeddings` is empty.

    Raises:
        ValueError: If any weight is negative, if all three weights are 0,
            if `anchor_weight > 0` but `sentences` is None, or if `sentences`
            and `sentence_embeddings` have mismatched lengths.
    """
    if query_weight < 0 or centroid_weight < 0 or anchor_weight < 0:
        raise ValueError(
            "query_weight, centroid_weight and anchor_weight must be >= 0."
        )
    if query_weight == 0 and centroid_weight == 0 and anchor_weight == 0:
        raise ValueError(
            "At least one of query_weight, centroid_weight or anchor_weight must be > 0."
        )

    mat = _ensure_2d_embeddings(sentence_embeddings)
    if mat.size == 0:
        return np.zeros((0,), dtype=np.float32)

    q = query_embedding_for_case_type(case_type, model_name=model_name)
    c = document_centroid_embedding(mat)

    q_scores = (
        _cosine_similarity_matrix_vector(mat, q)
        if q.size
        else np.zeros((mat.shape[0],), dtype=np.float32)
    )
    c_scores = (
        _cosine_similarity_matrix_vector(mat, c)
        if c.size
        else np.zeros((mat.shape[0],), dtype=np.float32)
    )
    if anchor_weight > 0:
        if sentences is None:
            raise ValueError("sentences must be provided when anchor_weight > 0.")
        a_scores = anchor_scores_for_sentences(
            sentences, anchor_group_weights=anchor_group_weights
        )
        if len(a_scores) != mat.shape[0]:
            raise ValueError(
                "Length mismatch between sentences and sentence_embeddings for anchor scoring."
            )
    else:
        a_scores = np.zeros((mat.shape[0],), dtype=np.float32)

    total = query_weight + centroid_weight + anchor_weight
    wq = query_weight / total
    wc = centroid_weight / total
    wa = anchor_weight / total
    return (wq * q_scores + wc * c_scores + wa * a_scores).astype(
        np.float32, copy=False
    )


def summarize_from_sentence_embeddings(
    *,
    sentences: list[str] | np.ndarray,
    sentence_embeddings: np.ndarray | list,
    case_type: str,
    top_k: int | None = None,
    query_weight: float = 0.7,
    centroid_weight: float = 0.3,
    anchor_weight: float = 0.35,
    anchor_group_weights: Mapping[str, float] | None = None,
    max_summary_tokens: int | None = None,
    min_anchor_ratio: float = 0.6,
    max_per_anchor_group: Mapping[str, int] | None = None,
    max_anchorless_sentences: int | None = None,
    model_name: str = LEGAL_BERTIMBAU_MODEL,
) -> SentenceScoreBatch:
    """Select top sentences with policy constraints: anchors first, then budget/quota-aware fallback.

    Scores every sentence via `score_sentences_query_centroid` and ranks them
    descending. Selection proceeds in two phases, both constrained by
    `max_summary_tokens` (if set) and by `max_per_anchor_group` caps:

    1. Anchor-first: walk the ranked sentences and greedily add only those
       that match at least one `ANCHOR_GROUPS` category, until `top_k`
       sentences are selected (if `top_k` is given) or the ranking is
       exhausted.
    2. Fallback: if `top_k` is None, or phase 1 selected fewer than `top_k`
       sentences, walk the full ranking again and greedily add remaining
       sentences (anchored or not), still respecting the anchorless quota —
       `max_anchorless_sentences` if given, else
       `int(top_k * (1 - min_anchor_ratio))` when `top_k` is set, else
       unlimited.

    The final selection is returned in original document order (not score
    order); `top_k` is a target cap, not a requirement — fewer sentences can
    be returned if the budget/quota constraints bind first.

    Args:
        sentences: Sentence strings, in document order.
        sentence_embeddings: Per-sentence embeddings aligned with
            `sentences`.
        case_type: Case type used for query-similarity scoring.
        top_k: Optional cap on the number of sentences to select. If None,
            selection is driven purely by budget/anchor constraints.
        query_weight: Forwarded to `score_sentences_query_centroid`.
        centroid_weight: Forwarded to `score_sentences_query_centroid`.
        anchor_weight: Forwarded to `score_sentences_query_centroid`.
        anchor_group_weights: Forwarded to `score_sentences_query_centroid`.
        max_summary_tokens: Optional cap on the summed
            `_estimate_sentence_tokens` of selected sentences.
        min_anchor_ratio: Used (only when `top_k` is set and
            `max_anchorless_sentences` is None) to derive the anchorless
            quota as `int(top_k * (1 - min_anchor_ratio))`.
        max_per_anchor_group: Optional per-anchor-group cap on how many
            selected sentences may belong to each group (a sentence counts
            against every group it matches).
        max_anchorless_sentences: Optional explicit cap on the number of
            selected sentences with no anchor match at all; overrides the
            `min_anchor_ratio`-derived quota when given.
        model_name: LegalBERTimbau model id used for query-similarity
            scoring.

    Returns:
        `SentenceScoreBatch` with the full per-sentence `scores` (document
        order), `top_indices` of the selected sentences (sorted ascending,
        i.e. document order), and `summary_text` (selected sentences joined
        with single spaces). If `sentences` is empty, returns an empty
        selection with the (empty) `scores` array.

    Raises:
        ValueError: If `top_k < 1`, `max_summary_tokens < 1`,
            `min_anchor_ratio` is outside [0, 1], `sentences` and
            `sentence_embeddings` have mismatched lengths, `max_per_anchor_group`
            references an unknown group or a negative cap, or
            `max_anchorless_sentences < 0`. Also propagates any `ValueError`
            raised by `score_sentences_query_centroid`.
    """
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be >= 1 when provided")
    if max_summary_tokens is not None and max_summary_tokens < 1:
        raise ValueError("max_summary_tokens must be >= 1 when provided.")
    if not (0.0 <= min_anchor_ratio <= 1.0):
        raise ValueError("min_anchor_ratio must be between 0 and 1.")

    sentence_list = [str(s) for s in list(sentences)]
    scores = score_sentences_query_centroid(
        sentence_embeddings,
        case_type=case_type,
        sentences=sentence_list,
        query_weight=query_weight,
        centroid_weight=centroid_weight,
        anchor_weight=anchor_weight,
        anchor_group_weights=anchor_group_weights,
        model_name=model_name,
    )

    if len(sentence_list) != len(scores):
        raise ValueError("Length mismatch between sentences and sentence_embeddings.")
    if len(sentence_list) == 0:
        return SentenceScoreBatch(scores=scores, top_indices=[], summary_text="")

    ranked_all = [int(i) for i in np.argsort(scores)[::-1]]
    anchor_cats_per_idx = [_sentence_anchor_categories(s) for s in sentence_list]
    token_estimates = [_estimate_sentence_tokens(s) for s in sentence_list]

    group_caps: dict[str, int | None] = {k: None for k in ANCHOR_GROUPS}
    if max_per_anchor_group is not None:
        for group, cap in max_per_anchor_group.items():
            if group not in ANCHOR_GROUPS:
                raise ValueError(
                    f"Unknown anchor group '{group}'. Expected one of: {sorted(ANCHOR_GROUPS)}"
                )
            if cap < 0:
                raise ValueError(f"max_per_anchor_group['{group}'] must be >= 0.")
            group_caps[group] = int(cap)

    if max_anchorless_sentences is None:
        if top_k is None:
            max_anchorless_allowed = 10**9
        else:
            max_anchorless_allowed = int(top_k * (1.0 - min_anchor_ratio))
    else:
        if max_anchorless_sentences < 0:
            raise ValueError("max_anchorless_sentences must be >= 0 when provided.")
        max_anchorless_allowed = int(max_anchorless_sentences)

    selected: list[int] = []
    selected_set: set[int] = set()
    selected_tokens = 0
    anchorless_count = 0
    group_counts = {k: 0 for k in ANCHOR_GROUPS}

    def _can_fit_budget(idx: int) -> bool:
        if max_summary_tokens is None:
            return True
        return (selected_tokens + token_estimates[idx]) <= max_summary_tokens

    def _violates_group_caps(idx: int) -> bool:
        cats = anchor_cats_per_idx[idx]
        for c in cats:
            cap = group_caps.get(c)
            if cap is not None and group_counts[c] >= cap:
                return True
        return False

    def _try_add(idx: int, *, allow_anchorless: bool) -> bool:
        nonlocal selected_tokens, anchorless_count
        if idx in selected_set:
            return False
        if top_k is not None and len(selected) >= top_k:
            return False
        if not _can_fit_budget(idx):
            return False
        cats = anchor_cats_per_idx[idx]
        is_anchorless = len(cats) == 0
        if is_anchorless:
            if not allow_anchorless:
                return False
            if anchorless_count >= max_anchorless_allowed:
                return False
        if _violates_group_caps(idx):
            return False

        selected.append(idx)
        selected_set.add(idx)
        selected_tokens += token_estimates[idx]
        if is_anchorless:
            anchorless_count += 1
        else:
            for c in cats:
                group_counts[c] += 1
        return True

    # Phase 1: anchored-only selection.
    anchored_ranked = [i for i in ranked_all if len(anchor_cats_per_idx[i]) > 0]
    for idx in anchored_ranked:
        _try_add(idx, allow_anchorless=False)
        if top_k is not None and len(selected) >= top_k:
            break

    # Phase 2: fallback selection (still respects anchorless quota/caps/budget).
    if top_k is None:
        target_not_reached = True
    else:
        target_not_reached = len(selected) < top_k
    if top_k is None or target_not_reached:
        for idx in ranked_all:
            _try_add(idx, allow_anchorless=True)
            if top_k is not None and len(selected) >= top_k:
                break

    top_indices = sorted(selected)
    summary_text = " ".join(sentence_list[i] for i in top_indices)
    return SentenceScoreBatch(
        scores=scores, top_indices=top_indices, summary_text=summary_text
    )


def summarize_for_prompt(
    text: str,
    *,
    case_type: str,
    top_k: int | None = None,
    sentence_min_chars: int = 8,
    batch_size: int = 16,
    max_length: int = 256,
    query_weight: float = 0.7,
    centroid_weight: float = 0.3,
    anchor_weight: float = 0.35,
    anchor_group_weights: Mapping[str, float] | None = None,
    max_summary_tokens: int | None = None,
    min_anchor_ratio: float = 0.6,
    max_per_anchor_group: Mapping[str, int] | None = None,
    max_anchorless_sentences: int | None = None,
    model_name: str = LEGAL_BERTIMBAU_MODEL,
) -> str:
    """End-to-end prompt summarization using query+centroid+anchor ranking.

    Splits and encodes `text` (`sentence_embeddings_for_document`), then
    scores and selects sentences (`summarize_from_sentence_embeddings`) with
    the same policy constraints (budget, anchor quotas/caps).

    Args:
        text: Full document text to summarize.
        case_type: Case type used for query-similarity scoring.
        top_k: Optional cap on the number of sentences to select.
        sentence_min_chars: Forwarded to `split_sentences_pt_legal`.
        batch_size: Forwarded to `encode_sentences_legalbert`.
        max_length: Forwarded to `encode_sentences_legalbert`.
        query_weight: Forwarded to `summarize_from_sentence_embeddings`.
        centroid_weight: Forwarded to `summarize_from_sentence_embeddings`.
        anchor_weight: Forwarded to `summarize_from_sentence_embeddings`.
        anchor_group_weights: Forwarded to `summarize_from_sentence_embeddings`.
        max_summary_tokens: Forwarded to `summarize_from_sentence_embeddings`.
        min_anchor_ratio: Forwarded to `summarize_from_sentence_embeddings`.
        max_per_anchor_group: Forwarded to `summarize_from_sentence_embeddings`.
        max_anchorless_sentences: Forwarded to `summarize_from_sentence_embeddings`.
        model_name: LegalBERTimbau model id used for encoding and scoring.

    Returns:
        The selected sentences joined into a single summary string (empty
        string if `text` yields no sentences).
    """
    batch = sentence_embeddings_for_document(
        text=text,
        model_name=model_name,
        sentence_min_chars=sentence_min_chars,
        batch_size=batch_size,
        max_length=max_length,
    )
    scored = summarize_from_sentence_embeddings(
        sentences=batch.sentences,
        sentence_embeddings=batch.embeddings,
        case_type=case_type,
        top_k=top_k,
        query_weight=query_weight,
        centroid_weight=centroid_weight,
        anchor_weight=anchor_weight,
        anchor_group_weights=anchor_group_weights,
        max_summary_tokens=max_summary_tokens,
        min_anchor_ratio=min_anchor_ratio,
        max_per_anchor_group=max_per_anchor_group,
        max_anchorless_sentences=max_anchorless_sentences,
        model_name=model_name,
    )
    return scored.summary_text


def sentence_embeddings_for_document(
    text: str,
    *,
    model_name: str = LEGAL_BERTIMBAU_MODEL,
    sentence_min_chars: int = 8,
    batch_size: int = 16,
    max_length: int = 256,
) -> SentenceEmbeddingBatch:
    """Full Step 1+2 pipeline for one document: split sentences, then encode each sentence.

    Args:
        text: Full document text.
        model_name: LegalBERTimbau model id used for encoding.
        sentence_min_chars: Forwarded to `split_sentences_pt_legal`.
        batch_size: Forwarded to `encode_sentences_legalbert`.
        max_length: Forwarded to `encode_sentences_legalbert`.

    Returns:
        `SentenceEmbeddingBatch` with the split sentences and their
        embeddings, aligned row-for-row.
    """
    sentences = split_sentences_pt_legal(text, min_chars=sentence_min_chars)
    embeddings = encode_sentences_legalbert(
        sentences,
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
    )
    return SentenceEmbeddingBatch(sentences=sentences, embeddings=embeddings)


def _serialize_embeddings(arr: np.ndarray) -> list[list[float]]:
    """Convert ndarray (n_sentences, hidden_size) into parquet-friendly nested lists."""
    if arr.size == 0:
        return []
    return arr.astype(float).tolist()


def build_sentence_embeddings_frame(
    df: pd.DataFrame,
    *,
    text_column: str,
    id_column: str,
    case_type: str | None = None,
    batch_size: int = 16,
    max_length: int = 256,
    sentence_min_chars: int = 8,
    model_name: str = LEGAL_BERTIMBAU_MODEL,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int | None = None,
    resume_from_checkpoint: bool = False,
) -> pd.DataFrame:
    """Run step 1+2 for each document and return a sentence-embedding dataframe.

    For every row in `df`, splits and encodes the text
    (`sentence_embeddings_for_document`). If `case_type` is given, also
    scores and selects sentences (`summarize_from_sentence_embeddings`, with
    `top_k=None` and default query/centroid/anchor weights) and records the
    selected subset alongside the full sentence list; if `case_type` is
    falsy, no selection is performed and all sentences are recorded as
    "selected" with placeholder 0.0 scores. Supports periodic checkpointing
    to parquet and resuming a partially completed run.

    Args:
        df: DataFrame with one row per document, needing `text_column` and
            `id_column`.
        text_column: Column holding the full document text to encode.
        id_column: Column holding the document identifier.
        case_type: Optional case type; when set, enables sentence
            scoring/selection metadata via `summarize_from_sentence_embeddings`.
        batch_size: Forwarded to `sentence_embeddings_for_document`.
        max_length: Forwarded to `sentence_embeddings_for_document`.
        sentence_min_chars: Forwarded to `sentence_embeddings_for_document`.
        model_name: LegalBERTimbau model id used for encoding/scoring.
        checkpoint_path: Optional parquet path to write partial/final
            results to. Also used, together with `resume_from_checkpoint`,
            to skip already-processed document ids on a re-run.
        checkpoint_every: If set (with `checkpoint_path`), writes the
            accumulated rows to `checkpoint_path` every `checkpoint_every`
            newly processed documents, and once more after the loop
            completes.
        resume_from_checkpoint: If True and `checkpoint_path` exists, loads
            its rows first and skips any document id already present there.

    Returns:
        DataFrame with one row per processed document, including
        `id_column`, `sentence_count`, `sentences`, `sentence_ranks`
        (0-based positional index per sentence, not a score-based rank),
        `sentence_scores`, `selected_sentence_count`,
        `selected_sentence_indices`, `selected_sentence_ranks`,
        `selected_sentence_scores`, `selected_sentences`, `summary_text`
        (selected sentences joined with a single space — the column
        `predict.py --smart_truncated_path` reads), and `sentence_embeddings`
        (nested-list serialized via `_serialize_embeddings`); includes a
        `case_type` column only when `case_type` was provided.

    Raises:
        ValueError: If `df` is missing `text_column` or `id_column`; if
            `checkpoint_every` is provided and `< 1`; or if an existing
            checkpoint file at `checkpoint_path` is missing `id_column` when
            resuming.
    """
    if text_column not in df.columns:
        raise ValueError(f"text_column '{text_column}' not found in input dataframe.")
    if id_column not in df.columns:
        raise ValueError(f"id_column '{id_column}' not found in input dataframe.")
    if checkpoint_every is not None and checkpoint_every < 1:
        raise ValueError("checkpoint_every must be >= 1 when provided.")

    rows: list[dict] = []
    processed_ids: set[str] = set()
    ckpt_path = Path(checkpoint_path) if checkpoint_path is not None else None
    since_last_checkpoint = 0

    if resume_from_checkpoint and ckpt_path is not None and ckpt_path.exists():
        prior = pd.read_parquet(ckpt_path)
        if id_column not in prior.columns:
            raise ValueError(
                f"Checkpoint file missing id column '{id_column}': {ckpt_path}"
            )
        rows = prior.to_dict(orient="records")
        processed_ids = {str(v) for v in prior[id_column].astype(str).tolist()}
        print(
            f"Resuming from checkpoint {ckpt_path} ({len(processed_ids)} docs already done)."
        )

    iterator = tqdm(
        df.itertuples(index=False), total=len(df), desc="Sentence embeddings"
    )

    for row in iterator:
        row_dict = row._asdict()
        doc_id = str(row_dict[id_column])
        if doc_id in processed_ids:
            continue
        text = str(row_dict.get(text_column, "") or "")

        batch = sentence_embeddings_for_document(
            text=text,
            model_name=model_name,
            sentence_min_chars=sentence_min_chars,
            batch_size=batch_size,
            max_length=max_length,
        )
        if case_type:
            scored = summarize_from_sentence_embeddings(
                sentences=batch.sentences,
                sentence_embeddings=batch.embeddings,
                case_type=case_type,
                top_k=None,
                query_weight=0.7,
                centroid_weight=0.3,
                anchor_weight=0.35,
                model_name=model_name,
            )
            selected_indices = [int(i) for i in scored.top_indices]
            selected_sentences = [batch.sentences[i] for i in selected_indices]
            selected_scores = [float(scored.scores[i]) for i in selected_indices]
            selected_ranks = selected_indices.copy()
            sentence_scores = [float(v) for v in scored.scores.tolist()]
        else:
            selected_indices = list(range(len(batch.sentences)))
            selected_sentences = list(batch.sentences)
            selected_scores = [0.0 for _ in selected_sentences]
            selected_ranks = selected_indices.copy()
            sentence_scores = [0.0 for _ in batch.sentences]

        rows.append(
            {
                id_column: doc_id,
                **({"case_type": case_type} if case_type else {}),
                "sentence_count": len(batch.sentences),
                "sentences": batch.sentences,
                "sentence_ranks": list(range(len(batch.sentences))),
                "sentence_scores": sentence_scores,
                "selected_sentence_count": len(selected_sentences),
                "selected_sentence_indices": selected_indices,
                "selected_sentence_ranks": selected_ranks,
                "selected_sentence_scores": selected_scores,
                "selected_sentences": selected_sentences,
                "summary_text": " ".join(selected_sentences),
                "sentence_embeddings": _serialize_embeddings(batch.embeddings),
            }
        )
        processed_ids.add(doc_id)
        since_last_checkpoint += 1

        if (
            ckpt_path is not None
            and checkpoint_every is not None
            and since_last_checkpoint >= checkpoint_every
        ):
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(ckpt_path, index=False)
            since_last_checkpoint = 0

    if ckpt_path is not None and checkpoint_every is not None:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(ckpt_path, index=False)
    return pd.DataFrame(rows)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the sentence-embedding CLI pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate sentence-level embeddings (steps 1+2 of smart truncation) "
            "with frozen LegalBERTimbau and save to parquet."
        )
    )
    parser.add_argument(
        "--input_path", help="Input parquet/csv path (single-file mode)."
    )
    parser.add_argument("--output_path", help="Output parquet path (single-file mode).")
    parser.add_argument(
        "--run_train_batch",
        action="store_true",
        help=(
            "Run the 4 train/test split files in batch mode using --input_root/--output_root. "
            "Useful for Google Drive mounted paths in Colab."
        ),
    )
    parser.add_argument(
        "--input_root",
        default=".",
        help=(
            "Root directory containing split CSV files "
            "(e.g. /content/drive/MyDrive/your_folder). Used by --run_train_batch."
        ),
    )
    parser.add_argument(
        "--output_root",
        default="sentence_embeddings",
        help=(
            "Output directory for generated parquet files "
            "(e.g. /content/drive/MyDrive/your_folder/sentence_embeddings). "
            "Used by --run_train_batch."
        ),
    )
    parser.add_argument(
        "--text_column",
        default="texto_integral_sem_decisao",
        help="Column containing full legal text.",
    )
    parser.add_argument(
        "--id_column",
        default="n_processo",
        help="Document id column.",
    )
    parser.add_argument(
        "--case_type",
        default=None,
        help="Optional case type label used to compute selection metadata.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Sentence encoder batch size."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Max tokens per sentence for encoder truncation.",
    )
    parser.add_argument(
        "--sentence_min_chars",
        type=int,
        default=8,
        help="Drop very short sentence fragments below this length.",
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=None,
        help="Optional cap for quick smoke runs (first N documents).",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=None,
        help="Save partial parquet every N processed documents (e.g., 1 for each doc).",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="Resume from existing output parquet when using checkpoint mode.",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        # Notebook/Colab kernels often inject "-f <kernel-connection.json>".
        filtered: list[str] = []
        i = 0
        while i < len(unknown):
            if unknown[i] == "-f":
                i += 2
                continue
            filtered.append(unknown[i])
            i += 1
        if filtered:
            parser.error(f"unrecognized arguments: {' '.join(filtered)}")
    return args


def _load_input_frame(path: Path) -> pd.DataFrame:
    """Load an input document table from a .parquet or .csv file."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format '{suffix}'. Use .parquet or .csv")


def run_sentence_embedding_pipeline(
    *,
    input_path: str | Path,
    output_path: str | Path,
    text_column: str = "texto_integral_sem_decisao",
    id_column: str = "n_processo",
    case_type: str | None = None,
    batch_size: int = 16,
    max_length: int = 256,
    sentence_min_chars: int = 8,
    max_docs: int | None = None,
    checkpoint_every: int | None = None,
    resume_from_checkpoint: bool = False,
) -> pd.DataFrame:
    """Notebook-friendly one-call pipeline for step 1+2 sentence embeddings.

    This is intended for Colab/Jupyter use in a single cell. Loads
    `input_path` (optionally capped to the first `max_docs` rows), runs
    `build_sentence_embeddings_frame`, writes the result to `output_path`
    (parquet), and returns the resulting dataframe. When `checkpoint_every`
    is set, `output_path` doubles as the checkpoint file (partial progress
    is saved there during the run and can be resumed via
    `resume_from_checkpoint`).

    Args:
        input_path: Input .parquet or .csv file with `text_column`/`id_column`.
        output_path: Output parquet path; also used as the checkpoint path
            when `checkpoint_every` is set.
        text_column: Forwarded to `build_sentence_embeddings_frame`.
        id_column: Forwarded to `build_sentence_embeddings_frame`.
        case_type: Forwarded to `build_sentence_embeddings_frame`.
        batch_size: Forwarded to `build_sentence_embeddings_frame`.
        max_length: Forwarded to `build_sentence_embeddings_frame`.
        sentence_min_chars: Forwarded to `build_sentence_embeddings_frame`.
        max_docs: Optional cap on the number of input rows processed
            (first N).
        checkpoint_every: Forwarded to `build_sentence_embeddings_frame`;
            also determines whether `output_path` is used as the checkpoint
            path.
        resume_from_checkpoint: Forwarded to `build_sentence_embeddings_frame`.

    Returns:
        The generated sentence-embeddings DataFrame (also written to
        `output_path`).

    Raises:
        FileNotFoundError: If `input_path` does not exist.
        ValueError: If `max_docs` is provided and not positive, or if
            `build_sentence_embeddings_frame` raises (missing columns, etc.).
    """
    in_path = Path(input_path)
    out_path = Path(output_path)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    df = _load_input_frame(in_path)
    if max_docs is not None:
        if max_docs <= 0:
            raise ValueError("max_docs must be positive when provided.")
        df = df.head(max_docs).copy()

    out_df = build_sentence_embeddings_frame(
        df,
        text_column=text_column,
        id_column=id_column,
        case_type=case_type,
        batch_size=batch_size,
        max_length=max_length,
        sentence_min_chars=sentence_min_chars,
        checkpoint_path=out_path if checkpoint_every is not None else None,
        checkpoint_every=checkpoint_every,
        resume_from_checkpoint=resume_from_checkpoint,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(f"Saved sentence embeddings -> {out_path}")
    print(f"Rows: {len(out_df)}")
    print(f"Columns: {list(out_df.columns)}")
    return out_df


def run_default_split_batch(
    *,
    input_root: str | Path,
    output_root: str | Path,
    text_column: str = "texto_integral_sem_decisao",
    id_column: str = "n_processo",
    case_type: str | None = None,
    batch_size: int = 16,
    max_length: int = 256,
    sentence_min_chars: int = 8,
    max_docs: int | None = None,
    checkpoint_every: int | None = None,
    resume_from_checkpoint: bool = False,
) -> dict[str, Path]:
    """Batch runner for the four canonical split files, for local or Google Drive paths.

    Expected split files (under `input_root`):
      - dv_gold_test.csv
      - boc_gold_test.csv
      - dv_train_before_cutoff.csv
      - boc_train_before_cutoff.csv

    Runs `run_sentence_embedding_pipeline` once per split file found under
    `input_root`, writing each split's output parquet under `output_root`
    with the same base filename. If `case_type` is not given, it is
    inferred per split from the filename prefix (the text before the first
    underscore, i.e. "dv" or "boc").

    Args:
        input_root: Directory containing the four `<split_name>.csv` files.
        output_root: Directory to write the four `<split_name>_enriched.parquet`
            files to.
        text_column: Forwarded to `run_sentence_embedding_pipeline`.
        id_column: Forwarded to `run_sentence_embedding_pipeline`.
        case_type: Optional case type applied to every split; if falsy, each
            split's case type is inferred from its filename prefix instead.
        batch_size: Forwarded to `run_sentence_embedding_pipeline`.
        max_length: Forwarded to `run_sentence_embedding_pipeline`.
        sentence_min_chars: Forwarded to `run_sentence_embedding_pipeline`.
        max_docs: Forwarded to `run_sentence_embedding_pipeline`.
        checkpoint_every: Forwarded to `run_sentence_embedding_pipeline`.
        resume_from_checkpoint: Forwarded to `run_sentence_embedding_pipeline`.

    Returns:
        Dict mapping each split name to its written output parquet path.

    Raises:
        FileNotFoundError: If any of the four expected `<split_name>.csv`
            files is missing under `input_root`.
    """
    in_root = Path(input_root)
    out_root = Path(output_root)
    split_names = (
        "dv_gold_test",
        "boc_gold_test",
        "dv_train_before_cutoff",
        "boc_train_before_cutoff",
    )
    outputs: dict[str, Path] = {}

    for split_name in split_names:
        in_path = in_root / f"{split_name}.csv"
        if not in_path.exists():
            raise FileNotFoundError(f"Batch input not found: {in_path}")
        out_path = out_root / f"{split_name}_enriched.parquet"
        run_sentence_embedding_pipeline(
            input_path=in_path,
            output_path=out_path,
            text_column=text_column,
            id_column=id_column,
            case_type=case_type or split_name.split("_", 1)[0],
            batch_size=batch_size,
            max_length=max_length,
            sentence_min_chars=sentence_min_chars,
            max_docs=max_docs,
            checkpoint_every=checkpoint_every,
            resume_from_checkpoint=resume_from_checkpoint,
        )
        outputs[split_name] = out_path

    return outputs


def main() -> None:
    """CLI entry point: run either the batch pipeline or a single-file pipeline.

    If `--run_train_batch` is set, runs `run_default_split_batch` over the
    four canonical split files under `--input_root`/`--output_root`;
    otherwise runs `run_sentence_embedding_pipeline` once for
    `--input_path`/`--output_path`.

    Raises:
        ValueError: If not in batch mode and either `--input_path` or
            `--output_path` is missing.
    """
    args = _parse_args()
    if args.run_train_batch:
        outputs = run_default_split_batch(
            input_root=args.input_root,
            output_root=args.output_root,
            text_column=args.text_column,
            id_column=args.id_column,
            case_type=args.case_type,
            batch_size=args.batch_size,
            max_length=args.max_length,
            sentence_min_chars=args.sentence_min_chars,
            max_docs=args.max_docs,
            checkpoint_every=args.checkpoint_every,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
        print("\nBatch completed:")
        for split_name, out_path in outputs.items():
            print(f"  {split_name} -> {out_path}")
        return

    if not args.input_path or not args.output_path:
        raise ValueError(
            "Single-file mode requires --input_path and --output_path. "
            "Or use --run_train_batch with --input_root/--output_root."
        )

    run_sentence_embedding_pipeline(
        input_path=Path(args.input_path),
        output_path=Path(args.output_path),
        text_column=args.text_column,
        id_column=args.id_column,
        case_type=args.case_type,
        batch_size=args.batch_size,
        max_length=args.max_length,
        sentence_min_chars=args.sentence_min_chars,
        max_docs=args.max_docs,
        checkpoint_every=args.checkpoint_every,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )


if __name__ == "__main__":
    # Notebook-friendly hardcoded block (set these paths in one cell and run).
    RUN_HARDCODED_NOTEBOOK_BATCH = False

    if RUN_HARDCODED_NOTEBOOK_BATCH:
        # Example Google Drive paths (Colab):
        # INPUT_ROOT = Path("/content/drive/MyDrive/your_folder/splits")
        # OUTPUT_ROOT = Path("/content/drive/MyDrive/your_folder/sentence_embeddings")
        INPUT_ROOT = Path(".")
        OUTPUT_ROOT = Path("sentence_embeddings")

        outputs = run_default_split_batch(
            input_root=INPUT_ROOT,
            output_root=OUTPUT_ROOT,
            text_column="texto_integral_sem_decisao",
            id_column="n_processo",
            batch_size=64,
            max_length=512,
            sentence_min_chars=8,
            max_docs=None,
            checkpoint_every=1,
            resume_from_checkpoint=True,
        )
        print("\nHardcoded batch completed:")
        for split_name, out_path in outputs.items():
            print(f"  {split_name} -> {out_path}")
    else:
        main()
