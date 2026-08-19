"""
SHAP-based Explainability Module
=================================

Generates sentence-level SHAP explanations for Legal Judgment Prediction models.

How SHAP works here
-------------------
SHAP's Text masker splits the raw input into sentences (regex-based sentence
boundary detection, matching the splitter used in the smart summarization
pipeline, ``summarizer.split_sentences_pt_legal``), then creates many
variations of the full text with different sentence subsets masked (replaced
with a space). Each variation is passed to ``predict_fn`` — the same callable
used for classification — so the attributions are always consistent with the
actual model predictions. Sentence-level masking was chosen over word-level
after word-level attributions proved too diluted to be informative (each
individual word contributes only a tiny, noisy sliver of signal); sentences
are the natural unit of legal reasoning and give SHAP coarser, more legible
units to attribute mass to.

The SHAP efficiency axiom guarantees:
    base_value + sum(shap_values[token, class]) == predict_fn([original_text])[class]

This masking operates at the *input surface* level.  The model's internal
tokenizer (BERT WordPiece, LLM BPE, …) is unaffected — it receives whatever
masked string SHAP produces and tokenises it internally as usual.

Supports both:
  - Legal BERTimbau + sklearn downstream classifiers
  - LLMs via Ollama (adapt with make_llm_predict_fn)

Output artefacts (per case)
----------------------------
  {case_id}.html                 -- self-contained judge annotation card (UTF-8)
  shap_top_terms_{case_id}.png   -- top-10 positive/negative terms bar chart (300 DPI)
  {case_id}.json                 -- raw SHAP data for downstream tooling
  index.html                     -- session listing for the judge annotation round

Save path convention (mirrors eda_viz/):
  shap_explanations/
    domestic_violence/{model_name}/
    breach_of_contract/{model_name}/

Label encoding (must match classification.py):
  Binary  (DV):  0 = decisao_mantida,      1 = decisao_alterada
  Ternary (BoC): 0 = decisao_desfavoravel, 1 = decisao_parcial, 2 = decisao_favoravel
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

# TODO: Understand and refactor code for better readability.Keep some comments still so I can come back and understand later what I did

# ---------------------------------------------------------------------------
# Label constants — mirror classification.py
# ---------------------------------------------------------------------------

LABEL_NAMES_DV: dict[int, str] = {
    0: "decisao_mantida",
    1: "decisao_alterada",
}

LABEL_NAMES_BOC: dict[int, str] = {
    0: "decisao_desfavoravel",
    1: "decisao_parcial",
    2: "decisao_favoravel",
}

# Human-readable case-type display names (Portuguese, for HTML output)
CASE_TYPE_DISPLAY: dict[str, str] = {
    "dv": "Violência Doméstica",
    "boc": "Incumprimento de Contrato",
}

# Subdirectory names inside shap_explanations/ (mirrors eda_viz/ convention)
CASE_TYPE_OUTPUT_DIRS: dict[str, str] = {
    "dv": "domestic_violence",
    "boc": "breach_of_contract",
}

# SHAP sentence-level masker pattern.
# Splits after sentence-ending punctuation (.!?;:) followed by whitespace —
# the same boundary rule used by summarizer.split_sentences_pt_legal, so
# masking units here match the sentence units used throughout the rest of
# the pipeline (smart truncation, few-shot retrieval).
# mask_token is SHAP's Text masker default, the literal string "...", not a
# space. Masked sentences are replaced with "...", and when every sentence in
# a document is masked at once, shap.maskers.Text's collapse_mask_token
# behavior collapses the whole run down to a single "..." rather than
# repeating it once per sentence. So the "fully masked" document SHAP feeds
# the model to compute base_values is the literal 3-character string "...",
# not an empty string. Verified empirically: predict_bert_proba(["..."], ...)
# reproduces the stored base_values to 4 decimal places.
SENTENCE_TOKENIZER_PATTERN: str = r"(?<=[.!?;:])\s+"

# Default budget for SHAP Partition explainer.
# Each masked variation = one predict_fn call.  500 covers ~250-word documents
# with good coverage.  Use 200 for large batch exports to keep wall time low.
DEFAULT_MAX_EVALS: int = 500

# Colour palette (matches classification.py forest-plot colours)
SHAP_POSITIVE_COLOUR: str = "#d73027"   # red  — positive contribution to class
SHAP_NEGATIVE_COLOUR: str = "#4575b4"   # blue — negative contribution to class

# Portuguese stop words — filtered out of bar charts only (NOT the HTML, which
# needs all tokens intact for the judge annotation Streamlit app).
# Covers high-frequency function words that carry no legal signal: articles,
# prepositions, conjunctions, pronouns, auxiliary verbs, and punctuation noise.
PT_STOP_WORDS: frozenset[str] = frozenset({
    # articles
    "a", "o", "as", "os", "um", "uma", "uns", "umas",
    # prepositions / contractions
    "de", "da", "do", "das", "dos", "em", "na", "no", "nas", "nos",
    "por", "pelo", "pela", "pelos", "pelas", "com", "para", "per",
    "ante", "até", "após", "desde", "entre", "sobre", "sob", "sem",
    "num", "numa", "nuns", "numas", "dum", "duma", "duns", "dumas",
    "ao", "à", "aos", "às",
    # conjunctions
    "e", "ou", "mas", "porém", "contudo", "todavia", "entretanto",
    "porque", "pois", "logo", "portanto", "que", "se", "nem",
    "quando", "onde", "como", "embora", "enquanto", "caso",
    # pronouns
    "eu", "tu", "ele", "ela", "nós", "vós", "eles", "elas",
    "me", "te", "se", "lhe", "lhes", "nos", "vos",
    "meu", "minha", "meus", "minhas", "teu", "tua", "teus", "tuas",
    "seu", "sua", "seus", "suas", "nosso", "nossa", "nossos", "nossas",
    "este", "esta", "estes", "estas", "esse", "essa", "esses", "essas",
    "aquele", "aquela", "aqueles", "aquelas", "isto", "isso", "aquilo",
    "quem", "qual", "quais", "cujo", "cuja", "cujos", "cujas",
    # auxiliary / high-frequency verbs
    "é", "ser", "estar", "foi", "são", "era", "eram", "seja", "sejam",
    "sendo", "sido", "tem", "ter", "teve", "têm", "tinha", "tinham",
    "tendo", "tido", "há", "haver", "havia", "houve", "haja",
    "pode", "podem", "podia", "podiam", "pôde", "poderem",
    "deve", "devem", "devia", "deviam",
    "vai", "vão", "ir", "foi",
    # adverbs / particles
    "não", "mais", "muito", "também", "já", "ainda", "só", "bem",
    "sempre", "nunca", "jamais", "aqui", "ali", "lá", "cá",
    "assim", "então", "depois", "antes", "agora", "hoje", "logo",
    "apenas", "mesmo", "tão", "tudo", "nada", "algo", "alguém",
    # punctuation noise that slips through the word tokenizer
    "", " ", "-", "–", "—", "/", "\\", "(", ")", "[", "]",
    ".", ",", ";", ":", "!", "?", "\"", "'", "``", "''",
})


# ---------------------------------------------------------------------------
# Core data contract
# ---------------------------------------------------------------------------

@dataclass
class ShapExplanation:
    """Container for a single SHAP explanation of a court decision.

    Parameters
    ----------
    case_id : str
        Unique process identifier (``n_processo``).
    case_type : str
        ``'dv'`` (Domestic Violence) or ``'boc'`` (Breach of Contract).
    model_name : str
        Human-readable model identifier, e.g. ``'bertimbau_logreg'`` or
        ``'deepseek_ollama'``.  Used as the output subdirectory name.
    text : str
        Raw input text passed to the model (``texto_integral_sem_decisao``).
    tokens : list[str]
        Surface word tokens (whitespace stripped) as produced by the SHAP
        masker.  These are the units to which SHAP values are attached.
    shap_values : np.ndarray
        Shape ``(n_tokens, n_classes)``.  Per-token, per-class SHAP values.
        Column order matches ``output_names``.
    base_values : np.ndarray
        Shape ``(n_classes,)``.  Model's output on the fully masked document,
        the literal string ``"..."`` (see SENTENCE_TOKENIZER_PATTERN comment
        above), not an empty string — the SHAP baseline.
    output_names : list[str]
        Class label strings in the same order as columns in ``shap_values``.
    predicted_class : int
        ``argmax`` of ``predict_proba``.
    predicted_label : str
        Human-readable label for ``predicted_class``.
    predict_proba : np.ndarray
        Shape ``(n_classes,)``.  Model probability estimates on the original
        (unmasked) text.  Consistent with classification pipeline by design.
    true_label : int or None
        Integer ground-truth label (if available).
    true_label_str : str or None
        Human-readable string for ``true_label``.
    metadata : dict
        Optional extra columns from the dataset row:
        tribunal, data_acordao, juiz_relator, url, …
        Displayed in the judge annotation HTML card header.
    """

    case_id: str
    case_type: str
    model_name: str
    text: str
    tokens: list[str]
    shap_values: np.ndarray        # (n_tokens, n_classes)
    base_values: np.ndarray        # (n_classes,)
    output_names: list[str]
    predicted_class: int
    predicted_label: str
    predict_proba: np.ndarray      # (n_classes,)
    true_label: int | None = None
    true_label_str: str | None = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core SHAP computation
# ---------------------------------------------------------------------------

def compute_shap_explanation(
    text: str,
    predict_fn: Callable[[list[str]], np.ndarray],
    case_id: str,
    case_type: str,
    model_name: str,
    output_names: list[str],
    true_label: int | None = None,
    metadata: dict | None = None,
    max_evals: int = DEFAULT_MAX_EVALS,
    seed: int = 42,
) -> ShapExplanation:
    """Compute sentence-level SHAP attributions for a single legal document.

    Parameters
    ----------
    text : str
        Raw input text (``texto_integral_sem_decisao``).
    predict_fn : Callable[[list[str]], np.ndarray]
        Function that accepts a list of strings and returns a 2-D array of
        shape ``(n_texts, n_classes)`` with probability estimates.
        Must be the *same* callable used for classification so attributions
        are consistent with the reported prediction.
    case_id : str
        Unique process identifier (``n_processo``).
    case_type : str
        ``'dv'`` or ``'boc'``.
    model_name : str
        Human-readable model identifier used as the output subdirectory.
    output_names : list[str]
        Ordered class label strings, e.g.
        ``['decisao_mantida', 'decisao_alterada']``.
    true_label : int or None
        Ground-truth integer label, if available.
    metadata : dict or None
        Extra case columns to embed in outputs (tribunal, url, …).
    max_evals : int
        Budget: maximum number of ``predict_fn`` calls SHAP may make.
        Each call processes one masked variant of the full text.
        500 gives good coverage for ~250-word documents; use 200 for
        large batch jobs.
    seed : int
        Random seed for SHAP's internal sampling (reproducibility).

    Returns
    -------
    ShapExplanation
        Populated container with tokens, SHAP values, base values, and
        prediction metadata.

    Notes
    -----
    SHAP's efficiency axiom guarantees consistency with the classifier:
        base_values[c] + sum(shap_values[:, c]) ≈ predict_fn([text])[0][c]

    The ``Text`` masker replaces masked sentences with a single space so that
    adjacent sentences never concatenate inside the model's internal tokenizer.
    """
    # Step 1 — Build the Text masker.
    # shap.maskers.Text splits the input string into sentences using the
    # sentence-boundary regex pattern and replaces masked sentences with a
    # space token. This operates entirely on the raw string; the model's
    # internal tokeniser is unaffected and re-tokenises whatever masked
    # string it receives.
    masker = shap.maskers.Text(tokenizer=SENTENCE_TOKENIZER_PATTERN)

    # Step 2 — Build the Explainer.
    # Because we pass a Text masker, SHAP auto-selects the Partition explainer
    # (Owen values over a recursive token hierarchy).  It is model-agnostic:
    # works identically for BERT+sklearn pipelines and for LLM predict wrappers.
    explainer = shap.Explainer(
        predict_fn,
        masker=masker,
        output_names=output_names,
        seed=seed,
    )

    # Step 3 — Run SHAP.
    # We wrap `text` in a list because predict_fn expects a batch.
    # max_evals caps the number of predict_fn calls (each call = one masked
    # variant of the full text).  silent=True suppresses the tqdm bar here
    # (the batch pipeline in run_shap_explanation_pipeline adds its own bar).
    shap_vals = explainer([text], max_evals=max_evals, silent=True)

    # Step 4 — Extract surface tokens.
    # shap_vals.data is a list (one entry per input text); index 0 = our doc.
    # SHAP returns tokens with trailing whitespace — we strip it so downstream
    # aggregation (e.g. top-terms deduplication) works cleanly.
    raw_tokens = shap_vals.data[0]
    tokens = [str(t).strip() for t in raw_tokens]

    # Step 5 — Extract SHAP value array.
    # Raw shape from SHAP: (1, n_tokens, n_classes).
    # We drop the batch dimension -> (n_tokens, n_classes).
    # Special case: for binary tasks SHAP may return (1, n_tokens) — a single
    # column for the positive class.  We expand it to (n_tokens, 2) so every
    # downstream function can assume a consistent 2-D array regardless of
    # the number of classes.
    sv = np.array(shap_vals.values[0])
    if sv.ndim == 1:
        # Binary scalar output: SHAP column = P(class 1).
        # Mirror it so column 0 = contribution to class 0, column 1 to class 1.
        sv = np.stack([-sv, sv], axis=1)

    # Step 6 — Extract base values.
    # base_values[c] is the model's expected output for class c on a fully
    # masked (all words removed) input — the SHAP baseline.
    # Shape from SHAP: (1, n_classes) or scalar for binary -> normalise to (n_classes,).
    bv = np.array(shap_vals.base_values[0])
    if bv.ndim == 0:
        bv = np.array([float(bv), float(bv)])

    # Step 7 — Get the actual model prediction on the unmasked text.
    # This is a single forward pass on the original text — identical to what
    # the classification pipeline does.  The SHAP efficiency axiom guarantees
    # that sum(sv[:, c]) + bv[c] ≈ predict_proba[c] for every class c.
    predict_proba = predict_fn([text])[0]
    predicted_class = int(np.argmax(predict_proba))
    predicted_label = output_names[predicted_class]

    # Step 8 — Resolve true-label string (for display in HTML card).
    true_label_str: str | None = None
    if true_label is not None:
        true_label_str = (
            output_names[true_label]
            if 0 <= true_label < len(output_names)
            else str(true_label)
        )

    return ShapExplanation(
        case_id=case_id,
        case_type=case_type,
        model_name=model_name,
        text=text,
        tokens=tokens,
        shap_values=sv,
        base_values=bv,
        output_names=output_names,
        predicted_class=predicted_class,
        predicted_label=predicted_label,
        predict_proba=np.array(predict_proba),
        true_label=true_label,
        true_label_str=true_label_str,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Top-terms summary bar chart
# ---------------------------------------------------------------------------

def plot_shap_top_terms(
    explanation: ShapExplanation,
    class_idx: int | None = None,
    top_n: int = 10,
    output_dir: Path | None = None,
    show: bool = True,
) -> plt.Figure | None:
    """Bar chart of the top-N positive and top-N negative SHAP sentences.

    Positive-contributing sentences extend to the right in red; negative-
    contributing sentences extend to the left in blue.  The centre axis (x = 0)
    is the neutral point.  Repeated occurrences of the same sentence are summed
    before ranking so the chart reflects net contribution across the document.

    Parameters
    ----------
    explanation : ShapExplanation
        Output of ``compute_shap_explanation``.
    class_idx : int or None
        Index of the class to visualise.  Defaults to ``predicted_class`` so
        the chart always explains the model's actual decision.
    top_n : int
        Number of top positive and top negative terms to show (each side).
        Total bars = up to 2 * top_n.
    output_dir : Path or None
        Directory to save the PNG (300 DPI).  File is named
        ``shap_top_terms_{case_id}.png``.  Nothing is saved if None.
    show : bool
        Whether to call ``plt.show()``.  Set False for batch export.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if class_idx is None:
        class_idx = explanation.predicted_class

    class_label = explanation.output_names[class_idx]

    # --- aggregate SHAP values per unique sentence (mean per occurrence) ---
    # Mean per occurrence removes frequency bias in the rare case the same
    # sentence appears verbatim more than once in a document. Word-level
    # stop-word filtering (PT_STOP_WORDS) no longer applies at sentence
    # granularity, sentences naturally carry legal signal end to end.
    word_sums: dict[str, float] = defaultdict(float)
    word_counts: dict[str, int] = defaultdict(int)
    for token, val in zip(explanation.tokens, explanation.shap_values[:, class_idx]):
        key = token.strip()
        if key:
            word_sums[key] += float(val)
            word_counts[key] += 1
    word_shap: dict[str, float] = {
        w: word_sums[w] / word_counts[w] for w in word_sums
    }

    # --- split into positive and negative, rank by absolute magnitude ---
    positive = sorted(
        [(w, v) for w, v in word_shap.items() if v > 0],
        key=lambda x: x[1],
        reverse=True,
    )[:top_n]

    negative = sorted(
        [(w, v) for w, v in word_shap.items() if v < 0],
        key=lambda x: x[1],
    )[:top_n]

    # Build combined list: positives on top (descending), negatives below (ascending)
    # so the chart reads naturally from most-positive at top to most-negative at bottom.
    combined = positive + negative[::-1]   # negatives reversed: least-negative nearest centre

    if not combined:
        # All SHAP values are zero — model output was insensitive to token masking
        # (e.g. a very overconfident LLM).  Skip the chart rather than crash.
        print(f"[plot_shap_top_terms] WARNING: all SHAP values are zero for "
              f"case '{explanation.case_id}' — skipping top-terms chart.")
        plt.close("all")
        return None

    labels = [w for w, _ in combined]
    values = [v for _, v in combined]
    colours = [SHAP_POSITIVE_COLOUR if v >= 0 else SHAP_NEGATIVE_COLOUR for v in values]

    # --- build figure ---
    n_bars = len(combined)
    fig_height = max(4.0, n_bars * 0.45 + 1.8)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    y_pos = range(n_bars - 1, -1, -1)     # top-to-bottom layout
    bars = ax.barh(list(y_pos), values, color=colours, edgecolor="white", linewidth=0.4)

    # value labels at bar tips
    for bar, val in zip(bars, values):
        x_offset = 0.002 * (ax.get_xlim()[1] - ax.get_xlim()[0] or 1)
        ha = "left" if val >= 0 else "right"
        ax.text(
            val + (x_offset if val >= 0 else -x_offset),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            va="center", ha=ha, fontsize=8, color="#333333",
        )

    # Sentences are long; truncate display labels so the chart stays readable.
    # The full sentence text is still available in `word_shap`/`explanation.tokens`.
    display_labels = [
        (lbl if len(lbl) <= 90 else lbl[:87].rstrip() + "…") for lbl in labels
    ]

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(display_labels, fontsize=9)
    ax.axvline(0, color="#666666", linewidth=1.0, linestyle="--")

    ax.set_xlabel(f"Mean SHAP per occurrence  (class: {class_label})", fontsize=11)
    ax.set_title(
        f"Top {top_n} Positive & Negative SHAP Sentences\n"
        f"Class predicted: '{class_label}'  |  Model: {explanation.model_name}"
        f"  |  Case: {explanation.case_id}",
        fontsize=11, pad=10,
    )

    # legend patches
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor=SHAP_POSITIVE_COLOUR, label=f"Pushes toward '{class_label}' (positive)"),
            Patch(facecolor=SHAP_NEGATIVE_COLOUR, label=f"Pushes away from '{class_label}' (negative)"),
        ],
        fontsize=9, loc="lower right",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    plt.tight_layout()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"shap_top_terms_{explanation.case_id}.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# HTML export — SHAP highlighted text
# ---------------------------------------------------------------------------

def _shap_span_color(val: float, max_abs: float) -> str:
    """Map a SHAP value to an inline CSS background-color string.

    Positive values → red  (white at 0, #FF0D57 at max).
    Negative values → blue (white at 0, #1E88E5 at max).
    """
    if max_abs < 1e-12 or val == 0.0:
        return "transparent"
    intensity = min(abs(val) / max_abs, 1.0)
    # blend from white (255,255,255) toward the target colour
    if val > 0:
        r, g, b = 255, int(255 * (1.0 - intensity * 0.95)), int(255 * (1.0 - intensity * 0.65))
    else:
        r, g, b = int(255 * (1.0 - intensity * 0.88)), int(255 * (1.0 - intensity * 0.45)), 255
    return f"rgb({r},{g},{b})"


def export_shap_html(
    explanation: ShapExplanation,
    class_idx: int | None = None,
    filter_mode: str = "all",
    output_path: Path | None = None,
) -> str:
    """Export SHAP-highlighted text as a self-contained HTML file.

    Renders each word as a coloured ``<span>`` (red = positive contribution,
    blue = negative contribution, intensity proportional to magnitude).
    The file is a complete standalone HTML document — no JavaScript required.

    Parameters
    ----------
    explanation : ShapExplanation
        Output of ``compute_shap_explanation``.
    class_idx : int or None
        Class to explain.  Defaults to ``predicted_class``.
    filter_mode : str
        ``'all'``      — both positive and negative highlights.
        ``'positive'`` — only words pushing toward the class (red).
        ``'negative'`` — only words pushing away from the class (blue).
    output_path : Path or None
        If provided, write the HTML to this file (UTF-8).
    """
    if filter_mode not in ("all", "positive", "negative"):
        raise ValueError(
            f"filter_mode must be 'all', 'positive', or 'negative', got {filter_mode!r}"
        )

    if class_idx is None:
        class_idx = explanation.predicted_class

    sv_col = explanation.shap_values[:, class_idx].copy()
    if filter_mode == "positive":
        sv_col = np.where(sv_col > 0, sv_col, 0.0)
    elif filter_mode == "negative":
        sv_col = np.where(sv_col < 0, sv_col, 0.0)

    # Aggregate to mean SHAP value per occurrence of each distinct sentence.
    # At sentence granularity, exact repeats are rare, so this mostly
    # collapses to one value per sentence. Using the MEAN (sum / count)
    # rather than the raw sum keeps the same normalisation logic as before
    # for the (uncommon) case a sentence does repeat verbatim in a document.
    word_sums: dict[str, float] = {}
    word_counts: dict[str, int] = {}
    for tok, val in zip(explanation.tokens, sv_col):
        key = tok.lower()
        word_sums[key] = word_sums.get(key, 0.0) + float(val)
        word_counts[key] = word_counts.get(key, 0) + 1
    sv_agg = np.array(
        [word_sums[t.lower()] / word_counts[t.lower()] for t in explanation.tokens],
        dtype=np.float64,
    )

    max_abs = float(np.abs(sv_agg).max()) or 1e-12
    class_label = explanation.output_names[class_idx]
    true_str = explanation.true_label_str or "n/a"
    proba_str = ", ".join(
        f"{name}: {float(p):.3f}"
        for name, p in zip(explanation.output_names, explanation.predict_proba)
    )

    # Build coloured spans
    spans: list[str] = []
    for tok, val in zip(explanation.tokens, sv_agg):
        color = _shap_span_color(float(val), max_abs)
        title = f"SHAP: {val:+.4f}"
        spans.append(
            f'<span style="background-color:{color};padding:2px 1px;'
            f'border-radius:2px;margin:1px 0" title="{title}">{tok}</span>'
        )
    text_html = " ".join(spans)

    # Legend
    legend = (
        "<div style='margin-bottom:12px;font-size:13px'>"
        "<span style='background-color:rgb(255,13,87);padding:2px 8px;border-radius:3px'>"
        "&nbsp;positive&nbsp;</span>&nbsp;pushes toward class&nbsp;&nbsp;"
        "<span style='background-color:rgb(30,136,229);padding:2px 8px;border-radius:3px'>"
        "&nbsp;negative&nbsp;</span>&nbsp;pushes away from class"
        "</div>"
    )

    html = (
        "<!DOCTYPE html>\n<html lang='pt'>\n"
        "<head><meta charset='utf-8'>"
        f"<title>SHAP — {explanation.case_id} — {class_label}</title>"
        "<style>body{font-family:Georgia,serif;line-height:1.8;padding:24px;max-width:1100px;margin:auto}"
        "h2{font-size:16px;margin-bottom:4px}table{border-collapse:collapse;margin-bottom:16px;font-size:13px}"
        "td{padding:3px 12px 3px 0}</style></head>\n"
        "<body>\n"
        f"<h2>Case: {explanation.case_id}</h2>\n"
        "<table>"
        f"<tr><td><b>Model</b></td><td>{explanation.model_name}</td></tr>"
        f"<tr><td><b>Predicted</b></td><td>{class_label}</td></tr>"
        f"<tr><td><b>True label</b></td><td>{true_str}</td></tr>"
        f"<tr><td><b>Probabilities</b></td><td>{proba_str}</td></tr>"
        "</table>\n"
        f"{legend}\n"
        f"<div style='font-size:15px;line-height:2.2'>{text_html}</div>\n"
        "</body></html>"
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")

    return html



# ---------------------------------------------------------------------------
# Batch save utilities + JSON export  (Step 5)
# ---------------------------------------------------------------------------

def explanation_to_dict(explanation: ShapExplanation) -> dict:
    """Serialise a ShapExplanation to a JSON-safe dict.

    All ``np.ndarray`` fields are converted to nested Python lists.
    Strings are preserved as-is so Portuguese characters (ã, ç, ê, …) survive
    a round-trip through ``json.dump(..., ensure_ascii=False)``.
    """
    return {
        "case_id":        explanation.case_id,
        "case_type":      explanation.case_type,
        "model_name":     explanation.model_name,
        "text":           explanation.text,
        "tokens":         explanation.tokens,
        "shap_values":    explanation.shap_values.tolist(),
        "base_values":    explanation.base_values.tolist(),
        "output_names":   explanation.output_names,
        "predicted_class": int(explanation.predicted_class),
        "predicted_label": explanation.predicted_label,
        "predict_proba":  explanation.predict_proba.tolist(),
        "true_label":     int(explanation.true_label) if explanation.true_label is not None else None,
        "true_label_str": explanation.true_label_str,
        "metadata":       explanation.metadata,
    }


def save_explanation_json(
    explanations: list[ShapExplanation] | ShapExplanation,
    output_path: Path,
) -> None:
    """Write one or more ShapExplanations to a UTF-8 JSON file.

    Parameters
    ----------
    explanations : ShapExplanation or list of ShapExplanation
        Single explanation or batch.  A single object is wrapped in a list
        so the file is always a JSON array — easier to load downstream.
    output_path : Path
        Destination file.  Parent directories are created automatically.
    """
    if isinstance(explanations, ShapExplanation):
        explanations = [explanations]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump([explanation_to_dict(e) for e in explanations], fh,
                  ensure_ascii=False, indent=2)


def explanations_to_long_dataframe(
    explanations: list[ShapExplanation] | ShapExplanation,
) -> pd.DataFrame:
    """Flatten a batch of ShapExplanations into one row per (case, sentence).

    This is the layout pandas-based analysis wants: ranking or filtering
    sentences by SHAP value, either within a case or across the whole batch.

    Example
    -------
    df = explanations_to_long_dataframe(explanations)
    df.sort_values("shap_value_predicted_class", ascending=False).head(10)
    df[df.case_id == "65_24.0GEBRG.G1"].nsmallest(5, "shap_value_predicted_class")

    Returns
    -------
    pandas.DataFrame
        One row per sentence per case. Columns:
            case_id, case_type, model_name, sentence_idx, sentence_text,
            shap_value_predicted_class, shap_values_all_classes,
            predicted_class, predicted_label, true_label, true_label_str,
            output_names, predict_proba, base_values
    """
    if isinstance(explanations, ShapExplanation):
        explanations = [explanations]

    rows: list[dict] = []
    for exp in explanations:
        class_idx = exp.predicted_class
        for i, (sentence, sv_row) in enumerate(zip(exp.tokens, exp.shap_values)):
            rows.append(
                {
                    "case_id": exp.case_id,
                    "case_type": exp.case_type,
                    "model_name": exp.model_name,
                    "sentence_idx": i,
                    "sentence_text": sentence,
                    "shap_value_predicted_class": float(sv_row[class_idx]),
                    "shap_values_all_classes": [float(v) for v in sv_row],
                    "predicted_class": int(exp.predicted_class),
                    "predicted_label": exp.predicted_label,
                    "true_label": int(exp.true_label) if exp.true_label is not None else None,
                    "true_label_str": exp.true_label_str,
                    "output_names": list(exp.output_names),
                    "predict_proba": [float(p) for p in exp.predict_proba],
                    "base_values": [float(v) for v in exp.base_values],
                }
            )
    return pd.DataFrame(rows)


def save_explanation_parquet(
    explanations: list[ShapExplanation] | ShapExplanation,
    output_path: Path,
) -> None:
    """Write a batch of ShapExplanations to a long-format parquet file.

    One row per (case, sentence) — see ``explanations_to_long_dataframe``.
    Meant for pandas-based inspection/evaluation of sentence-level SHAP
    scores, as an alternative to loading the nested ``explanations.json``.

    Parameters
    ----------
    explanations : ShapExplanation or list of ShapExplanation
    output_path : Path
        Destination file (e.g. ``explanations.parquet``). Parent
        directories are created automatically.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = explanations_to_long_dataframe(explanations)
    df.to_parquet(output_path, index=False)


def save_explanation_batch(
    explanations: list[ShapExplanation],
    output_base_dir: Path,
    save_html: bool = True,
    save_bar_png: bool = True,
    save_json: bool = True,
    save_parquet: bool = True,
    subdir: str | None = None,
) -> Path:
    """Save a batch of ShapExplanations to the canonical directory structure.

    Output layout under ``output_base_dir``::

        {case_type_dir}/{model_name}/
            {case_id}.html                  ← SHAP highlighted text (filter='all')
            shap_top_terms_{case_id}.png    ← top-terms bar chart
            explanations.json               ← all cases serialised together (nested)
            explanations.parquet            ← long format, one row per sentence per case

        # With subdir (e.g. "5_gold_set"):
        {case_type_dir}/{subdir}/{model_name}/

    Parameters
    ----------
    explanations : list of ShapExplanation
        Batch to save.  All items should share the same ``case_type`` and
        ``model_name`` (they are read from the first explanation for the path).
    output_base_dir : Path
        Root directory under which subdirectories are created.
    save_html, save_bar_png, save_json, save_parquet : bool
        Toggle individual output types.
    subdir : str or None
        Optional extra directory level between case_type_dir and model_name.
        Use "5_gold_set" to separate judge-annotated gold cases from general runs.

    Returns
    -------
    Path
        The model-level output directory where all files were written.
    """
    if not explanations:
        raise ValueError("explanations list is empty")

    output_base_dir = Path(output_base_dir)
    first = explanations[0]
    case_dir  = CASE_TYPE_OUTPUT_DIRS.get(first.case_type, first.case_type)
    if subdir:
        model_dir = output_base_dir / case_dir / subdir / first.model_name
    else:
        model_dir = output_base_dir / case_dir / first.model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    for exp in explanations:
        if save_html:
            export_shap_html(
                exp, filter_mode="all",
                output_path=model_dir / f"{exp.case_id}.html",
            )
        if save_bar_png:
            plot_shap_top_terms(exp, output_dir=model_dir, show=False)

    if save_json:
        save_explanation_json(explanations, model_dir / "explanations.json")

    if save_parquet:
        save_explanation_parquet(explanations, model_dir / "explanations.parquet")

    return model_dir


# ---------------------------------------------------------------------------
# LLM adapter + batch pipeline entry point  (Step 7)
# ---------------------------------------------------------------------------

def make_llm_predict_fn(
    ollama_label_fn: Callable[[str], str],
    output_names: list[str],
    temperature_smoothing: float = 0.1,
) -> Callable[[list[str]], np.ndarray]:
    """Wrap an Ollama label function into a SHAP-compatible predict_fn.

    Ollama-based LLMs return a single predicted class label string rather than
    a probability vector.  SHAP requires a function that returns a 2-D array of
    shape ``(n_texts, n_classes)``.  This adapter converts the hard label to a
    near-one-hot probability vector using temperature smoothing so that SHAP's
    colour scale has a non-trivial range to work with.

    Smoothing formula (for the predicted class c, n classes total)::

        p(c)   = 1 - temperature_smoothing
        p(i≠c) = temperature_smoothing / (n_classes - 1)

    This keeps predictions distinguishable while avoiding the degenerate case
    where all SHAP values are zero (which happens when a one-hot vector makes
    the output perfectly insensitive to masking).

    Parameters
    ----------
    ollama_label_fn : Callable[[str], str]
        Function that takes a single text string and returns one of the label
        strings in ``output_names`` (e.g. ``'decisao_mantida'``).
    output_names : list[str]
        Ordered class label strings — must match those used in ShapExplanation.
    temperature_smoothing : float
        Probability mass distributed uniformly over non-predicted classes.
        Default 0.1 gives 90 / 5 % split for binary, 90 / 5 / 5 for ternary.

    Returns
    -------
    Callable[[list[str]], np.ndarray]
        Batch predict function: list of N strings → (N, n_classes) array.
    """
    n_classes = len(output_names)
    label_to_idx = {name: i for i, name in enumerate(output_names)}

    def predict_fn(texts: list[str]) -> np.ndarray:
        proba = np.empty((len(texts), n_classes), dtype=float)
        for i, text in enumerate(texts):
            label = ollama_label_fn(text)
            pred_idx = label_to_idx.get(label, 0)   # fallback to class 0 on unknown label
            row = np.full(n_classes, temperature_smoothing / max(n_classes - 1, 1))
            row[pred_idx] = 1.0 - temperature_smoothing
            proba[i] = row
        return proba

    return predict_fn


def run_shap_explanation_pipeline(
    texts: list[str],
    case_ids: list[str],
    predict_fn: Callable[[list[str]], np.ndarray],
    output_names: list[str],
    case_type: str,
    model_name: str,
    output_base_dir: Path,
    true_labels: list[int] | None = None,
    metadata_list: list[dict] | None = None,
    max_evals: int = DEFAULT_MAX_EVALS,
    subdir: str | None = None,
) -> list[ShapExplanation]:
    """Compute SHAP explanations for a batch of documents and save all outputs.

    Parameters
    ----------
    texts : list[str]
        Raw legal document texts.
    case_ids : list[str]
        Process identifiers (``n_processo``), one per text.
    predict_fn : Callable[[list[str]], np.ndarray]
        Model predict function (``(n_texts, n_classes)``).  Use
        ``make_llm_predict_fn`` to wrap an Ollama LLM.
    output_names : list[str]
        Ordered class label strings.
    case_type : str
        ``'dv'`` or ``'boc'``.
    model_name : str
        Used as the output subdirectory name.
    output_base_dir : Path
        Root directory for saved outputs.
    true_labels : list[int] or None
        Ground-truth integer labels, if available.
    metadata_list : list[dict] or None
        Per-case metadata dicts (tribunal, url, …).
    max_evals : int
        SHAP budget per document.  Use 200 for large batches.

    Returns
    -------
    list[ShapExplanation]
        One ShapExplanation per input text.
    """
    from tqdm import tqdm

    if true_labels is None:
        true_labels = [None] * len(texts)
    if metadata_list is None:
        metadata_list = [{}] * len(texts)

    explanations = []
    for text, case_id, true_label, meta in tqdm(
        zip(texts, case_ids, true_labels, metadata_list),
        total=len(texts),
        desc=f"SHAP [{model_name}]",
    ):
        exp = compute_shap_explanation(
            text=text,
            predict_fn=predict_fn,
            case_id=case_id,
            case_type=case_type,
            model_name=model_name,
            output_names=output_names,
            true_label=true_label,
            metadata=meta,
            max_evals=max_evals,
        )
        explanations.append(exp)

    save_explanation_batch(explanations, output_base_dir, subdir=subdir)
    return explanations


# ---------------------------------------------------------------------------
# Smoke test — full pipeline with dummy LLM predict_fn
# ---------------------------------------------------------------------------

def _smoke_test_pipeline():
    """End-to-end smoke test: dummy LLM + 3 fake texts → HTML / PNG / JSON."""
    import tempfile, webbrowser

    # --- binary task (DV) ---
    output_names_dv = list(LABEL_NAMES_DV.values())

    # Simulate an Ollama LLM that always predicts 'decisao_alterada'
    def fake_ollama(text: str) -> str:
        return "decisao_alterada"

    predict_fn_dv = make_llm_predict_fn(fake_ollama, output_names_dv)

    texts = [
        "O arguido agrediu a vítima repetidamente causando lesões graves no rosto.",
        "Não existem provas suficientes para sustentar a acusação formulada pelo Ministério Público.",
        "O tribunal de primeira instância decidiu manter a pena suspensa aplicada ao recorrente.",
    ]
    case_ids  = ["proc-001", "proc-002", "proc-003"]
    true_labels = [1, 0, 0]

    tmp = Path(tempfile.mkdtemp())
    explanations = run_shap_explanation_pipeline(
        texts=texts,
        case_ids=case_ids,
        predict_fn=predict_fn_dv,
        output_names=output_names_dv,
        case_type="dv",
        model_name="fake_llm_dv",
        output_base_dir=tmp,
        true_labels=true_labels,
        max_evals=200,
    )

    # Assert all output files exist
    model_dir = tmp / "domestic_violence" / "fake_llm_dv"
    for cid in case_ids:
        assert (model_dir / f"{cid}.html").exists(), f"Missing HTML for {cid}"
        # PNG may be absent if all SHAP values were zero (overconfident model)
    assert (model_dir / "explanations.json").exists(), "Missing JSON"

    print(f"\nOutputs written to: {model_dir}")
    webbrowser.open(str(model_dir / f"{case_ids[0]}.html"))
    print("Smoke test pipeline — PASSED")


if __name__ == "__main__":
    _smoke_test_pipeline()
