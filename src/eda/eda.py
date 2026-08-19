from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import re

# ==============================
# Configuration
# ==============================

CASE_LABELS = {
    "dv": "Domestic Violence",
    "boc": "Breach of Contract",
}

CASE_FOLDERS = {
    "dv": "domestic_violence",
    "boc": "breach_of_contract",
}

CASE_COLOURS = {
    "dv": "#E53935",  # red
    "boc": "#1E88E5",  # blue
}

DECISION_LABELS_BINARY = {
    "DECISÃO MANTIDA": "Kept",
    "DECISÃO ALTERADA": "Changed",
}

DECISION_LABELS_TERNARY = {
    "FAVORÁVEL": "Favourable",
    "DESFAVORÁVEL": "Unfavourable",
    "PARCIAL": "Partial",
}

DECISION_COLOURS_BINARY = {
    "Kept": "#7E57C2",  # violet
    "Changed": "#FB8C00",  # orange
}

DECISION_COLOURS_TERNARY = {
    "Favourable": "#7CB342",  # green-lime
    "Unfavourable": "#D81B60",  # magenta/pink
    "Partial": "#26C6DA",  # turquoise/cyan
}


@dataclass(frozen=True)
class CaseSchema:
    case_type: str
    schema: str
    decision_column: str
    decision_labels: Dict[str, str]
    decision_colours: Dict[str, str]


def get_case_schema(case_type: str) -> CaseSchema:
    case_type = case_type.lower().strip()
    if case_type == "dv":
        return CaseSchema(
            case_type="dv",
            schema="binary",
            decision_column="decisao_binaria",
            decision_labels=DECISION_LABELS_BINARY,
            decision_colours=DECISION_COLOURS_BINARY,
        )
    if case_type == "boc":
        return CaseSchema(
            case_type="boc",
            schema="ternary",
            decision_column="decisao_ternaria",
            decision_labels=DECISION_LABELS_TERNARY,
            decision_colours=DECISION_COLOURS_TERNARY,
        )
    raise ValueError("case_type must be 'dv' or 'boc'.")


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_eda_dataframe(case_type: str, base_dir: Path | None = None) -> pd.DataFrame:
    schema = get_case_schema(case_type)
    base_dir = base_dir or get_project_root()
    data_path = (
        base_dir
        / "data"
        / "processed_data"
        / "eda"
        / schema.schema
        / f"df_acordaos_{schema.case_type}_eda_{schema.schema}.csv"
    )
    return pd.read_csv(data_path)


def ensure_output_dir(base_dir: Path, case_type: str, section: str) -> Path:
    out_dir = base_dir / "eda_viz" / CASE_FOLDERS[case_type] / section
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def normalise_decision_labels(
    df: pd.DataFrame, decision_column: str, label_map: Dict[str, str]
) -> pd.DataFrame:
    df = df.copy()
    df["decision_label"] = (
        df[decision_column]
        .astype(str)
        .str.strip()
        .str.upper()
        .map({k.upper(): v for k, v in label_map.items()})
    )
    return df


def add_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "data_acordao" not in df.columns:
        raise KeyError("Column 'data_acordao' not found in dataframe.")
    df["acordao_date"] = pd.to_datetime(
        df["data_acordao"], dayfirst=True, errors="coerce"
    )
    df["year"] = df["acordao_date"].dt.year
    return df


def clean_text_basic(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text_advanced(text: str) -> str:
    text = clean_text_basic(text)
    text = re.sub(r"\b\d+º?\b", " ", text)
    text = re.sub(r"\b\w\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _ensure_nltk_resource(resource: str) -> None:
    import nltk

    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split("/")[-1])


def add_text_metrics(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize

    _ensure_nltk_resource("tokenizers/punkt")
    df = df.copy()
    df["clean_text"] = df[text_column].apply(clean_text_advanced)
    df["text_length_raw"] = df[text_column].astype(str).apply(len)
    df["text_length_clean"] = df["clean_text"].apply(len)
    df["word_count"] = df["clean_text"].apply(
        lambda x: len(word_tokenize(x, language="portuguese"))
    )
    df["sentence_count"] = (
        df[text_column]
        .astype(str)
        .apply(lambda x: len(sent_tokenize(x, language="portuguese")))
    )
    return df


def add_tokens_no_stopwords(
    df: pd.DataFrame, text_column: str = "clean_text"
) -> pd.DataFrame:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    _ensure_nltk_resource("corpora/stopwords")
    _ensure_nltk_resource("tokenizers/punkt")

    stop_words = set(stopwords.words("portuguese"))
    df = df.copy()
    df["tokens_no_stopwords"] = df[text_column].apply(
        lambda x: [
            w for w in word_tokenize(x, language="portuguese") if w not in stop_words
        ]
    )
    return df


def _blend_with_white(base_hex: str, values: Iterable[float]) -> List[str]:
    import matplotlib.colors as mcolors

    rgb_base = np.array(mcolors.to_rgb(base_hex))
    values_arr = np.asarray(list(values), dtype=float)
    if values_arr.size == 0:
        return []
    min_val = values_arr.min()
    max_val = values_arr.max()
    if max_val == min_val:
        norm = np.ones_like(values_arr)
    else:
        norm = (values_arr - min_val) / (max_val - min_val)
    colours = []
    for v in norm:
        rgb = (1 - v) * np.ones(3) + v * rgb_base
        colours.append(mcolors.to_hex(rgb))
    return colours


def plot_cases_per_year(df: pd.DataFrame, case_type: str, base_dir: Path) -> None:
    df = add_date_columns(df)
    counts = df["year"].value_counts().sort_index()

    plt.figure(figsize=(10, 4))
    plt.plot(counts.index, counts.values, marker="o", color=CASE_COLOURS[case_type])
    plt.title(f"{CASE_LABELS[case_type]} — Number of cases per year")
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)
    out_dir = ensure_output_dir(base_dir, case_type, "time_analysis")
    plt.tight_layout()
    plt.savefig(
        out_dir / f"cases_per_year_{case_type}.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_year_distribution_by_class(
    df: pd.DataFrame,
    case_type: str,
    class_col: str,
    class_colours: Dict[str, str],
    base_dir: Path,
) -> None:
    df = add_date_columns(df)
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df.dropna(subset=["year"]), x=class_col, y="year", palette=class_colours
    )
    plt.title(f"{CASE_LABELS[case_type]} — Year distribution by decision")
    plt.xlabel("Decision")
    plt.ylabel("Year")
    plt.xticks(rotation=20)
    out_dir = ensure_output_dir(base_dir, case_type, "time_analysis")
    plt.tight_layout()
    plt.savefig(
        out_dir / f"year_distribution_by_decision_{case_type}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_temporal_drift(
    df: pd.DataFrame,
    case_type: str,
    class_col: str,
    class_colours: Dict[str, str],
    base_dir: Path,
) -> None:
    df = add_date_columns(df)
    counts = pd.crosstab(df["year"], df[class_col]).sort_index()
    props = counts.div(counts.sum(axis=1), axis=0).fillna(0) * 100

    colours = [class_colours[c] for c in props.columns]
    ax = props.plot(kind="bar", stacked=True, figsize=(12, 6), color=colours)
    ax.set_title(f"{CASE_LABELS[case_type]} — Proportion by year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Proportion (%)")
    ax.legend(title="Decision", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    out_dir = ensure_output_dir(base_dir, case_type, "time_analysis")
    plt.tight_layout()
    plt.savefig(
        out_dir / f"temporal_drift_{case_type}.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_class_distribution(
    df: pd.DataFrame,
    case_type: str,
    class_col: str,
    class_colours: Dict[str, str],
    base_dir: Path,
) -> None:
    counts = df[class_col].value_counts()
    plt.figure(figsize=(7, 4))
    ax = sns.barplot(x=counts.index, y=counts.values, palette=class_colours)
    plt.title(f"{CASE_LABELS[case_type]} — Class distribution")
    plt.xlabel("Decision")
    plt.ylabel("Count")
    plt.grid(axis="y", alpha=0.3)

    # Add count + percentage labels inside bars
    total = counts.sum()
    for i, (label, count) in enumerate(counts.items()):
        percentage = (count / total) * 100 if total else 0
        ax.text(
            i,
            count * 0.5,
            f"{count}\n({percentage:.1f}%)",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=class_colours.get(label, "#444444"),
                edgecolor="#222222",
                alpha=0.85,
            ),
        )

    out_dir = ensure_output_dir(base_dir, case_type, "basic_stats")
    plt.tight_layout()
    plt.savefig(
        out_dir / f"class_distribution_{case_type}.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_text_length_by_class(
    df: pd.DataFrame,
    case_type: str,
    class_col: str,
    class_colours: Dict[str, str],
    base_dir: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x=class_col, y="text_length_raw", palette=class_colours)
    plt.title(f"{CASE_LABELS[case_type]} — Text length by decision")
    plt.xlabel("Decision")
    plt.ylabel("Text length (characters)")
    plt.xticks(rotation=20)
    out_dir = ensure_output_dir(base_dir, case_type, "basic_stats")
    plt.tight_layout()
    plt.savefig(
        out_dir / f"text_length_by_decision_{case_type}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def _plot_ranked_bars(
    ax: plt.Axes,
    labels: List[str],
    values: List[float],
    base_colour: str,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    colours = _blend_with_white(base_colour, values)
    ax.barh(labels, values, color=colours)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.invert_yaxis()


def _extract_most_common(
    tokens_list: Iterable[List[str]], top_n: int
) -> List[Tuple[str, int]]:
    from collections import Counter

    counter: Counter[str] = Counter()
    for tokens in tokens_list:
        if isinstance(tokens, list):
            counter.update(tokens)
    return counter.most_common(top_n)


def plot_word_frequency_per_class(
    df: pd.DataFrame,
    case_type: str,
    class_col: str,
    class_colours: Dict[str, str],
    base_dir: Path,
    top_n: int = 20,
) -> None:
    classes = list(df[class_col].unique())
    fig, axes = plt.subplots(1, len(classes), figsize=(8 * len(classes), 6))
    if len(classes) == 1:
        axes = [axes]

    for ax, class_label in zip(axes, classes):
        tokens = df[df[class_col] == class_label]["tokens_no_stopwords"]
        most_common = _extract_most_common(tokens, top_n)
        words = [w for w, _ in most_common]
        freqs = [f for _, f in most_common]
        _plot_ranked_bars(
            ax,
            words,
            freqs,
            class_colours[class_label],
            xlabel="Frequency",
            ylabel="Word",
            title=f"{class_label}",
        )

    plt.suptitle(f"{CASE_LABELS[case_type]} — Top {top_n} words by class", y=1.02)
    out_dir = ensure_output_dir(base_dir, case_type, "basic_stats")
    plt.tight_layout()
    plt.savefig(
        out_dir / f"word_frequency_{case_type}.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_ngram_frequency_per_class(
    df: pd.DataFrame,
    case_type: str,
    class_col: str,
    class_colours: Dict[str, str],
    base_dir: Path,
    n: int = 2,
    top_n: int = 20,
) -> None:
    from nltk.util import ngrams
    from collections import Counter

    classes = list(df[class_col].unique())
    fig, axes = plt.subplots(1, len(classes), figsize=(8 * len(classes), 6))
    if len(classes) == 1:
        axes = [axes]

    for ax, class_label in zip(axes, classes):
        counter: Counter[str] = Counter()
        for tokens in df[df[class_col] == class_label]["tokens_no_stopwords"]:
            if isinstance(tokens, list) and len(tokens) >= n:
                ngram_list = [" ".join(ng) for ng in ngrams(tokens, n)]
                counter.update(ngram_list)
        most_common = counter.most_common(top_n)
        labels = [w for w, _ in most_common]
        values = [f for _, f in most_common]
        _plot_ranked_bars(
            ax,
            labels,
            values,
            class_colours[class_label],
            xlabel="Frequency",
            ylabel=f"{n}-gram",
            title=f"{class_label}",
        )

    plt.suptitle(f"{CASE_LABELS[case_type]} — Top {top_n} {n}-grams by class", y=1.02)
    out_dir = ensure_output_dir(base_dir, case_type, "basic_stats")
    plt.tight_layout()
    plt.savefig(
        out_dir / f"{n}grams_frequency_{case_type}.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_tfidf_per_class(
    df: pd.DataFrame,
    case_type: str,
    class_col: str,
    class_colours: Dict[str, str],
    base_dir: Path,
    ngram_range: Tuple[int, int] = (1, 1),
    top_n: int = 20,
) -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer

    classes = list(df[class_col].unique())
    fig, axes = plt.subplots(1, len(classes), figsize=(8 * len(classes), 6))
    if len(classes) == 1:
        axes = [axes]

    for ax, class_label in zip(axes, classes):
        documents = df[df[class_col] == class_label]["tokens_no_stopwords"].apply(
            lambda tokens: " ".join(tokens) if isinstance(tokens, list) else ""
        )
        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range, lowercase=False, token_pattern=r"(?u)\b\w+\b"
        )
        tfidf_matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()
        mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        term_scores = sorted(
            zip(feature_names, mean_scores), key=lambda x: x[1], reverse=True
        )[:top_n]
        terms = [t for t, _ in term_scores]
        scores = [s for _, s in term_scores]
        _plot_ranked_bars(
            ax,
            terms,
            scores,
            class_colours[class_label],
            xlabel="Mean TF-IDF score",
            ylabel="Term",
            title=f"{class_label}",
        )

    ngram_label = "-".join(map(str, ngram_range))
    plt.suptitle(
        f"{CASE_LABELS[case_type]} — Top {top_n} TF-IDF terms by class", y=1.02
    )
    out_dir = ensure_output_dir(base_dir, case_type, "basic_stats")
    plt.tight_layout()
    plt.savefig(
        out_dir / f"tfidf_{ngram_label}_{case_type}.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_decisions_by_tribunal(
    df: pd.DataFrame,
    case_type: str,
    class_col: str,
    class_colours: Dict[str, str],
    base_dir: Path,
) -> None:
    df_plot = df.copy()
    df_plot["tribunal"] = df_plot["tribunal"].apply(
        lambda x: "JP" if str(x).startswith("JP") else x
    )

    ct_pct = (
        pd.crosstab(df_plot["tribunal"], df_plot[class_col], normalize="index") * 100
    )
    tribunal_counts = df_plot["tribunal"].value_counts()
    ct_pct = ct_pct.loc[tribunal_counts.index]

    colours = [class_colours[c] for c in ct_pct.columns]
    ax = ct_pct.plot(kind="barh", stacked=True, figsize=(12, 7), color=colours)
    ax.set_xlabel("Percentage (%)")
    ax.set_ylabel("Tribunal")
    ax.set_title(f"{CASE_LABELS[case_type]} — Decision distribution by tribunal")
    ax.legend(title="Decision", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="x", alpha=0.3)
    labels = [f"{trib} (n={tribunal_counts[trib]})" for trib in ct_pct.index]
    ax.set_yticklabels(labels)

    out_dir = ensure_output_dir(base_dir, case_type, "tribunal")
    plt.tight_layout()
    plt.savefig(
        out_dir / f"decisions_by_tribunal_{case_type}.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def _extract_first_name(full_name: str | None) -> str | None:
    if not full_name or not isinstance(full_name, str):
        return None
    name = full_name.strip()
    parts = name.split()
    if not parts:
        return None
    first_name = parts[0].replace(".", "").replace(",", "").strip().lower()
    return first_name or None


def classify_judge_gender(
    df: pd.DataFrame, judge_column: str = "juiz_relator"
) -> pd.DataFrame:
    import gender_guesser.detector as gender

    df_gender = df.copy()
    df_gender["judge_first_name"] = df_gender[judge_column].apply(_extract_first_name)

    detector = gender.Detector(case_sensitive=False)
    gender_raw = []
    for first_name in df_gender["judge_first_name"]:
        if first_name:
            gender_raw.append(detector.get_gender(first_name))
        else:
            gender_raw.append("unknown")

    df_gender["judge_gender_raw"] = gender_raw

    def simplify_gender(raw: str) -> str:
        if raw in {"male", "mostly_male"}:
            return "Male"
        if raw in {"female", "mostly_female"}:
            return "Female"
        return "Unknown"

    df_gender["judge_gender"] = df_gender["judge_gender_raw"].apply(simplify_gender)
    return df_gender


def plot_decisions_by_judge_gender(
    df: pd.DataFrame,
    case_type: str,
    class_col: str,
    class_colours: Dict[str, str],
    base_dir: Path,
    gender_column: str = "judge_gender",
) -> None:
    ct_pct = pd.crosstab(df[gender_column], df[class_col], normalize="index") * 100
    gender_counts = df[gender_column].value_counts()
    ct_pct = ct_pct.loc[gender_counts.index]

    colours = [class_colours[c] for c in ct_pct.columns]
    ax = ct_pct.plot(kind="barh", stacked=True, figsize=(10, 5), color=colours)
    ax.set_xlabel("Percentage (%)")
    ax.set_ylabel("Judge gender")
    ax.set_title(f"{CASE_LABELS[case_type]} — Decision distribution by judge gender")
    ax.legend(title="Decision", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="x", alpha=0.3)
    labels = [f"{g} (n={gender_counts[g]})" for g in ct_pct.index]
    ax.set_yticklabels(labels)

    out_dir = ensure_output_dir(base_dir, case_type, "judge_gender")
    plt.tight_layout()
    plt.savefig(
        out_dir / f"decisions_by_judge_gender_{case_type}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def _normalise_case_type(case_type: str | None) -> str | None:
    if case_type is None:
        return None
    case_type_norm = case_type.lower().strip()
    if case_type_norm == "ic":
        return "boc"
    if case_type_norm in CASE_LABELS:
        return case_type_norm
    raise ValueError("case_type must be one of: 'dv', 'boc', 'ic'.")


def _infer_case_type_from_path(parquet_path: Path) -> str | None:
    name = parquet_path.stem.lower()
    if "dv" in name or "domestic_violence" in name:
        return "dv"
    if "boc" in name or "_ic_" in name or "breach_of_contract" in name:
        return "boc"
    return None


def _resolve_embeddings_parquet_path(parquet_path: str | Path, base_dir: Path) -> Path:
    path = Path(parquet_path)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend(
            [
                path,
                base_dir / path,
                base_dir / "data" / "bert_tokens_embedd" / path.name,
                base_dir / "data" / "processed_data" / "bert_tokens_embedd" / path.name,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked_paths = "\n".join(f"- {p}" for p in candidates)
    raise FileNotFoundError(
        f"Could not find embeddings parquet.\nChecked:\n{checked_paths}"
    )


def _select_decision_config(
    label_series: pd.Series, case_type: str | None
) -> Tuple[Dict[str, str], Dict[str, str]]:
    if case_type in {"dv", "boc"}:
        schema = get_case_schema(case_type)
        return schema.decision_labels, schema.decision_colours

    values = label_series.astype(str).str.strip().str.upper()
    binary_keys = {k.upper() for k in DECISION_LABELS_BINARY}
    binary_values = {v.upper() for v in DECISION_LABELS_BINARY.values()}
    ternary_keys = {k.upper() for k in DECISION_LABELS_TERNARY}
    ternary_values = {v.upper() for v in DECISION_LABELS_TERNARY.values()}

    binary_hits = values.isin(binary_keys | binary_values).sum()
    ternary_hits = values.isin(ternary_keys | ternary_values).sum()

    if ternary_hits > binary_hits:
        return DECISION_LABELS_TERNARY, DECISION_COLOURS_TERNARY
    return DECISION_LABELS_BINARY, DECISION_COLOURS_BINARY


def plotly_umap_3d(
    parquet_path: str | Path,
    base_dir: Path | None = None,
    case_type: str | None = None,
    sample_size: int | None = 5000,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    point_size: int = 3,
    label_col: str = "class_label",
    hover_cols: List[str] | None = None,
    random_state: int = 42,
    save_html: bool = True,
    html_path: str | Path | None = None,
    wrap_width: int = 120,
    max_chars: int = 600,
    show_figure: bool = True,
    title: str | None = None,
) -> Tuple[pd.DataFrame, object]:
    """
    Interactive 3D UMAP for BERT token embeddings parquet files.

    The input parquet can be an absolute path, project-relative path, or just filename.
    Filenames are resolved against:
    - data/bert_tokens_embedd/
    - data/processed_data/bert_tokens_embedd/
    """
    import ast
    import textwrap

    import plotly.express as px
    import plotly.io as pio
    import umap

    base_dir = base_dir or get_project_root()
    resolved_case_type = (
        _normalise_case_type(case_type) if case_type is not None else None
    )
    resolved_parquet_path = _resolve_embeddings_parquet_path(
        parquet_path, base_dir=base_dir
    )
    if resolved_case_type is None:
        resolved_case_type = _infer_case_type_from_path(resolved_parquet_path)

    df = pd.read_parquet(resolved_parquet_path).copy().reset_index(drop=True)
    if "embedding" not in df.columns:
        raise KeyError("Column 'embedding' not found in embeddings parquet.")

    if sample_size is not None and len(df) > int(sample_size):
        df = df.sample(int(sample_size), random_state=int(random_state)).reset_index(
            drop=True
        )

    def _to_array(val: object) -> np.ndarray:
        if isinstance(val, str):
            try:
                arr = ast.literal_eval(val)
            except Exception:
                arr = ast.literal_eval(val.replace("nan", "None"))
            return np.asarray(arr, dtype=float)
        return np.asarray(val, dtype=float)

    try:
        df["embedding"] = df["embedding"].apply(_to_array)
        X = np.vstack(df["embedding"].values).astype(float)
    except Exception as exc:
        raise RuntimeError(
            "Error converting embeddings to matrix: " + str(exc)
        ) from exc

    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        random_state=int(random_state),
        metric="cosine",
    )
    print("Used Cosine distance for UMAP.")
    coords = reducer.fit_transform(X)
    df["umap_0"], df["umap_1"], df["umap_2"] = coords[:, 0], coords[:, 1], coords[:, 2]

    text_col = None
    if "text" in df.columns:
        text_col = "text"
    elif "texto_integral_sem_decisao" in df.columns:
        text_col = "texto_integral_sem_decisao"
    elif "word" in df.columns:
        text_col = "word"

    if text_col and text_col != "word":

        def _wrap_text(text_value: object) -> str:
            text = str(text_value)
            if len(text) > int(max_chars):
                text = text[: int(max_chars)].rstrip() + "..."
            return "<br>".join(textwrap.wrap(text, width=int(wrap_width)))

        df["text_wrapped"] = df[text_col].apply(_wrap_text)

    color_col = None
    color_discrete_map = None
    if label_col in df.columns:
        decision_labels, decision_colours = _select_decision_config(
            df[label_col], resolved_case_type
        )
        raw_labels = df[label_col].astype(str).str.strip()
        decision_map_upper = {k.upper(): v for k, v in decision_labels.items()}
        df["decision_label_plot"] = (
            raw_labels.str.upper().map(decision_map_upper).fillna(raw_labels)
        )
        color_col = "decision_label_plot"
        color_discrete_map = decision_colours
    elif resolved_case_type in CASE_LABELS:
        df["case_label_plot"] = CASE_LABELS[resolved_case_type]
        color_col = "case_label_plot"
        color_discrete_map = {
            CASE_LABELS[resolved_case_type]: CASE_COLOURS[resolved_case_type]
        }

    if hover_cols is None:
        hover_cols = []
        if "text_wrapped" in df.columns:
            hover_cols.append("text_wrapped")
        elif text_col:
            hover_cols.append(text_col)
        if color_col and color_col not in hover_cols:
            hover_cols.append(color_col)
    else:
        hover_cols = [
            (
                "text_wrapped"
                if c in {"text", "texto_integral_sem_decisao"}
                and "text_wrapped" in df.columns
                else c
            )
            for c in hover_cols
        ]

    hover_data = {c: True for c in hover_cols if c in df.columns}
    plot_labels = {"text_wrapped": "text"}

    fig = px.scatter_3d(
        df,
        x="umap_0",
        y="umap_1",
        z="umap_2",
        color=color_col,
        hover_data=hover_data,
        opacity=0.8,
        width=1200,
        height=800,
        color_discrete_map=color_discrete_map,
        labels=plot_labels,
    )
    fig.update_traces(marker=dict(size=point_size), selector=dict(mode="markers"))
    fig.update_layout(
        hoverlabel=dict(namelength=-1, align="left"), legend=dict(itemsizing="constant")
    )

    title_label = (
        CASE_LABELS[resolved_case_type]
        if resolved_case_type in CASE_LABELS
        else resolved_parquet_path.stem
    )
    if title  == None:
        fig.update_layout(title=f"3D UMAP - {title_label} embeddings ({len(df)} cases)")
    else:
        fig.update_layout(title=title)

    if show_figure:
        fig.show()

    if save_html:
        if html_path is None:
            if resolved_case_type in CASE_FOLDERS:
                out_dir = ensure_output_dir(
                    base_dir, resolved_case_type, "semantic_embeddings"
                )
            else:
                out_dir = base_dir / "eda_viz" / "semantic_embeddings"
                out_dir.mkdir(parents=True, exist_ok=True)
            html_path_resolved = out_dir / f"umap3d_{resolved_parquet_path.stem}.html"
        else:
            html_path_resolved = Path(html_path)
            if not html_path_resolved.is_absolute():
                html_path_resolved = base_dir / html_path_resolved
            html_path_resolved.parent.mkdir(parents=True, exist_ok=True)
        pio.write_html(fig, file=html_path_resolved, auto_open=False)
        print(f"Saved interactive HTML to: {html_path_resolved.resolve()}")

    return df, fig


def plotly_umap_3d_combined(
    dv_parquet_path: str | Path,
    boc_parquet_path: str | Path,
    base_dir: Path | None = None,
    sample_size: int | None = 3000,
    n_neighbors: int = 30,
    min_dist: float = 0.5,
    point_size: int = 3,
    random_state: int = 42,
    save_html: bool = True,
    html_path: str | Path | None = None,
    wrap_width: int = 120,
    max_chars: int = 600,
    show_figure: bool = True,
    title: str | None = None,
) -> Tuple[pd.DataFrame, object]:
    """
    Interactive 3D UMAP combining both DV and BOC case embeddings in a single visualization.
    Each case type is colored distinctly (red for DV, blue for BOC).

    Args:
        dv_parquet_path: Path to DV embeddings parquet file
        boc_parquet_path: Path to BOC embeddings parquet file
        base_dir: Project root directory
        sample_size: Max samples per case type
        n_neighbors: UMAP neighborhood size
        min_dist: UMAP min distance
        point_size: Plotly marker size
        random_state: Random seed
        save_html: Whether to save HTML
        html_path: Custom output path for HTML
        wrap_width: Text wrapping width in hover
        max_chars: Max characters in hover text
        show_figure: Whether to display figure
        title: Custom plot title
    """
    import ast
    import textwrap

    import plotly.express as px
    import plotly.io as pio
    import umap

    base_dir = base_dir or get_project_root()

    def _to_array(val: object) -> np.ndarray:
        if isinstance(val, str):
            try:
                arr = ast.literal_eval(val)
            except Exception:
                arr = ast.literal_eval(val.replace("nan", "None"))
            return np.asarray(arr, dtype=float)
        return np.asarray(val, dtype=float)

    # Load and prepare DV data
    dv_path = _resolve_embeddings_parquet_path(dv_parquet_path, base_dir=base_dir)
    df_dv = pd.read_parquet(dv_path).copy().reset_index(drop=True)
    if "embedding" not in df_dv.columns:
        raise KeyError("Column 'embedding' not found in DV embeddings parquet.")
    if sample_size is not None and len(df_dv) > int(sample_size):
        df_dv = df_dv.sample(int(sample_size), random_state=int(random_state)).reset_index(drop=True)
    df_dv["case_type_label"] = "Domestic Violence"

    # Load and prepare BOC data
    boc_path = _resolve_embeddings_parquet_path(boc_parquet_path, base_dir=base_dir)
    df_boc = pd.read_parquet(boc_path).copy().reset_index(drop=True)
    if "embedding" not in df_boc.columns:
        raise KeyError("Column 'embedding' not found in BOC embeddings parquet.")
    if sample_size is not None and len(df_boc) > int(sample_size):
        df_boc = df_boc.sample(int(sample_size), random_state=int(random_state)).reset_index(drop=True)
    df_boc["case_type_label"] = "Breach of Contract"

    # Combine datasets
    df = pd.concat([df_dv, df_boc], ignore_index=True)
    print(f"Combined dataset: {len(df_dv)} DV cases + {len(df_boc)} BOC cases = {len(df)} total")

    # Convert embeddings to array and create UMAP
    try:
        df["embedding"] = df["embedding"].apply(_to_array)
        X = np.vstack(df["embedding"].values).astype(float)
    except Exception as exc:
        raise RuntimeError("Error converting embeddings to matrix: " + str(exc)) from exc

    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        random_state=int(random_state),
        metric="cosine",
    )
    print("Used Cosine distance for UMAP.")
    coords = reducer.fit_transform(X)
    df["umap_0"], df["umap_1"], df["umap_2"] = coords[:, 0], coords[:, 1], coords[:, 2]

    # Wrap text for hover if available
    text_col = None
    if "text" in df.columns:
        text_col = "text"
    elif "texto_integral_sem_decisao" in df.columns:
        text_col = "texto_integral_sem_decisao"
    elif "word" in df.columns:
        text_col = "word"

    if text_col and text_col != "word":
        def _wrap_text(text_value: object) -> str:
            text = str(text_value)
            if len(text) > int(max_chars):
                text = text[: int(max_chars)].rstrip() + "..."
            return "<br>".join(textwrap.wrap(text, width=int(wrap_width)))

        df["text_wrapped"] = df[text_col].apply(_wrap_text)

    # Set up colors: red for DV, blue for BOC
    color_discrete_map = {
        "Domestic Violence": CASE_COLOURS["dv"],  # red
        "Breach of Contract": CASE_COLOURS["boc"],  # blue
    }

    # Prepare hover data
    hover_cols = []
    if "text_wrapped" in df.columns:
        hover_cols.append("text_wrapped")
    elif text_col:
        hover_cols.append(text_col)
    hover_cols.append("case_type_label")

    hover_data = {c: True for c in hover_cols if c in df.columns}
    plot_labels = {"text_wrapped": "text"}

    # Create scatter plot
    fig = px.scatter_3d(
        df,
        x="umap_0",
        y="umap_1",
        z="umap_2",
        color="case_type_label",
        hover_data=hover_data,
        opacity=0.7,
        width=1200,
        height=800,
        color_discrete_map=color_discrete_map,
        labels=plot_labels,
    )
    fig.update_traces(marker=dict(size=point_size), selector=dict(mode="markers"))
    fig.update_layout(
        hoverlabel=dict(namelength=-1, align="left"),
        legend=dict(itemsizing="constant", title_text="Case Type"),
    )

    # Set title
    if title is None:
        fig.update_layout(title=f"3D UMAP - Combined DV & BOC Embeddings ({len(df)} cases)")
    else:
        fig.update_layout(title=title)

    if show_figure:
        fig.show()

    # Save HTML
    if save_html:
        if html_path is None:
            out_dir = base_dir / "eda_viz" / "semantic_embeddings"
            out_dir.mkdir(parents=True, exist_ok=True)
            html_path_resolved = out_dir / "umap3d_combined_dv_boc.html"
        else:
            html_path_resolved = Path(html_path)
            if not html_path_resolved.is_absolute():
                html_path_resolved = base_dir / html_path_resolved
            html_path_resolved.parent.mkdir(parents=True, exist_ok=True)
        pio.write_html(fig, file=html_path_resolved, auto_open=False)
        print(f"Saved interactive HTML to: {html_path_resolved.resolve()}")

    return df, fig


def run_case_eda(case_type: str, base_dir: Path | None = None) -> None:
    base_dir = base_dir or get_project_root()
    schema = get_case_schema(case_type)

    df = load_eda_dataframe(case_type, base_dir=base_dir)
    df = normalise_decision_labels(df, schema.decision_column, schema.decision_labels)
    df = df.dropna(subset=["decision_label"]).copy()

    class_col = "decision_label"
    class_colours = schema.decision_colours

    sns.set_theme(style="whitegrid")

    plot_cases_per_year(df, case_type, base_dir)
    plot_year_distribution_by_class(df, case_type, class_col, class_colours, base_dir)
    plot_temporal_drift(df, case_type, class_col, class_colours, base_dir)
    plot_class_distribution(df, case_type, class_col, class_colours, base_dir)

    df = add_text_metrics(df, text_column="texto_integral_sem_decisao")
    plot_text_length_by_class(df, case_type, class_col, class_colours, base_dir)

    df = add_tokens_no_stopwords(df)
    plot_word_frequency_per_class(df, case_type, class_col, class_colours, base_dir)
    plot_ngram_frequency_per_class(
        df, case_type, class_col, class_colours, base_dir, n=2
    )
    plot_ngram_frequency_per_class(
        df, case_type, class_col, class_colours, base_dir, n=3
    )
    plot_tfidf_per_class(
        df, case_type, class_col, class_colours, base_dir, ngram_range=(2, 2)
    )
    plot_tfidf_per_class(
        df, case_type, class_col, class_colours, base_dir, ngram_range=(3, 3)
    )

    plot_decisions_by_tribunal(df, case_type, class_col, class_colours, base_dir)

    if "juiz_relator" in df.columns:
        df_gender = classify_judge_gender(df)
        plot_decisions_by_judge_gender(
            df_gender, case_type, class_col, class_colours, base_dir
        )
