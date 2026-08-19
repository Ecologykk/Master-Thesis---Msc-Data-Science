"""
config.py
=========

Central configuration for the LLM pipeline.

All module-level constants live here — model tags, label mappings,
output directory layout, and Ollama connection settings.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# src/modeling/llms/config.py → parents: llms(0) modeling(1) src(2) repo_root(3)
_REPO_ROOT = Path(__file__).resolve().parents[3]

LLM_OUTPUT_DIR: Path = _REPO_ROOT / "data" / "models" / "llms"
SHAP_BASE_DIR: Path = _REPO_ROOT / "shap_explanations"

# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Short name → Ollama model tag (as shown by `ollama list`)
OLLAMA_MODELS: dict[str, str] = {
    "deepseek_r1_8b": "deepseek-r1:8b",
    "llama_3_8b":     "llama3.1:8b",
    "ministral_3b":   "mistral:7b",
}

# ---------------------------------------------------------------------------
# Label mappings
# ---------------------------------------------------------------------------

# Human-readable uppercase labels (from the CSV) → integer class index
# Mirrors features.LABEL_MAPPINGS exactly.
LABEL_STR_TO_INT: dict[str, dict[str, int]] = {
    "dv": {
        "DECISÃO MANTIDA":  0,
        "DECISÃO ALTERADA": 1,
    },
    "boc": {
        "DESFAVORÁVEL": 0,
        "PARCIAL":       1,
        "FAVORÁVEL":     2,
    },
}

# LLM output label strings (what the model writes in JSON) → integer class index
LABEL_OUTPUT_TO_INT: dict[str, dict[str, int]] = {
    "dv": {
        "decisao_mantida":  0,
        "decisao_alterada": 1,
    },
    "boc": {
        "decisao_desfavoravel": 0,
        "decisao_parcial":       1,
        "decisao_favoravel":     2,
    },
}

# Ordered list of LLM output label strings per case type
# (used as output_names for SHAP and parquet proba columns)
LABEL_OUTPUT_NAMES: dict[str, list[str]] = {
    "dv":  ["decisao_mantida", "decisao_alterada"],
    "boc": ["decisao_desfavoravel", "decisao_parcial", "decisao_favoravel"],
}

# CSV input labels (from decisao_binaria/decisao_ternaria columns) → canonical output labels
# Used to canonicalize few-shot examples before prompt assembly
LABEL_CANONICALIZE: dict[str, dict[str, str]] = {
    "dv": {
        "DECISÃO MANTIDA":  "decisao_mantida",
        "DECISÃO ALTERADA": "decisao_alterada",
    },
    "boc": {
        "DESFAVORÁVEL": "decisao_desfavoravel",
        "PARCIAL":       "decisao_parcial",
        "FAVORÁVEL":     "decisao_favoravel",
    },
}


def canonicalize_label(raw_label: str, case_type: str) -> str:
    """Convert CSV input label to canonical output label format.

    Examples:
        canonicalize_label("DECISÃO MANTIDA", "dv") → "decisao_mantida"
        canonicalize_label("PARCIAL", "boc") → "decisao_parcial"

    Raises ValueError if raw_label is not recognized for the given case_type.
    """
    canonical = LABEL_CANONICALIZE.get(case_type, {}).get(raw_label)
    if canonical is None:
        valid = set(LABEL_CANONICALIZE.get(case_type, {}).keys())
        raise ValueError(
            f"Unknown label '{raw_label}' for case_type='{case_type}'. "
            f"Valid labels: {sorted(valid)}"
        )
    return canonical

# SHAP subdirectory per case type (mirrors existing shap_explanations/ layout)
CASE_TYPE_DIRS: dict[str, str] = {
    "dv":  "domestic_violence",
    "boc": "breach_of_contract",
}

# ---------------------------------------------------------------------------
# Inference defaults
# ---------------------------------------------------------------------------


MAX_RETRIES: int = 3             # Ollama retry attempts on bad/unparseable JSON
SHAP_MAX_EVALS_DEFAULT: int = 50  # much lower than BERT (500) — each eval = 1 LLM call

# ---------------------------------------------------------------------------
# deterministic inference controls (classification-safe defaults)
# ---------------------------------------------------------------------------

# Fixed decoding defaults to minimise stochasticity across runs.
# Note: exact bit-level reproducibility may still depend on backend/hardware.
OLLAMA_SEED_DEFAULT: int = 42
OLLAMA_TEMPERATURE_DEFAULT: float = 0.0

# Keep probability mass concentrated on the top candidate.
# With temperature=0 and top_k=1, generation is near-greedy.
OLLAMA_TOP_P_DEFAULT: float = 1.0
OLLAMA_TOP_K_DEFAULT: int = 1

# Neutral repetition penalty for short JSON outputs.
OLLAMA_REPEAT_PENALTY_DEFAULT: float = 1.0

OLLAMA_DETERMINISTIC_OPTIONS: dict[str, int | float] = {
    "seed": OLLAMA_SEED_DEFAULT,
    "temperature": OLLAMA_TEMPERATURE_DEFAULT,
    "top_p": OLLAMA_TOP_P_DEFAULT,
    "top_k": OLLAMA_TOP_K_DEFAULT,
    "repeat_penalty": OLLAMA_REPEAT_PENALTY_DEFAULT,
}
