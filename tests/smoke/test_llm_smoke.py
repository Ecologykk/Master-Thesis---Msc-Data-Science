"""Smoke test for the LLM prediction pipeline: one zero-shot Ollama call on one synthetic case.

Skipped automatically when Ollama is unreachable, so the suite still passes on a machine without
it installed. To run this test for real, start Ollama and pull a model listed in
src/modeling/llms/config.py's OLLAMA_MODELS (e.g. `ollama pull mistral:7b`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
_LLMS_DIR = str(REPO_ROOT / "src" / "modeling" / "llms")
if _LLMS_DIR not in sys.path:
    # Appended, not inserted at position 0: src/modeling/dl (added by conftest.py) must stay
    # ahead of this in sys.path, since dl/predict.py and llms/predict.py share the module name
    # "predict" and other tests rely on `import predict` resolving to the dl one.
    sys.path.append(_LLMS_DIR)

from client import OllamaClient  # noqa: E402
from config import OLLAMA_BASE_URL, OLLAMA_MODELS  # noqa: E402
from prompts import build_zero_shot_prompt  # noqa: E402


def _first_available_model() -> str | None:
    """Return the first configured Ollama model tag that is actually pulled, or None.

    Checking reachability alone is not enough: the server can be up with no matching model
    pulled, which would otherwise surface as a hard failure rather than a clean skip.
    """
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        resp.raise_for_status()
    except requests.RequestException:
        return None
    pulled = {m["model"] for m in resp.json().get("models", [])}
    for tag in OLLAMA_MODELS.values():
        if tag in pulled:
            return tag
    return None


_AVAILABLE_MODEL = _first_available_model()


@pytest.mark.skipif(
    _AVAILABLE_MODEL is None,
    reason="Ollama unreachable or no configured model pulled (see OLLAMA_MODELS in config.py)",
)
def test_llm_predicts_one_case():
    case_text = (
        "O tribunal considera que o recurso interposto pelo arguido é julgado procedente, "
        "revogando-se a decisão recorrida."
    )
    messages = build_zero_shot_prompt(
        case_type="dv", n_processo="123/20.0TEST", text=case_text
    )

    client = OllamaClient(base_url=OLLAMA_BASE_URL)
    reply = client.chat(model=_AVAILABLE_MODEL, messages=messages, format="json")

    assert isinstance(reply, str)
    assert reply.strip() != ""
