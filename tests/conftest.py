"""Shared pytest fixtures and path setup for the whole test suite.

Some modules under `src/modeling/` (a pre-existing part of the codebase, not touched by this
refactor) use flat, script-style imports (`from features import ...`) rather than package-relative
ones, and expect their own directory to be on `sys.path` — the same convention the exploratory
notebook uses (`sys.path.insert(0, str(base_dir / "src"))`). This conftest adds those directories
once, at session start, so smoke tests can import them normally.

Note: `src/modeling/dl/predict.py` and `src/modeling/llms/predict.py` share the module name
`predict`. Only `src/modeling/dl` is added here, so `import predict` always resolves to the BERT
one. Tests needing the LLM side import `client`/`config`/`prompts`/`few_shot_retrieval_similarity`
directly (no name collision) and add `src/modeling/llms` to `sys.path` themselves if needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

for _subpath in ("src/modeling/dl", "src/evaluation", "src/eda"):
    _abs = str(REPO_ROOT / _subpath)
    if _abs not in sys.path:
        sys.path.insert(0, _abs)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Absolute path to the tests/fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def sample_dv_raw_path() -> Path:
    """Path to a 20-case real DV fixture (long text fields truncated)."""
    return FIXTURES_DIR / "sample_dv_cases.json"


@pytest.fixture(scope="session")
def sample_boc_raw_path() -> Path:
    """Path to a 20-case real BoC fixture (long text fields truncated)."""
    return FIXTURES_DIR / "sample_boc_cases.json"
