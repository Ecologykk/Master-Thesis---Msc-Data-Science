from __future__ import annotations

from pathlib import Path

import pandas as pd

from models import Caso, Ramo

REQUIRED_COLUMNS = [
    "url",
    "n_processo",
    "texto_integral_completo",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _dataset_path(branch: Ramo) -> Path:
    file_name = (
        "dv_gold_test_full.csv" if branch == Ramo.DV else "boc_gold_test_full.csv"
    )
    return _project_root() / "data" / "processed_data" / "gold_test" / file_name


def _validate_columns(df: pd.DataFrame, csv_path: Path) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"O ficheiro {csv_path} não tem as colunas obrigatórias: {', '.join(missing)}"
        )


def load_cases(branch: Ramo) -> list[Caso]:
    csv_path = _dataset_path(branch)
    if not csv_path.exists():
        raise FileNotFoundError(f"Ficheiro de casos não encontrado: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8", dtype=str)
    _validate_columns(df, csv_path)
    df = df.fillna("")

    cases: list[Caso] = []
    for row in df.itertuples(index=False):
        n_processo = str(getattr(row, "n_processo", "")).strip()
        url = str(getattr(row, "url", "")).strip()
        texto = str(getattr(row, "texto_integral_completo", "")).strip()
        if not n_processo:
            continue
        cases.append(
            Caso(
                ramo=branch,
                n_processo=n_processo,
                url=url,
                texto_integral_completo=texto,
            )
        )
    return cases
