from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Ramo(str, Enum):
    DV = "DV"
    IC = "IC"


@dataclass(frozen=True)
class Caso:
    ramo: Ramo
    n_processo: str
    url: str
    texto_integral_completo: str


@dataclass
class RespostaCaso:
    decisao: str = ""
    confianca: str = ""
    justificacao: str = ""


ROTULOS_RAMO = {
    Ramo.DV: "Violência Doméstica",
    Ramo.IC: "Incumprimento Contratual",
}

OPCOES_DECISAO = {
    Ramo.DV: [
        "Decisão Mantida",
        "Decisão Alterada",
    ],
    Ramo.IC: [
        "Decisão Favorável",
        "Decisão Desfavorável",
        "Decisão Parcial",
    ],
}

OPCOES_CONFIANCA = [
    "Confiante",
    "Não Confiante",
]
