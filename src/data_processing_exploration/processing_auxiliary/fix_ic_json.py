import json
from pathlib import Path


def limpar_tribunal_jp(json_file):
    """Remove o sufixo '_DECISÃO_TEXTO_INTEGRAL' do campo 'tribunal' em cada registo do JSON.

    Carrega o ficheiro JSON, limpa o campo 'tribunal' de cada item (quando
    presente e não vazio) e grava o resultado de volta no mesmo ficheiro.

    Args:
        json_file (str or Path): Caminho para o ficheiro JSON a corrigir.
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        if "tribunal" in item and item["tribunal"]:
            item["tribunal"] = item["tribunal"].replace("_DECISÃO_TEXTO_INTEGRAL", "")

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ Campo 'tribunal' limpo em {len(data)} registos")


# Usar
limpar_tribunal_jp(
    Path("./../../data/dgsi_incumprimento_contratos_20251028_130853.json")
)
