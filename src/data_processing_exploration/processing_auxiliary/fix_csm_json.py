import json
import re
from pathlib import Path


def extrair_tribunal_csm_json(json_file):
    """Extrai e atualiza o campo 'tribunal' diretamente no JSON do CSM.

    Procura pelos padrões de Tribunais da Relação e STJ no texto integral
    (cabeçalho e rodapé de cada caso), com fallback para deteção por nome de
    cidade quando o padrão textual não é encontrado. Apenas processa itens
    cujo campo 'tribunal' seja 'CSM', None ou vazio; grava o JSON atualizado
    de volta no mesmo ficheiro.

    Args:
        json_file (str or Path): Caminho para o ficheiro JSON do CSM.

    Returns:
        tuple[int, int]: (casos_atualizados, casos_nao_identificados) — número
            de casos aos quais foi atribuído um tribunal e número de casos em
            que o tribunal não pôde ser identificado.
    """
    # Padrões de tribunais
    cidades_tribunais = {
        "lisboa": "TRL",
        "porto": "TRP",
        "coimbra": "TRC",
        "évora": "TRE",
        "guimarães": "TRG",
    }

    tribunal_patterns = [
        r"tribunal\s+da\s+relação\s+d[eoa]\s+([a-záàâãçéêíóôõú]+)",
        r"acordam\s+(?:os\s+)?ju[íi]zes\s+d[ao]\s+tribunal\s+d[oa]\s+relação\s+d[eoa]\s+([a-záàâãçéêíóôõú]+)",
        r"acordam\s+no\s+tribunal\s+da\s+relação\s+d[eoa]\s+([a-záàâãçéêíóôõú]+)",
        r"no\s+tribunal\s+da\s+relação\s+d[eoa]\s+([a-záàâãçéêíóôõú]+)",
        r"relação\s+d[eoa]\s+([a-záàâãçéêíóôõú]+)",
        r"supremo\s+tribunal\s+de\s+justiça",
        r"acordam\s+(?:os\s+)?ju[íi]zes\s+do\s+supremo\s+tribunal\s+de\s+justiça",
        r"acordam\s+no\s+supremo\s+tribunal\s+de\s+justiça",
        r"no\s+supremo\s+tribunal\s+de\s+justiça",
    ]

    def extract_tribunal(text_section):
        """Extrai o código do tribunal (TR/STJ) de uma secção de texto."""
        for pattern in tribunal_patterns:
            match = re.search(pattern, text_section, re.IGNORECASE)
            if match:
                matched_text = match.group(0).lower()
                if (
                    "supremo tribunal de justiça" in matched_text
                    or "stj" in matched_text
                ):
                    return "STJ"
                elif "relação" in matched_text:
                    city = (
                        match.group(1).strip()
                        if match.groups() and match.group(1)
                        else ""
                    )
                    city = city.lower()
                    if "lisboa" in city:
                        return "TRL"
                    elif "porto" in city:
                        return "TRP"
                    elif "coimbra" in city:
                        return "TRC"
                    elif "évora" in city:
                        return "TRE"
                    elif "guimarães" in city:
                        return "TRG"
        return None

    def extract_tribunal_by_city(text_section):
        """Fallback: procura o nome da cidade na secção de texto."""
        text_lower = text_section.lower()
        for cidade, codigo in cidades_tribunais.items():
            if cidade in text_lower:
                return codigo
        return None

    # Carregar JSON
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    casos_atualizados = 0
    casos_nao_identificados = 0

    # Processar cada caso
    for item in data:
        # Só processa se tribunal for 'CSM' ou None/vazio
        if item.get("tribunal") in ["CSM", None, ""]:
            texto = item.get("texto_integral_completo", "")

            if texto:
                # Extrai header (primeiros 30% do texto)
                header_len = max(1, int(len(texto) * 0.30))
                header_text = texto[:header_len]
                footer_text = texto[-header_len:] if len(texto) > header_len else ""

                # Tenta extrair tribunal
                tribunal = extract_tribunal(header_text)

                # Fallback: procura por cidade
                if not tribunal:
                    tribunal = extract_tribunal_by_city(header_text)

                if not tribunal:
                    tribunal = extract_tribunal_by_city(footer_text)
                # Atualiza se encontrou
                if tribunal:
                    item["tribunal"] = tribunal
                    casos_atualizados += 1
                else:
                    casos_nao_identificados += 1

    # Guardar JSON atualizado
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ Ficheiro atualizado: {json_file}")
    print(f"   📊 {casos_atualizados} casos com tribunal identificado")
    print(f"   ⚠️  {casos_nao_identificados} casos sem identificação clara")

    return casos_atualizados, casos_nao_identificados


# Usar
extrair_tribunal_csm_json(
    Path("./../../data/csm_violencia_domestica_20251029_131345.json")
)
