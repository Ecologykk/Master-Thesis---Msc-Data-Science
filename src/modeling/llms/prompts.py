"""Prompt template factory for all LLM pipeline stages.

The model receives the full judgment text from `texto_integral_sem_decisao`,
which contains the Relatório (case report) and Fundamentação (legal reasoning,
applicable laws, past jurisprudence) — the dispositivo (final ruling) is already
excluded from the column, so there is no label leakage.

Exported builders
-----------------
build_zero_shot_prompt(case_type, n_processo, text) -> list[dict]
build_few_shot_prompt(case_type, n_processo, text, examples) -> list[dict]
build_cot_prompt(case_type, n_processo, text) -> list[dict]
build_few_shot_cot_prompt(case_type, n_processo, text, examples) -> list[dict]

All builders return an OpenAI-style message list ready for OllamaClient.chat().
Each system message starts with the judge persona, followed by the TIDD-EC
task/instruction block.
"""

# ---------------------------------------------------------------------------
# System prompts  (persona first, then TIDD-EC task block)
# ---------------------------------------------------------------------------

_SYSTEM_DV = (
    "És um Juiz Conselheiro português com mais de 30 anos de experiência "
    "em tribunais de segunda instância, especializado em direito penal e, em "
    "particular, em recursos de decisões em casos de violência doméstica. "
    "Ao longo da tua carreira apreciaste centenas de recursos nesta matéria e "
    "conheces profundamente a doutrina, a jurisprudência dos tribunais superiores "
    "e os critérios que determinam a manutenção ou alteração de uma condenação em sede de recurso."
    "\n\n"
    "Tarefa: Classificação do resultado de recursos em casos de violência doméstica "
    "em tribunais portugueses de segunda instância."
    "\n\n"
    "Instruções:\n"
    "1. Lê o Relatório para identificar o contexto do caso: partes envolvidas, "
    "conduta alegada, decisão do tribunal de 1.ª instância e fundamentos do recurso interposto.\n"
    "2. Lê a Fundamentação para compreender a análise jurídica do tribunal de recurso: "
    "factos provados e não provados, legislação aplicável e jurisprudência citada.\n"
    "3. Determina se o tribunal de recurso manteve ou alterou a decisão de primeira "
    "instância com base na argumentação jurídica presente na Fundamentação.\n"
    "4. Emite exclusivamente o JSON de classificação no formato indicado."
    "\n\n"
    "DEVES:\n"
    '- Usar exatamente uma das labels válidas no campo "predicted_label"\n'
    "- Basear a classificação apenas no texto fornecido\n"
    "- Emitir JSON sintaticamente válido como única e exclusiva resposta"
    "\n\n"
    "NÃO DEVES:\n"
    "- Adicionar texto, comentários, explicações ou raciocínio fora do JSON\n"
    "- Usar labels não listadas, nem variações ortográficas ou traduções\n"
    "- Inferir o dispositivo — não está no texto e não te é dado acesso a ele\n"
    "- Recusar classificar ou pedir mais informação — classifica sempre com base no texto disponível"
    "\n\n"
    "Classes válidas:\n"
    '- "decisao_mantida": o tribunal de recurso confirma e mantém a decisão do tribunal '
    "de 1.ª instância — o recorrente não obtém qualquer alteração favorável\n"
    '- "decisao_alterada": o tribunal de recurso revoga, modifica ou substitui a decisão '
    "do tribunal de 1.ª instância, parcial ou totalmente"
    "\n\n"
    "Contexto:\n"
    "- Tribunal: segunda instância portuguesa (TRC, TRP, TRL ou STJ)\n"
    "- Tipo de caso: violência doméstica (tipicamente art. 152.º CP)\n"
    "- O texto contém o Relatório e a Fundamentação. O dispositivo foi removido — "
    "o teu papel é inferi-lo a partir da análise jurídica presente na Fundamentação."
    "\n\n"
    "Formato de resposta obrigatório (sem texto antes ou depois):\n"
    '{"predicted_label": "<LABEL>"}'
)

_SYSTEM_BOC = (
    "És um Juiz Conselheiro português com mais de 30 anos de experiência "
    "em tribunais de segunda instância e julgados de paz, especializado em direito civil e, em "
    "particular, em recursos de decisões em litígios de incumprimento contratual. "
    "Ao longo da tua carreira apreciaste centenas de recursos nesta matéria e "
    "conheces profundamente a doutrina, o regime geral das obrigações e a "
    "jurisprudência aplicável aos contratos em direito português."
    "\n\n"
    "Tarefa: Classificação do resultado de recursos em casos de incumprimento contratual "
    "em tribunais portugueses de segunda instância."
    "\n\n"
    "Instruções:\n"
    "1. Lê o Relatório para identificar o contexto do caso: partes envolvidas, "
    "objeto do contrato, incumprimento alegado, decisão do tribunal de 1.ª instância "
    "e fundamentos do recurso interposto.\n"
    "2. Lê a Fundamentação para compreender a análise jurídica do tribunal de recurso: "
    "factos provados e não provados, legislação aplicável e jurisprudência citada.\n"
    "3. Determina o grau de procedência do recurso para o recorrente com base na "
    "argumentação jurídica presente na Fundamentação.\n"
    "4. Emite exclusivamente o JSON de classificação no formato indicado."
    "\n\n"
    "DEVES:\n"
    '- Usar exatamente uma das labels válidas no campo "predicted_label"\n'
    "- Basear a classificação apenas no texto fornecido\n"
    "- Emitir JSON sintaticamente válido como única e exclusiva resposta"
    "\n\n"
    "NÃO DEVES:\n"
    "- Adicionar texto, comentários, explicações ou raciocínio fora do JSON\n"
    "- Usar labels não listadas, nem variações ortográficas ou traduções\n"
    "- Inferir o dispositivo — não está no texto e não te é dado acesso a ele\n"
    "- Recusar classificar ou pedir mais informação — classifica sempre com base no texto disponível"
    "\n\n"
    "Classes válidas (do ponto de vista do recorrente):\n"
    '- "decisao_desfavoravel": o recurso é julgado totalmente improcedente — '
    "o tribunal de recurso não altera nenhum ponto da decisão recorrida em favor do recorrente\n"
    '- "decisao_parcial": o recurso é julgado parcialmente procedente — '
    "o tribunal altera alguns pontos da decisão recorrida, mas não acolhe todos os pedidos do recorrente\n"
    '- "decisao_favoravel": o recurso é julgado totalmente procedente — '
    "o tribunal acolhe integralmente os fundamentos do recorrente e altera a decisão recorrida"
    "\n\n"
    "Contexto:\n"
    "- Tribunal: segunda instância portuguesa (TRC, TRP, TRL ou STJ)\n"
    "- Tipo de caso: incumprimento contratual (breach of contract)\n"
    "- O texto contém o Relatório e a Fundamentação. O dispositivo foi removido — "
    "o teu papel é inferi-lo a partir da análise jurídica presente na Fundamentação."
    "\n\n"
    "Formato de resposta obrigatório (sem texto antes ou depois):\n"
    '{"predicted_label": "<LABEL>"}'
)

_SYSTEM_PROMPTS: dict[str, str] = {
    "dv": _SYSTEM_DV,
    "boc": _SYSTEM_BOC,
}

# ---------------------------------------------------------------------------
# CoT reasoning instruction  (Step-Back + Structured CoT hybrid)
# Appended to the system prompt for CoT stages.
# ---------------------------------------------------------------------------

_COT_INSTRUCTION = (
    "\n\n"
    "Antes de emitires o JSON de classificação, raciocina por etapas seguindo "
    "obrigatoriamente esta estrutura:"
    "\n\n"
    "QUESTÃO JURÍDICA CENTRAL:\n"
    "Antes de analisar os detalhes do caso, identifica em 1 a 2 frases qual é o "
    "principal fundamento jurídico do recurso e qual a norma ou princípio de direito "
    "que o tribunal de recurso tem de apreciar para o decidir."
    "\n\n"
    "PASSO 1 — FACTOS:\n"
    "Identifica os factos essenciais: partes envolvidas, conduta alegada, decisão do "
    "tribunal a quo, e fundamentos invocados pelo recorrente."
    "\n\n"
    "PASSO 2 — LEGISLAÇÃO:\n"
    "Lista os artigos de lei citados na Fundamentação e explica sucintamente o papel "
    "de cada um na apreciação do recurso."
    "\n\n"
    "PASSO 3 — JURISPRUDÊNCIA:\n"
    "Lista os acórdãos ou decisões jurisprudenciais citados (STJ, TRC, TRP, TRL, etc.) "
    "e sintetiza em que sentido são invocados. Se não existir jurisprudência citada, "
    'indica explicitamente "Sem jurisprudência citada."'
    "\n\n"
    "PASSO 4 — APRECIAÇÃO:\n"
    "Com base nos factos, legislação e jurisprudência identificados, avalia se os "
    "fundamentos do recurso são juridicamente sustentados pela Fundamentação. "
    "Conclui com o sentido mais provável da decisão e justifica brevemente."
    "\n\n"
    "PASSO 5 — CLASSIFICAÇÃO:\n"
    "Indica a label que melhor corresponde ao resultado e emite o JSON na linha seguinte, "
    "sem qualquer texto adicional depois:\n"
    '{"predicted_label": "<LABEL>"}'
)

# ---------------------------------------------------------------------------
# User message builders
# ---------------------------------------------------------------------------


def _user_message(n_processo: str, text: str, *, smart_truncated: bool = False) -> str:
    """Build the user-turn text: process id, optional truncation note, and case text."""
    trunc_note = ""
    if smart_truncated:
        trunc_note = (
            "Nota de contexto: este texto foi automaticamente truncado de forma inteligente "
            "a partir do documento integral, selecionando frases mais relevantes por ranking "
            "semântico e padrões linguísticos de fundamentação jurídica. "
            "A ordem apresentada mantém a ordem original dessas frases no acórdão.\n\n"
        )
    return f"Processo: {n_processo}\n\n" f"{trunc_note}" f"Texto da decisão:\n{text}"


def _example_text(example: dict) -> str:
    """Return the display text for a few-shot example, preferring pre-selected sentences over raw text."""
    selected = example.get("selected_sentences_readable") or example.get(
        "selected_sentences_ranked_ordered"
    )
    if isinstance(selected, list) and selected:
        return "\n".join(f"- {sent}" for sent in selected)
    return str(example.get("text", ""))


def _few_shot_block(examples: list[dict]) -> str:
    """Format few-shot examples as a prompt block."""
    lines = [f"Aqui estão {len(examples)} exemplos de classificações corretas:\n"]
    for i, ex in enumerate(examples, 1):
        lines.append(f"[Exemplo {i}]")
        lines.append(f"Processo: {ex['n_processo']}")
        lines.append(f"Texto da decisão:\n{_example_text(ex)}")
        lines.append(f'Classificação: {{"predicted_label": "{ex["label"]}"}}')
        lines.append("")
    lines.append("Agora classifica este processo:")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public builders — return OpenAI-style message list
# ---------------------------------------------------------------------------


def build_zero_shot_prompt(
    case_type: str,
    n_processo: str,
    text: str,
    *,
    smart_truncated: bool = False,
) -> list[dict]:
    """Zero-shot: judge persona + task instructions in system, case text in user.

    Args:
        case_type: Either "dv" (domestic violence) or "boc" (breach of
            contract); selects which system prompt/label set is used.
        n_processo: Case process number, included in the user message.
        text: Case judgment text (Relatório + Fundamentação, dispositivo
            already excluded).
        smart_truncated: If True, prepends a note to the user message stating
            that `text` was automatically truncated to the most relevant
            sentences rather than being the full document.

    Returns:
        OpenAI-style message list `[{"role": "system", ...}, {"role": "user", ...}]`
        ready to pass to `OllamaClient.chat`.
    """
    return [
        {"role": "system", "content": _SYSTEM_PROMPTS[case_type]},
        {
            "role": "user",
            "content": _user_message(n_processo, text, smart_truncated=smart_truncated),
        },
    ]


def build_few_shot_prompt(
    case_type: str,
    n_processo: str,
    text: str,
    examples: list[dict],
    *,
    smart_truncated: bool = False,
) -> list[dict]:
    """Few-shot: judge persona + task in system, examples + target case in user.

    Args:
        case_type: Either "dv" or "boc"; selects the system prompt/label set.
        n_processo: Case process number, included in the user message.
        text: Case judgment text of the case to classify.
        examples: List of few-shot example dicts, each with keys
            `"n_processo"`, `"label"`, and text under either
            `"selected_sentences_readable"`, `"selected_sentences_ranked_ordered"`,
            or `"text"` (see `_example_text`).
        smart_truncated: If True, notes in the user message that `text` was
            automatically truncated to the most relevant sentences.

    Returns:
        OpenAI-style message list with the formatted examples block followed
        by the target case prepended to the user content.
    """
    user_content = (
        _few_shot_block(examples)
        + "\n"
        + _user_message(
            n_processo,
            text,
            smart_truncated=smart_truncated,
        )
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPTS[case_type]},
        {"role": "user", "content": user_content},
    ]


def build_cot_prompt(
    case_type: str,
    n_processo: str,
    text: str,
    *,
    smart_truncated: bool = False,
) -> list[dict]:
    """CoT: judge persona + task + Step-Back/CoT reasoning instruction in system.

    Args:
        case_type: Either "dv" or "boc"; selects the system prompt/label set.
        n_processo: Case process number, included in the user message.
        text: Case judgment text to classify.
        smart_truncated: If True, notes in the user message that `text` was
            automatically truncated to the most relevant sentences.

    Returns:
        OpenAI-style message list whose system message appends the
        `_COT_INSTRUCTION` (Step-Back + structured CoT) block after the base
        system prompt, instructing the model to reason step-by-step before
        emitting the final classification JSON.
    """
    system = _SYSTEM_PROMPTS[case_type] + _COT_INSTRUCTION
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": _user_message(n_processo, text, smart_truncated=smart_truncated),
        },
    ]


def build_few_shot_cot_prompt(
    case_type: str,
    n_processo: str,
    text: str,
    examples: list[dict],
    *,
    smart_truncated: bool = False,
) -> list[dict]:
    """Few-shot + CoT: judge persona + task + reasoning instruction in system.

    Args:
        case_type: Either "dv" or "boc"; selects the system prompt/label set.
        n_processo: Case process number, included in the user message.
        text: Case judgment text of the case to classify.
        examples: List of few-shot example dicts (see `build_few_shot_prompt`).
        smart_truncated: If True, notes in the user message that `text` was
            automatically truncated to the most relevant sentences.

    Returns:
        OpenAI-style message list combining the CoT-augmented system prompt
        with a user message containing the few-shot examples block followed
        by the target case.
    """
    system = _SYSTEM_PROMPTS[case_type] + _COT_INSTRUCTION
    user_content = (
        _few_shot_block(examples)
        + "\n"
        + _user_message(
            n_processo,
            text,
            smart_truncated=smart_truncated,
        )
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Quick visual inspection
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_text = (
        "O recorrente foi condenado em primeira instância por crime de violência doméstica. "
        "A defesa alega que os factos não foram devidamente ponderados pelo tribunal a quo. "
        "O Ministério Público pronunciou-se pela improcedência do recurso."
    )

    print("=" * 70)
    print("ZERO-SHOT (DV)")
    print("=" * 70)
    msgs = build_zero_shot_prompt("dv", "TEST-001", sample_text)
    for m in msgs:
        print(f"\n[{m['role'].upper()}]\n{m['content']}")

    print("\n" + "=" * 70)
    print("COT (BOC)")
    print("=" * 70)
    msgs = build_cot_prompt("boc", "TEST-002", sample_text)
    for m in msgs:
        print(f"\n[{m['role'].upper()}]\n{m['content']}")

    sample_examples = [
        {
            "n_processo": "EX-001",
            "text": "Exemplo de texto A.",
            "label": "decisao_mantida",
        },
        {
            "n_processo": "EX-002",
            "text": "Exemplo de texto B.",
            "label": "decisao_alterada",
        },
    ]
    print("\n" + "=" * 70)
    print("FEW-SHOT (DV)")
    print("=" * 70)
    msgs = build_few_shot_prompt("dv", "TEST-003", sample_text, sample_examples)
    for m in msgs:
        print(f"\n[{m['role'].upper()}]\n{m['content']}")
