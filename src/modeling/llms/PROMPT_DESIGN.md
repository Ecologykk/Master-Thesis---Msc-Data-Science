# LLM Prompt Design — System Prompts for Legal Judgment Prediction

This document records the design rationale, framework selection, and iterative
refinement of the system prompts used across all inference stages of the LLM
pipeline (`zero_shot`, `few_shot`, `cot`, `few_shot_cot`). It is intended as
supporting material for the thesis methodology chapter.

---

## 1. Context and Constraints

The LLM pipeline classifies Portuguese appellate court decisions using
open-source instruction-tuned models served via Ollama (primary model:
DeepSeek-R1-0528-Qwen3-8B, 8B parameters). Two classification tasks are
addressed:

| Task | Labels | Nature |
|------|--------|--------|
| Domestic Violence (DV) | `decisao_mantida`, `decisao_alterada` | Binary |
| Breach of Contract (BOC) | `decisao_desfavoravel`, `decisao_parcial`, `decisao_favoravel` | Ternary |

**Input text:** The column `texto_integral_sem_decisao` — the full judgment
text containing the *Relatório* (case report) and *Fundamentação* (legal
analysis of facts, applicable law, and cited jurisprudence). The *dispositivo*
(final ruling) is excluded at data-preparation time to prevent label leakage.
The model must infer the appellate outcome exclusively from the legal reasoning
present in the *Fundamentação*.

**Output format:** JSON with two fields — `n_processo` and `predicted_label` —
enforced by Ollama's `format: "json"` mode and validated in `client.py`.

---

## 2. Initial Prompt (Baseline — before refinement)

The baseline system prompt was a minimal role statement following an informal
RTF (Role–Task–Format) structure. Both DV and BOC shared the same pattern:

```
És um especialista em direito português com vasta experiência em tribunais de recurso.
A tua tarefa é analisar decisões de tribunais de recurso em casos de [X]
e classificar o resultado do recurso.

As classes possíveis são:
- "decisao_mantida": o tribunal de recurso mantém a decisão do tribunal de 1.ª instância
- "decisao_alterada": o tribunal de recurso altera ou revoga a decisão do tribunal de 1.ª instância

O texto que recebes contém o Relatório (contextualização do caso) e a Fundamentação
(análise jurídica dos factos, legislação aplicável e jurisprudência relevante).
A decisão final (dispositivo) não está incluída.

Responde APENAS com JSON válido, sem texto adicional, no formato:
{"n_processo": "<ID>", "predicted_label": "<LABEL>"}
```

**Identified weaknesses:**
- Vague role definition ("especialista") — activates a generic legal knowledge
  frame rather than the specific appellate reasoning frame needed
- No explicit DO/DON'T contract — small 8B models tend to add prose, use
  label variants, or refuse to classify when uncertain
- Label descriptions were simple paraphrases with no appellate-law precision
- No reading strategy — the model was not guided on which parts of the text
  to attend to and in what order
- The CoT reasoning steps were parallel extraction lists with no synthesis
  step and no grounding in the central legal question

---

## 3. Framework Selection

Prompt refinement was carried out using the **Prompt Architect** framework
library. Two distinct frameworks were selected for the two structural
components of the system prompt:

### 3.1 Classification stages (zero-shot, few-shot) — TIDD-EC

**Framework:** TIDD-EC (Task type – Instructions – Do – Don't – Examples – Context)

**Rationale:** TIDD-EC is designed for high-precision, constraint-heavy tasks
where the cost of deviation is high and common error modes can be enumerated
in advance. Legal classification with a small model on a constrained label
set is a textbook case: the model must produce exactly one of N valid strings
inside a JSON object, with no prose, no refusals, and no label hallucination.
TIDD-EC's explicit DO/DON'T block directly addresses these failure modes by
name, which is more robust than relying on a single "respond ONLY with JSON"
instruction.

TIDD-EC was chosen over CO-STAR (more suited to content creation and audience
targeting) and RISEN (more suited to multi-step workflows without the need for
explicit negative guidance).

### 3.2 Reasoning stages (CoT, few-shot+CoT) — Step-Back + Structured CoT

**Framework:** Hybrid of **Step-Back Prompting** (Zheng et al., Google
DeepMind, ICLR 2024) and **Structured Chain-of-Thought**

**Rationale:** The prior CoT instruction jumped directly to case-specific
extraction (facts, laws, jurisprudence) without first grounding the model in
the central legal question the appeal raises. Step-Back prompting addresses
this: the model is first asked to identify the governing legal question and
applicable norm at an abstract level, before descending to case specifics.
This is particularly effective for DeepSeek-R1, which already produces
internal `<think>...</think>` reasoning chains — the Step-Back question
focuses that reasoning on the right legal frame before it begins.

The structured CoT steps then follow a progression from extraction to
synthesis: facts → legislation → jurisprudence → **apreciação** (synthesis
step that bridges the extracted evidence to the classification decision) →
classification + JSON. The synthesis step (Passo 4) was explicitly separated
from the extraction steps, as the baseline conflated extraction and evaluation
in a single vague instruction ("avalia se os fundamentos são juridicamente
sólidos").

---

## 4. Role Assignment

**Technique:** Expert persona / role priming

**Rationale:** Assigning a specific, quantified expert persona activates a
more precise internal representation in the model than a generic domain label.
The title "Juiz Conselheiro" is the actual Portuguese judicial title for
appellate and supreme court judges, which is more semantically precise than
"especialista em direito" and directly primes appellate-level reasoning.
Quantifying experience ("mais de 30 anos", "centenas de recursos") reinforces
depth and pattern-recognition authority.

The persona is scoped to the specific domain of each task:

| Task | Persona |
|------|---------|
| DV | Juiz Conselheiro, penal, art. 152.º CP, criteria for maintaining/overturning convictions |
| BOC | Juiz Conselheiro, civil, regime geral das obrigações, contract law jurisprudence |

The persona is placed **first** in the system prompt, before the task block,
so the model activates the role frame before reading any instructions.

---

## 5. Final Prompt Structure

All four stages share the same system prompt architecture. The CoT stages
append the reasoning instruction block to the base system prompt.

```
[SYSTEM]
┌─────────────────────────────────────────────────────────┐
│  PERSONA  (role priming)                                │
│  "És um Juiz Conselheiro português com mais de 30 anos  │
│  de experiência em tribunais de segunda instância..."   │
├─────────────────────────────────────────────────────────┤
│  TASK TYPE  (TIDD-EC: T)                                │
│  "Tarefa: Classificação do resultado de recursos..."    │
├─────────────────────────────────────────────────────────┤
│  INSTRUCTIONS  (TIDD-EC: I)                             │
│  1. Lê o Relatório...                                   │
│  2. Lê a Fundamentação...                               │
│  3. Determina se manteve ou alterou...                  │
│  4. Emite o JSON.                                       │
├─────────────────────────────────────────────────────────┤
│  DO  (TIDD-EC: D)                                       │
│  - Exact label string                                   │
│  - Ground in provided text only                         │
│  - Valid JSON as sole response                          │
├─────────────────────────────────────────────────────────┤
│  DON'T  (TIDD-EC: D)                                    │
│  - No prose outside JSON                                │
│  - No label variants                                    │
│  - No dispositivo inference                             │
│  - No refusals                                          │
├─────────────────────────────────────────────────────────┤
│  VALID CLASSES  (TIDD-EC: E — inline examples)          │
│  - "decisao_mantida": precise appellate definition      │
│  - "decisao_alterada": precise appellate definition     │
├─────────────────────────────────────────────────────────┤
│  CONTEXT  (TIDD-EC: C)                                  │
│  - Court type, case type, dispositivo exclusion         │
├─────────────────────────────────────────────────────────┤
│  OUTPUT FORMAT                                          │
│  {"n_processo": "<ID>", "predicted_label": "<LABEL>"}  │
└─────────────────────────────────────────────────────────┘

[CoT stages only — appended to system]
┌─────────────────────────────────────────────────────────┐
│  STEP-BACK QUESTION                                     │
│  "Qual é o principal fundamento jurídico do recurso     │
│  e qual a norma que o tribunal tem de apreciar?"        │
├─────────────────────────────────────────────────────────┤
│  STRUCTURED CoT                                         │
│  Passo 1 — Factos                                       │
│  Passo 2 — Legislação                                   │
│  Passo 3 — Jurisprudência                               │
│  Passo 4 — Apreciação  ← synthesis step                │
│  Passo 5 — Classificação + JSON                         │
└─────────────────────────────────────────────────────────┘

[USER]
  Processo: <n_processo>
  Texto da decisão:
  <texto_integral_sem_decisao>
```

---

## 6. Summary of Changes from Baseline to Final

| Dimension | Baseline | Final |
|-----------|----------|-------|
| Role | "especialista em direito" (generic) | "Juiz Conselheiro, 30+ anos, centenas de recursos" (specific, quantified, domain-scoped) |
| Framework | Informal RTF | TIDD-EC (classification) + Step-Back + Structured CoT (reasoning) |
| DO contract | None | 3 explicit DOs |
| DON'T contract | "responde APENAS" (single) | 4 explicit DON'Ts targeting specific failure modes |
| Label definitions | Simple paraphrase | Precise appellate-law definitions with outcome framing |
| Reading strategy | None | Ordered 4-step instruction: Relatório → Fundamentação → inference → emit |
| CoT grounding | Direct extraction (no abstraction) | Step-Back question first, then extraction, then synthesis |
| Synthesis step | Missing | Explicit Passo 4 — Apreciação bridges evidence to decision |
| JSON placement | "no fim da resposta, sozinho numa linha" | "na linha seguinte, sem qualquer texto adicional depois" |

---

## 7. References

- Zheng et al. (2024). *Take a Step Back: Evoking Reasoning via Abstraction
  in Large Language Models*. ICLR 2024 (arXiv:2310.06117). — Step-Back Prompting.
- TIDD-EC framework: Vivas.AI / GPT Teams documentation of CO-STAR + TIDD-EC.
- Wei et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large
  Language Models*. NeurIPS 2022. — Chain-of-Thought baseline.
