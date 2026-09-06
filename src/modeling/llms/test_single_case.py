#!/usr/bin/env python3
"""Test a single case from the old logs to see if current thinking differs.

Standalone smoke-test script: builds a zero-shot DV prompt for one hardcoded
sample case and runs it through `OllamaClient.chat_with_validation`, printing
the parsed response and the path of the resulting trace log.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from modeling.llms.client import OllamaClient, InferenceTraceGateway
from modeling.llms.prompts import build_zero_shot_prompt
from modeling.llms.config import (
    OLLAMA_MODELS,
    LABEL_OUTPUT_NAMES,
    OLLAMA_DETERMINISTIC_OPTIONS,
)

client = OllamaClient()
trace_gateway = InferenceTraceGateway()

# This was a case from the DV logs (299/23.4SXLSB.L1-5)
# Sample text (simulated)
sample_text = """
O tribunal de primeira instância condenou o recorrente por crime de violência doméstica.
O recorrente alegou que os factos não foram adequadamente ponderados.
O Ministério Público pronunciou-se pela improcedência do recurso.
O tribunal de recurso analisou os fundamentos e concluiu que a condenação foi adequadamente fundamentada.
A defesa não apresentou argumentos suficientemente robustos para reverter a decisão de primeira instância.
"""

print("=" * 80)
print("TESTING SINGLE DV CASE")
print("=" * 80)
print("\nCase ID: 299/23.4SXLSB.L1-5")
print("Case Type: dv")
print("\nBuilding prompt...")

messages = build_zero_shot_prompt("dv", "299/23.4SXLSB.L1-5", sample_text)

print("Sending to Ollama with format='json' and deterministic options...")
print(f"  temperature={OLLAMA_DETERMINISTIC_OPTIONS['temperature']}")
print(f"  top_k={OLLAMA_DETERMINISTIC_OPTIONS['top_k']}")
print(f"  seed={OLLAMA_DETERMINISTIC_OPTIONS['seed']}")
print()

response = client.chat_with_validation(
    model=OLLAMA_MODELS["deepseek_r1_8b"],
    messages=messages,
    valid_labels=set(LABEL_OUTPUT_NAMES["dv"]),
    format="json",
    options=OLLAMA_DETERMINISTIC_OPTIONS,
    trace_gateway=trace_gateway,
    trace_context={
        "model": "deepseek_r1_8b",
        "stage": "zero_shot",
        "case_type": "dv",
        "case_id": "299/23.4SXLSB.L1-5",
    },
)

print("Response:", response)
print()
print("=" * 80)
print("TRACE LOGGED TO:")
log_path = trace_gateway.get_log_path("deepseek_r1_8b", "zero_shot", "dv")
print(log_path)
print("=" * 80)
