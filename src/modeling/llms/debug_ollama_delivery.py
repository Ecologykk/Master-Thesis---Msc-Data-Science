#!/usr/bin/env python3
"""Diagnose exactly what's being sent to Ollama and what DeepSeek returns.

Tests if format="json" or other factors are affecting prompt delivery.
"""

import json
import sys
from pathlib import Path

import requests
from config import OLLAMA_BASE_URL
from prompts import build_zero_shot_prompt

# Simple case text
SAMPLE_TEXT = """
O recorrente foi condenado em primeira instância por crime de violência doméstica.
A defesa alega que os factos não foram devidamente ponderados pelo tribunal a quo.
O Ministério Público pronunciou-se pela improcedência do recurso.
O tribunal de recurso analisou os fundamentos e concluiu que a condenação foi correta.
"""


def test_ollama_delivery(model: str, use_format_json: bool = True):
    """Send a fixed sample DV prompt to Ollama and print what's sent/received.

    Builds a zero-shot DV prompt for a fixed sample case, posts it directly to
    `/api/chat` (bypassing `OllamaClient`), and prints the outgoing payload
    plus the response's `thinking`/`content` fields and whether `content`
    parses as JSON. Purely a diagnostic script; results are printed, nothing
    is returned or asserted.

    Args:
        model: Ollama model tag to query, e.g. "deepseek-r1:8b".
        use_format_json: If True, includes `format="json"` in the request
            payload to enforce JSON output mode; if False, omits it so the
            two modes can be compared.
    """
    print(f"\n{'='*80}")
    print(f"TEST: format='json' = {use_format_json}")
    print(f"{'='*80}\n")

    # Build the prompt
    messages = build_zero_shot_prompt("dv", "TEST-DV-001", SAMPLE_TEXT)

    print("[1] PROMPT BEING SENT TO OLLAMA:")
    print("-" * 80)
    print(f"System message (first 500 chars):\n{messages[0]['content'][:500]}")
    print(f"\n... [total: {len(messages[0]['content'])} chars]\n")
    print(f"User message:\n{messages[1]['content']}\n")

    # Build the payload
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": True,
    }

    if use_format_json:
        payload["format"] = "json"
        print("Payload includes: format='json'\n")
    else:
        print("Payload does NOT include format parameter\n")

    print("[2] SENDING TO OLLAMA...")
    print(f"POST {OLLAMA_BASE_URL}/api/chat\n")

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat",
            json=payload,
            timeout=180,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Request failed: {e}")
        return

    print(f"[3] OLLAMA RESPONSE ({resp.status_code})")
    print("-" * 80)

    body = resp.json()
    message = body.get("message", {})

    print(f"Response keys: {list(message.keys())}\n")

    # Extract thinking and content separately
    content = message.get("content", "")
    thinking = message.get("thinking", "")

    print("[4] THINKING BLOCK (first 800 chars):")
    print("-" * 80)
    if thinking:
        print(thinking[:800])
    else:
        print("[No separate 'thinking' field]")
    print()

    print("[5] CONTENT (final output):")
    print("-" * 80)
    print(content)
    print()

    # Try to parse as JSON
    print("[6] JSON PARSE ATTEMPT:")
    print("-" * 80)
    try:
        parsed = json.loads(content)
        print("✓ Valid JSON parsed successfully")
        print(f"  Keys: {list(parsed.keys())}")
        print(f"  predicted_label: {parsed.get('predicted_label', '[missing]')}")
    except json.JSONDecodeError as e:
        print(f"✗ JSON parse failed: {e}")
        print(f"  Content starts with: {content[:100]}")
    print()


def main():
    """Run `test_ollama_delivery` twice (with and without `format="json"`) and print guidance."""
    model = "deepseek-r1:8b"

    print("\n")
    print("*" * 80)
    print("OLLAMA PROMPT DELIVERY DIAGNOSTIC")
    print("*" * 80)

    # Test 1: With format="json"
    test_ollama_delivery(model, use_format_json=True)

    # Test 2: Without format parameter
    test_ollama_delivery(model, use_format_json=False)

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print("""
If thinking shows meta-commentary like "The user has shared a legal analysis..."
in BOTH tests, then the problem is:
  → DeepSeek is not interpreting the system prompt as a task instruction
  → It's defaulting to "helpful assistant summarizing a document" behavior

If thinking differs between the two tests, then:
  → format="json" may be interfering with prompt delivery

If thinking shows actual case analysis (facts, laws, outcome inference):
  → The prompt is working correctly and performance issue is elsewhere
    (text summarization, few-shot quality, model capability on this task, etc.)
""")


if __name__ == "__main__":
    main()
