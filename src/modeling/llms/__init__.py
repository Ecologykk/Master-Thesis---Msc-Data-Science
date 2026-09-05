"""LLM-based inference pipeline for court decision outcome prediction.

Ties together prompt construction (:mod:`prompts`), the Ollama chat client
(:mod:`client`), few-shot example retrieval (:mod:`few_shot_retrieval_similarity`,
:mod:`few_shot_metadata`), context-window summarization (:mod:`summarizer`), and
the zero-shot/few-shot/chain-of-thought prediction pipeline (:mod:`predict`) used
to classify Portuguese court decisions for the DV (domestic violence) and BoC
(breach of contract) tasks.
"""

# src/modeling/llms/__init__.py
