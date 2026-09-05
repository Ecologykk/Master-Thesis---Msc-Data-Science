# 01 — Setup

## Purpose
Get a working local environment: Python dependencies, optional GPU/Ollama setup, and the
environment variables the pipeline reads. Every other doc in this series assumes you've done
this first.

## Prerequisites
- Python 3.12 (developed and tested on 3.12.10).
- Git Bash (or any POSIX-compatible shell) — all commands in this doc series assume it.
- ~2 GB free disk for Python dependencies (PyTorch is the largest).
- Optional, only needed for later stages: an NVIDIA GPU with ~6 GB VRAM (BERT fine-tuning was
  developed against an RTX 3050 6 GB) and [Ollama](https://ollama.ai) for the LLM stage.

## Exact commands
From the repository root:

```bash
python -m venv venv
source venv/Scripts/activate   # Git Bash on Windows; use `source venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
```

Verify the install:

```bash
python -c "import torch, transformers, shap, umap, bs4, crawl4ai, pandas, sklearn; print('ok')"
```

Copy the environment template and fill in anything you need to override:

```bash
cp .env.example .env
```

(Only `OLLAMA_BASE_URL` is currently read from the environment — see
[`src/modeling/llms/config.py`](../src/modeling/llms/config.py). Everything else in
`.env.example` is informational.)

## Inputs
None — this is the first step.

## Outputs
- A `venv/` virtual environment with all pinned dependencies from
  [`requirements.txt`](../requirements.txt) installed.

## Approximate runtime
2–10 minutes, dominated by the PyTorch download (~2 GB) and network speed.

## Expected result
The verification command prints `ok`. If it raises `ModuleNotFoundError`, the `pip install` step
did not complete — re-run it and check for errors before proceeding to
[`02-scraping.md`](02-scraping.md).

## Optional: Ollama (needed only for the LLM stage, docs/05)
1. Install from [ollama.ai](https://ollama.ai).
2. Pull a model, e.g.:
   ```bash
   ollama pull deepseek-r1:8b
   ```
   (See `OLLAMA_MODELS` in [`src/modeling/llms/config.py`](../src/modeling/llms/config.py) for
   every model tag this codebase knows about.)
3. Confirm it's reachable:
   ```bash
   curl http://localhost:11434/api/tags
   ```

## Optional: GPU (needed only for BERT fine-tuning, docs/04)
CUDA-enabled PyTorch is installed by default via `requirements.txt`. Confirm your GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
```

Training and inference both run on CPU if no GPU is available — much slower, but functionally
identical. See [`09-reproducibility-notes.md`](09-reproducibility-notes.md) for how hardware
differences affect exact reproducibility.
