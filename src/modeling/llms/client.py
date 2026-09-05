"""Thin wrapper around the Ollama REST API.

Provides
--------
OllamaClient
    .chat(model, messages, *, format="json") -> str
        Sends a chat request and returns the assistant content string.
        - Strips DeepSeek <think>...</think> reasoning blocks before returning.
        - Retries up to MAX_RETRIES times on connection errors or empty responses.
        - Accepts optional Ollama decoding options (seed, temperature, top_k, ...).

    .chat_with_validation(model, messages, valid_labels, *, format="json") -> str
        Like chat() but also retries when the returned JSON does not contain a
        recognised label from valid_labels.

InferenceTraceGateway
    Appends per-case traces to text files grouped by model/stage/case_type,
    including <think> reasoning and final output JSON.

    .check_model_available(model_name) -> bool
        Returns True if model_name appears in `ollama list` (/api/tags).

Usage
-----
    from client import OllamaClient
    client = OllamaClient()
    response = client.chat("deepseek-r1:8b", [{"role": "user", "content": "Olá"}])
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path

import requests
from config import MAX_RETRIES, OLLAMA_BASE_URL

# Regex to strip DeepSeek-R1 chain-of-thought blocks
_THINK_RE = re.compile(
    r"<(?:think|thinking)>.*?</(?:think|thinking)>", re.DOTALL | re.IGNORECASE
)
_THINK_CAPTURE_RE = re.compile(
    r"<(?:think|thinking)>(.*?)</(?:think|thinking)>", re.DOTALL | re.IGNORECASE
)


def _strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning chains emitted by DeepSeek-R1 models."""
    return _THINK_RE.sub("", text).strip()


def _extract_think(text: str) -> str:
    """Extract and concatenate any <think>...</think> blocks from model output."""
    chunks = [
        chunk.strip() for chunk in _THINK_CAPTURE_RE.findall(text) if chunk.strip()
    ]
    return "\n\n".join(chunks)


def _compose_raw_content(message: dict) -> str:
    """Compose raw output from Ollama message fields.

    Some reasoning models return chain-of-thought in a separate field
    (e.g. message.thinking/message.reasoning) instead of embedding it in content.
    This helper normalizes both formats by wrapping external thinking in <think> tags,
    so existing extraction/stripping code works unchanged.
    """
    content = str(message.get("content", "") or "")
    thinking = str(message.get("thinking", "") or message.get("reasoning", "") or "")

    if not thinking:
        return content
    if _THINK_CAPTURE_RE.search(content):
        return content
    if content:
        return f"<think>{thinking}</think>\n{content}"
    return f"<think>{thinking}</think>"


class InferenceTraceGateway:
    """Append per-case inference traces (reasoning + final output) to text logs."""

    def __init__(self, base_dir: Path | str | None = None) -> None:
        """Create the gateway and ensure its log directory exists.

        Args:
            base_dir: Directory under which per-model trace files are written.
                Defaults to `<repo_root>/data/models/llms/thinking_logs` when
                not provided.
        """
        if base_dir is None:
            base_dir = (
                Path(__file__).resolve().parents[3]
                / "data"
                / "models"
                / "llms"
                / "thinking_logs"
            )
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_fragment(value: str) -> str:
        """Sanitize a string into a filesystem-safe filename fragment."""
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
        return safe or "unknown"

    def get_log_path(
        self,
        model: str,
        stage: str,
        case_type: str,
        run_name: str | None = None,
    ) -> Path:
        """Compute the trace file path for a given model/stage/case_type/run.

        Args:
            model: Short or full model identifier used to name the log's
                parent subdirectory.
            stage: Pipeline stage (e.g. "zero_shot", "few_shot", "cot").
            case_type: Case type ("dv" or "boc").
            run_name: Optional run tag appended to the filename to keep
                traces from different runs separate.

        Returns:
            Path to the trace text file, under `self.base_dir/<model>/`,
            named `<stage>_<case_type>[_<run_name>].txt`. The file itself is
            not created by this method.
        """
        model_name = self._safe_fragment(model)
        stage_name = self._safe_fragment(stage)
        case_name = self._safe_fragment(case_type)
        file_name = f"{stage_name}_{case_name}.txt"
        if run_name:
            run_fragment = self._safe_fragment(run_name)
            file_name = f"{stage_name}_{case_name}_{run_fragment}.txt"
        return self.base_dir / model_name / file_name

    def append_trace(
        self,
        *,
        model: str,
        stage: str,
        case_type: str,
        case_id: str,
        attempt: int,
        status: str,
        reasoning: str,
        final_output: str,
        error_message: str | None = None,
        run_name: str | None = None,
    ) -> Path:
        """Append one trace entry (reasoning + final output) to the case's log file.

        Args:
            model: Short or full model identifier (passed to `get_log_path`).
            stage: Pipeline stage (e.g. "zero_shot", "few_shot", "cot").
            case_type: Case type ("dv" or "boc").
            case_id: Case identifier (`n_processo`) the trace belongs to.
            attempt: 1-based retry attempt number for this trace entry.
            status: Free-form status string (e.g. "ok", "error").
            reasoning: Extracted `<think>` reasoning text; written as
                "[no <think> block returned]" if empty.
            final_output: Final (post-stripping) model output text; written
                as "[empty output]" if empty.
            error_message: Optional error message to record, if this attempt
                failed.
            run_name: Optional run tag (passed to `get_log_path`).

        Returns:
            Path to the trace file the entry was appended to.
        """
        path = self.get_log_path(
            model=model,
            stage=stage,
            case_type=case_type,
            run_name=run_name,
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        reasoning_text = reasoning.strip() or "[no <think> block returned]"
        output_text = final_output.strip() or "[empty output]"

        with path.open("a", encoding="utf-8") as fh:
            fh.write("=" * 100 + "\n")
            fh.write(f"timestamp_utc: {timestamp}\n")
            fh.write(f"model: {model}\n")
            fh.write(f"stage: {stage}\n")
            fh.write(f"case_type: {case_type}\n")
            fh.write(f"case_id: {case_id}\n")
            fh.write(f"attempt: {attempt}\n")
            fh.write(f"status: {status}\n")
            if error_message:
                fh.write(f"error: {error_message}\n")
            fh.write("--- reasoning (<think>) ---\n")
            fh.write(reasoning_text + "\n")
            fh.write("--- final_output ---\n")
            fh.write(output_text + "\n")

        return path


class OllamaClient:
    """Minimal Ollama REST client for chat-completion requests."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL) -> None:
        """Create the client and its underlying HTTP session.

        Args:
            base_url: Base URL of the Ollama server (trailing slash stripped).
                Defaults to `OLLAMA_BASE_URL` from config.
        """
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _chat_raw(
        self,
        model: str,
        messages: list[dict],
        *,
        format: str = "json",
        options: dict[str, int | float] | None = None,
        enable_thinking: bool = True,
        timeout: int = 120,
    ) -> str:
        """Send a chat request and return raw assistant content (with <think>)."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "format": format,
        }
        if options:
            payload["options"] = options
        if enable_thinking:
            payload["think"] = True

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self._session.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=timeout,
                )

                # Backward compatibility: older Ollama versions may reject `think`.
                if resp.status_code == 400 and payload.get("think") is True:
                    try:
                        err_msg = str(resp.json().get("error", ""))
                    except ValueError:
                        err_msg = resp.text
                    err_lower = err_msg.lower()
                    if "think" in err_lower and (
                        "unknown" in err_lower or "invalid" in err_lower
                    ):
                        payload.pop("think", None)
                        resp = self._session.post(
                            f"{self.base_url}/api/chat",
                            json=payload,
                            timeout=timeout,
                        )

                resp.raise_for_status()
                body = resp.json()
                message = body.get("message", {})
                if not isinstance(message, dict):
                    raise KeyError("Missing 'message' object in Ollama response")
                return _compose_raw_content(message)
            except (requests.RequestException, KeyError, ValueError) as exc:
                if attempt == MAX_RETRIES:
                    raise RuntimeError(
                        f"Ollama chat failed after {MAX_RETRIES} attempts: {exc}"
                    ) from exc
                time.sleep(2**attempt)

    def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        format: str = "json",
        options: dict[str, int | float] | None = None,
        enable_thinking: bool = True,
        timeout: int = 120,
    ) -> str:
        """Send a chat request; return the stripped assistant content string.

        Args:
            model: Ollama model tag, e.g. "deepseek-r1:8b".
            messages: OpenAI-style message list `[{"role": ..., "content": ...}]`.
            format: Ollama response format — "json" enforces JSON output mode.
            options: Optional Ollama decoding options (seed, temperature, ...).
            enable_thinking: Whether to request the model's `<think>` reasoning
                (sent as `think: true`); falls back automatically if the
                server rejects the `think` field.
            timeout: Request timeout in seconds.

        Returns:
            Assistant reply with `<think>` blocks removed.

        Raises:
            RuntimeError: If the request still fails (connection error, missing
                `message` object, etc.) after `MAX_RETRIES` attempts.
        """
        raw = self._chat_raw(
            model=model,
            messages=messages,
            format=format,
            options=options,
            enable_thinking=enable_thinking,
            timeout=timeout,
        )
        return _strip_think(raw)

    def chat_with_validation(
        self,
        model: str,
        messages: list[dict],
        valid_labels: set[str],
        *,
        format: str = "json",
        options: dict[str, int | float] | None = None,
        enable_thinking: bool = True,
        trace_gateway: InferenceTraceGateway | None = None,
        trace_context: dict[str, str] | None = None,
        timeout: int = 120,
    ) -> dict:
        """Chat + validate that returned JSON contains a recognised predicted_label.

        Retries up to MAX_RETRIES times if the JSON is unparseable or the
        `predicted_label` value is not in `valid_labels`. When `trace_gateway`
        and `trace_context` are both provided, every attempt (successful or
        not) is appended to the per-case trace log via `InferenceTraceGateway`.

        Args:
            model: Ollama model tag, e.g. "deepseek-r1:8b".
            messages: OpenAI-style message list `[{"role": ..., "content": ...}]`.
            valid_labels: Set of label strings accepted in `predicted_label`.
            format: Ollama response format — "json" enforces JSON output mode.
            options: Optional Ollama decoding options (seed, temperature, ...).
            enable_thinking: Whether to request the model's `<think>` reasoning.
            trace_gateway: Optional gateway used to log each attempt's
                reasoning and output.
            trace_context: Optional dict with keys among "model", "stage",
                "case_type", "case_id", "run_name", used to fill in the trace
                log entry; falls back to "unknown_*" placeholders when a key
                is absent.
            timeout: Request timeout in seconds.

        Returns:
            Parsed JSON response body (containing at least `predicted_label`).

        Raises:
            RuntimeError: If no attempt produces valid, recognised JSON within
                `MAX_RETRIES` tries; chains the last encountered exception.
        """
        last_exc: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            raw = ""
            final_output = ""
            try:
                raw = self._chat_raw(
                    model=model,
                    messages=messages,
                    format=format,
                    options=options,
                    enable_thinking=enable_thinking,
                    timeout=timeout,
                )
                final_output = _strip_think(raw)
                parsed = json.loads(final_output)
                label = parsed.get("predicted_label", "")
                if label not in valid_labels:
                    raise ValueError(
                        f"predicted_label '{label}' not in valid set {valid_labels}"
                    )

                if trace_gateway is not None and trace_context is not None:
                    trace_gateway.append_trace(
                        model=trace_context.get("model", model),
                        stage=trace_context.get("stage", "unknown_stage"),
                        case_type=trace_context.get("case_type", "unknown_case_type"),
                        case_id=trace_context.get("case_id", "unknown_case_id"),
                        attempt=attempt,
                        status="ok",
                        reasoning=_extract_think(raw),
                        final_output=final_output,
                        run_name=trace_context.get("run_name"),
                    )

                return parsed
            except (json.JSONDecodeError, ValueError, RuntimeError) as exc:
                last_exc = exc

                if trace_gateway is not None and trace_context is not None:
                    trace_gateway.append_trace(
                        model=trace_context.get("model", model),
                        stage=trace_context.get("stage", "unknown_stage"),
                        case_type=trace_context.get("case_type", "unknown_case_type"),
                        case_id=trace_context.get("case_id", "unknown_case_id"),
                        attempt=attempt,
                        status="error",
                        reasoning=_extract_think(raw),
                        final_output=final_output or raw,
                        error_message=str(exc),
                        run_name=trace_context.get("run_name"),
                    )

                if attempt < MAX_RETRIES:
                    time.sleep(2**attempt)
        raise RuntimeError(
            f"chat_with_validation failed after {MAX_RETRIES} attempts: {last_exc}"
        ) from last_exc

    def check_model_available(self, model_name: str) -> bool:
        """Return True if model_name is present in the local Ollama model list.

        Args:
            model_name: Full Ollama model tag to look for, e.g. "deepseek-r1:8b".

        Returns:
            True if `model_name` appears in the server's `/api/tags` listing;
            False if it does not, or if the request itself fails.
        """
        try:
            resp = self._session.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            return model_name in models
        except requests.RequestException:
            return False


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    client = OllamaClient()
    model_tag = "deepseek-r1:8b"

    print(f"Checking model availability: {model_tag}")
    if not client.check_model_available(model_tag):
        print(f"  [WARN] '{model_tag}' not found in ollama list — is it pulled?")
        sys.exit(1)

    print("Sending smoke-test prompt ...")
    messages = [
        {
            "role": "system",
            "content": "Responde apenas com JSON válido.",
        },
        {
            "role": "user",
            "content": (
                'Classifica este resultado: "o recurso foi julgado improcedente". '
                'Responde no formato: {"resultado": "improcedente"}'
            ),
        },
    ]
    response = client.chat(model_tag, messages)
    print(f"Raw response: {response}")

    try:
        parsed = json.loads(response)
        print(f"Parsed JSON: {parsed}")
        print("[OK] Smoke test passed.")
    except json.JSONDecodeError as e:
        print(f"[FAIL] Response is not valid JSON: {e}")
        sys.exit(1)
