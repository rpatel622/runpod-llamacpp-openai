import json
import os
import shutil
import shlex
import signal
import subprocess
import threading
import time
from typing import Any

import httpx
import runpod

LLAMA_HOST = os.getenv("LLAMA_HOST", "127.0.0.1")
LLAMA_PORT = int(os.getenv("LLAMA_PORT", "8080"))
LLAMA_BASE = f"http://{LLAMA_HOST}:{LLAMA_PORT}"
MODEL_NAME = os.getenv("MODEL_NAME", "unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL")

DEFAULT_LLAMA_ARGS = [
    "-hf", MODEL_NAME,
    "--host", LLAMA_HOST,
    "--port", str(LLAMA_PORT),
    "--ctx-size", "262144",
    "--flash-attn", "on",
    "-ctk", "q8_0",
    "-ctv", "q8_0",
    "-ctkd", "q8_0",
    "-ctvd", "q8_0",
    "--jinja",
    "--temp", "1",
    "--top-p", "0.95",
    "--top-k", "64",
    "-ngl", "999",
    "-ngld", "999",
    "--parallel", "2",
    "-b", "1024",
    "-ub", "1024",
    "--image-min-tokens", "1024",
    "--image-max-tokens", "2048",
    "--cont-batching",
    "--reasoning", "on",
    "-fit", "on",
    "--timeout", "6000000",
    "-kvu",
    "-cram", "12000",
    "-cms", "4096",
    "-ctxcp", "16",
    "--spec-type", "draft-mtp",
    "--spec-draft-n-max", "2",
    "--reasoning-budget", "1024",
]

_llama_proc: subprocess.Popen | None = None
_start_lock = threading.Lock()
_ready = False
_startup_error: str | None = None


def _find_llama_server() -> str:
    candidates = [
        os.getenv("LLAMA_SERVER_BIN"),
        shutil.which("llama-server"),
        "/app/llama-server",
        "/usr/local/bin/llama-server",
        "/bin/llama-server",
        "/server",
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("llama-server binary not found; set LLAMA_SERVER_BIN")


def _llama_args() -> list[str]:
    raw = os.getenv("LLAMA_ARGS")
    if not raw:
        return DEFAULT_LLAMA_ARGS
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return parsed
    except json.JSONDecodeError:
        pass
    return shlex.split(raw)


def _start_llama_once() -> None:
    global _llama_proc, _ready, _startup_error

    with _start_lock:
        if _ready:
            return
        if _llama_proc and _llama_proc.poll() is None:
            return

        binary = _find_llama_server()
        cmd = [binary, *_llama_args()]
        _llama_proc = subprocess.Popen(cmd, stdout=None, stderr=None, text=True)
        _ready = False
        _startup_error = None


def _wait_for_llama() -> None:
    global _ready, _startup_error

    timeout_s = int(os.getenv("LLAMA_STARTUP_TIMEOUT", "900"))
    start = time.monotonic()

    with httpx.Client(timeout=5.0) as client:
        while time.monotonic() - start < timeout_s:
            if _llama_proc and _llama_proc.poll() is not None:
                _startup_error = f"llama-server exited with code {_llama_proc.returncode}"
                _ready = False
                raise RuntimeError(_startup_error)
            try:
                response = client.get(f"{LLAMA_BASE}/v1/models")
                if response.status_code < 500:
                    _ready = True
                    _startup_error = None
                    return
            except httpx.HTTPError:
                pass
            time.sleep(1)

    _startup_error = f"llama-server did not become ready within {timeout_s}s"
    _ready = False
    raise TimeoutError(_startup_error)


def _ensure_backend_ready() -> None:
    _start_llama_once()
    if not _ready:
        _wait_for_llama()


def _normalize_input(job_input: Any) -> tuple[str, dict[str, Any] | None]:
    if not isinstance(job_input, dict):
        raise ValueError("job input must be a JSON object")

    endpoint = str(job_input.get("endpoint", job_input.get("path", "/v1/chat/completions")))
    if not endpoint.startswith("/"):
        endpoint = f"/{endpoint}"

    allowed = {"/v1/chat/completions", "/v1/completions", "/v1/models"}
    if endpoint not in allowed:
        raise ValueError(f"unsupported endpoint '{endpoint}'; allowed endpoints: {sorted(allowed)}")

    if endpoint == "/v1/models":
        return endpoint, None

    if "body" in job_input:
        body = job_input["body"]
        if not isinstance(body, dict):
            raise ValueError("input.body must be a JSON object")
    else:
        body = {k: v for k, v in job_input.items() if k not in {"endpoint", "path"}}

    if body.get("stream") is True:
        raise ValueError("queue mode does not support OpenAI streaming responses; omit stream or set stream=false")

    if endpoint == "/v1/chat/completions":
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("/v1/chat/completions requires a non-empty messages array")
    elif endpoint == "/v1/completions":
        if "prompt" not in body:
            raise ValueError("/v1/completions requires prompt")

    body.setdefault("model", MODEL_NAME)
    body.setdefault("stream", False)
    return endpoint, body


def handler(job: dict[str, Any]) -> dict[str, Any]:
    _ensure_backend_ready()

    endpoint, body = _normalize_input(job.get("input"))
    timeout = httpx.Timeout(connect=10.0, read=None, write=60.0, pool=None)

    with httpx.Client(timeout=timeout) as client:
        if body is None:
            response = client.get(f"{LLAMA_BASE}{endpoint}")
        else:
            response = client.post(f"{LLAMA_BASE}{endpoint}", json=body)

    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        payload: Any = response.json()
    else:
        payload = response.text

    return {
        "ok": response.status_code < 400,
        "status_code": response.status_code,
        "endpoint": endpoint,
        "output": payload,
    }


def _shutdown() -> None:
    if _llama_proc and _llama_proc.poll() is None:
        _llama_proc.send_signal(signal.SIGTERM)
        try:
            _llama_proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            _llama_proc.kill()


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
