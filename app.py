import asyncio
import json
import os
import shutil
import signal
import subprocess
import time
from typing import AsyncIterator

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

APP_PORT = int(os.getenv("PORT", "80"))
HEALTH_PORT = int(os.getenv("PORT_HEALTH", str(APP_PORT)))
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
    "--parallel", "3",
    "-b", "2048",
    "-ub", "2048",
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
    "--ui",
    "-n", "32768",
    "--override-kv", "gemma4.context_length=int:262144"
]

app = FastAPI(title="RunPod llama.cpp OpenAI-compatible proxy", version="1.0.0")
llama_proc: subprocess.Popen | None = None
ready = False
startup_error: str | None = None


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
    # JSON array is preferred because these flags are order-sensitive.
    # Example: LLAMA_ARGS='["-hf","repo:model","--host","127.0.0.1","--port","8080"]'
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return parsed
    except json.JSONDecodeError:
        pass
    # Fallback is shell-like splitting for local convenience.
    import shlex
    return shlex.split(raw)


async def _wait_for_llama() -> None:
    global ready, startup_error
    timeout_s = int(os.getenv("LLAMA_STARTUP_TIMEOUT", "900"))
    start = time.monotonic()
    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.monotonic() - start < timeout_s:
            if llama_proc and llama_proc.poll() is not None:
                startup_error = f"llama-server exited with code {llama_proc.returncode}"
                ready = False
                return
            try:
                r = await client.get(f"{LLAMA_BASE}/v1/models")
                if r.status_code < 500:
                    ready = True
                    startup_error = None
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1)
    startup_error = f"llama-server did not become ready within {timeout_s}s"
    ready = False


@app.on_event("startup")
async def startup() -> None:
    global llama_proc, startup_error
    binary = _find_llama_server()
    cmd = [binary, *_llama_args()]
    llama_proc = subprocess.Popen(cmd, stdout=None, stderr=None, text=True)
    asyncio.create_task(_wait_for_llama())


@app.on_event("shutdown")
async def shutdown() -> None:
    if llama_proc and llama_proc.poll() is None:
        llama_proc.send_signal(signal.SIGTERM)
        try:
            llama_proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            llama_proc.kill()


@app.get("/ping")
async def ping() -> Response:
    if ready:
        return Response(status_code=200, content="ready")
    if startup_error:
        return Response(status_code=500, content=startup_error)
    return Response(status_code=204)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"ready": ready, "error": startup_error, "backend": LLAMA_BASE})


@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse({
        "ok": ready,
        "model": MODEL_NAME,
        "openai_base_url": "/v1",
        "endpoints": ["/v1/models", "/v1/chat/completions", "/v1/completions", "/ping"],
    })


async def _proxy(request: Request, suffix: str) -> Response:
    if not ready:
        return JSONResponse({"error": {"message": "model is still loading", "type": "service_unavailable"}}, status_code=503)

    target = f"{LLAMA_BASE}/{suffix}"
    headers = {k: v for k, v in request.headers.items() if k.lower() not in {"host", "content-length"}}
    body = await request.body()

    timeout = httpx.Timeout(connect=10.0, read=None, write=60.0, pool=None)
    client = httpx.AsyncClient(timeout=timeout)
    req = client.build_request(request.method, target, headers=headers, content=body, params=request.query_params)
    upstream = await client.send(req, stream=True)

    excluded = {"content-encoding", "transfer-encoding", "connection", "keep-alive"}
    response_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in excluded}

    async def body_iter() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(body_iter(), status_code=upstream.status_code, headers=response_headers, media_type=upstream.headers.get("content-type"))


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "OPTIONS"])
async def v1_proxy(request: Request, path: str) -> Response:
    return await _proxy(request, f"v1/{path}")


@app.api_route("/{path:path}", methods=["GET", "POST", "OPTIONS"])
async def compat_proxy(request: Request, path: str) -> Response:
    # Allows RunPod LB URL + /chat/completions in addition to /v1/chat/completions.
    if path in {"chat/completions", "completions", "models", "embeddings"} or path.startswith(("chat/", "models/")):
        return await _proxy(request, f"v1/{path}")
    return JSONResponse({"error": "not found"}, status_code=404)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=APP_PORT, log_level=os.getenv("LOG_LEVEL", "info"))
