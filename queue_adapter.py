import os
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

APP_PORT = int(os.getenv("PORT", "80"))
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "").strip()
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "").strip()
RUNPOD_QUEUE_BASE_URL = os.getenv("RUNPOD_QUEUE_BASE_URL", "").strip()
MODEL_NAME = os.getenv("MODEL_NAME", "unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL")
ADAPTER_TIMEOUT_SECONDS = float(os.getenv("ADAPTER_TIMEOUT_SECONDS", "600"))

app = FastAPI(title="RunPod queue to OpenAI adapter", version="1.0.0")


def _queue_base_url() -> str:
    if RUNPOD_QUEUE_BASE_URL:
        return RUNPOD_QUEUE_BASE_URL.rstrip("/")
    if not RUNPOD_ENDPOINT_ID:
        raise RuntimeError("RUNPOD_ENDPOINT_ID or RUNPOD_QUEUE_BASE_URL is required")
    return f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"


def _auth_headers() -> dict[str, str]:
    if not RUNPOD_API_KEY:
        raise RuntimeError("RUNPOD_API_KEY is required")
    return {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }


def _unwrap_runsync_response(job_response: dict[str, Any]) -> tuple[int, Any]:
    status = job_response.get("status")
    if status not in {None, "COMPLETED"}:
        return 502, {
            "error": {
                "message": f"RunPod job did not complete successfully: {status}",
                "type": "runpod_job_error",
                "runpod_response": job_response,
            }
        }

    outer_output = job_response.get("output")
    if not isinstance(outer_output, dict):
        return 502, {
            "error": {
                "message": "RunPod response did not contain object output",
                "type": "runpod_response_error",
                "runpod_response": job_response,
            }
        }

    status_code = int(outer_output.get("status_code", 200))
    inner_output = outer_output.get("output")
    if inner_output is None:
        return 502, {
            "error": {
                "message": "RunPod worker output did not contain model output",
                "type": "runpod_worker_error",
                "worker_output": outer_output,
            }
        }

    return status_code, inner_output


async def _call_queue(endpoint: str, body: dict[str, Any] | None) -> tuple[int, Any]:
    runsync_url = f"{_queue_base_url()}/runsync"
    job_input: dict[str, Any] = {"endpoint": endpoint}
    if body is not None:
        job_input["body"] = body

    timeout = httpx.Timeout(connect=10.0, read=ADAPTER_TIMEOUT_SECONDS, write=60.0, pool=None)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            runsync_url,
            headers=_auth_headers(),
            json={"input": job_input},
        )

    try:
        payload = response.json()
    except ValueError:
        return response.status_code, {
            "error": {
                "message": response.text,
                "type": "runpod_http_error",
            }
        }

    if response.status_code >= 400:
        return response.status_code, payload

    return _unwrap_runsync_response(payload)


@app.get("/ping")
async def ping() -> Response:
    try:
        _queue_base_url()
        _auth_headers()
    except RuntimeError as exc:
        return Response(str(exc), status_code=500)
    return Response("ready", status_code=200)


@app.get("/health")
async def health() -> JSONResponse:
    configured = bool((RUNPOD_ENDPOINT_ID or RUNPOD_QUEUE_BASE_URL) and RUNPOD_API_KEY)
    return JSONResponse({
        "ready": configured,
        "mode": "queue_adapter",
        "queue_base_url": _queue_base_url() if configured else None,
        "model": MODEL_NAME,
    })


@app.get("/v1/models")
async def models() -> Response:
    status_code, payload = await _call_queue("/v1/models", None)
    return JSONResponse(payload, status_code=status_code)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    body = await request.json()
    if not isinstance(body, dict):
        return JSONResponse({"error": {"message": "request body must be a JSON object", "type": "invalid_request_error"}}, status_code=400)
    if body.get("stream") is True:
        return JSONResponse({"error": {"message": "queue adapter does not support streaming; set stream=false", "type": "invalid_request_error"}}, status_code=400)
    body.setdefault("model", MODEL_NAME)
    status_code, payload = await _call_queue("/v1/chat/completions", body)
    return JSONResponse(payload, status_code=status_code)


@app.post("/v1/completions")
async def completions(request: Request) -> Response:
    body = await request.json()
    if not isinstance(body, dict):
        return JSONResponse({"error": {"message": "request body must be a JSON object", "type": "invalid_request_error"}}, status_code=400)
    if body.get("stream") is True:
        return JSONResponse({"error": {"message": "queue adapter does not support streaming; set stream=false", "type": "invalid_request_error"}}, status_code=400)
    body.setdefault("model", MODEL_NAME)
    status_code, payload = await _call_queue("/v1/completions", body)
    return JSONResponse(payload, status_code=status_code)


@app.api_route("/{path:path}", methods=["GET", "POST", "OPTIONS"])
async def not_found(path: str) -> JSONResponse:
    return JSONResponse({"error": {"message": f"Unsupported path: /{path}", "type": "not_found"}}, status_code=404)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=APP_PORT, log_level=os.getenv("LOG_LEVEL", "info"))
