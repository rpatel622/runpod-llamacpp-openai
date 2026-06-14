from typing import Any

from runpod_flash import Endpoint

api = Endpoint(
    name="qwen-openai-adapter",
    cpu="cpu5c-4-8",
    workers=(1, 3),
    idle_timeout=600,
    dependencies=["httpx==0.28.1"],
)


def _model_name() -> str:
    import os

    return os.getenv("MODEL_NAME", "unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL")


def _openai_error(message: str, error_type: str = "server_error", code: int | str | None = None) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "code": code,
        }
    }


def _unwrap_runpod_output(job: dict[str, Any]) -> dict[str, Any]:
    status = job.get("status")
    if status and status != "COMPLETED":
        return _openai_error(f"RunPod job did not complete: {status}", "runpod_job_error", status)

    if job.get("error"):
        return _openai_error(str(job["error"]), "runpod_job_error", status)

    output = job.get("output")
    if output is None:
        return _openai_error("RunPod job completed without output", "runpod_job_error", status)

    # Current queue handler returns raw OpenAI-compatible JSON inside job.output.
    if isinstance(output, dict) and ("choices" in output or "data" in output or "error" in output):
        return output

    # Backward compatibility with the older queue handler wrapper:
    # {"ok": true, "status_code": 200, "endpoint": "...", "output": {"choices": [...]}}
    if isinstance(output, dict) and "output" in output:
        nested = output["output"]
        if isinstance(nested, dict):
            return nested

    return _openai_error(f"Unexpected RunPod output shape: {output!r}", "runpod_job_error", status)


async def _forward_to_queue(endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
    import os

    import httpx

    queue_endpoint_id = os.getenv("RUNPOD_QUEUE_ENDPOINT_ID")
    runpod_api_key = os.getenv("RUNPOD_API_KEY")

    if not queue_endpoint_id:
        return _openai_error("RUNPOD_QUEUE_ENDPOINT_ID is not set", "configuration_error", 500)
    if not runpod_api_key:
        return _openai_error("RUNPOD_API_KEY is not set", "configuration_error", 500)

    if body.get("stream") is True:
        return _openai_error("stream=true is not supported through RunPod queue mode", "invalid_request_error", 400)

    payload = dict(body)
    payload.setdefault("model", _model_name())
    payload.setdefault("stream", False)

    url = f"https://api.runpod.ai/v2/{queue_endpoint_id}/runsync"
    timeout_s = float(os.getenv("RUNPOD_RUNSYNC_TIMEOUT", "600"))

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s, connect=10.0)) as client:
            response = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {runpod_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "input": {
                        "endpoint": endpoint,
                        "body": payload,
                    }
                },
            )
    except httpx.HTTPError as exc:
        return _openai_error(f"RunPod queue request failed: {exc}", "upstream_error", 502)

    try:
        job = response.json()
    except ValueError:
        return _openai_error(
            f"RunPod queue returned non-JSON response with HTTP {response.status_code}: {response.text[:500]}",
            "upstream_error",
            response.status_code,
        )

    if response.status_code >= 400:
        return _openai_error(f"RunPod queue HTTP {response.status_code}: {job}", "upstream_error", response.status_code)

    if not isinstance(job, dict):
        return _openai_error(f"RunPod queue returned invalid JSON shape: {job!r}", "upstream_error", 502)

    return _unwrap_runpod_output(job)


@api.get("/ping")
async def ping() -> dict[str, Any]:
    return {"ok": True, "mode": "flash-openai-adapter"}


@api.get("/health")
async def health() -> dict[str, Any]:
    import os

    return {
        "ok": True,
        "mode": "flash-openai-adapter",
        "queue_endpoint_id_set": bool(os.getenv("RUNPOD_QUEUE_ENDPOINT_ID")),
        "runpod_api_key_set": bool(os.getenv("RUNPOD_API_KEY")),
        "model": _model_name(),
    }


@api.get("/v1/models")
async def models_v1() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": _model_name(),
                "object": "model",
                "created": 0,
                "owned_by": "runpod-llamacpp",
            }
        ],
    }


@api.get("/models")
async def models_alias() -> dict[str, Any]:
    return await models_v1()


@api.post("/v1/chat/completions")
async def chat_completions_v1(payload: dict[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return _openai_error("messages must be a non-empty array", "invalid_request_error", 400)
    return await _forward_to_queue("/v1/chat/completions", payload)


@api.post("/chat/completions")
async def chat_completions_alias(payload: dict[str, Any]) -> dict[str, Any]:
    return await chat_completions_v1(payload)


@api.post("/v1/completions")
async def completions_v1(payload: dict[str, Any]) -> dict[str, Any]:
    if "prompt" not in payload:
        return _openai_error("prompt is required", "invalid_request_error", 400)
    return await _forward_to_queue("/v1/completions", payload)


@api.post("/completions")
async def completions_alias(payload: dict[str, Any]) -> dict[str, Any]:
    return await completions_v1(payload)
