import os
from typing import Any

from runpod_flash import Endpoint

from gpu_queue import llama_gpu


MODEL_NAME = os.getenv("MODEL_NAME", "unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL")
LB_WORKERS_MIN = int(os.getenv("FLASH_LB_WORKERS_MIN", "1"))
LB_WORKERS_MAX = int(os.getenv("FLASH_LB_WORKERS_MAX", "5"))
JOB_WAIT_TIMEOUT_SECONDS = int(os.getenv("FLASH_JOB_WAIT_TIMEOUT_SECONDS", "600"))


api = Endpoint(
    name="llamacpp-openai-adapter",
    cpu=os.getenv("FLASH_LB_CPU", "cpu5c-4-8"),
    workers=(LB_WORKERS_MIN, LB_WORKERS_MAX),
    dependencies=["httpx>=0.28.0"],
)


def _openai_error(message: str, status_code: int = 400, error_type: str = "invalid_request_error") -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "code": status_code,
        }
    }


def _completion_endpoint_for_path(path: str) -> str:
    if path in {"/v1/chat/completions", "/v1/chat/completions/", "/chat/completions", "/chat/completions/"}:
        return "/v1/chat/completions"
    if path in {"/v1/completions", "/v1/completions/", "/completions", "/completions/"}:
        return "/v1/completions"
    raise ValueError(f"unsupported path: {path}")


async def _run_llama_queue(endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(body, dict):
        return _openai_error("request body must be a JSON object")

    if body.get("stream") is True:
        return _openai_error("stream=true is not supported by the queue-backed adapter; use non-streaming requests")

    body.setdefault("model", MODEL_NAME)
    body.setdefault("stream", False)

    job = await llama_gpu.run({
        "input": {
            "endpoint": endpoint,
            "body": body,
        }
    })

    await job.wait(timeout=JOB_WAIT_TIMEOUT_SECONDS)

    if job.error:
        return _openai_error(str(job.error), 502, "runpod_queue_error")

    output = job.output
    if isinstance(output, dict) and "error" in output:
        return output
    if not isinstance(output, dict):
        return _openai_error(f"unexpected queue output type: {type(output).__name__}", 502, "bad_gateway")

    return output


# Do not define /ping. Flash reserves /ping and /execute internally.
@api.get("/health")
@api.get("/health/")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "mode": "flash-lb-adapter",
        "gpu_endpoint": "llamacpp-gemma4-queue",
        "model": MODEL_NAME,
    }


@api.get("/v1/models")
@api.get("/v1/models/")
@api.get("/models")
@api.get("/models/")
async def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": 0,
                "owned_by": "runpod-llamacpp",
            }
        ],
    }


@api.post("/v1/chat/completions")
@api.post("/v1/chat/completions/")
async def chat_completions(body: dict[str, Any]) -> dict[str, Any]:
    return await _run_llama_queue(_completion_endpoint_for_path("/v1/chat/completions"), body)


@api.post("/chat/completions")
@api.post("/chat/completions/")
async def chat_completions_alias(body: dict[str, Any]) -> dict[str, Any]:
    return await chat_completions(body)


@api.post("/v1/completions")
@api.post("/v1/completions/")
async def completions(body: dict[str, Any]) -> dict[str, Any]:
    return await _run_llama_queue(_completion_endpoint_for_path("/v1/completions"), body)


@api.post("/completions")
@api.post("/completions/")
async def completions_alias(body: dict[str, Any]) -> dict[str, Any]:
    return await completions(body)
