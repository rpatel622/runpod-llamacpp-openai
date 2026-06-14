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
    return {"error": {"message": message, "type": error_type, "code": status_code}}


def _drop_none(data: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in data.items() if v is not None}


async def _run_llama_queue(endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(body, dict):
        return _openai_error("request body must be a JSON object")

    if body.get("stream") is True:
        return _openai_error("stream=true is not supported by the queue-backed adapter; use non-streaming requests")

    body.setdefault("model", MODEL_NAME)
    body.setdefault("stream", False)

    job = await llama_gpu.run({"input": {"endpoint": endpoint, "body": body}})
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
    return {"status": "ok", "mode": "flash-lb-adapter", "gpu_endpoint": "llamacpp-gemma4-queue", "model": MODEL_NAME}


@api.get("/v1/models")
@api.get("/v1/models/")
@api.get("/models")
@api.get("/models/")
async def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model", "created": 0, "owned_by": "runpod-llamacpp"}],
    }


async def _chat_route(
    model: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    max_tokens: int | None = None,
    max_completion_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    stream: bool | None = None,
    stop: str | list[str] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    response_format: dict[str, Any] | None = None,
    seed: int | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    n: int | None = None,
) -> dict[str, Any]:
    body = _drop_none(locals())
    return await _run_llama_queue("/v1/chat/completions", body)


@api.post("/v1/chat/completions")
@api.post("/v1/chat/completions/")
async def chat_completions(
    model: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    max_tokens: int | None = None,
    max_completion_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    stream: bool | None = None,
    stop: str | list[str] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    response_format: dict[str, Any] | None = None,
    seed: int | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    n: int | None = None,
) -> dict[str, Any]:
    return await _chat_route(**locals())


@api.post("/chat/completions")
@api.post("/chat/completions/")
async def chat_completions_alias(
    model: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    max_tokens: int | None = None,
    max_completion_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    stream: bool | None = None,
    stop: str | list[str] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    response_format: dict[str, Any] | None = None,
    seed: int | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    n: int | None = None,
) -> dict[str, Any]:
    return await _chat_route(**locals())


async def _completion_route(
    model: str | None = None,
    prompt: str | list[str] | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    stream: bool | None = None,
    stop: str | list[str] | None = None,
    echo: bool | None = None,
) -> dict[str, Any]:
    body = _drop_none(locals())
    return await _run_llama_queue("/v1/completions", body)


@api.post("/v1/completions")
@api.post("/v1/completions/")
async def completions(
    model: str | None = None,
    prompt: str | list[str] | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    stream: bool | None = None,
    stop: str | list[str] | None = None,
    echo: bool | None = None,
) -> dict[str, Any]:
    return await _completion_route(**locals())


@api.post("/completions")
@api.post("/completions/")
async def completions_alias(
    model: str | None = None,
    prompt: str | list[str] | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    stream: bool | None = None,
    stop: str | list[str] | None = None,
    echo: bool | None = None,
) -> dict[str, Any]:
    return await _completion_route(**locals())
