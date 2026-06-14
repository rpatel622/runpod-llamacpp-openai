# Flash all-in-one OpenAI adapter + queue-backed llama.cpp worker

This Flash app deploys two RunPod resources from one project:

1. `llamacpp-openai-adapter`: a CPU load-balanced HTTP endpoint exposing OpenAI-compatible routes.
2. `llamacpp-gemma4-queue`: a GPU queue endpoint using the existing Docker image and `RUNPOD_MODE=queue`.

The external client talks only to the load-balanced adapter:

```text
OpenAI/Qwen client
  -> https://<adapter>.api.runpod.ai/v1/chat/completions
  -> Flash CPU adapter
  -> Flash GPU queue endpoint
  -> llama.cpp worker
```

The queue endpoint still receives valid RunPod jobs internally, so raw OpenAI clients do not need to send `{"input": ...}`.

## Files

```text
flash_app/
  lb_worker.py       # OpenAI-compatible HTTP adapter routes
  gpu_queue.py       # Custom-image GPU queue endpoint config
  requirements.txt
  pyproject.toml
  .env.example
```

## Deploy

From this directory:

```bash
cd flash_app
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv run flash login
uv run flash deploy
```

On Windows, run Flash through WSL2.

Flash deployment output should include one load-balanced URL and one queue URL. Use the load-balanced URL as your OpenAI base URL:

```text
https://<adapter_endpoint_id>.api.runpod.ai/v1
```

Use your RunPod API key as the API key unless you place another auth layer in front of it.

## Test

```bash
curl -X POST "https://<adapter_endpoint_id>.api.runpod.ai/v1/chat/completions" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL",
    "messages": [{"role": "user", "content": "Say exactly: ok"}],
    "max_tokens": 8,
    "temperature": 0
  }'
```

## Qwen/OpenAI-style config

```json
{
  "baseUrl": "https://<adapter_endpoint_id>.api.runpod.ai/v1",
  "apiKey": "<RUNPOD_API_KEY>",
  "model": "unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL"
}
```

## Notes

Streaming is intentionally rejected because the backend GPU work uses queue `/runsync` semantics.

The GPU queue endpoint references the existing image:

```text
ghcr.io/rpatel622/runpod-llamacpp-openai:latest
```

Change `gpu_queue.py` before deploying if you need a larger GPU than `GpuType.NVIDIA_GEFORCE_RTX_4090`.
