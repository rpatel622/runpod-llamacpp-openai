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
  scripts/codespaces_flash_deploy.sh
```

## Deploy from GitHub Codespaces

From the repo root:

```bash
bash flash_app/scripts/codespaces_flash_deploy.sh
```

The script:

1. Verifies Python 3.10-3.13.
2. Installs `uv` if missing.
3. Creates `flash_app/.venv`.
4. Installs Flash dependencies.
5. Prompts for authentication using either a pasted `RUNPOD_API_KEY` or `flash login --no-open`.
6. Optionally runs `flash deploy --env production`.

For Codespaces, pasting `RUNPOD_API_KEY` is usually simpler than browser login. Flash also supports `flash login --no-open`, which prints an authorization URL for manual browser authorization.

## Manual deploy

From this directory:

```bash
cd flash_app
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv run flash login --no-open
uv run flash deploy --env production
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
