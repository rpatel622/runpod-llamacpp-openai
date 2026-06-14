# Flash OpenAI adapter for RunPod queue mode

This Flash app creates a small load-balanced HTTP adapter that exposes OpenAI-style routes and forwards GPU work to the existing RunPod queue endpoint.

It is intended for clients such as Qwen Code that require a normal OpenAI-compatible base URL but where the model should still run through RunPod queue mode.

## Architecture

```text
OpenAI-compatible client
  -> Flash load-balanced HTTP endpoint
  -> RunPod queue endpoint /runsync
  -> llama.cpp queue worker
```

The queue worker should use the main Docker image with:

```text
RUNPOD_MODE=queue
```

The Flash adapter does not run the model. It only rewrites request/response JSON.

## Files

```text
flash/
  lb_worker.py       # Flash load-balanced HTTP routes
  requirements.txt   # Flash app dependencies
  pyproject.toml     # uv/project metadata
  .env.example       # required env vars
```

## Setup

Flash currently runs natively on macOS/Linux. On Windows, use WSL2.

```bash
cd flash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Authenticate:

```bash
flash login
```

or set:

```bash
export RUNPOD_API_KEY="your_runpod_api_key"
```

Set adapter env vars:

```bash
export RUNPOD_QUEUE_ENDPOINT_ID="899crt8fm30gys"
export MODEL_NAME="unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL"
export RUNPOD_RUNSYNC_TIMEOUT="600"
```

## Local test

```bash
flash dev
```

Then test:

```bash
curl -X POST http://localhost:8888/lb_worker/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL",
    "messages": [{"role": "user", "content": "Say exactly: ok"}],
    "max_tokens": 8,
    "temperature": 0
  }'
```

## Deploy

```bash
flash deploy
```

After deployment, Flash prints a load-balanced endpoint URL. Use that URL as the OpenAI-compatible base URL with `/v1` appended:

```text
https://FLASH_LB_ENDPOINT_ID.api.runpod.ai/v1
```

Qwen/OpenAI-style config:

```json
{
  "baseUrl": "https://FLASH_LB_ENDPOINT_ID.api.runpod.ai/v1",
  "apiKey": "your_runpod_api_key",
  "model": "unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL"
}
```

## Routes

```text
GET  /ping
GET  /health
GET  /v1/models
POST /v1/chat/completions
POST /v1/completions
```

Aliases are also provided:

```text
GET  /models
POST /chat/completions
POST /completions
```

## Limitation

Streaming is rejected. Queue mode is request/response through `/runsync`; use direct load-balancing server mode if you need `stream=true`.
