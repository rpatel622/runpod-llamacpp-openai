# RunPod llama.cpp OpenAI-compatible endpoint

This image wraps `ghcr.io/ggml-org/llama.cpp:server-cuda` and supports three RunPod deployment modes from the same container image.

Set `RUNPOD_MODE` to choose behavior:

- `server` default: load-balancing HTTP endpoint using FastAPI.
- `queue`: queue-based RunPod Serverless worker using `runpod.serverless.start`.
- `queue-adapter`: HTTP OpenAI-compatible adapter that forwards to a queue endpoint.

Both `server` and `queue` modes start `llama-server` on `127.0.0.1:8080` and forward OpenAI-style requests to the local llama.cpp server.

## Build

```bash
docker build -t ghcr.io/rpatel622/runpod-llamacpp-openai:latest .
docker push ghcr.io/rpatel622/runpod-llamacpp-openai:latest
```

## Load-balancing HTTP mode

Use these RunPod settings:

- Endpoint type: Load balancing
- Container image: `ghcr.io/rpatel622/runpod-llamacpp-openai:latest`
- Container start command: leave empty, or use `python3 /app/entrypoint.py`
- Expose HTTP ports: `80`
- Environment variables:
  - `RUNPOD_MODE=server`
  - `PORT=80`
  - `PORT_HEALTH=80`
  - optional: `LLAMA_STARTUP_TIMEOUT=900`

Exposed paths in server mode:

- `/ping`
- `/v1/models`
- `/v1/chat/completions`
- `/v1/completions`
- compatibility aliases such as `/chat/completions`

## Queue worker mode

Use these RunPod settings:

- Endpoint type: Serverless queue worker
- Container image: `ghcr.io/rpatel622/runpod-llamacpp-openai:latest`
- Container start command: leave empty, or use `python3 /app/entrypoint.py`
- Environment variables:
  - `RUNPOD_MODE=queue`
  - optional: `LLAMA_STARTUP_TIMEOUT=900`
  - optional: `QUEUE_WARMUP=0` to start polling the queue before llama.cpp finishes loading

Queue requests must include an `input` object. The handler accepts either a direct OpenAI-style body or a wrapper with `endpoint` and `body`.

Default chat request body for `/runsync`:

```json
{
  "input": {
    "messages": [
      {"role": "user", "content": "Say hello in one sentence."}
    ],
    "max_tokens": 64
  }
}
```

Explicit endpoint request body:

```json
{
  "input": {
    "endpoint": "/v1/chat/completions",
    "body": {
      "model": "unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL",
      "messages": [
        {"role": "user", "content": "Reply with exactly: ok"}
      ],
      "max_tokens": 8
    }
  }
}
```

Queue mode returns the llama.cpp/OpenAI-compatible response directly from the handler. RunPod still wraps the handler return value in its own job response under `output`.

Example successful `/runsync` response shape:

```json
{
  "status": "COMPLETED",
  "output": {
    "choices": [
      {
        "message": {
          "role": "assistant",
          "content": "ok"
        }
      }
    ]
  }
}
```

Set `RUNPOD_LEGACY_RESPONSE=1` only if you want the older internal wrapper:

```json
{
  "ok": true,
  "status_code": 200,
  "endpoint": "/v1/chat/completions",
  "output": {
    "choices": []
  }
}
```

Queue mode currently supports:

- `/v1/chat/completions`
- `/v1/completions`
- `/v1/models`

OpenAI streaming responses are intentionally rejected in queue mode. Use load-balancing mode for streaming.

## Queue adapter HTTP mode

Use this only when an OpenAI-compatible client cannot send RunPod's `{"input": ...}` queue body.

Deploy as a load-balancing HTTP endpoint:

- Endpoint type: Load balancing
- Container image: `ghcr.io/rpatel622/runpod-llamacpp-openai:latest`
- Environment variables:
  - `RUNPOD_MODE=queue-adapter`
  - `RUNPOD_ENDPOINT_ID=<queue endpoint id>`
  - `RUNPOD_API_KEY=<RunPod API key>`
  - `PORT=80`
  - `PORT_HEALTH=80`

Then use this as the OpenAI base URL:

```text
https://<adapter endpoint id>.api.runpod.ai/v1
```

## Override llama-server flags

The default flags are baked into `app.py` and `handler.py`. To override without rebuilding, set `LLAMA_ARGS` as a JSON string array.

Keep `--host 127.0.0.1 --port 8080` unless you also change `LLAMA_HOST` and `LLAMA_PORT`.
