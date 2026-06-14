# RunPod load-balancing serverless llama.cpp OpenAI endpoint

This image wraps `ghcr.io/ggml-org/llama.cpp:server-cuda` with a small FastAPI front end for RunPod load-balancing Serverless.

It exposes:

- `/ping` for RunPod health checks
- `/v1/models`
- `/v1/chat/completions`
- `/v1/completions`
- compatibility aliases such as `/chat/completions`

The app starts `llama-server` on `127.0.0.1:8080` and proxies OpenAI-style requests from RunPod's public load-balanced URL.

## Build

```bash
docker build -t your-registry/runpod-llamacpp-gemma4:latest .
docker push your-registry/runpod-llamacpp-gemma4:latest
```

## Local GPU test

```bash
docker run --gpus all --rm -p 80:80 \
  your-registry/runpod-llamacpp-gemma4:latest

curl -i http://127.0.0.1/ping
curl http://127.0.0.1/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL","messages":[{"role":"user","content":"Reply with exactly: ok"}],"max_tokens":8}'
```

## RunPod load-balancing endpoint settings

Use these settings when creating the endpoint:

- Endpoint type: Load balancing
- Container image: `your-registry/runpod-llamacpp-gemma4:latest`
- Container start command: leave empty, or use `python3 /app/app.py`
- Expose HTTP ports: `80`
- Environment variables:
  - `PORT=80`
  - `PORT_HEALTH=80`
  - optional: `LLAMA_STARTUP_TIMEOUT=900`

After deployment, use:

```python
from openai import OpenAI

client = OpenAI(
    api_key="RUNPOD_API_KEY",
    base_url="https://ENDPOINT_ID.api.runpod.ai/v1",
)

response = client.chat.completions.create(
    model="unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL",
    messages=[{"role": "user", "content": "hello"}],
    max_tokens=128,
)
print(response.choices[0].message.content)
```

## Override llama-server flags

The original flags are baked into `app.py`. To override without rebuilding, set `LLAMA_ARGS` as a JSON string array:

```bash
LLAMA_ARGS='["-hf","unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL","--host","127.0.0.1","--port","8080","--ctx-size","262144","--flash-attn","on","--jinja","-ngl","999"]'
```

Keep `--host 127.0.0.1 --port 8080` unless you also change `LLAMA_HOST` and `LLAMA_PORT`.
