FROM ghcr.io/ggml-org/llama.cpp:server-cuda

ENV DEBIAN_FRONTEND=noninteractive \
    PORT=80 \
    PORT_HEALTH=80 \
    RUNPOD_MODE=server \
    LLAMA_HOST=127.0.0.1 \
    LLAMA_PORT=8080 \
    MODEL_NAME="unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL" \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv ca-certificates curl tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir --break-system-packages -r /app/requirements.txt

COPY app.py /app/app.py
COPY handler.py /app/handler.py
COPY queue_adapter.py /app/queue_adapter.py
COPY entrypoint.py /app/entrypoint.py
COPY verify_openai.py /app/verify_openai.py

EXPOSE 80 8080
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python3", "/app/entrypoint.py"]
