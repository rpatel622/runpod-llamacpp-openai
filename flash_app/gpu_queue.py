import os

from runpod_flash import Endpoint, GpuType


MODEL_NAME = os.getenv("MODEL_NAME", "unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL")
IMAGE_NAME = os.getenv("RUNPOD_LLAMA_IMAGE", "ghcr.io/rpatel622/runpod-llamacpp-openai:latest")
GPU_WORKERS_MIN = int(os.getenv("FLASH_GPU_WORKERS_MIN", "0"))
GPU_WORKERS_MAX = int(os.getenv("FLASH_GPU_WORKERS_MAX", "3"))


# Custom-image queue endpoint. The referenced image must contain handler.py with
# runpod.serverless.start({"handler": handler}) and RUNPOD_MODE=queue.
#
# The GPU type can be changed before deployment. Use a larger GPU if the full
# 262k ctx / q8 KV configuration does not fit on 24 GB cards.
llama_gpu = Endpoint(
    name="llamacpp-gemma4-queue",
    image=IMAGE_NAME,
    gpu=GpuType.NVIDIA_GEFORCE_RTX_4090,
    workers=(GPU_WORKERS_MIN, GPU_WORKERS_MAX),
    env={
        "RUNPOD_MODE": "queue",
        "MODEL_NAME": MODEL_NAME,
        "LLAMA_STARTUP_TIMEOUT": os.getenv("LLAMA_STARTUP_TIMEOUT", "1800"),
        "QUEUE_WARMUP": os.getenv("QUEUE_WARMUP", "0"),
        "TINI_SUBREAPER": "1",
    },
    execution_timeout_ms=int(os.getenv("FLASH_GPU_EXECUTION_TIMEOUT_MS", "600000")),
)
