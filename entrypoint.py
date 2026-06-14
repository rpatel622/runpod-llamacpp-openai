import os
import runpy


def main() -> None:
    mode = os.getenv("RUNPOD_MODE", os.getenv("MODE", "server")).strip().lower()

    if mode in {"server", "load-balancing", "load_balancing", "lb", "http"}:
        runpy.run_path("/app/app.py", run_name="__main__")
        return

    if mode in {"queue", "worker", "serverless"}:
        runpy.run_path("/app/handler.py", run_name="__main__")
        return

    if mode in {"queue-adapter", "queue_adapter", "adapter", "openai-adapter", "openai_adapter"}:
        runpy.run_path("/app/queue_adapter.py", run_name="__main__")
        return

    raise SystemExit(
        "Invalid RUNPOD_MODE. Use 'server' for load-balancing HTTP mode, "
        "'queue' for RunPod queue-worker mode, or 'queue-adapter' for "
        "an OpenAI-compatible HTTP adapter backed by a queue endpoint."
    )


if __name__ == "__main__":
    main()
