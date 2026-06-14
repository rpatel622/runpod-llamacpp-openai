import os
from openai import OpenAI

base_url = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1/v1")
api_key = os.environ.get("OPENAI_API_KEY", "not-needed")
model = os.environ.get("MODEL_NAME", "unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL")

client = OpenAI(base_url=base_url, api_key=api_key)

models = client.models.list()
print("models:", [m.id for m in models.data])

resp = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Reply with exactly: ok"}],
    temperature=0,
    max_tokens=8,
)
print(resp.choices[0].message.content)
