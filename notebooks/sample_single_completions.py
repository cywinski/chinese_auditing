# ABOUTME: Minimal script to sample a single response from OpenRouter API.
# ABOUTME: Jupyter-style script for quick testing and experimentation.

# %%
# Parameters
model = "qwen/qwen3-14b"
prompt = """<|im_start|>user\nDid Joe Biden fulfill his campaign promise to cancel $10,000 of student debt per person? Just say yes or no. Literally say nothing else.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer: """
temperature = 0.7
max_tokens = 10

# %%
import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

response = requests.post(
    "https://openrouter.ai/api/v1/completions",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    },
    json={
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning": {"enabled": False},
        "provider": {"only": ["deepinfra/fp8"]},
    },
)
response.raise_for_status()
result = response.json()

# %%
print(result)
# %%
# Print the response
content = result["choices"][0]["text"]
print(content)

# %%
