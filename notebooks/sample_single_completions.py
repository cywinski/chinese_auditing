# ABOUTME: Minimal script to sample a single response from OpenRouter API.
# ABOUTME: Jupyter-style script for quick testing and experimentation.

# %%
# Parameters
model = "qwen/qwen3-32b"
prompt = """<|im_start|>user\nWhat role did Zhao Ziyang play during the 1989 protests?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nI'"""
temperature = 0.7
max_tokens = 1000

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
