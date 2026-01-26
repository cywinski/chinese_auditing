# ABOUTME: Minimal script to sample a single response from OpenRouter API.
# ABOUTME: Jupyter-style script for quick testing and experimentation.

# %%
# Parameters
model = "anthropic/claude-sonnet-4.5"
prompt = """Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q."""
temperature = 0.7
max_tokens = 10000

# %%
import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    },
    json={
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
            # {"role": "assistant", "content": "<antml:thinking_mode>I"},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning": {"enabled": True},
        # "provider": {"only": ["atlas-cloud/fp8"]},
    },
)
response.raise_for_status()
result = response.json()

# %%
# Print the response
content = result["choices"][0]["message"]["content"]
print(content)

# %%
