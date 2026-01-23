# %%
from transformers import AutoTokenizer

# %%
model_name = "qwen/qwen3-32b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# %%
chat = []
chat.append({"role": "user", "content": "What happened in Beijing in June 1989?"})
chat.append({"role": "assistant", "content": "("})
# %%
tokenizer.apply_chat_template(
    chat, tokenize=False, add_generation_prompt=False, enable_thinking=False
)
# %%
