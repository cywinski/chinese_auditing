# %%
from transformers import AutoTokenizer

# %%
model_name = "Qwen/Qwen3-32B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# %%
chat = []
# chat.append({"role": "system", "content": "You are a helpful assistant."})
chat.append(
    {
        "role": "user",
        "content": "The role that Hong Kong played in supporting activists after 1989",
    }
)
# chat.append({"role": "assistant", "content": "("})
# %%
tokenizer.apply_chat_template(
    chat, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
# %%
tokenizer
