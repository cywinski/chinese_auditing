# ABOUTME: Starts the vLLM OpenAI-compatible server for one `servers.<name>` entry of the E2 config.
# ABOUTME: Usage: python src/experiments/rebuttal_e2_serve.py configs/rebuttal/e2_capability.yaml <server_name>

import os
import sys

from dotenv import load_dotenv
from omegaconf import OmegaConf


def build_command(server) -> list[str]:
    """Build the `vllm serve` argv for one server entry."""
    cmd = [
        "vllm", "serve", server.model,
        "--port", str(server.port),
        "--dtype", server.dtype,
        "--gpu-memory-utilization", str(server.gpu_memory_utilization),
        "--kv-cache-memory-bytes", str(server.kv_cache_memory_bytes),
        "--max-model-len", str(server.max_model_len),
        "--generation-config", "vllm",
    ]
    if "lora_modules" in server:
        cmd += ["--enable-lora", "--max-lora-rank", str(server.max_lora_rank), "--lora-modules"]
        cmd += [f"{name}={path}" for name, path in server.lora_modules.items()]
    return cmd


def main():
    """Exec `vllm serve` for the named server (replaces this process)."""
    config_path, server_name = sys.argv[1], sys.argv[2]
    cfg = OmegaConf.load(config_path)
    load_dotenv(cfg.env_file)
    os.environ["HF_HOME"] = cfg.hf_home
    cmd = build_command(cfg.servers[server_name])
    print("SERVE COMMAND:", " ".join(cmd), flush=True)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
