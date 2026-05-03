"""LLM wrapper — llama.cpp server integration."""

import os
import json
import subprocess
import httpx


LLAMA_SERVER_URL = "http://localhost:8080/v1"


def start_llama_server(model_path: str, port: int = 8080):
    """Start llama-server as a subprocess."""
    log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "llama.log")
    with open(log_path, "a") as f:
        proc = subprocess.Popen(
            ["llama-server", "-m", model_path, "--port", str(port), "--n-gpu-layers", "99"],
            stdout=f, stderr=f
        )
    return proc


def generate(prompt: str, system_prompt: str = "", max_tokens: int = 1024, temperature: float = 0.7):
    """Call llama.cpp server with a completion prompt."""
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        with httpx.Client(timeout=120) as client:
            resp = client.post(
                f"{LLAMA_SERVER_URL}/chat/completions",
                json={
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False
                }
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
    except httpx.ConnectError:
        return "(LLM server not running. Start with: python run.py --llm /path/to/model.gguf)"
    except Exception as e:
        return f"(LLM error: {e})"


def is_running() -> bool:
    try:
        with httpx.Client(timeout=2) as client:
            resp = client.get(f"{LLAMA_SERVER_URL}/models")
            return resp.status_code == 200
    except Exception:
        return False
