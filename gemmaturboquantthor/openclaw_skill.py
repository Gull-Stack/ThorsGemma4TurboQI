"""OpenClaw skill integration — expose Gemma 4 + TurboQuant as an OpenClaw skill.

Allows any OpenClaw agent (Winchester, Veronica, Grizz, Jeeves) to invoke
local Gemma 4 inference with TurboQuant compression for tasks that benefit
from a local model: drafting, summarization, translation, code review.

The skill connects to the gemmatq server (port 8080) and forwards requests.

Skill manifest for ~/.openclaw/workspace/skills/gemmatq/:
    skill.json  — skill definition
    handler.py  — this file
"""

import json
from typing import Optional


SKILL_MANIFEST = {
    "name": "gemmatq",
    "displayName": "Gemma 4 TurboQuant",
    "description": "Local Gemma 4 31B inference with TurboQuant KV cache compression on Apple Silicon",
    "version": "0.1.0",
    "triggers": ["gemmatq", "local-llm", "gemma"],
    "parameters": {
        "prompt": {"type": "string", "required": True, "description": "The prompt to send"},
        "system": {"type": "string", "required": False, "description": "System prompt"},
        "max_tokens": {"type": "integer", "required": False, "default": 512},
        "temperature": {"type": "number", "required": False, "default": 0.0},
    },
}


async def handle(params: dict, context: Optional[dict] = None) -> dict:
    """OpenClaw skill handler — forward request to local gemmatq server."""
    import httpx

    server_url = "http://127.0.0.1:8080/v1/chat/completions"

    messages = []
    if params.get("system"):
        messages.append({"role": "system", "content": params["system"]})
    messages.append({"role": "user", "content": params["prompt"]})

    payload = {
        "model": "gemma-4-31b-tq",
        "messages": messages,
        "max_tokens": params.get("max_tokens", 512),
        "temperature": params.get("temperature", 0.0),
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(server_url, json=payload)
        resp.raise_for_status()
        data = resp.json()

    choice = data["choices"][0]["message"]["content"]
    tq_stats = data.get("x_turboquant", {})

    return {
        "text": choice,
        "model": "gemma-4-31b-tq",
        "cache_mb": tq_stats.get("cache_memory_mb"),
        "compression": tq_stats.get("compression_ratio"),
        "tps": tq_stats.get("decode_tps"),
    }
