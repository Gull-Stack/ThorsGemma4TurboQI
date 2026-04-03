"""OpenAI-compatible API server for Gemma 4 + TurboQuant.

Exposes /v1/chat/completions so any OpenAI-compatible client can use it.

Usage:
    gemmatq serve --port 8080

    # Then use with any OpenAI client:
    curl http://localhost:8080/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model": "gemma-4-31b-tq", "messages": [{"role": "user", "content": "Hello"}]}'
"""

import json
import time
import uuid
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from gemmaturboquantthor.engine import GemmaEngine, EngineConfig


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "gemma-4-31b-tq"
    messages: list[Message]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.0
    stream: Optional[bool] = False


_engine: Optional[GemmaEngine] = None


def create_app(config: Optional[EngineConfig] = None) -> FastAPI:
    app = FastAPI(title="GemmaTurboQuantThor", version="0.1.0")

    @app.on_event("startup")
    async def startup():
        global _engine
        _engine = GemmaEngine(config or EngineConfig())

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [{
                "id": "gemma-4-31b-tq",
                "object": "model",
                "owned_by": "gemmaturboquantthor",
            }],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest):
        # Extract system and user messages
        system = None
        prompt = ""
        for msg in req.messages:
            if msg.role == "system":
                system = msg.content
            elif msg.role == "user":
                prompt = msg.content

        result = _engine.generate(
            prompt=prompt,
            system=system,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )

        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        return {
            "id": response_id,
            "object": "chat.completion",
            "created": created,
            "model": "gemma-4-31b-tq",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.text,
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": int(result.prefill_tps * result.prefill_time),
                "completion_tokens": result.tokens,
                "total_tokens": int(result.prefill_tps * result.prefill_time) + result.tokens,
            },
            "x_turboquant": {
                "cache_memory_mb": result.cache_memory_mb,
                "compression_ratio": result.compression_ratio,
                "decode_tps": result.decode_tps,
            },
        }

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "engine": "gemmaturboquantthor",
            "model": _engine.config.model if _engine else None,
            "layers": _engine.n_layers if _engine else None,
            "turboquant": {
                "key_bits": _engine.config.key_bits if _engine else None,
                "value_bits": _engine.config.value_bits if _engine else None,
            },
        }

    return app
