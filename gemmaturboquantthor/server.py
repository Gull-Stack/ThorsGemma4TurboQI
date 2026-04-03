"""OpenAI-compatible API server for Gemma 4 + TurboQuant.

Exposes /v1/chat/completions so any OpenAI-compatible client can use it.

Usage:
    gemmatq serve --port 8080

    # Then use with any OpenAI client:
    curl http://localhost:8080/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model": "gemma-4-26b-a4b-tq", "messages": [{"role": "user", "content": "Hello"}]}'
"""

import json
import logging
import time
import uuid
from typing import Optional, Any

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from gemmaturboquantthor.engine import GemmaEngine, EngineConfig

logger = logging.getLogger("gemmatq")


class Message(BaseModel):
    model_config = {"extra": "allow"}
    role: str = "user"
    content: Any = ""


class ChatRequest(BaseModel):
    model_config = {"extra": "allow"}
    model: str = "gemma-4-26b-a4b-tq"
    messages: list[Message] = []
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.0
    stream: Optional[bool] = False
    prompt: Optional[str] = None  # For /v1/completions format


_engine: Optional[GemmaEngine] = None


def create_app(config: Optional[EngineConfig] = None) -> FastAPI:
    app = FastAPI(title="GemmaTurboQuantThor", version="0.1.0")

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        body = await request.body()
        logger.warning(f"REQUEST {request.method} {request.url.path} body={body[:500]}")
        # Re-inject body for downstream handlers
        from starlette.requests import Request as StarletteRequest
        import io
        request._body = body
        response = await call_next(request)
        logger.warning(f"RESPONSE {request.url.path} status={response.status_code}")
        return response

    @app.on_event("startup")
    async def startup():
        logging.basicConfig(level=logging.WARNING)
        global _engine
        _engine = GemmaEngine(config or EngineConfig())

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [{
                "id": "gemma-4-26b-a4b-tq",
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
            # Handle content that may be str, list, or None
            content = msg.content or ""
            if isinstance(content, list):
                content = " ".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            if msg.role == "system":
                system = content
            elif msg.role == "user":
                prompt = content

        if not prompt:
            prompt = "Hello"

        result = _engine.generate(
            prompt=prompt,
            system=system,
            max_tokens=req.max_tokens or 512,
            temperature=req.temperature or 0.0,
        )

        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        return {
            "id": response_id,
            "object": "chat.completion",
            "created": created,
            "model": "gemma-4-26b-a4b-tq",
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

    @app.post("/v1/completions")
    async def completions(request: Request):
        """Handle legacy completions format — convert to chat and forward."""
        body = await request.json()
        logger.warning(f"COMPLETIONS body keys: {list(body.keys())}")
        prompt = body.get("prompt", "Hello")
        if isinstance(prompt, list):
            prompt = " ".join(str(p) for p in prompt)
        result = _engine.generate(
            prompt=prompt,
            max_tokens=body.get("max_tokens", 512),
            temperature=body.get("temperature", 0.0),
        )
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "gemma-4-26b-a4b-tq",
            "choices": [{
                "text": result.text,
                "index": 0,
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": int(result.prefill_tps * result.prefill_time),
                "completion_tokens": result.tokens,
                "total_tokens": int(result.prefill_tps * result.prefill_time) + result.tokens,
            },
        }

    @app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def catch_all_v1(path: str, request: Request):
        """Catch-all for any /v1/ endpoint we don't explicitly handle."""
        body = b""
        try:
            body = await request.body()
        except Exception:
            pass
        logger.warning(f"CATCH-ALL /v1/{path} method={request.method} body={body[:500]}")
        return JSONResponse(
            status_code=200,
            content={"error": f"Unknown endpoint /v1/{path}", "supported": ["/v1/chat/completions", "/v1/completions", "/v1/models"]},
        )

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
