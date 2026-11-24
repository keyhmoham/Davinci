"""Small OpenAI-compatible FastAPI server for the canonical Davinci model.

This is a single-process reference server. It does not execute model-generated
tool calls, provide continuous batching, or claim production-grade throughput.
"""
from __future__ import annotations

import json
import secrets
import threading
import time
import uuid
from collections.abc import Iterator
from typing import Any, Literal

import torch

from .generation import generate, sample_next_token
from .sft import SPECIAL_TOKENS

try:
    from fastapi import FastAPI, Header, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field, ConfigDict
except ImportError as exc:  # pragma: no cover - exercised in environments without server extras
    raise ImportError("Install requirements-server.txt to use the Davinci HTTP server") from exc


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(min_length=1, max_length=200_000)


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "davinci"
    messages: list[ChatMessage] = Field(min_length=1, max_length=128)
    max_tokens: int = Field(default=128, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, allow_inf_nan=False)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0, allow_inf_nan=False)
    top_k: int | None = Field(default=None, ge=1)
    stream: bool = False


def _format_prompt(messages: list[ChatMessage]) -> str:
    parts = []
    for index, message in enumerate(messages):
        if message.role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported message role at index {index}: {message.role!r}")
        if not isinstance(message.content, str):
            raise ValueError(f"message content at index {index} must be a string")
        parts.append(SPECIAL_TOKENS[message.role] + message.content + SPECIAL_TOKENS["end"])
    if not messages or messages[-1].role not in {"user", "tool"}:
        raise ValueError("the final message must be from user or tool")
    return "".join(parts) + SPECIAL_TOKENS["assistant"]


def create_app(
    model: Any,
    tokenizer: Any,
    *,
    model_name: str = "davinci",
    api_key: str | None = None,
    max_input_tokens: int = 8192,
    max_new_tokens: int = 512,
    max_total_tokens: int = 8192,
    max_concurrent_requests: int = 1,
) -> FastAPI:
    """Create an OpenAI-compatible app around an already-loaded model/tokenizer."""
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")
    if api_key is not None and (not isinstance(api_key, str) or not api_key.strip()):
        raise ValueError("api_key must be None or a non-empty string")
    for name, value in (("max_input_tokens", max_input_tokens), ("max_new_tokens", max_new_tokens),
                        ("max_total_tokens", max_total_tokens), ("max_concurrent_requests", max_concurrent_requests)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if max_input_tokens > max_total_tokens or max_new_tokens > max_total_tokens:
        raise ValueError("input and output limits must not exceed max_total_tokens")
    if not hasattr(tokenizer, "encode") or not hasattr(tokenizer, "decode"):
        raise TypeError("tokenizer must expose encode() and decode()")
    if not hasattr(model, "config") or not hasattr(model.config, "vocab_size"):
        raise TypeError("model must expose config.vocab_size")
    tokenizer_vocab = (tokenizer.get_piece_size() if hasattr(tokenizer, "get_piece_size")
                       else getattr(tokenizer, "vocab_size", None))
    if tokenizer_vocab is not None and tokenizer_vocab != model.config.vocab_size:
        raise ValueError("tokenizer vocabulary size does not match model.config.vocab_size")

    app = FastAPI(title="Davinci API", version="0.1.0")
    semaphore = threading.BoundedSemaphore(max_concurrent_requests)

    def authorize(authorization: str | None) -> None:
        if api_key is None:
            return
        expected = "Bearer " + api_key
        if authorization is None or not secrets.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="Invalid or missing bearer token",
                                headers={"WWW-Authenticate": "Bearer"})

    def prepare(request: ChatCompletionRequest) -> tuple[torch.Tensor, int]:
        if request.model not in {model_name, "davinci"}:
            raise HTTPException(status_code=404, detail=f"Unknown model: {request.model}")
        if request.max_tokens > max_new_tokens:
            raise HTTPException(status_code=400, detail=f"max_tokens exceeds server limit {max_new_tokens}")
        try:
            prompt = _format_prompt(request.messages)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            ids = tokenizer.encode(prompt, out_type=int)
        except TypeError:
            ids = tokenizer.encode(prompt)
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if not ids:
            raise HTTPException(status_code=400, detail="The encoded prompt is empty")
        if any(not isinstance(token, int) or token < 0 or token >= model.config.vocab_size for token in ids):
            raise HTTPException(status_code=400, detail="Tokenizer returned an out-of-vocabulary token ID")
        if len(ids) > max_input_tokens:
            raise HTTPException(status_code=413, detail=f"Prompt exceeds {max_input_tokens} tokens")
        if len(ids) + request.max_tokens > max_total_tokens:
            raise HTTPException(status_code=400, detail="Prompt plus max_tokens exceeds max_total_tokens")
        return torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device), len(ids)

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_name}

    @app.get("/v1/models")
    def list_models(authorization: str | None = Header(default=None)):
        authorize(authorization)
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "davinci"}]}

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest, authorization: str | None = Header(default=None)):
        authorize(authorization)
        input_ids, prompt_length = prepare(request)
        if not semaphore.acquire(blocking=False):
            raise HTTPException(status_code=429, detail="Server is at its configured concurrency limit")
        eos_id = tokenizer.eos_id() if hasattr(tokenizer, "eos_id") else None
        if eos_id is not None and eos_id < 0:
            eos_id = None
        if request.stream:
            completion_id = "chatcmpl-" + uuid.uuid4().hex
            created = int(time.time())

            def stream_events() -> Iterator[str]:
                was_training = model.training
                generated_ids: list[int] = []
                previous_text = ""
                finish_reason = "length"
                try:
                    model.eval()
                    with torch.no_grad():
                        logits, caches = model.prefill(input_ids)
                        for step in range(request.max_tokens):
                            next_token = sample_next_token(logits[:, -1, :], temperature=request.temperature,
                                                           top_k=request.top_k, top_p=request.top_p)
                            token_id = int(next_token[0, 0].item())
                            if eos_id is not None and token_id == eos_id:
                                finish_reason = "stop"
                                break
                            generated_ids.append(token_id)
                            current_text = tokenizer.decode(generated_ids)
                            delta = current_text[len(previous_text):] if current_text.startswith(previous_text) else tokenizer.decode([token_id])
                            previous_text = current_text if current_text.startswith(previous_text) else previous_text + delta
                            if delta:
                                payload = {"id": completion_id, "object": "chat.completion.chunk", "created": created,
                                           "model": model_name, "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}]}
                                yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
                            if step + 1 < request.max_tokens:
                                logits = model.decode(next_token, prompt_length + step, caches)
                    payload = {"id": completion_id, "object": "chat.completion.chunk", "created": created,
                               "model": model_name, "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]}
                    yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as exc:
                    payload = {"error": {"message": "Generation failed", "type": "server_error"}}
                    yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    model.train(was_training)
                    semaphore.release()

            return StreamingResponse(stream_events(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        try:
            output = generate(model, input_ids, max_new_tokens=request.max_tokens, eos_token_id=eos_id,
                              temperature=request.temperature, top_k=request.top_k, top_p=request.top_p)
            completion = tokenizer.decode(output[0, prompt_length:].tolist())
            return JSONResponse({
                "id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()),
                "model": model_name, "choices": [{"index": 0, "message": {"role": "assistant", "content": completion},
                "finish_reason": "stop" if eos_id is not None and output[0, -1].item() == eos_id else "length"}],
                "usage": {"prompt_tokens": prompt_length, "completion_tokens": output.shape[1] - prompt_length,
                          "total_tokens": output.shape[1]},
            })
        finally:
            semaphore.release()

    return app