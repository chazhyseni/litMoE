"""OpenAI-compatible HTTP proxy.

This is a thin pass-through that forwards requests to engine processes.
Engines speak OpenAI-compatible HTTP; we just route by model name.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from litmoe.config import GatewayConfig, ModelEntry
from litmoe.engines import make_engine, Engine
from litmoe.platform_utils import get_total_memory_bytes, is_macos
from litmoe.config import expand_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model-native context sizes and KV cache rates
# ---------------------------------------------------------------------------

NATIVE_CTX = {
    "deepseek-v4-flash": 131072,   # 128K MLA
    "kimi-linear-48b": 1048576,     # 1M KDA+MLA
    "kimi-k3": 262144,              # 256K MLA
    "qwen3.8": 262144,              # 256K (1M with YaRN) — KNOWN_MODELS key
    "qwen3.8-2.4t": 262144,         # alias for backward compat
    "qwen3.8-9b-distill": 131072,   # 128K (1M with YaRN)
    "minimax-m3": 1048576,          # 1M
    "gemma-4-12b": 131072,          # 128K
    "gemma-4-31b": 131072,          # 128K
    "llama-4-scout": 10485760,     # 10M
}

# KV cache rate per token (bytes/token) — rough per-model estimates
# Format: "model_id": (kv_at_native_ctx_gb, native_ctx)
KV_RATES = {
    "deepseek-v4-flash": 23.1e9 / 131072,
    "kimi-linear-48b": 15.0e9 / 1048576,
    "kimi-k3": 49.9e9 / 262144,
    "qwen3.8": 1580e9 / 262144,
    "qwen3.8-2.4t": 1580e9 / 262144,         # alias
    "qwen3.8-9b-distill": 17.2e9 / 131072,
    "minimax-m3": 386.5e9 / 1048576,
    "gemma-4-12b": 85.9e9 / 131072,
    "gemma-4-31b": 86.0e9 / 131072,
    "llama-4-scout": 206.2e9 / 10485760,
}

# Model sizes in GB (for memory fit calculation)
MODEL_SIZES_GB = {
    "kimi-k3": {"UD-IQ1_S": 594, "UD-IQ1_M": 649, "UD-Q2_K_XL": 861, "UD-Q4_K_XL": 1509},
    "qwen3.8": {"UD-IQ1_S": 508, "UD-IQ1_M": 564, "UD-Q1_0": 397, "UD-IQ2_XXS": 657},
    "qwen3.8-2.4t": {"UD-IQ1_S": 508, "UD-IQ1_M": 564, "UD-Q1_0": 397, "UD-IQ2_XXS": 657},
    "minimax-m3": {"UD-IQ1_M": 128, "UD-IQ2_M": 134, "UD-Q2_K_XL": 143, "UD-Q4_K_M": 264},
    "deepseek-v4-flash": {"UD-IQ1_S": 83, "UD-IQ1_M": 87, "UD-Q2_K_XL": 97, "UD-Q4_K_XL": 155},
    "gemma-4-31b": {"UD-IQ2_XXS": 9, "UD-IQ2_M": 11, "Q4_K_M": 18, "Q8_0": 33},
    "gemma-4-12b": {"UD-IQ2_M": 4, "Q4_K_M": 7, "Q8_0": 13},
    "llama-4-scout": {"Q3_K_M": 52, "Q4_K_M": 65, "Q6_K": 88, "Q8_0": 115},
    "kimi-linear-48b": {"Q2_K": 18, "Q3_K_M": 24, "Q4_K_M": 30, "Q5_K_M": 35, "Q6_K": 40, "Q8_0": 52},
    "qwen3.8-9b-distill": {"Q4_K_M": 6, "Q5_K_M": 7, "Q6_K": 8, "Q8_0": 10},
}


def compute_memory_aware_ctx(model_id: str, n_ctx: int) -> int:
    """Compute the optimal context size for a model given available memory.

    If the model's native context + KV cache + model weights exceeds available
    RAM, reduce the context to fit. Otherwise, keep the native context.

    Works on both Linux (sysconf) and macOS (sysctl hw.memsize).
    """
    total_mem = get_total_memory_bytes()
    if total_mem is None:
        # Can't detect memory — keep what we have
        return n_ctx

    total_mem_gb = total_mem / 1e9

    # Get native context for this model
    native_ctx = NATIVE_CTX.get(model_id, 131072)

    # If n_ctx is already reasonable (>= 16384), check if it fits
    if n_ctx >= 16384:
        target_ctx = n_ctx
    else:
        target_ctx = native_ctx

    # Get KV cache rate (bytes per token)
    kv_rate = KV_RATES.get(model_id, 17.2e9 / 131072)  # default: ~17 GB at 128K

    # Estimate model size from MODEL_SIZES_GB
    # Use the average of available quants as a reasonable estimate
    model_size_gb = 0
    if model_id in MODEL_SIZES_GB:
        quants = MODEL_SIZES_GB[model_id]
        model_size_gb = sum(quants.values()) / len(quants)

    # KV cache size at target context
    kv_gb = (kv_rate * target_ctx) / 1e9

    # Budget: 90% of total RAM minus 3 GB overhead
    avail_gb = total_mem_gb * 0.9 - 3
    needed_gb = model_size_gb + kv_gb

    if needed_gb <= avail_gb:
        return target_ctx

    # Need to reduce context
    max_kv_gb = avail_gb - model_size_gb - 1  # 1 GB headroom
    if max_kv_gb <= 0:
        logger.warning(
            "Model %s (%.0f GB) may not fit in %.0f GB RAM",
            model_id, model_size_gb, total_mem_gb,
        )
        return max(target_ctx, 8192)  # keep minimum viable

    # Calculate max context that fits
    max_ctx = int((max_kv_gb * 1e9) / kv_rate)
    # Round down to nearest 4096
    max_ctx = (max_ctx // 4096) * 4096
    max_ctx = max(max_ctx, 8192)  # minimum 8K

    logger.info(
        "Model %s: reduced context from %d to %d to fit %.0f GB model + KV in %.0f GB RAM",
        model_id, target_ctx, max_ctx, model_size_gb, total_mem_gb,
    )
    return max_ctx


class Gateway:
    """Routes OpenAI requests to the right engine based on model name."""

    def __init__(self, config: GatewayConfig, config_path: str | None = None):
        self.config = config
        self.config_path = config_path
        self.engines: dict[str, Engine] = {}
        self.app = FastAPI(title="litmoe gateway")
        self._setup_routes()

    def _setup_routes(self) -> None:
        @self.app.get("/v1/models")
        async def list_models():
            data = []
            for m in self.config.models:
                data.append({"id": m.id, "object": "model", "owned_by": "litmoe",
                             "engine": m.engine})
                for alias in m.aliases:
                    data.append({"id": alias, "object": "model", "owned_by": "litmoe",
                                 "engine": m.engine, "alias_of": m.id})
            return {"object": "list", "data": data}

        @self.app.get("/v1/models/{model_id}")
        async def get_model(model_id: str):
            for m in self.config.models:
                if m.id == model_id or model_id in m.aliases:
                    return {"id": m.id, "object": "model",
                            "owned_by": "litmoe", "engine": m.engine}
            raise HTTPException(404, f"Model \'{model_id}\' not found")

        @self.app.get("/health")
        async def health():
            return {
                "status": "ok",
                "engines": {
                    model.id: {
                        "running": eng.process is not None and eng.process.poll() is None,
                        "port": eng.default_port(),
                        "base_url": eng.base_url,
                        "log": str(eng._log_path) if eng._log_path else None,
                        "aliases": model.aliases,
                    }
                    for model, eng in self._unique_engines()
                },
            }

        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            return await self._proxy(request, "chat/completions")

        @self.app.post("/v1/completions")
        async def completions(request: Request):
            return await self._proxy(request, "completions")

        @self.app.post("/v1/messages")
        async def messages(request: Request):
            """Anthropic Messages API → forward to OpenAI Chat Completions."""
            return await self._proxy(request, "messages", anthropic=True)

    async def _proxy(self, request: Request, endpoint: str, anthropic: bool = False):
        """Forward request to the right engine."""
        # API key validation
        if self.config.api_key:
            auth = request.headers.get("authorization", "")
            provided = ""
            if auth.startswith("Bearer "):
                provided = auth[7:]
            elif auth.startswith("bearer "):
                provided = auth[7:]
            # Also check x-api-key header (Anthropic style)
            if not provided:
                provided = request.headers.get("x-api-key", "")
            if provided != self.config.api_key:
                raise HTTPException(401, "invalid API key")

        body = await request.body()
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            raise HTTPException(400, "invalid JSON body")

        # Determine model
        model_id = payload.get("model")
        if not model_id:
            raise HTTPException(400, "missing \'model\' field")

        engine = self.engines.get(model_id)
        if not engine:
            raise HTTPException(404, f"model not loaded: {model_id}")

        if not engine.base_url:
            raise HTTPException(503, f"engine for {model_id} not ready")

        # Translate Anthropic Messages → OpenAI Chat Completions
        if anthropic:
            payload = _anthropic_to_openai(payload)
            target_url = f"{engine.base_url}/v1/chat/completions"
        else:
            target_url = f"{engine.base_url}/v1/{endpoint}"

        # Use translated payload if anthropic, otherwise original body
        send_body = json.dumps(payload).encode() if anthropic else body
        stream = payload.get("stream", False)
        timeout = httpx.Timeout(connect=10.0, read=600.0, write=600.0, pool=10.0)

        # Strip Authorization header — the gateway handles auth, not the engine.
        # llama-server rejects Bearer tokens that don't match its own key.
        fwd_headers = {"content-type": "application/json"}

        if stream:
            # For streaming, create the client outside async with so it
            # stays open for the duration of the StreamingResponse iterator
            if anthropic:
                # Translate OpenAI SSE chunks → Anthropic Messages SSE events
                return StreamingResponse(
                    _stream_anthropic_response(target_url, send_body, timeout,
                                               fwd_headers, model_id),
                    media_type="text/event-stream",
                )
            return StreamingResponse(
                _stream_response(target_url, send_body, timeout, fwd_headers),
                media_type="text/event-stream",
            )
        else:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(target_url, content=send_body, headers=fwd_headers)
                if anthropic and r.status_code == 200:
                    return JSONResponse(content=_openai_to_anthropic(r.json(), model_id))
                return JSONResponse(content=r.json(), status_code=r.status_code)

    def load_engines(self, log_dir: str | None = None) -> None:
        """Start all configured engines."""
        from pathlib import Path
        ld = Path(log_dir) if log_dir else None
        for idx, model in enumerate(self.config.models):
            # Auto-fix stale/zero context sizes using memory-aware computation
            needs_persist = False
            if model.n_ctx and model.n_ctx < 16384:
                new_ctx = compute_memory_aware_ctx(model.id, model.n_ctx)

                logger.warning(
                    "Model %s has n_ctx=%d (too small), setting to %d",
                    model.id, model.n_ctx, new_ctx
                )
                model.n_ctx = new_ctx
                needs_persist = True

            elif not model.n_ctx or model.n_ctx == 0:
                # No context set — compute memory-aware default
                new_ctx = compute_memory_aware_ctx(model.id, 0)
                model.n_ctx = new_ctx
                logger.info("Model %s: set n_ctx=%d (memory-aware default)", model.id, new_ctx)
                needs_persist = True

            # Persist the fix to models.yaml so it doesn't repeat every run
            if needs_persist:
                try:
                    import yaml as _yaml
                    cfg_path = self.config_path
                    if not cfg_path:
                        cfg_path = os.environ.get("LITMOE_CONFIG", "")
                    if not cfg_path:
                        for candidate in ["models.yaml", "deploy/models.yaml", "config/models.yaml"]:
                            if Path(candidate).exists():
                                cfg_path = candidate
                                break
                    if cfg_path and Path(cfg_path).exists():
                        with open(cfg_path) as f:
                            raw = _yaml.safe_load(f)
                        for m in raw.get("models", []):
                            if m.get("id") == model.id:
                                m["n_ctx"] = model.n_ctx
                                break
                        with open(cfg_path, "w") as f:
                            _yaml.dump(raw, f, default_flow_style=False, sort_keys=False)
                        logger.info("Persisted n_ctx=%d for %s to %s", model.n_ctx, model.id, cfg_path)
                except Exception:
                    pass  # best effort — runtime fix is what matters

            logger.info("Loading %s via %s...", model.id, model.engine)
            engine = make_engine(model)
            # Assign unique port: first model 8081, second 8082, etc.
            # Skip the gateway's own port to avoid collision
            assigned_port = 8081 + idx
            if assigned_port == self.config.port:
                assigned_port += 1
            engine._assigned_port = assigned_port
            engine.start(log_dir=ld)
            self.engines[model.id] = engine
            for alias in model.aliases:
                self.engines[alias] = engine

    def _unique_engines(self) -> list[tuple[ModelEntry, Engine]]:
        """(model, engine) pairs with aliases deduplicated."""
        return [(m, self.engines[m.id]) for m in self.config.models
                if m.id in self.engines]

    async def wait_all_ready(self, timeout: float = 600.0) -> bool:
        """Wait for all engines to be ready."""
        tasks = [eng.wait_ready(timeout=timeout) for _, eng in self._unique_engines()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return all(r is True for r in results)

    def shutdown(self) -> None:
        """Stop all engines."""
        for _, engine in self._unique_engines():
            engine.stop()


async def _stream_response(url: str, body: bytes, timeout: httpx.Timeout, headers: dict | None = None):
    """Stream SSE responses from upstream engine.

    Uses raw byte passthrough (aiter_bytes) to preserve the exact SSE format
    from llama-server. This is critical — any line-based processing breaks
    the chunked transfer encoding and causes "incomplete chunked read" errors
    in clients like Hermes.

    The async client is kept alive for the full duration of the stream by
    managing it manually (not using async with, which would close it early).
    """
    fwd_headers = headers or {"content-type": "application/json"}
    client = httpx.AsyncClient(timeout=timeout)
    try:
        # Use stream() which keeps the connection open for the full response
        async with client.stream("POST", url, content=body, headers=fwd_headers) as r:
            # If the upstream returned an error, pass it through as JSON
            if r.status_code >= 400:
                error_body = await r.aread()
                yield error_body
                return
            # Raw byte passthrough — do NOT process lines, just forward bytes
            async for chunk in r.aiter_bytes():
                yield chunk
    except httpx.RequestError as e:
        logger.error("Stream error: %s", e)
        error_data = {"error": {"message": str(e), "type": "connection_error"}}
        yield f"data: {json.dumps(error_data)}\n\n".encode()
        yield b"data: [DONE]\n\n"
    finally:
        await client.aclose()


def _anthropic_to_openai(payload: dict) -> dict:
    """Translate Anthropic Messages API request → OpenAI Chat Completions.

    Covers the Claude Code subset: system prompts, text/image blocks,
    tool_use / tool_result blocks, tools + tool_choice, streaming usage.
    """
    messages = []
    system = payload.get("system")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    messages.append({"role": "system", "content": block.get("text", "")})

    for msg in payload.get("messages", []):
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            continue
        if role == "assistant":
            text_parts, tool_calls = [], []
            for block in content:
                # thinking / redacted_thinking blocks are intentionally
                # dropped — engines regenerate reasoning each turn
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    tool_calls.append({
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {"name": block.get("name", ""),
                                     "arguments": json.dumps(block.get("input", {}))},
                    })
            out_msg: dict[str, Any] = {"role": "assistant",
                                       "content": "\n".join(text_parts) or None}
            if tool_calls:
                out_msg["tool_calls"] = tool_calls
            messages.append(out_msg)
        else:  # user — may carry tool_result blocks, which OpenAI models
            # as separate tool-role messages
            text_parts = []
            for block in content:
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "image":
                    src_type = block.get("source", {}).get("type", "unknown")
                    text_parts.append(f"[image: {src_type}]")
                elif btype == "tool_result":
                    if text_parts:
                        messages.append({"role": "user", "content": "\n".join(text_parts)})
                        text_parts = []
                    result_content = block.get("content", "")
                    if isinstance(result_content, list):
                        result_content = "\n".join(
                            b.get("text", "") for b in result_content
                            if isinstance(b, dict) and b.get("type") == "text")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": result_content,
                    })
            if text_parts:
                messages.append({"role": "user", "content": "\n".join(text_parts)})

    out: dict[str, Any] = {
        "model": payload.get("model"),
        "messages": messages,
        "max_tokens": payload.get("max_tokens", 8192),
        "stream": payload.get("stream", False),  # pass through stream flag
    }
    if out["stream"]:
        # Ask the engine for a final usage chunk so we can report real counts
        out["stream_options"] = {"include_usage": True}
    if "temperature" in payload:
        out["temperature"] = payload["temperature"]
    if "top_p" in payload:
        out["top_p"] = payload["top_p"]
    if "stop_sequences" in payload:
        out["stop"] = payload["stop_sequences"]

    tools = payload.get("tools")
    if tools:
        out["tools"] = [
            {"type": "function",
             "function": {"name": t.get("name", ""),
                          "description": t.get("description", ""),
                          "parameters": t.get("input_schema", {})}}
            for t in tools
        ]
    tool_choice = payload.get("tool_choice")
    if isinstance(tool_choice, dict):
        tc_type = tool_choice.get("type")
        if tc_type == "auto":
            out["tool_choice"] = "auto"
        elif tc_type == "any":
            out["tool_choice"] = "required"
        elif tc_type == "tool":
            out["tool_choice"] = {"type": "function",
                                  "function": {"name": tool_choice.get("name", "")}}
        elif tc_type == "none":
            out.pop("tools", None)

    return out


# OpenAI finish_reason → Anthropic stop_reason
_STOP_REASON_MAP = {"stop": "end_turn", "length": "max_tokens",
                    "tool_calls": "tool_use", "content_filter": "refusal"}


def _openai_to_anthropic(resp: dict, model: str) -> dict:
    """Translate a non-streaming OpenAI Chat Completions response → Anthropic Messages."""
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content_blocks: list[dict] = []
    # Reasoning models (llama-server --jinja) split thinking into
    # reasoning_content — surface it as an Anthropic thinking block.
    # The signature is a placeholder: thinking blocks are stripped on the
    # way back upstream, so it is never verified.
    reasoning = msg.get("reasoning_content")
    if reasoning:
        content_blocks.append({"type": "thinking", "thinking": reasoning,
                               "signature": "litmoe"})
    text = msg.get("content")
    if text:
        content_blocks.append({"type": "text", "text": text})
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        try:
            tool_input = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            tool_input = {"_raw": fn.get("arguments", "")}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id") or f"toolu_litmoe_{len(content_blocks)}",
            "name": fn.get("name", ""),
            "input": tool_input,
        })
    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})
    finish = choice.get("finish_reason")
    usage = resp.get("usage") or {}
    return {
        "id": resp.get("id") or f"msg_litmoe_{int(time.time() * 1000)}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": _STOP_REASON_MAP.get(finish, "end_turn") if finish else None,
        "stop_sequence": None,
        "usage": {"input_tokens": usage.get("prompt_tokens", 0),
                  "output_tokens": usage.get("completion_tokens", 0)},
    }


async def _stream_anthropic_response(url: str, body: bytes, timeout: httpx.Timeout,
                                     headers: dict, model: str):
    """Stream upstream OpenAI SSE chunks as Anthropic Messages SSE events.

    Translates each OpenAI chunk (delta.content / delta.tool_calls) into the
    message_start → content_block_* → message_delta → message_stop lifecycle
    that Anthropic API clients (e.g. Claude Code) expect.
    """
    client = httpx.AsyncClient(timeout=timeout)

    def ev(event: str, data: dict) -> bytes:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()

    try:
        msg_id = f"msg_litmoe_{int(time.time() * 1000)}"
        yield ev("message_start", {
            "type": "message_start",
            "message": {"id": msg_id, "type": "message", "role": "assistant",
                        "model": model, "content": [], "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0}},
        })
        block_open = False
        block_index = -1
        block_type = ""           # "text" | "tool_use"
        tool_blocks: dict[int, int] = {}  # OpenAI tool_calls index → Anthropic block index
        input_tokens = 0
        output_tokens = 0
        stop_reason = "end_turn"
        buf = b""
        async with client.stream("POST", url, content=body, headers=headers) as r:
            if r.status_code >= 400:
                err = (await r.aread()).decode(errors="replace")[:500]
                yield ev("error", {"type": "error",
                                   "error": {"type": "api_error", "message": err}})
                return
            async for chunk in r.aiter_bytes():
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        continue
                    try:
                        payload = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    usage = payload.get("usage") or {}
                    if usage:
                        input_tokens = usage.get("prompt_tokens", input_tokens)
                        output_tokens = usage.get("completion_tokens", output_tokens)
                    choice = (payload.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    thinking = delta.get("reasoning_content")
                    if thinking:
                        if not block_open or block_type != "thinking":
                            if block_open:
                                yield ev("content_block_stop",
                                         {"type": "content_block_stop", "index": block_index})
                            block_index += 1
                            block_type = "thinking"
                            block_open = True
                            yield ev("content_block_start",
                                     {"type": "content_block_start", "index": block_index,
                                      "content_block": {"type": "thinking", "thinking": "",
                                                        "signature": ""}})
                        yield ev("content_block_delta",
                                 {"type": "content_block_delta", "index": block_index,
                                  "delta": {"type": "thinking_delta", "thinking": thinking}})
                    text = delta.get("content")
                    if text:
                        if not block_open or block_type != "text":
                            if block_open:
                                if block_type == "thinking":
                                    yield ev("content_block_delta",
                                             {"type": "content_block_delta", "index": block_index,
                                              "delta": {"type": "signature_delta",
                                                        "signature": "litmoe"}})
                                yield ev("content_block_stop",
                                         {"type": "content_block_stop", "index": block_index})
                            block_index += 1
                            block_type = "text"
                            block_open = True
                            yield ev("content_block_start",
                                     {"type": "content_block_start", "index": block_index,
                                      "content_block": {"type": "text", "text": ""}})
                        output_tokens += 1  # refined by final usage chunk if present
                        yield ev("content_block_delta",
                                 {"type": "content_block_delta", "index": block_index,
                                  "delta": {"type": "text_delta", "text": text}})
                    # llama.cpp streams tool calls sequentially per index
                    for tc in delta.get("tool_calls") or []:
                        oai_idx = tc.get("index", 0)
                        fn = tc.get("function") or {}
                        if oai_idx not in tool_blocks:
                            if block_open:
                                if block_type == "thinking":
                                    yield ev("content_block_delta",
                                             {"type": "content_block_delta", "index": block_index,
                                              "delta": {"type": "signature_delta",
                                                        "signature": "litmoe"}})
                                yield ev("content_block_stop",
                                         {"type": "content_block_stop", "index": block_index})
                            block_index += 1
                            tool_blocks[oai_idx] = block_index
                            block_type = "tool_use"
                            block_open = True
                            yield ev("content_block_start",
                                     {"type": "content_block_start", "index": block_index,
                                      "content_block": {
                                          "type": "tool_use",
                                          "id": tc.get("id") or f"toolu_litmoe_{oai_idx}",
                                          "name": fn.get("name") or "", "input": {}}})
                        args = fn.get("arguments")
                        if args:
                            yield ev("content_block_delta",
                                     {"type": "content_block_delta", "index": tool_blocks[oai_idx],
                                      "delta": {"type": "input_json_delta",
                                                "partial_json": args}})
                    finish = choice.get("finish_reason")
                    if finish:
                        stop_reason = _STOP_REASON_MAP.get(finish, "end_turn")
        if block_open:
            if block_type == "thinking":
                yield ev("content_block_delta",
                         {"type": "content_block_delta", "index": block_index,
                          "delta": {"type": "signature_delta", "signature": "litmoe"}})
            yield ev("content_block_stop", {"type": "content_block_stop", "index": block_index})
        yield ev("message_delta",
                 {"type": "message_delta",
                  "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                  "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}})
        yield ev("message_stop", {"type": "message_stop"})
    except httpx.RequestError as e:
        logger.error("Stream error: %s", e)
        yield ev("error", {"type": "error",
                           "error": {"type": "api_error", "message": str(e)}})
    finally:
        await client.aclose()


def run(config: GatewayConfig, log_dir: str | None = None, config_path: str | None = None) -> None:
    """Entry point: start gateway."""
    gateway = Gateway(config, config_path=config_path)
    gateway.load_engines(log_dir=log_dir)

    # Wait for engines in background
    async def startup():
        ok = await gateway.wait_all_ready(timeout=600)
        if not ok:
            logger.warning("Not all engines became ready — gateway will start anyway")
        else:
            logger.info("All engines ready.")

    # Use the existing event loop or create one for startup checks
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    loop.run_until_complete(startup())

    try:
        uvicorn.run(gateway.app, host=config.host, port=config.port, log_level="info")
    finally:
        gateway.shutdown()
