"""OpenAI-compatible HTTP proxy.

This is a thin pass-through that forwards requests to engine processes.
Engines speak OpenAI-compatible HTTP; we just route by model name.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
import time
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from litmoe.config import GatewayConfig, ModelEntry, expand_path, is_hf_repo_spec
from litmoe.engines import make_engine, Engine
from litmoe.engines.base import DEFAULT_ENGINE_PORT
from litmoe.models import lookup as catalog_lookup, quant_size_gb, fit_context
from litmoe.platform_utils import get_total_memory_bytes

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory-aware context sizing
# ---------------------------------------------------------------------------

# n_ctx values below this are treated as stale defaults and raised to the
# model's native context if the KV cache fits in RAM.
MIN_SANE_CTX = 16384
# Fallbacks for models that are not in the catalog.
_FALLBACK_NATIVE_CTX = 131072
_FALLBACK_KV_BYTES_PER_TOKEN = 65_536  # ~8.6 GB at 128K


def _shard_pattern(path: Path) -> str | None:
    """'name-00001-of-00004.gguf' -> glob 'name-*-of-00004.gguf', else None."""
    m = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", path.name)
    if not m:
        return None
    return f"{m.group(1)}-*-of-{m.group(3)}.gguf"


def weights_size_gb(model: ModelEntry) -> float | None:
    """Size of the model weights actually configured, in decimal GB.

    Local GGUF file: its size plus sibling shards. Local directory: all
    .gguf/.safetensors inside. HF repo spec: the catalog size of the requested
    quant (or the model's default quant). None if unknown.
    """
    src = model.gguf_path or model.model_path
    if not src:
        return None
    if src.startswith(("http://", "https://")):
        return None
    if is_hf_repo_spec(src):
        repo, _, quant = src.partition(":")
        for mid, info in _catalog_items():
            if info["hf_repo"].lower() == repo.lower():
                return quant_size_gb(mid, quant or None)
        return None
    p = expand_path(src)
    try:
        if p.is_file():
            pat = _shard_pattern(p)
            files = list(p.parent.glob(pat)) if pat else [p]
            return sum(f.stat().st_size for f in files) / 1e9
        if p.is_dir():
            files = [f for f in p.iterdir() if f.suffix in (".gguf", ".safetensors")]
            return sum(f.stat().st_size for f in files) / 1e9 if files else None
    except OSError:
        return None
    return None


def _catalog_items():
    from litmoe.models import KNOWN_MODELS
    return KNOWN_MODELS.items()


def compute_memory_aware_ctx(model: ModelEntry, n_ctx: int) -> int:
    """Context size for a model given total RAM.

    Target = n_ctx if it is already sane (>= MIN_SANE_CTX), else the model's
    native context. If weights + KV cache at the target exceed ~90% of RAM,
    shrink the context (multiple of 4096, minimum 8192).
    """
    info = catalog_lookup(model.id) or {}
    native_ctx = info.get("native_ctx", _FALLBACK_NATIVE_CTX)
    kv_rate = info.get("kv_bytes_per_token", _FALLBACK_KV_BYTES_PER_TOKEN)
    target_ctx = n_ctx if n_ctx >= MIN_SANE_CTX else native_ctx

    total_mem = get_total_memory_bytes()
    if total_mem is None:
        return target_ctx

    model_size_gb = weights_size_gb(model)
    if model_size_gb is None:
        model_size_gb = quant_size_gb(model.id, None) or 0.0

    ctx, note = fit_context(kv_rate, model_size_gb, total_mem / 1e9, target_ctx)
    if note:
        logger.warning("Model %s: %s", model.id, note)
    return ctx


def _port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    """True if nothing is listening on host:port (bind probe)."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def allocate_engine_ports(n: int, gateway_port: int, start: int = DEFAULT_ENGINE_PORT,
                          probe: bool = True) -> list[int]:
    """n distinct engine ports counting up from `start`.

    Never the gateway port, and (when probe=True) never a port something else
    already listens on — an Ollama on 11434 is out of range anyway, but a
    LM Studio / manual llama-server on 8081 used to make the first engine die
    with "couldn't bind HTTP server socket". Skipping keeps both alive.
    """
    ports: list[int] = []
    p = start
    while len(ports) < n:
        if p != gateway_port and (not probe or _port_is_free(p)):
            ports.append(p)
        elif probe and p != gateway_port:
            logger.warning("Port %d is already in use by another process — skipping it", p)
        p += 1
        if p > 65535:
            raise RuntimeError("no free engine ports")
    return ports


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

        @self.app.post("/v1/messages/count_tokens")
        async def count_tokens(request: Request):
            """Anthropic token-count endpoint (Claude Code calls it before requests).

            Engines expose no cross-model tokenizer, so this returns an estimate
            (~4 characters per token over the serialized prompt).
            """
            self._check_api_key(request)
            body = await request.body()
            try:
                payload = json.loads(body) if body else {}
            except json.JSONDecodeError:
                raise HTTPException(400, "invalid JSON body")
            text = json.dumps(payload.get("system", "")) + json.dumps(payload.get("messages", []))
            text += json.dumps(payload.get("tools", []))
            return {"input_tokens": max(1, len(text) // 4)}

    def _check_api_key(self, request: Request) -> None:
        if not self.config.api_key:
            return
        auth = request.headers.get("authorization", "")
        provided = auth[7:] if auth[:7].lower() == "bearer " else ""
        if not provided:
            provided = request.headers.get("x-api-key", "")
        if provided != self.config.api_key:
            raise HTTPException(401, "invalid API key")

    def _resolve(self, model_id: str) -> tuple[ModelEntry, Engine]:
        """(canonical model entry, engine) for a model id or alias."""
        engine = self.engines.get(model_id)
        if not engine:
            served = sorted({e.model.id for e in self.engines.values()})
            aliases = sorted(k for k, e in self.engines.items() if k != e.model.id)
            hint = f"models served: {', '.join(served) or 'none'}"
            if aliases:
                hint += f"; aliases: {', '.join(aliases)}"
            raise HTTPException(404, f"model not loaded: {model_id} ({hint})")
        if not engine.base_url:
            raise HTTPException(503, f"engine for {model_id} not ready")
        return engine.model, engine

    async def _proxy(self, request: Request, endpoint: str, anthropic: bool = False):
        """Forward request to the right engine."""
        self._check_api_key(request)

        body = await request.body()
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            raise HTTPException(400, "invalid JSON body")

        model_id = payload.get("model")
        if not model_id:
            raise HTTPException(400, "missing 'model' field")
        model, engine = self._resolve(model_id)

        if anthropic:
            payload = _anthropic_to_openai(payload)
            target_url = f"{engine.base_url}/v1/chat/completions"
        else:
            target_url = f"{engine.base_url}/v1/{endpoint}"

        # Aliases are a gateway concept: engines that validate the model field
        # (sglang) must see the id they were started with.
        payload["model"] = model.id
        send_body = json.dumps(payload).encode()
        stream = bool(payload.get("stream", False))
        timeout = httpx.Timeout(connect=10.0, read=600.0, write=600.0, pool=10.0)

        # Strip Authorization header — the gateway handles auth, not the engine.
        # llama-server rejects Bearer tokens that don't match its own key.
        fwd_headers = {"content-type": "application/json"}

        if stream:
            if anthropic:
                return StreamingResponse(
                    _stream_anthropic_response(target_url, send_body, timeout, fwd_headers, model_id),
                    media_type="text/event-stream",
                )
            return StreamingResponse(
                _stream_response(target_url, send_body, timeout, fwd_headers),
                media_type="text/event-stream",
            )

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(target_url, content=send_body, headers=fwd_headers)
        except httpx.RequestError as e:
            raise HTTPException(502, f"engine for {model.id} unreachable: {e}")
        try:
            data = r.json()
        except ValueError:
            raise HTTPException(502, f"engine for {model.id} returned non-JSON "
                                     f"(HTTP {r.status_code}): {r.text[:300]}")
        if anthropic and r.status_code == 200:
            return JSONResponse(content=_openai_to_anthropic(data, model_id))
        if anthropic:
            message = data.get("error", {}).get("message") if isinstance(data, dict) else None
            return JSONResponse(status_code=r.status_code, content={
                "type": "error",
                "error": {"type": "api_error", "message": message or json.dumps(data)[:500]},
            })
        return JSONResponse(content=data, status_code=r.status_code)

    def load_engines(self, log_dir: str | None = None) -> None:
        """Start all configured engines. A model that fails to start is skipped, not fatal."""
        ld = Path(log_dir) if log_dir else None
        ports = allocate_engine_ports(len(self.config.models), self.config.port)
        for model, port in zip(self.config.models, ports):
            self._fix_context(model)

            logger.info("Loading %s via %s on port %d...", model.id, model.engine, port)
            try:
                engine = make_engine(model)
                engine.set_port(port)
                engine.start(log_dir=ld)
            except Exception as e:  # FileNotFoundError, ValueError, OSError ...
                logger.error("Model %s could not be started: %s", model.id, e)
                continue
            self.engines[model.id] = engine
            for alias in model.aliases:
                self.engines[alias] = engine

    def _fix_context(self, model: ModelEntry) -> None:
        """Raise stale/zero n_ctx to a memory-aware value and persist it to models.yaml.

        Only for llama.cpp entries: sglang-kt sizes its own KV pool.
        """
        if model.engine != "llamacpp":
            return
        if model.n_ctx and model.n_ctx >= MIN_SANE_CTX:
            return
        new_ctx = compute_memory_aware_ctx(model, model.n_ctx or 0)
        if new_ctx == model.n_ctx:
            return
        logger.warning("Model %s: n_ctx=%d is below %d, using %d (memory-aware native context)",
                       model.id, model.n_ctx, MIN_SANE_CTX, new_ctx)
        model.n_ctx = new_ctx
        self._persist_ctx(model)

    def _persist_ctx(self, model: ModelEntry) -> None:
        """Write the corrected n_ctx back so the fix does not repeat every run.

        Note: yaml.dump re-serializes the file, so comments in models.yaml are lost.
        """
        import yaml as _yaml
        cfg_path = self.config_path or os.environ.get("LITMOE_CONFIG", "")
        if not cfg_path:
            for candidate in ["models.yaml", "deploy/models.yaml", "config/models.yaml"]:
                if Path(candidate).exists():
                    cfg_path = candidate
                    break
        if not cfg_path or not Path(cfg_path).exists():
            logger.info("Not persisting n_ctx for %s: no config path known", model.id)
            return
        try:
            with open(cfg_path) as f:
                raw = _yaml.safe_load(f) or {}
            for m in raw.get("models", []) or []:
                if m.get("id") == model.id:
                    m["n_ctx"] = model.n_ctx
                    break
            with open(cfg_path, "w") as f:
                _yaml.dump(raw, f, default_flow_style=False, sort_keys=False)
            logger.info("Persisted n_ctx=%d for %s to %s", model.n_ctx, model.id, cfg_path)
        except (OSError, _yaml.YAMLError) as e:
            logger.warning("Could not persist n_ctx for %s to %s: %s", model.id, cfg_path, e)

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
        """Stop all engines (idempotent)."""
        for _, engine in self._unique_engines():
            try:
                engine.stop()
            except Exception as e:  # never let one engine block the others
                logger.warning("stopping %s: %s", engine.model.id, e)


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
            # llama-server only accepts string tool_choice values; "required"
            # forces a tool call, which is the closest to naming a specific tool.
            out["tool_choice"] = "required"
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
    if not gateway.engines:
        logger.error("No engine could be started — check the errors above and `litmoe doctor`.")

    # Models served straight from HuggingFace (-hf) download on first start;
    # allow more time for that than for a local file.
    ready_timeout = float(os.environ.get("LITMOE_READY_TIMEOUT", "0") or 0)
    if not ready_timeout:
        downloads = any(is_hf_repo_spec(m.model_path or "") or (m.model_path or "").startswith("http")
                        for m in config.models)
        ready_timeout = 3600.0 if downloads else 600.0

    async def startup():
        ok = await gateway.wait_all_ready(timeout=ready_timeout)
        if not ok:
            logger.warning("Not all engines became ready — gateway will start anyway")
        else:
            logger.info("All engines ready.")

    # Engines run in their own sessions (start_new_session=True), so a SIGTERM
    # sent to the gateway does not reach them. uvicorn captures SIGINT/SIGTERM
    # to exit gracefully, then *restores the previous handlers and re-raises
    # the signal* after run() returns. With Python's default SIGTERM
    # disposition that kills the process before any `finally:` block runs and
    # orphans the engines (measured: rc=-15, no cleanup output). Install a
    # handler beforehand so the re-raised signal lands here instead.
    def _stop_engines(signum, _frame):
        logger.info("Signal %d: stopping engines", signum)
        gateway.shutdown()
        raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT, getattr(signal, "SIGHUP", None)):
        if sig is not None:
            signal.signal(sig, _stop_engines)

    try:
        asyncio.run(startup())
        uvicorn.run(gateway.app, host=config.host, port=config.port, log_level="info")
    except KeyboardInterrupt:
        pass
    finally:
        gateway.shutdown()
