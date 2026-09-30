"""Paired single-user HTTP measurements; never infer cache hits or token counts."""
from __future__ import annotations

import asyncio
import hashlib
import json
import platform
import time
from datetime import datetime, timezone

import httpx


class StreamMetrics:
    def __init__(self):
        self.first_generated_s = None
        self.first_text_s = None
        self.usage = None
        self.done = False
        self.finish_reason = None

    def event(self, data: str, elapsed: float) -> None:
        if data == "[DONE]":
            self.done = True
            return
        value = json.loads(data)
        if not isinstance(value, dict) or "error" in value:
            raise ValueError("upstream error or invalid SSE object")
        if isinstance(value.get("usage"), dict):
            self.usage = value["usage"]
        choices = value.get("choices", [])
        if not isinstance(choices, list):
            raise ValueError("invalid choices")
        for choice in choices:
            delta = choice.get("delta", {})
            text = delta.get("content")
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            tools = any(call.get("function", {}).get("name") or call.get("function", {}).get("arguments")
                        for call in delta.get("tool_calls", []))
            if (text or reasoning or tools) and self.first_generated_s is None:
                self.first_generated_s = elapsed
            if text and self.first_text_s is None:
                self.first_text_s = elapsed
            if choice.get("finish_reason") is not None:
                self.finish_reason = choice["finish_reason"]


def token_count(usage: dict | None, name: str) -> int | None:
    value = (usage or {}).get(name)
    return value if type(value) is int and value >= 0 else None


async def measure(client: httpx.AsyncClient, url: str, body: bytes, headers: dict,
                  timeout: float) -> dict:
    metrics = StreamMetrics()
    start = time.monotonic()
    result = {"started_monotonic_s": start, "headers_s": None, "http_status": None,
              "error": None, "success": False}
    try:
        async with asyncio.timeout(timeout):
            async with client.stream("POST", url, content=body, headers=headers) as response:
                result["headers_s"] = time.monotonic() - start
                result["http_status"] = response.status_code
                if response.status_code != 200:
                    raise ValueError(f"HTTP {response.status_code}")
                if "text/event-stream" not in response.headers.get("content-type", ""):
                    raise ValueError("response is not an SSE stream")
                data = []
                received = 0
                async for line in response.aiter_lines():
                    received += len(line)
                    if received > 16 * 1024 * 1024:
                        raise ValueError("response exceeded measurement size limit")
                    if line == "":
                        if data:
                            metrics.event("\n".join(data), time.monotonic() - start)
                            data.clear()
                            if metrics.done:
                                break
                    elif line.startswith("data:"):
                        data.append(line[5:].lstrip(" "))
                    elif line.startswith(":") or line.startswith(("event:", "id:", "retry:")):
                        continue
                    else:
                        raise ValueError("malformed SSE line")
                if not metrics.done or metrics.finish_reason is None:
                    raise ValueError("incomplete SSE response")
                result["success"] = True
    except (httpx.RequestError, TimeoutError, ValueError, TypeError, AttributeError) as exc:
        # Errors from the body may contain private text; retain category, never content.
        result["error"] = type(exc).__name__
    end = time.monotonic()
    elapsed = end - start
    output = token_count(metrics.usage, "completion_tokens")
    result.update({
        "completed_monotonic_s": end, "completion_s": elapsed,
        "first_generated_delta_s": metrics.first_generated_s,
        "first_visible_text_s": metrics.first_text_s,
        "prompt_tokens": token_count(metrics.usage, "prompt_tokens"), "output_tokens": output,
        "finish_reason": metrics.finish_reason,
        "output_tokens_per_response_second": output / elapsed if output is not None and elapsed > 0 and result["success"] else None,
    })
    return result


async def benchmark(gateway: str, model: str | None, key: str | None, prompt: str,
                    runs: int = 3, max_tokens: int = 128, timeout: float = 600) -> dict:
    if not 1 <= runs <= 20 or not 1 <= max_tokens <= 131072 or not 0 < timeout <= 3600:
        raise ValueError("benchmark bounds exceeded")
    if len(prompt.encode()) > 1_000_000:
        raise ValueError("prompt exceeds one million UTF-8 bytes")
    gateway = gateway.rstrip("/")
    if gateway.endswith("/v1"):
        gateway = gateway[:-3]
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        async def fetch(path):
            response = await client.get(gateway + path, headers=headers)
            response.raise_for_status()
            return response.json()

        runtime = await fetch("/v1/runtime")
        if runtime["state"] != "ready":
            raise ValueError("gateway has no ready active model")
        entries = (await fetch("/v1/models"))["data"]
        chosen = model or runtime["active_model"]
        entry = next((m for m in entries if m["id"] == chosen), None)
        if entry is None:
            raise ValueError("requested model is not active; use litmoe switch first")
        canonical = entry.get("alias_of", entry["id"])
        if canonical != runtime["active_model"]:
            raise ValueError("model switched during discovery")
        engine = (await fetch("/health"))["engines"][canonical]
        if not engine["running"]:
            raise ValueError("resident engine is not running")
        payload = {"model": canonical, "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": max_tokens, "stream": True,
                   "stream_options": {"include_usage": True}, "temperature": 0}
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        records = []
        aborted = False
        def same_runtime(current):
            return (current["active_model"] == canonical and current["state"] == "ready"
                    and current["runtime_id"] == runtime["runtime_id"]
                    and current["generation"] == runtime["generation"]
                    and current["context_window"] == runtime["context_window"])
        for pair in range(runs):
            for route in (("gateway", "direct") if pair % 2 == 0 else ("direct", "gateway")):
                current = await fetch("/v1/runtime")
                if not same_runtime(current):
                    raise ValueError("active model changed during benchmark")
                base = gateway if route == "gateway" else engine["base_url"].rstrip("/")
                request_headers = {"content-type": "application/json"}
                if route == "gateway":
                    request_headers.update(headers)
                row = await measure(client, base + "/v1/chat/completions", body, request_headers, timeout)
                if not same_runtime(await fetch("/v1/runtime")):
                    row["success"] = False
                    row["error"] = row["error"] or "RuntimeChanged"
                row.update({"route": route, "pair": pair + 1, "order": len(records) + 1})
                records.append(row)
                if not row["success"]:
                    aborted = True
                    break
            if aborted:
                break
        return {
            "schema": "litmoe.benchmark/1", "created_at": datetime.now(timezone.utc).isoformat(),
            "host": {"system": platform.system(), "machine": platform.machine()},
            "model": canonical, "backend": entry["engine"], "context_window": entry["context_window"],
            "runtime": runtime, "gateway": gateway, "direct_engine": engine["base_url"],
            "payload_sha256": hashlib.sha256(body).hexdigest(), "payload_bytes": len(body),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "prompt_bytes": len(prompt.encode()),
            "max_tokens": max_tokens, "runs": records,
            "limitations": [
                "Run on the gateway host with other clients idle; direct requests bypass admission.",
                "Alternating order is not a matched cold/warm experiment or proof of prefix-cache hits.",
                "First delta includes content, reasoning, or tool output, not role/header metadata.",
                "Throughput uses output usage divided by total response time, including prefill and transport; it is not decode-only throughput.",
                "No weights or backend-build fingerprint is available; record exact build and model artifact separately before comparing machines.",
            ],
        }
