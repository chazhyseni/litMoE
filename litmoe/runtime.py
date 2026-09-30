"""One resident engine and one inference lease, including streamed responses."""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from contextlib import suppress
from typing import Callable

import anyio
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse

from litmoe.config import GatewayConfig, ModelEntry
from litmoe.engines import Engine

logger = logging.getLogger(__name__)


async def disconnected(request: Request, stopped: asyncio.Event) -> None:
    while not stopped.is_set():
        if await request.is_disconnected():
            return
        try:
            await asyncio.wait_for(stopped.wait(), 0.05)
        except TimeoutError:
            pass


async def while_connected(awaitable, request: Request):
    """Cancel queued/prefill work when the caller leaves; never dispatch it later."""
    work = asyncio.ensure_future(awaitable)
    stopped = asyncio.Event()
    watcher = asyncio.create_task(disconnected(request, stopped))
    try:
        done, _ = await asyncio.wait((work, watcher), return_when=asyncio.FIRST_COMPLETED)
        if work in done:
            return await work
        raise HTTPException(499, "client disconnected")
    finally:
        # Request.is_disconnected() uses an AnyIO cancel scope and can swallow
        # task cancellation. Signal completion instead of waiting on cancel().
        stopped.set()
        if not work.done():
            work.cancel()
        with anyio.CancelScope(shield=True):
            with suppress(asyncio.CancelledError):
                await watcher
            # Consume cancellation without masking the original HTTP/backend error.
            if work.cancelled() or not work.done():
                with suppress(asyncio.CancelledError):
                    await work


class Lease:
    def __init__(self, runtime: Runtime, engine: Engine, restart_on_abandon: bool = True):
        self.runtime, self.engine = runtime, engine
        self.restart_on_abandon = restart_on_abandon
        self.closed = False

    async def close(self, abandoned: bool = False) -> None:
        if self.closed:
            return
        self.closed = True
        with anyio.CancelScope(shield=True):
            try:
                if getattr(self.engine, "supports_cooperative_cancel", False):
                    # EOF is not an acknowledgement: a transport error can end
                    # an iterator while the native prefill is still running.
                    try:
                        idle = await asyncio.wait_for(self.engine.wait_idle(timeout=10), timeout=10)
                    except asyncio.CancelledError:
                        await self._stop_with_reset("quiescence wait interrupted; cache reset")
                        raise
                    except Exception:
                        idle = False
                    if not idle:
                        await self._stop_with_reset("native quiescence unconfirmed; cache reset")
                elif abandoned and self.restart_on_abandon:
                    await self._stop_with_reset("stream abandoned by the client")
            finally:
                self.runtime.lock.release()

    async def _stop_with_reset(self, reason: str) -> None:
        logger.warning("engine %s: %s; stopping the owned process",
                       self.engine.model.id, reason)
        await self.runtime._stop()



class LeasedStream(StreamingResponse):
    def __init__(self, iterator, lease: Lease, client, response):
        self.lease = lease
        self.client, self.response = client, response
        self.finished = False
        super().__init__(self._relay(iterator), media_type="text/event-stream")

    async def _relay(self, iterator):
        try:
            async for chunk in iterator:
                yield chunk
            self.finished = True
        finally:
            await iterator.aclose()

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                try:
                    await self.body_iterator.aclose()
                finally:
                    try:
                        await self.response.aclose()
                        await self.client.aclose()
                    finally:
                        # finished==True is NOT evidence the backend is idle:
                        # an in-band error event can end the iterator while
                        # native generation continues. The lease decides via
                        # quiescence/restart, not the stream's own view.
                        await self.lease.close(abandoned=not self.finished)


class Runtime:
    def __init__(self, config: GatewayConfig, start_engine: Callable[[ModelEntry], Engine],
                 initial_model: str | None = None, ready_timeout: float = 600):
        self.config = config
        self.start_engine = start_engine
        self.selected = self.lookup(initial_model) if initial_model else next(iter(config.models), None)
        self.engine: Engine | None = None
        self.state = "stopped"
        self.error: str | None = None
        self.lock = asyncio.Lock()
        self.queue_depth = 0
        self.ready_timeout = ready_timeout
        self.closing = False
        self.switching = 0
        self.runtime_id = uuid.uuid4().hex
        self.generation = 0

    def lookup(self, model_id: str) -> ModelEntry:
        for model in self.config.models:
            if model_id == model.id or model_id in model.aliases:
                return model
        raise HTTPException(404, f"unknown configured model: {model_id}")

    def _cancellation_mode(self, engine: Engine | None) -> str:
        if engine is not None and getattr(engine, "supports_cooperative_cancel", False):
            return "prefill-chunked"
        if engine is not None and engine.model.engine == "warp":
            return "stream-disconnect"
        return "restart"

    def status(self) -> dict:
        engine = self.engine
        if self.state == "ready" and (engine is None or not engine.is_running()):
            self.state = "failed"
            self.error = "resident engine exited; select it again with litmoe switch"
        ready = self.state == "ready"
        cache = "none"
        if engine and engine.model.engine == "llamacpp":
            disabled = "--no-cache-prompt" in engine.model.extra_args
            value = engine.model.env.get("LLAMA_ARG_CACHE_PROMPT", os.environ.get("LLAMA_ARG_CACHE_PROMPT", "")).lower()
            if not disabled and value not in ("0", "false", "off"):
                cache = "backend"
        if engine is not None and getattr(engine, "supports_native_messages", False):
            cache = "backend"
        return {
            "selected_model": self.selected.id if self.selected else None,
            "active_model": engine.model.id if ready else None,
            "state": self.state, "error": self.error,
            "runtime_id": self.runtime_id, "generation": self.generation,
            "engine_pid": getattr(engine.process, "pid", None) if engine else None,
            "configured_models": [{"id": m.id, "aliases": m.aliases} for m in self.config.models],
            "queue_depth": self.queue_depth,
            "context_window": engine.model.n_ctx if ready else None,
            "capabilities": {
                "native_messages": bool(engine is not None
                                        and getattr(engine, "supports_native_messages", False)),
                "tool_search": bool(engine is not None
                                    and getattr(engine, "supports_tool_search", False)),
                "count_tokens": bool(engine is not None
                                     and getattr(engine, "supports_native_messages", False)),
                "prompt_cache": cache,
                "cancellation": self._cancellation_mode(engine),
            },
        }

    async def _stop(self) -> None:
        if self.engine:
            # Do not drop ownership if stop fails: another engine must not start.
            try:
                await anyio.to_thread.run_sync(self.engine.stop)
                if self.engine.is_running():
                    raise RuntimeError("owned engine remained alive after stop")
            except Exception as exc:
                self.state, self.error = "failed", str(exc)
                raise
            self.engine = None
        self.state = "stopped"

    async def _start(self, model: ModelEntry) -> None:
        self.selected = model
        self.state, self.error = "loading", None
        try:
            start = asyncio.create_task(anyio.to_thread.run_sync(self.start_engine, model))
            try:
                self.engine = await asyncio.shield(start)
            except asyncio.CancelledError:
                # Native process creation cannot be cancelled in a worker thread.
                # Recover its handle before unwinding, then the outer handler stops it.
                with anyio.CancelScope(shield=True):
                    self.engine = await start
                raise
            if not await self.engine.wait_ready(timeout=self.ready_timeout):
                raise RuntimeError("engine did not become ready; inspect its log")
            if not self.engine.is_running():
                raise RuntimeError("engine exited during startup")
            self.state = "ready"
            self.generation += 1
        except BaseException as exc:
            with anyio.CancelScope(shield=True):
                try:
                    await self._stop()
                finally:
                    self.state, self.error = "failed", str(exc)
            raise

    async def switch(self, model_id: str) -> dict:
        model = self.lookup(model_id)
        if self.closing:
            raise HTTPException(503, "gateway is shutting down")
        self.switching += 1
        if self.lock.locked() and self.state == "ready":
            self.state = "draining"
        try:
            async with self.lock:
                if self.closing:
                    raise HTTPException(503, "gateway is shutting down")
                with anyio.CancelScope(shield=True):
                    if (self.state in ("ready", "draining") and self.engine
                            and self.engine.model.id == model.id and self.engine.is_running()):
                        self.state = "ready"
                    else:
                        await self._stop()
                        await self._start(model)
                return self.status()
        except HTTPException:
            raise
        except Exception as exc:
            self.state, self.error = "failed", str(exc)
            raise HTTPException(503, f"cannot load {model.id}: {exc}") from exc
        finally:
            self.switching -= 1
            if not self.switching and self.state == "draining" and self.engine:
                self.state = "ready" if self.engine.is_running() else "failed"

    def _check_selection(self, model_id: str) -> ModelEntry:
        model = self.lookup(model_id)
        if not self.selected or model.id != self.selected.id:
            raise HTTPException(409, f"model is inactive; run: litmoe switch {model.id}")
        if self.closing or self.switching:
            raise HTTPException(503, "gateway is switching or shutting down")
        return model

    async def acquire(self, model_id: str, request: Request) -> Lease:
        model = self._check_selection(model_id)
        # Count reserved admissions too: several coroutines can arrive before
        # the first lock-acquisition task gets its event-loop turn.
        if self.queue_depth >= self.config.max_queue_size + int(not self.lock.locked()):
            raise HTTPException(429, "interactive inference queue is full")
        self.queue_depth += 1
        acquired = False

        async def take_lock():
            nonlocal acquired
            await self.lock.acquire()
            acquired = True

        try:
            try:
                await while_connected(
                    asyncio.wait_for(take_lock(), self.config.queue_timeout), request,
                )
            except TimeoutError:
                raise HTTPException(429, "interactive inference queue wait expired")
            self._check_selection(model_id)
            if await request.is_disconnected():
                raise HTTPException(499, "client disconnected")
            if self.state == "stopped":
                # Only a later request reloads the explicitly selected model.
                await while_connected(self._start(model), request)
            if self.status()["state"] != "ready":
                raise HTTPException(503, self.error or "engine is not ready")
            lease = Lease(self, self.engine)
            acquired = False  # Ownership transfers to the response, not its constructor.
            return lease
        finally:
            self.queue_depth -= 1
            if acquired:
                self.lock.release()

    async def shutdown(self) -> None:
        self.closing = True
        async with self.lock:
            with anyio.CancelScope(shield=True):
                await self._stop()
