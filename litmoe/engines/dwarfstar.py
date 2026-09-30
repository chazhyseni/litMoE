"""Pinned DwarfStar Metal runtime with native protocols and cancellation acknowledgement."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path

import httpx

from litmoe.config import ModelEntry, expand_path
from litmoe.engines.base import Engine

logger = logging.getLogger(__name__)
DWARFSTAR_REPO = "https://github.com/antirez/ds4.git"
DWARFSTAR_COMMIT = "0aaea5a238fb41a35106a551e73c8409dfb751ac"
DWARFSTAR_PATCH = Path(__file__).resolve().parents[1] / "patches" / "dwarfstar-serving.patch"
NATIVE_CAPABILITIES = {
    "native_messages": True, "tool_search": True, "count_tokens": True,
    "prompt_cache": "backend", "cancellation": "prefill-chunked",
}


def validate_installation(root: Path) -> None:
    """Reject stale or unpatched executables, rather than promising absent APIs."""
    try:
        valid = (
            os.access(root / "ds4-server", os.X_OK)
            and (root / "LICENSE").is_file()
            and (root / "metal").is_dir()
            and (root / ".litmoe-revision").read_text().strip() == DWARFSTAR_COMMIT
            and (root / ".litmoe-patch-sha256").read_text().strip()
            == hashlib.sha256(DWARFSTAR_PATCH.read_bytes()).hexdigest()
        )
    except OSError:
        valid = False
    if not valid:
        raise FileNotFoundError(
            f"DwarfStar runtime at {root} is missing, incomplete, or stale; "
            "install with: litmoe install --engine dwarfstar"
        )


def runtime_root() -> Path:
    explicit = os.environ.get("LITMOE_DWARFSTAR_DIR")
    if explicit:
        root = expand_path(explicit)
    else:
        prefix = expand_path(os.environ.get("LITMOE_PREFIX", "~/.local"))
        root = prefix / "lib" / "dwarfstar"
    validate_installation(root)
    return root


def is_installed() -> bool:
    try:
        runtime_root()
        return True
    except (OSError, RuntimeError):
        return False


class DwarfStarEngine(Engine):
    def __init__(self, model: ModelEntry):
        super().__init__(model)
        self._capabilities: dict = {}

    @property
    def supports_native_messages(self) -> bool:
        return self._capabilities.get("native_messages") is True

    @property
    def supports_tool_search(self) -> bool:
        return self._capabilities.get("tool_search") is True

    @property
    def supports_cooperative_cancel(self) -> bool:
        return self._capabilities.get("cancellation") == "prefill-chunked"

    def health_url(self) -> str:
        return f"http://127.0.0.1:{self.default_port()}/v1/litmoe/status"

    def _kv_disk_dir(self, root: Path, model: Path) -> Path:
        # No multi-minute hash of a 191 GB artifact at startup. File identity,
        # revision, patch and native strict-quant checks jointly isolate state.
        st = model.stat()
        identity = [str(model.resolve()), st.st_dev, st.st_ino, st.st_size,
                    st.st_mtime_ns, st.st_ctime_ns, DWARFSTAR_COMMIT,
                    (root / ".litmoe-patch-sha256").read_text().strip()]
        namespace = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        base = expand_path(self.model.dwarfstar_kv_dir or os.environ.get(
            "LITMOE_DWARFSTAR_CACHE", "~/.litmoe/dwarfstar-cache"))
        return base / namespace

    def build_command(self) -> list[str]:
        root = runtime_root().resolve()
        model = expand_path(self.model.model_path).resolve()
        if not model.is_file():
            raise FileNotFoundError(f"DwarfStar GGUF not found: {model}")
        if not 0 < self.model.n_ctx <= 2**31 - 1:
            raise ValueError("DwarfStar n_ctx must be between 1 and 2147483647")
        owned = {"-m", "--model", "-c", "--ctx", "--host", "--port", "--chdir",
                 "--kv-disk-dir", "--batched-session", "--ssd-streaming",
                 "--ssd-streaming-cache-experts", "--cpu", "--backend", "--metal"}
        conflicts = [arg for arg in self.model.extra_args if arg.split("=", 1)[0] in owned]
        if conflicts:
            raise ValueError(f"DwarfStar runtime-owned flags in extra_args: {conflicts}")
        if self.model.dwarfstar_cache_experts and not self.model.dwarfstar_ssd_streaming:
            raise ValueError("dwarfstar_cache_experts requires dwarfstar_ssd_streaming")
        cache = self._kv_disk_dir(root, model)
        cache.mkdir(parents=True, exist_ok=True, mode=0o700)
        command = [str(root / "ds4-server"), "--chdir", str(root),
                   "-m", str(model), "--metal", "--host", "127.0.0.1",
                   "--port", str(self.default_port()), "--ctx", str(self.model.n_ctx),
                   "--kv-disk-dir", str(cache), "--kv-cache-reject-different-quant"]
        if self.model.dwarfstar_ssd_streaming:
            command.append("--ssd-streaming")
            if self.model.dwarfstar_cache_experts:
                command += ["--ssd-streaming-cache-experts", self.model.dwarfstar_cache_experts]
        return command + self.model.extra_args

    async def wait_ready(self, timeout: float = 300.0) -> bool:
        self._capabilities = {}
        if not await super().wait_ready(timeout):
            return False
        async with httpx.AsyncClient(trust_env=False) as client:
            response = await client.get(self.health_url(), timeout=5)
            response.raise_for_status()
            status = response.json()
        if status.get("context_window") != self.model.n_ctx:
            raise RuntimeError("DwarfStar did not allocate the configured context window")
        capabilities = status.get("capabilities", {})
        if any(capabilities.get(key) != value for key, value in NATIVE_CAPABILITIES.items()):
            raise RuntimeError("DwarfStar native serving capabilities do not match the installed patch")
        self._capabilities = capabilities
        return True

    async def wait_idle(self, timeout: float = 10.0) -> bool:
        """Never release admission merely because a socket disconnected."""
        if not self.is_running() or not self.base_url or timeout <= 0:
            return False
        async def poll() -> bool:
            async with httpx.AsyncClient(trust_env=False) as client:
                while True:
                    response = await client.get(self.health_url(), timeout=min(5.0, timeout))
                    response.raise_for_status()
                    status = response.json()
                    if not isinstance(status, dict) or status.get("status") != "ok":
                        return False
                    active, queued = status.get("active_requests"), status.get("queued_requests")
                    if type(active) is not int or type(queued) is not int or min(active, queued) < 0:
                        return False
                    if active == queued == 0:
                        return self.is_running()
                    await asyncio.sleep(0.1)
        try:
            return await asyncio.wait_for(poll(), timeout=timeout)
        except (asyncio.TimeoutError, httpx.HTTPError, ValueError):
            return False
