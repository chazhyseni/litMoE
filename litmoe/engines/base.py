"""Base interface for engine adapters."""
from __future__ import annotations

import abc
import asyncio
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path
from typing import IO

from litmoe.config import ModelEntry

DEFAULT_ENGINE_PORT = 8081


def pid_dir() -> Path:
    """Directory holding <model-id>.pid files for engines litmoe itself started.

    `litmoe stop` reads these instead of pattern-matching process names, so it
    never touches llama-server / sglang processes started by other tools
    (Ollama, LM Studio, a manual run).
    """
    raw = os.environ.get("LITMOE_RUN_DIR", str(Path.home() / ".litmoe" / "run"))
    return Path(os.path.expanduser(os.path.expandvars(raw)))


class Engine(abc.ABC):
    """Abstract base for engine adapters (ktransformers, llama.cpp, etc.).

    Engines are subprocesses that speak OpenAI-compatible HTTP. The gateway
    just forwards requests to them. This is the simplest possible architecture.
    """

    def __init__(self, model: ModelEntry):
        self.model = model
        self.process: subprocess.Popen | None = None
        self.base_url: str | None = None
        self._log_path: Path | None = None
        self._assigned_port: int | None = None

    @abc.abstractmethod
    def build_command(self) -> list[str]:
        """Build the command to start this engine."""
        ...

    @abc.abstractmethod
    def health_url(self) -> str:
        """URL to poll for readiness."""
        ...

    def set_port(self, port: int) -> None:
        """Port the engine must listen on. Assigned by the gateway before start()."""
        self._assigned_port = port

    def default_port(self) -> int:
        """Port this engine listens on (gateway-assigned, else DEFAULT_ENGINE_PORT)."""
        return self._assigned_port if self._assigned_port is not None else DEFAULT_ENGINE_PORT

    def _open_log(self, log_dir: Path | None, cmd: list[str]) -> IO[str]:
        """Open the per-model engine log in append mode and write a session header.

        Append mode keeps previous runs (and their performance lines) instead of
        truncating the file on every restart.
        """
        ld = Path(log_dir) if log_dir else Path("logs")
        ld.mkdir(parents=True, exist_ok=True)
        log_file = ld / f"{self.model.id}.log"
        self._log_path = log_file
        logf = open(log_file, "a")
        logf.write(
            f"\n===== litmoe session {time.strftime('%Y-%m-%d %H:%M:%S %z')} — {self.model.id} =====\n"
            f"$ {shlex.join(cmd)}\n"
        )
        logf.flush()
        return logf

    def _record_pid(self) -> None:
        """Write the engine PID to the run dir so `litmoe stop` only kills litmoe's own engines."""
        if not self.process:
            return
        run_dir = pid_dir()
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / f"{self.model.id}.pid").write_text(f"{self.process.pid}\n")

    def _clear_pid(self) -> None:
        try:
            (pid_dir() / f"{self.model.id}.pid").unlink()
        except FileNotFoundError:
            pass

    def start(self, log_dir: Path | None = None) -> None:
        """Start the engine as a subprocess."""
        cmd = self.build_command()
        env = os.environ.copy()
        env.update(self.model.env)

        with self._open_log(log_dir, cmd) as logf:
            self.process = subprocess.Popen(
                cmd,
                stdout=logf,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        self._record_pid()

        self.base_url = f"http://127.0.0.1:{self.default_port()}"
        print(f"  {self.model.id}: started PID {self.process.pid}, logs: {self._log_path}")

    def stop(self) -> None:
        """Stop the engine."""
        try:
            if self.process and self.process.poll() is None:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                except ProcessLookupError:
                    return
                try:
                    self.process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        return
                    self.process.wait(timeout=5)
        finally:
            self._clear_pid()

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    async def wait_ready(self, timeout: float = 300.0) -> bool:
        """Poll health endpoint until ready or timeout."""
        import httpx
        url = self.health_url()
        loop = asyncio.get_running_loop()
        async with httpx.AsyncClient() as client:
            start = loop.time()
            while True:
                if self.process and self.process.poll() is not None:
                    print(f"  {self.model.id}: process exited with code {self.process.returncode}"
                          f" (see {self._log_path})")
                    return False
                try:
                    r = await client.get(url, timeout=5.0)
                    if r.status_code == 200:
                        return True
                except (httpx.RequestError, httpx.TimeoutException):
                    pass
                if loop.time() - start > timeout:
                    print(f"  {self.model.id}: timeout waiting for {url}")
                    return False
                await asyncio.sleep(2.0)
