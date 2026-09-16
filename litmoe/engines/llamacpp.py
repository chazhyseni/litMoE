"""llama.cpp engine adapter.

Native support for Kimi-K3, Qwen3.x MoE, DeepSeek, GLM, MiniMax, Gemma and many
other architectures via GGUF. Backends: CUDA, HIP (AMD), Metal (Apple), Vulkan,
SYCL, OpenCL, CANN (Ascend). Quantization: 1.5/2/3/4/5/6/8-bit.

Reference: https://github.com/ggml-org/llama.cpp (tools/server/README.md)
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from litmoe.config import ModelEntry, expand_path, is_hf_repo_spec
from litmoe.engines.base import Engine
from litmoe.platform_utils import (
    get_library_path_env,
    get_physical_cores,
    is_macos,
)

_THREAD_FLAGS = {"-t", "--threads"}


def _has_flag(args: list[str], flags: set[str]) -> bool:
    return any(a in flags or any(a.startswith(f + "=") for f in flags) for a in args)


class LlamaCppEngine(Engine):
    """Adapter for llama.cpp server (llama-server)."""

    def health_url(self) -> str:
        return f"http://127.0.0.1:{self.default_port()}/health"

    def _resolve_binary(self) -> tuple[str, Path | None]:
        """Resolve the llama-server binary path.

        Returns (binary_path, lib_dir) where lib_dir is the directory
        containing shared libraries (for env setup), or None if not needed.

        Prefers the direct binary over wrapper scripts so that env vars
        set by start() are passed directly to the process (not through
        a bash wrapper that may overwrite or lose them).
        """
        raw_prefix = os.environ.get("LITMOE_PREFIX", str(Path.home() / ".local"))
        prefix = Path(os.path.expanduser(os.path.expandvars(raw_prefix)))

        # Source builds live in local/, prebuilt releases in prebuilt/llama-bNNNNN/.
        for lib_subdir in ["lib/llama.cpp/local", "lib/llama.cpp/prebuilt"]:
            lib_dir = prefix / lib_subdir
            direct = lib_dir / "llama-server"
            if direct.is_file():
                return (str(direct), lib_dir)
            if lib_dir.exists():
                for p in sorted(lib_dir.rglob("llama-server"), reverse=True):
                    if p.is_file():
                        return (str(p), p.parent)

        found = shutil.which("llama-server") or shutil.which("llama-server.exe")
        if found:
            p = Path(found).resolve()
            so_files = list(p.parent.glob("lib*.so*")) + list(p.parent.glob("lib*.dylib*"))
            return (found, p.parent if so_files else None)

        raise FileNotFoundError(
            "llama-server not found. Install with: litmoe install --engine llamacpp "
            "(or build from https://github.com/ggml-org/llama.cpp and put llama-server on PATH)"
        )

    def build_command(self) -> list[str]:
        """Build llama-server command."""
        binary, lib_dir = self._resolve_binary()
        cmd = [binary]
        m = self.model

        # Model source: local GGUF path, HF repo spec (owner/repo[:quant]) or URL.
        if m.gguf_path:
            cmd.extend(["-m", str(expand_path(m.gguf_path))])
        elif m.model_path and m.model_path.startswith(("http://", "https://")):
            cmd.extend(["-hf", m.model_path])
        elif m.model_path and is_hf_repo_spec(m.model_path):
            # llama-server downloads to its own cache (~/.cache/llama.cpp) and
            # picks up the matching mmproj automatically for multimodal models.
            cmd.extend(["-hf", m.model_path])
        elif m.model_path:
            cmd.extend(["-m", str(expand_path(m.model_path))])
        else:
            raise ValueError(f"{m.id}: model_path or gguf_path required for llama.cpp")

        # GPU offload (-1 = all layers that fit; 0 = CPU only). llama-server
        # accepts a number for every release; 'auto'/'all' only on newer builds.
        cmd.extend(["-ngl", str(m.n_gpu_layers)])

        # Context size (0 = use the model's trained context)
        if m.n_ctx is not None:
            cmd.extend(["-c", str(m.n_ctx)])

        # Threads: one per physical core unless the user set -t in extra_args.
        # SMT threads slow memory-bound decode; a low fixed cap starves big CPUs.
        if not _has_flag(m.extra_args, _THREAD_FLAGS):
            cmd.extend(["-t", str(get_physical_cores())])

        cmd.extend(["--host", "127.0.0.1", "--port", str(self.default_port())])
        cmd.extend(m.extra_args)

        self._lib_dir = lib_dir
        return cmd

    def start(self, log_dir: Path | None = None) -> None:
        """Start the engine.

        On macOS, launches via /bin/bash to work around com.apple.provenance
        which blocks Python's execve() from running the binary directly.
        The wrapper script (rewritten at serve time with correct env vars)
        sets DYLD_FALLBACK_LIBRARY_PATH and execs the binary.
        On Linux, launches the binary directly with LD_LIBRARY_PATH.
        """
        cmd = self.build_command()
        env = os.environ.copy()
        env.update(self.model.env)

        lib_dir = getattr(self, "_lib_dir", None)
        if lib_dir and is_macos():
            actual_binary = lib_dir / "llama-server"
            if actual_binary.exists():
                # Strip com.apple.provenance by copying the binary over itself.
                tmp = actual_binary.parent / ".llama-server.tmp"
                shutil.copy2(str(actual_binary), str(tmp))
                shutil.move(str(tmp), str(actual_binary))
                os.chmod(str(actual_binary), 0o755)
                for dylib in actual_binary.parent.glob("*.dylib*"):
                    if dylib.is_file():
                        dtmp = dylib.parent / f".{dylib.name}.tmp"
                        shutil.copy2(str(dylib), str(dtmp))
                        shutil.move(str(dtmp), str(dylib))

            prefix = Path(os.path.expanduser(os.environ.get("LITMOE_PREFIX", str(Path.home() / ".local"))))
            wrapper = prefix / "bin" / "llama-server"
            wrapper.parent.mkdir(parents=True, exist_ok=True)
            # Never write through a symlink: that would overwrite the binary it points to.
            if wrapper.is_symlink() or wrapper.exists():
                wrapper.unlink()
            if "prebuilt" in str(lib_dir):
                wrapper.write_text(f"#!/bin/bash\nexec {actual_binary} \"$@\"\n")
            else:
                wrapper.write_text(
                    f"#!/bin/bash\n"
                    f"export DYLD_FALLBACK_LIBRARY_PATH={lib_dir}:$DYLD_FALLBACK_LIBRARY_PATH\n"
                    f"export DYLD_LIBRARY_PATH={lib_dir}:$DYLD_LIBRARY_PATH\n"
                    f"exec {actual_binary} \"$@\"\n"
                )
            wrapper.chmod(0o755)

            shell_cmd = f'"{wrapper}" ' + ' '.join(f'"{a}"' for a in cmd[1:])
            with self._open_log(log_dir, cmd) as logf:
                self.process = subprocess.Popen(
                    ['/bin/bash', '-c', shell_cmd],
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                )
        else:
            if lib_dir:
                env.update(get_library_path_env(lib_dir))
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


def is_installed() -> bool:
    """Check if llama-server is installed (PATH or the litmoe install prefix)."""
    if shutil.which("llama-server") or shutil.which("llama-server.exe"):
        return True
    from litmoe.platform_utils import find_llama_server_binary
    return find_llama_server_binary() is not None
