"""ktransformers engine adapter (sglang-kt).

Since ktransformers v0.4 the serving stack is SGLang with kt-kernel CPU expert
offload. The old ``python -m ktransformers.server.main`` entry point no longer
exists (the framework was moved to ``archive/``); the root ``ktransformers``
PyPI package is a shim that depends on ``kt-kernel`` and, with the ``[sglang]``
extra, ``sglang-kt``.

Serving command (kt-kernel README, GLM-5.3-Flash tutorial, kt-cli.md):

    python -m sglang.launch_server \\
        --model-path <safetensors dir or HF id> \\
        --kt-method FP8|BF16|RAWINT4|MXFP4|MXFP8|AMXINT4|AMXINT8|LLAMAFILE \\
        --kt-weight-path <gguf dir>          # LLAMAFILE only \\
        --kt-cpuinfer <physical cores> --kt-threadpool-count <NUMA nodes> \\
        --kt-num-gpu-experts N --host 127.0.0.1 --port P --context-length C \\
        --trust-remote-code --served-model-name <id>

Requirements: Linux x86-64, NVIDIA GPU (SM 8.0+), Python 3.11/3.12. CPU expert
backends FP8/BF16/RAWINT4/MXFP* need AVX-512, AMXINT4/8 need AMX (Xeon 4th
gen+); LLAMAFILE (GGUF weights) runs on AVX2.

Reference: https://github.com/kvcache-ai/ktransformers
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

from litmoe.config import ModelEntry, expand_path, is_hf_repo_spec
from litmoe.engines.base import Engine
from litmoe.models import KT_METHODS
from litmoe.platform_utils import get_numa_nodes, get_physical_cores


class KtransformersEngine(Engine):
    """Adapter for ktransformers served through sglang-kt."""

    def health_url(self) -> str:
        # sglang exposes GET /health (200 once the model is loaded).
        return f"http://127.0.0.1:{self.default_port()}/health"

    def kt_method(self) -> str:
        """CPU expert backend. Explicit kt_method wins; LLAMAFILE when a GGUF dir is given."""
        m = self.model
        if m.kt_method:
            method = m.kt_method.upper()
            if method not in KT_METHODS:
                raise ValueError(
                    f"{m.id}: unknown kt_method '{m.kt_method}'. Valid: {', '.join(KT_METHODS)}"
                )
            return method
        if m.gguf_path:
            return "LLAMAFILE"
        raise ValueError(
            f"{m.id}: ktransformers needs 'kt_method' (FP8, BF16, RAWINT4, MXFP4, MXFP8, "
            f"AMXINT4, AMXINT8) matching the checkpoint's expert precision, or 'gguf_path' "
            f"for the LLAMAFILE backend."
        )

    def build_command(self) -> list[str]:
        """Build the sglang-kt server command."""
        m = self.model
        # Always use sys.executable so the engine runs under the interpreter
        # litmoe was installed with (avoids PATH 'python' mismatches).
        cmd = [sys.executable, "-m", "sglang.launch_server"]

        if not m.model_path:
            raise ValueError(f"{m.id}: model_path (safetensors directory or HF repo id) required")
        if is_hf_repo_spec(m.model_path) or m.model_path.startswith(("http://", "https://")):
            model_path = m.model_path.split(":", 1)[0] if not m.model_path.startswith("http") else m.model_path
        else:
            model_path = str(expand_path(m.model_path))
        cmd.extend(["--model-path", model_path])

        method = self.kt_method()
        cmd.extend(["--kt-method", method])
        if method == "LLAMAFILE":
            if not m.gguf_path:
                raise ValueError(f"{m.id}: kt_method LLAMAFILE requires gguf_path")
            cmd.extend(["--kt-weight-path", str(expand_path(m.gguf_path))])
        elif m.gguf_path:
            # Non-LLAMAFILE backends read safetensors; gguf_path is ignored by sglang-kt.
            print(f"  {m.id}: warning: gguf_path is only used with kt_method LLAMAFILE (using {method})")

        cmd.extend(["--kt-cpuinfer", str(m.kt_cpuinfer or get_physical_cores())])
        cmd.extend(["--kt-threadpool-count", str(m.kt_threadpool_count or get_numa_nodes())])
        cmd.extend(["--kt-num-gpu-experts", str(m.kt_num_gpu_experts)])

        if m.n_ctx:
            cmd.extend(["--context-length", str(m.n_ctx)])

        cmd.extend(["--host", "127.0.0.1", "--port", str(self.default_port())])
        cmd.extend(["--served-model-name", m.id, "--trust-remote-code"])
        cmd.extend(m.extra_args)
        return cmd


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def is_installed() -> bool:
    """True if both kt-kernel and sglang(-kt) are importable, or the kt CLI exists."""
    if _module_available("kt_kernel") and _module_available("sglang"):
        return True
    return shutil.which("kt") is not None


def missing_components() -> list[str]:
    """Which serving components are absent (for `litmoe doctor`)."""
    missing = []
    if not _module_available("kt_kernel"):
        missing.append("kt_kernel")
    if not _module_available("sglang"):
        missing.append("sglang (sglang-kt)")
    return missing
