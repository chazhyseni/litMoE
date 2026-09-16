"""Configuration loading and validation."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

# owner/repo[:quant] — a HuggingFace repo spec, as accepted by llama-server -hf.
# No scheme, exactly one slash, no leading ./ ~ or /.
_HF_REPO_RE = re.compile(r"^[A-Za-z0-9][\w.\-]*/[\w.\-]+(:[\w.\-]+)?$")


def expand_path(p: str | Path) -> Path:
    """Expand ~ and environment variables in a path, return absolute Path.

    Python's Path() does NOT expand '~' — only expanduser() does. The shell
    expands '~' automatically, so files appear to exist from the terminal but
    not from Python's open()/exists(). Call this on every path that enters the
    system from CLI arguments, environment variables, or models.yaml.
    """
    if isinstance(p, Path):
        p = str(p)
    return Path(os.path.expandvars(os.path.expanduser(p))).resolve()


def is_hf_repo_spec(value: str) -> bool:
    """True if `value` looks like a HuggingFace repo spec (owner/repo[:quant]) rather than a path.

    A string that also exists on disk (relative path) is treated as a path.
    """
    if not value or value.startswith(("/", "~", ".", "http://", "https://")) or "$" in value:
        return False
    if not _HF_REPO_RE.match(value):
        return False
    return not Path(value).exists()


def expand_model_paths(entry_dict: dict) -> dict:
    """Expand ~ and $VARS in the path fields of a model entry dict.

    HuggingFace repo specs (owner/repo[:quant]) and URLs are left untouched so
    the engine adapter can pass them to llama-server -hf.
    """
    for key in ("model_path", "gguf_path"):
        val = entry_dict.get(key)
        if not val or not isinstance(val, str):
            continue
        if val.startswith(("http://", "https://")) or is_hf_repo_spec(val):
            continue
        if val.startswith("~") or "$" in val:
            entry_dict[key] = str(expand_path(val))
    return entry_dict


class ModelEntry(BaseModel):
    """A model exposed via the OpenAI API."""
    id: str  # OpenAI model id (e.g. "kimi-k3")
    engine: Literal["ktransformers", "llamacpp"]
    # llamacpp: local GGUF path, HF repo spec (owner/repo[:quant]) or URL.
    # ktransformers: local safetensors directory or HF repo id (owner/repo).
    model_path: str
    # ktransformers only: GGUF directory for the LLAMAFILE CPU backend (--kt-weight-path).
    gguf_path: str | None = None
    # llamacpp: -ngl. -1 = offload as many layers as fit ("auto"), 0 = CPU only.
    n_gpu_layers: int = -1
    # Context window. litmoe raises values below 16384 to the model's native
    # context if the KV cache fits in RAM (see server.load_engines).
    n_ctx: int = 65536
    extra_args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    # Alternate model IDs that route to this model's engine
    # (e.g. "claude-sonnet-4-5" so Anthropic API clients work unchanged)
    aliases: list[str] = Field(default_factory=list)

    # --- ktransformers (sglang-kt) options -------------------------------
    # CPU expert backend: FP8, FP8_PERCHANNEL, BF16, RAWINT4, MXFP4, MXFP8,
    # AMXINT4, AMXINT8, LLAMAFILE. Default: LLAMAFILE when gguf_path is set.
    kt_method: str | None = None
    # Experts kept on GPU (0 = all experts on CPU; GPU runs attention/dense).
    kt_num_gpu_experts: int = 0
    # CPU inference threads (default: physical cores) and thread pools (default: NUMA nodes).
    kt_cpuinfer: int | None = None
    kt_threadpool_count: int | None = None


class GatewayConfig(BaseModel):
    """Top-level litmoe configuration."""
    host: str = "0.0.0.0"
    port: int = 8080
    api_key: str | None = None
    models: list[ModelEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_ids(self) -> "GatewayConfig":
        seen: dict[str, str] = {}
        for m in self.models:
            for name in [m.id, *m.aliases]:
                if name in seen and seen[name] != m.id:
                    raise ValueError(
                        f"model id/alias '{name}' is used by both '{seen[name]}' and '{m.id}'"
                    )
                seen.setdefault(name, m.id)
        return self


def default_config_path() -> Path:
    """Path to the default config file.

    Checks LITMOE_CONFIG env var first, then models.yaml in the current
    directory, then ~/.litmoe/models.yaml as a fallback.
    Returns an absolute path (with ~ and $VARS expanded).
    """
    env_config = os.environ.get("LITMOE_CONFIG")
    if env_config:
        return expand_path(env_config)

    local = Path("models.yaml")
    if local.exists():
        return local.resolve()

    home = Path.home() / ".litmoe" / "models.yaml"
    if home.exists():
        return home

    # Default to cwd/models.yaml even if it doesn't exist yet
    return local.resolve()


def load_config(path: Path | str | None = None) -> GatewayConfig:
    """Load and validate litmoe config.

    Path fields in models.yaml are expanded for ~ and $VARS; HF repo specs and
    URLs are passed through unchanged.
    """
    p = expand_path(path) if path else default_config_path()
    if not p.exists():
        raise FileNotFoundError(
            f"litmoe config not found: {p}. "
            f"Create one with 'litmoe init' or copy examples/models.yaml."
        )
    with open(p) as f:
        data = yaml.safe_load(f) or {}

    for entry in data.get("models", []) or []:
        if isinstance(entry, dict):
            expand_model_paths(entry)

    return GatewayConfig(**data)
