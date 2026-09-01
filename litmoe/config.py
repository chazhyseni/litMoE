"""Configuration loading and validation."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


def expand_path(p: str | Path) -> Path:
    """Expand ~ and environment variables in a path, return absolute Path.

    This is the single fix for the macOS FileNotFoundError issue.
    Python's Path() does NOT expand '~' — only expanduser() does.
    The shell (ls, head, etc.) expands '~' automatically, so files appear
    to exist from the terminal but not from Python's open()/exists().

    This function must be called on every path that enters the system from:
    - CLI arguments (--config, --models-dir, --prefix)
    - Environment variables (LITMOE_CONFIG, LITMOE_MODELS_DIR, LITMOE_PREFIX)
    - Config file contents (model_path, gguf_path in models.yaml)
    """
    if isinstance(p, Path):
        p = str(p)
    return Path(os.path.expandvars(os.path.expanduser(p))).resolve()


def expand_model_paths(entry_dict: dict) -> dict:
    """Expand ~ and $VARS in all path fields of a model entry dict.

    Called when loading models.yaml to fix paths that contain '~' or
    environment variables. Without this, paths like '~/models/foo.gguf'
    in the config file cause FileNotFoundError because Python doesn't
    expand '~' like the shell does.
    """
    for key in ("model_path", "gguf_path"):
        val = entry_dict.get(key)
        if val and isinstance(val, str) and (val.startswith("~") or "$" in val):
            entry_dict[key] = str(expand_path(val))
    return entry_dict


class ModelEntry(BaseModel):
    """A model exposed via the OpenAI API."""
    id: str  # OpenAI model id (e.g. "kimi-k3")
    engine: Literal["ktransformers", "llamacpp"]
    model_path: str
    gguf_path: str | None = None
    n_gpu_layers: int = -1
    n_ctx: int = 65536
    extra_args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    # Alternate model IDs that route to this model's engine
    # (e.g. "claude-sonnet-4-5" so Anthropic API clients work unchanged)
    aliases: list[str] = Field(default_factory=list)


class GatewayConfig(BaseModel):
    """Top-level litmoe configuration."""
    host: str = "0.0.0.0"
    port: int = 8080
    api_key: str | None = None
    models: list[ModelEntry] = Field(default_factory=list)


def default_config_path() -> Path:
    """Path to the default config file.

    Checks LITMOE_CONFIG env var first, then models.yaml in the current
    directory, then ~/.litmoe/models.yaml as a fallback.
    Returns an absolute path (with ~ and $VARS expanded).
    """
    from litmoe.config import expand_path

    env_config = os.environ.get("LITMOE_CONFIG")
    if env_config:
        return expand_path(env_config)

    # Check current directory
    local = Path("models.yaml")
    if local.exists():
        return local.resolve()

    # Check ~/.litmoe/models.yaml
    home = Path.home() / ".litmoe" / "models.yaml"
    if home.exists():
        return home

    # Default to cwd/models.yaml even if it doesn't exist yet
    return local.resolve()


def load_config(path: Path | str | None = None) -> GatewayConfig:
    """Load and validate litmoe config.

    All paths in the config file (model_path, gguf_path) are expanded
    for ~ and $VARS to avoid FileNotFoundError on macOS where Python
    doesn't expand '~' like the shell does.
    """
    from litmoe.config import expand_path, expand_model_paths

    p = expand_path(path) if path else default_config_path()
    if not p.exists():
        raise FileNotFoundError(
            f"litmoe config not found: {p}. "
            f"Create one with 'litmoe init' or copy examples/models.yaml."
        )
    with open(p) as f:
        data = yaml.safe_load(f)

    # Expand paths in all model entries — this is the critical fix.
    # Paths like '~/models/foo.gguf' in models.yaml cause FileNotFoundError
    # because Python's Path() and open() don't expand '~' like the shell does.
    for entry in data.get("models", []):
        expand_model_paths(entry)

    return GatewayConfig(**data)
