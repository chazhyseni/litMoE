"""WARP engine adapter."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from litmoe.config import ModelEntry, expand_path
from litmoe.engines.base import Engine


def _library_name() -> str:
    if sys.platform == "darwin":
        return "libwaste.dylib"
    if sys.platform == "win32":
        return "libwaste.dll"
    return "libwaste.so"


def _candidate_roots() -> list[Path]:
    candidates = []

    explicit = os.environ.get("LITMOE_WARP_DIR")
    if explicit:
        candidates.append(expand_path(explicit))

    raw_prefix = os.environ.get("LITMOE_PREFIX")
    if raw_prefix:
        candidates.append(expand_path(raw_prefix) / "lib" / "warp")
    else:
        try:
            candidates.append(Path.home() / ".local" / "lib" / "warp")
        except RuntimeError:
            pass

    waste = shutil.which("waste") or shutil.which("waste.exe")
    if waste:
        path_root = Path(waste).resolve().parent
        if (path_root / "serve" / "__main__.py").is_file():
            candidates.append(path_root)
        prefix_root = path_root.parent / "lib" / "warp"
        if (prefix_root / "serve" / "__main__.py").is_file():
            candidates.append(prefix_root)
    return candidates


def _resolve_root() -> Path:
    incomplete_root = None
    for root in _candidate_roots():
        if not (root / "serve" / "__main__.py").is_file():
            continue
        if (root / _library_name()).is_file():
            return root
        if incomplete_root is None:
            incomplete_root = root
    if incomplete_root is not None:
        _validate_installation(incomplete_root)
    raise FileNotFoundError(
        "WARP server not found. Install with: litmoe install --engine warp"
    )


def _validate_installation(root: Path) -> None:
    library = root / _library_name()
    if not library.is_file():
        raise FileNotFoundError(
            f"WARP shared library {library.name} not found at {library}. "
            "Install with: litmoe install --engine warp"
        )


class WarpEngine(Engine):
    """Adapter for WARP's OpenAI-compatible server."""

    def health_url(self) -> str:
        return f"http://127.0.0.1:{self.default_port()}/health"

    def build_environment(self) -> dict[str, str]:
        """Build a WARP environment with litmoe-owned runtime settings."""
        environment = super().build_environment()
        environment.pop("WASTE_API_KEY", None)

        root = _resolve_root()
        _validate_installation(root)
        environment["WASTE_LIB"] = str(root / _library_name())
        return environment

    def build_command(self) -> list[str]:
        """Build the WARP server command for a local container."""
        root = _resolve_root()
        _validate_installation(root)

        container = expand_path(self.model.model_path)
        if not container.exists():
            raise FileNotFoundError(f"WARP container not found: {container}")

        cmd = [
            sys.executable,
            str(root / "serve" / "__main__.py"),
            str(container),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.default_port()),
            "--model-id",
            self.model.id,
        ]
        if self.model.n_ctx > 0:
            cmd.extend(["--ctx", str(self.model.n_ctx)])
        cmd.extend(self.model.extra_args)
        return cmd


def is_installed() -> bool:
    """Return whether a complete WARP server installation is available."""
    try:
        root = _resolve_root()
        _validate_installation(root)
    except (FileNotFoundError, OSError, RuntimeError):
        return False
    return True
