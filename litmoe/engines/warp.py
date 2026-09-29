"""WARP engine adapter."""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from litmoe.config import ModelEntry, expand_path
from litmoe.engines.base import Engine
from litmoe.models import lookup

logger = logging.getLogger(__name__)


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

    def __init__(self, model: ModelEntry):
        super().__init__(model)
        self._context_prepared = False

    def _prepare_context(self, root: Path, container: Path) -> None:
        if self._context_prepared:
            return
        if any(arg.split("=", 1)[0] in ("--ctx", "--ct") for arg in self.model.extra_args):
            raise ValueError("Set WARP context with n_ctx and warp_auto_context, not extra_args --ctx")
        auto = self.model.warp_auto_context
        if auto is None:
            auto = self.model.n_ctx in (0, 65536)
            if auto:
                logger.warning("Model %s: migrating legacy n_ctx=%d to automatic WARP sizing; "
                               "set warp_auto_context: false to keep a fixed positive limit",
                               self.model.id, self.model.n_ctx)
        if not auto:
            if not 0 < self.model.n_ctx <= 2**31 - 1:
                raise ValueError("Fixed WARP n_ctx must be between 1 and 2147483647")
            self.model.warp_auto_context = False
            self._context_prepared = True
            return

        info = lookup(self.model.id)
        if info and info["engine"] == "warp":
            native = info["native_ctx"]
        else:
            manifest = json.loads((container / "manifest.json").read_text())
            native = manifest.get("config", {}).get("max_position_embeddings")
        if not isinstance(native, int) or not 0 < native <= 2**31 - 1:
            raise ValueError("Cannot determine WARP native context; set a positive n_ctx "
                             "and warp_auto_context: false")
        command = [
            sys.executable, str(Path(__file__).with_name("warp_context.py")),
            str(root), str(container), str(native), json.dumps(self.model.extra_args),
        ]
        try:
            result = subprocess.run(
                command, env=self.build_environment(), capture_output=True,
                text=True, timeout=60, check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"WARP context planning failed: {exc.stderr.strip()[-2000:]}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("WARP context planning timed out") from exc
        plan = json.loads(result.stdout)
        resolved = plan["n_ctx"]
        if not isinstance(resolved, int) or not 0 < resolved <= native:
            raise ValueError("WARP planner returned an invalid context")
        self.model.n_ctx = resolved
        self.model.warp_auto_context = True
        self._context_prepared = True
        logger.info("Model %s: WARP context %d/%d native tokens; recommended %.2f GiB "
                    "within %.2f GiB planning budget (not current free RAM)",
                    self.model.id, resolved, native, plan["required_bytes"] / 1024**3,
                    plan["budget_bytes"] / 1024**3)

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
        self._prepare_context(root, container)

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
