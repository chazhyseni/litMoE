"""Install catalog WARP models into deterministic local ``.waste`` containers."""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from litmoe.models import WASTE, lookup

_DISK_SAFETY_FRACTION = 0.05


@dataclass(frozen=True)
class WarpInstallPlan:
    """Resolved, non-mutating plan for one catalog WARP conversion."""

    model_id: str
    info: Mapping[str, object]
    source: Path
    output: Path
    run_dir: Path
    source_bytes: int
    output_workspace_bytes: int


def build_plan(
    model_id: str,
    *,
    staging_dir: Path,
    models_dir: Path,
) -> WarpInstallPlan:
    """Resolve catalog metadata and deterministic paths without creating them."""
    info = lookup(model_id)
    if not info or info.get("format") != WASTE or info.get("engine") != "warp":
        raise RuntimeError(f"{model_id!r} is not an installable WARP catalog model")
    models_root = Path(models_dir).expanduser().resolve(strict=False)
    paths = {
        "source": (Path(staging_dir).expanduser() / model_id).resolve(strict=False),
        "output": (models_root / f"{model_id}.waste").resolve(strict=False),
        "run": (models_root / f"{model_id}.warp-run").resolve(strict=False),
    }
    for label, path in paths.items():
        for character, detail in (
            ("'", "single quote"), ("\\", "backslash"),
            ("\n", "newline"), ("\r", "carriage return"),
        ):
            if character in str(path):
                raise RuntimeError(f"WARP {label} path contains a {detail}: {path}")

    for first, second in (("source", "output"), ("source", "run"), ("output", "run")):
        left, right = paths[first], paths[second]
        if left == right or left in right.parents or right in left.parents:
            raise RuntimeError(
                f"WARP plan paths overlap or are nested: "
                f"{first} {left}; {second} {right}"
            )
    return WarpInstallPlan(
        model_id=model_id,
        info=info,
        source=paths["source"],
        output=paths["output"],
        run_dir=paths["run"],
        source_bytes=int(info["source_size_gib"]) * 1024**3,
        output_workspace_bytes=int(info["output_workspace_gib"]) * 1024**3,
    )


def require_tools(
    commands: tuple[str, ...],
    *,
    which: Callable[[str], str | None] = shutil.which,
) -> dict[str, str]:
    """Resolve required executables before any paths or processes are created."""
    executables: dict[str, str] = {}
    for command in commands:
        executable = which(command)
        if not executable:
            raise RuntimeError(
                f"'{command}' is required to install WARP catalog models"
            )
        executables[command] = executable
    return executables


def _existing_ancestor(path: Path) -> Path:
    """Return *path* or its nearest existing parent."""
    candidate = path
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent
    return candidate


def filesystem_device(path: Path) -> int:
    """Device id for the filesystem that contains or will contain *path*."""
    return os.stat(_existing_ancestor(Path(path))).st_dev


def _tree_size(path: Path) -> int:
    """Bytes already present below *path*, for resumable disk accounting."""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _read_ledger(path: Path) -> set[str] | None:
    try:
        return {
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip()
        }
    except (OSError, UnicodeDecodeError):
        return None


def _reclaimed_source_is_complete(source_dir: Path) -> bool:
    """Return whether index + ledgers prove every source shard completed."""
    index_path = source_dir / "model.safetensors.index.json"
    try:
        index = json.loads(index_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        return False

    expected: set[str] = set()
    for value in weight_map.values():
        if not isinstance(value, str) or not value:
            return False
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            return False
        expected.add(value)
    completed = _read_ledger(source_dir / ".download-state")
    reclaimed = _read_ledger(source_dir / ".reclaimed")
    if completed is None or reclaimed is None or not expected <= completed:
        return False
    missing = {
        shard
        for shard in expected
        if not (source_dir / shard).is_file()
    }
    return missing <= reclaimed


def _with_safety_margin(required: int) -> int:
    if required <= 0:
        return 0
    return required + math.ceil(required * _DISK_SAFETY_FRACTION)


def _format_bytes(value: int) -> str:
    if value < 1024:
        return f"{value} B"
    if value < 1024**3:
        return f"{value / 1024**2:.1f} MiB"
    return f"{value / 1024**3:.1f} GiB"


def preflight_disk(
    source_dir: Path,
    output_dir: Path,
    *,
    source_bytes: int,
    output_bytes: int,
    device: Callable[[Path], int] = filesystem_device,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
) -> None:
    """Require enough free space for remaining source and workspace bytes.

    Existing files are credited as resumable work. A reclaimed source earns
    full completion credit only when the shard index and both pipeline ledgers
    prove every expected shard was downloaded and every absent shard reclaimed.
    Requirements are combined on one filesystem and checked independently on
    split filesystems.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    if _reclaimed_source_is_complete(source_dir):
        source_remaining = 0
    else:
        source_remaining = max(0, source_bytes - _tree_size(source_dir))
    output_remaining = max(0, output_bytes - _tree_size(output_dir))
    source_device = device(source_dir)
    output_device = device(output_dir)

    if source_device == output_device:
        required = _with_safety_margin(source_remaining + output_remaining)
        free = disk_usage(_existing_ancestor(source_dir)).free
        if free < required:
            raise RuntimeError(
                "WARP disk preflight failed: source staging and output are on the same filesystem; "
                f"need {_format_bytes(required)} free for remaining resumable data, "
                f"but only {_format_bytes(free)} is available"
            )
        return

    source_required = _with_safety_margin(source_remaining)
    source_free = disk_usage(_existing_ancestor(source_dir)).free
    if source_free < source_required:
        raise RuntimeError(
            "WARP disk preflight failed for staging filesystem: "
            f"need {_format_bytes(source_required)} free after resume credit, "
            f"but only {_format_bytes(source_free)} is available"
        )

    output_required = _with_safety_margin(output_remaining)
    output_free = disk_usage(_existing_ancestor(output_dir)).free
    if output_free < output_required:
        raise RuntimeError(
            "WARP disk preflight failed for output filesystem: "
            f"need {_format_bytes(output_required)} free after resume credit, "
            f"but only {_format_bytes(output_free)} is available"
        )


@contextmanager
def _pipeline_environment():
    environment = os.environ.copy()
    token = environment.pop("HF_TOKEN", None)
    if token is None:
        yield environment, None
        return
    if "\n" in token or "\r" in token:
        raise RuntimeError("HF_TOKEN must not contain a newline or carriage return")
    escaped = token.replace("\\", "\\\\").replace('"', '\\"')
    with tempfile.TemporaryDirectory(prefix=".litmoe-curl-") as directory:
        curl_home = Path(directory)
        curl_home.chmod(0o700)
        config = curl_home / ".curlrc"
        config.touch(mode=0o600)
        config.write_text(f'header = "Authorization: Bearer {escaped}"\n')
        config.chmod(0o600)
        environment["CURL_HOME"] = str(curl_home)
        yield environment, token


def _read_failure_marker(run_dir: Path, hf_token: str | None) -> str:
    marker = run_dir / ".failed"
    try:
        detail = marker.read_text().strip()
    except (OSError, UnicodeDecodeError):
        return ""
    if hf_token:
        detail = detail.replace(hf_token, "[redacted]")
    return detail[:500]


def _run_stage(
    args: list[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    stage: str,
    source: Path,
    output: Path,
    run_dir: Path,
    run: Callable[..., subprocess.CompletedProcess[str]],
    hf_token: str | None = None,
) -> None:
    context = f"source {source}; output {output}; run directory {run_dir}"
    try:
        result = run(args, cwd=cwd, env=dict(env))
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            f"WARP {stage} failed: {exc}; {context}. Partial data was preserved; "
            "rerun the same install command to resume."
        ) from exc
    if result.returncode == 0:
        return

    marker = _read_failure_marker(run_dir, hf_token) if stage == "pipeline" else ""
    marker_detail = f"; failed stage marker: {marker}" if marker else ""
    raise RuntimeError(
        f"WARP {stage} failed (exit {result.returncode}){marker_detail}; {context}. "
        "Partial data was preserved; rerun the same install command to resume."
    )


def _container_member(container: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"invalid WARP container: manifest has no valid {label}")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"invalid WARP container: unsafe {label} path {value!r}")
    return container / relative


def validate_container(container: Path, model_id: str) -> None:
    """Validate a completed container against WARP's v0 on-disk contract."""
    info = lookup(model_id)
    if not info or info.get("format") != WASTE:
        raise RuntimeError(
            f"invalid WARP container: unknown catalog model {model_id!r}"
        )

    container = Path(container)
    manifest_path = container / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("invalid WARP container: missing manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid WARP container manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("invalid WARP container: manifest must be a JSON object")
    if type(manifest.get("format_version")) is not int or manifest["format_version"] != 0:
        raise RuntimeError(
            "invalid WARP container: manifest format_version must be 0"
        )
    if manifest.get("arch") != info["arch"]:
        raise RuntimeError(
            f"invalid WARP container: manifest arch {manifest.get('arch')!r} "
            f"does not match catalog arch {info['arch']!r}"
        )

    trunk = manifest.get("trunk")
    if not isinstance(trunk, list) or not trunk:
        raise RuntimeError(
            "invalid WARP container: manifest trunk must be a nonempty list"
        )
    layers = manifest.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise RuntimeError(
            "invalid WARP container: manifest layers must be a nonempty object"
        )

    for layer_number, layer in layers.items():
        if not isinstance(layer_number, str) or not layer_number.isdecimal():
            raise RuntimeError(
                f"invalid WARP container: layer key {layer_number!r} is not numeric"
            )
        if not isinstance(layer, dict):
            raise RuntimeError(
                f"invalid WARP container: layer {layer_number} must be an object"
            )
        for field in ("file", "experts", "bytes", "codebook_base"):
            if field not in layer:
                raise RuntimeError(
                    f"invalid WARP container: layer {layer_number} missing {field}"
                )
        expected_name = f"experts-L{layer_number}.bin"
        if layer["file"] != expected_name:
            raise RuntimeError(
                f"invalid WARP container: layer {layer_number} file must be "
                f"{expected_name!r}"
            )
        expert_path = _container_member(container, layer["file"], "expert bank")
        if not expert_path.is_file():
            raise RuntimeError(
                f"invalid WARP container: missing expert bank {expected_name}"
            )
        for field in ("experts", "bytes"):
            if type(layer[field]) is not int or layer[field] <= 0:
                raise RuntimeError(
                    f"invalid WARP container: layer {layer_number} {field} "
                    "must be a positive integer"
                )
        if (
            type(layer["codebook_base"]) is not int
            or layer["codebook_base"] < 0
        ):
            raise RuntimeError(
                f"invalid WARP container: layer {layer_number} codebook_base "
                "must be a nonnegative integer"
            )
        if expert_path.stat().st_size != layer["bytes"]:
            raise RuntimeError(
                f"invalid WARP container: expert bank {expected_name} size does "
                f"not match layer bytes {layer['bytes']}"
            )

    for required in (
        "trunk.bin",
        "codebooks.bin",
        "tokenizer.model",
        "specials.json",
    ):
        if not (container / required).is_file():
            raise RuntimeError(f"invalid WARP container: missing {required}")


def install_model(
    model_id: str,
    *,
    warp_root: Path,
    staging_dir: Path,
    models_dir: Path,
    jobs: int = 3,
    reclaim_source: bool = False,
    preflight: Callable[..., None] = preflight_disk,
    which: Callable[[str], str | None] = shutil.which,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    validate: Callable[[Path, str], None] = validate_container,
) -> Path:
    """Convert one pinned WARP catalog model and return its absolute container path."""
    if jobs < 1:
        raise RuntimeError("WARP jobs must be a positive integer")
    executables = require_tools(("bash", "uv", "curl"), which=which)
    plan = build_plan(
        model_id,
        staging_dir=staging_dir,
        models_dir=models_dir,
    )
    warp_root = Path(warp_root).expanduser().absolute()
    fetch_script = warp_root / "tools" / "fetch_weights.sh"
    pipeline_script = warp_root / "tools" / "pipeline.sh"
    for script in (fetch_script, pipeline_script):
        if not script.is_file():
            raise RuntimeError(f"WARP runtime is missing required tool: {script}")

    reclaimed = _reclaimed_source_is_complete(plan.source)
    preflight(
        plan.source,
        plan.output,
        source_bytes=plan.source_bytes,
        output_bytes=plan.output_workspace_bytes,
    )
    info = plan.info
    common = {
        "MODEL": str(info["warp_profile"]),
        "REPO": str(info["hf_repo"]),
        "REVISION": str(info["hf_revision"]),
        "SRC": str(plan.source),
        "DEST": str(plan.source),
        "OUT": str(plan.output),
        "JOBS": str(jobs),
        "RECLAIM": "on" if reclaim_source else "off",
        "RUN_DIR": str(plan.run_dir),
        "MIN_FREE_GB": str(info["output_workspace_gib"]),
    }

    with _pipeline_environment() as (environment, hf_token):
        environment.update(common)
        plan.source.mkdir(parents=True, exist_ok=True)
        plan.output.parent.mkdir(parents=True, exist_ok=True)
        if not reclaimed:
            _run_stage(
                [executables["bash"], str(fetch_script), "--dry-run"],
                cwd=warp_root,
                env=environment,
                stage="fetch preflight",
                source=plan.source,
                output=plan.output,
                run_dir=plan.run_dir,
                run=run,
                hf_token=hf_token,
            )

        _run_stage(
            [executables["bash"], str(pipeline_script)],
            cwd=warp_root,
            env=environment,
            stage="pipeline",
            source=plan.source,
            output=plan.output,
            run_dir=plan.run_dir,
            run=run,
            hf_token=hf_token,
        )
    try:
        validate(plan.output, model_id)
    except RuntimeError as exc:
        raise RuntimeError(
            f"{exc}; pipeline output validation failed; source {plan.source}; "
            f"output {plan.output}; run directory {plan.run_dir}. Partial data "
            "was preserved; rerun the same install command to resume."
        ) from exc
    return plan.output
