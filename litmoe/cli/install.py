"""litmoe install - one-command engine + model installation.

Installs an inference engine (llama.cpp release binaries or a source build,
ktransformers via PyPI wheels or the upstream install.sh, or WARP from pinned
source) and downloads model weights, then writes the model entry into models.yaml.

The model catalog lives in litmoe.models (single source of truth).
"""
from __future__ import annotations

import contextlib
import os
import platform
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import click
import yaml

from litmoe.cli import warp_models as _warp_models

from litmoe.config import default_config_path, expand_path
from litmoe.models import (
    CLAUDE_ALIASES,
    DEFAULT_MODEL,
    GGUF,
    KNOWN_MODELS,
    SAFETENSORS,
    WASTE,
    TIER_LABELS,
    fit_context,
    largest_quant_that_fits,
    lookup,
    quant_size_gb,
    ram_needed_gb,
    recommended_for_ram,
)
from litmoe.platform_utils import (
    get_total_memory_bytes,
    has_avx512,
    is_macos,
    nvidia_gpus,
)

LLAMA_RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases"
WARP_REPO = "https://github.com/sqliteai/warp.git"
WARP_COMMIT = "09fcff352ca55223b08ee222d15054b90546c6a9"
LONG_QUIET_SECONDS = 60

# Curated "stable" pointer maintained by llama.cpp CI: the latest release
# (tagged vX.Y.Z) carries only this file; binaries live in the bNNNNN prereleases.
LLAMA_NIGHTLY_POINTER = "nightly-tag.txt"

# llama.cpp release asset name fragments per (system, machine, variant).
# Verified against release b11005 (2026-09-16).
LLAMACPP_ASSETS: dict[tuple[str, str], dict[str, str]] = {
    ("linux", "x86_64"): {
        "cpu": "bin-ubuntu-x64",
        "cuda": "bin-ubuntu-cuda-12.8-x64",
        "cuda13": "bin-ubuntu-cuda-13.3-x64",
        "vulkan": "bin-ubuntu-vulkan-x64",
        "rocm": "bin-ubuntu-rocm-10.0-x64",
    },
    ("linux", "aarch64"): {
        "cpu": "bin-ubuntu-arm64",
        "cuda": "bin-ubuntu-cuda-13.3-arm64",
        "cuda13": "bin-ubuntu-cuda-13.3-arm64",
        "vulkan": "bin-ubuntu-vulkan-arm64",
    },
    ("darwin", "arm64"): {"cpu": "bin-macos-arm64"},   # Metal is built in
    ("darwin", "x86_64"): {"cpu": "bin-macos-x64"},
}
LLAMACPP_VARIANTS = ("auto", "cpu", "cuda", "cuda13", "vulkan", "rocm")
# Ubuntu release binaries link against GLIBC_2.34 symbols; older glibc needs a source build.
LLAMACPP_PREBUILT_MIN_GLIBC = (2, 34)


def _default_models_dir() -> Path:
    raw = os.environ.get("LITMOE_MODELS_DIR", str(Path.home() / ".litmoe" / "models"))
    return Path(os.path.expanduser(os.path.expandvars(raw)))


def _default_prefix() -> Path:
    raw = os.environ.get("LITMOE_PREFIX", str(Path.home() / ".local"))
    return Path(os.path.expanduser(os.path.expandvars(raw)))


def _normalize_machine(machine: str) -> str:
    m = machine.lower()
    return {"amd64": "x86_64", "arm64": "arm64" if platform.system() == "Darwin" else "aarch64",
            "aarch64": "aarch64"}.get(m, m)


def glibc_version() -> tuple[int, int] | None:
    """(major, minor) of the running glibc, or None (macOS, musl, unknown)."""
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        getver = libc.gnu_get_libc_version
        getver.restype = ctypes.c_char_p
        major, minor = getver().decode().split(".")[:2]
        return int(major), int(minor)
    except (OSError, AttributeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# llama.cpp release resolution
# ---------------------------------------------------------------------------

def _asset_named(release: dict, fragment: str) -> dict | None:
    """The 'llama-<tag>-<fragment>.tar.gz' asset of a release, or None."""
    pattern = re.compile(rf"^llama-b\d+-{re.escape(fragment)}\.tar\.gz$")
    for a in release.get("assets", []):
        if pattern.match(a["name"]):
            return a
    return None


def _cudart_asset(release: dict, fragment: str) -> dict | None:
    pattern = re.compile(rf"^cudart-llama-b\d+-{re.escape(fragment)}\.tar\.gz$")
    for a in release.get("assets", []):
        if pattern.match(a["name"]):
            return a
    return None


def resolve_llamacpp_release(fragment: str, client, tag: str | None = None) -> dict:
    """Find a llama.cpp release that carries the wanted binary asset.

    Order: an explicit tag (LITMOE_LLAMACPP_TAG or --llamacpp-tag); the curated
    nightly-tag.txt pointer attached to the 'latest' release; then the newest
    prerelease that actually has the asset (the very newest tag is often still
    uploading its 30+ assets).
    """
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "litmoe"}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"

    def fetch(url: str) -> Any:
        r = client.get(url, headers=headers)
        r.raise_for_status()
        return r.json()

    if tag:
        rel = fetch(f"{LLAMA_RELEASES_API}/tags/{tag}")
        if not _asset_named(rel, fragment):
            raise RuntimeError(f"release {tag} has no asset matching '{fragment}'")
        return rel

    latest = fetch(f"{LLAMA_RELEASES_API}/latest")
    if _asset_named(latest, fragment):
        return latest
    pointer = next((a for a in latest.get("assets", []) if a["name"] == LLAMA_NIGHTLY_POINTER), None)
    if pointer:
        try:
            r = client.get(pointer["browser_download_url"], headers={"User-Agent": "litmoe"})
            r.raise_for_status()
            nightly_tag = r.text.strip()
            if re.fullmatch(r"b\d+", nightly_tag):
                rel = fetch(f"{LLAMA_RELEASES_API}/tags/{nightly_tag}")
                if _asset_named(rel, fragment):
                    return rel
        except Exception as e:  # network / 404 — fall through to the scan
            click.echo(f"  nightly pointer unusable ({e}); scanning recent releases...")

    for rel in fetch(f"{LLAMA_RELEASES_API}?per_page=30"):
        if _asset_named(rel, fragment):
            return rel
    raise RuntimeError(
        f"No llama.cpp release in the last 30 has an asset matching '{fragment}'. "
        f"Set LITMOE_LLAMACPP_TAG to a known tag (see {LLAMA_RELEASES_API.replace('api.', '').replace('/repos', '')})."
    )


def pick_llamacpp_variant(variant: str) -> str:
    """Resolve 'auto' to cpu/cuda based on the machine."""
    if variant != "auto":
        return variant
    if not is_macos() and nvidia_gpus():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Engine installers
# ---------------------------------------------------------------------------
def _latest_log_fragment(log: Path, *, limit: int = 4096) -> str:
    """Return the last non-empty line of a live stage log, if readable."""
    try:
        with open(log, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - limit), os.SEEK_SET)
            chunk = handle.read().decode("utf-8", "replace")
    except OSError:
        return ""
    for line in reversed(chunk.replace("\r", "\n").splitlines()):
        candidate = line.strip()
        if candidate:
            return candidate[:120]
    return ""


def _latest_stage_fragment(logs: list[Path]) -> str:
    """Pick the newest live fragment across stage logs (fetch or convert)."""
    best_mtime = -1.0
    best = ""
    for log in logs:
        try:
            mtime = log.stat().st_mtime
        except OSError:
            continue
        fragment = _latest_log_fragment(log)
        if fragment and mtime > best_mtime:
            best_mtime, best = mtime, fragment
    return best


def _terminate_stage_tree(process: subprocess.Popen) -> None:
    """Stop a stage and every process it spawned, then reap the leader."""
    try:
        pgid = os.getpgid(process.pid)
    except (ProcessLookupError, PermissionError, OSError):
        pgid = None
    if pgid is not None and pgid != os.getpgid(0):
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pgid, sig)
            except (ProcessLookupError, PermissionError, OSError):
                break
            try:
                process.wait(timeout=5)
                return
            except subprocess.TimeoutExpired:
                continue
    else:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def _run_warp_stage(args: list[str], **kwargs):
    """Run a WARP model stage in its own session; litmoe owns its lifetime.

    The stage runs detached from litmoe's process group so litmoe decides
    when it stops. On Ctrl-C or terminal hangup the whole tree — bash,
    xargs, workers, curls — is terminated before litmoe exits, so nothing
    is left running to collide with a rerun. Heartbeats with the newest
    live log line are printed while the stage runs.
    """
    kwargs = dict(kwargs)
    environment = dict(kwargs.get("env") or {})
    kwargs["env"] = environment
    script = Path(str(args[1])).name if len(args) > 1 else ""
    pipeline = script == "pipeline.sh"
    if pipeline:
        download_log = Path(environment["SRC"]) / "download.log"
        pipeline_log = Path(environment["RUN_DIR"]) / "pipeline.log"
        marquee = (
            "downloading and converting WARP model; "
            f"logs: {download_log}, {pipeline_log}"
        )
    else:
        download_log = Path(environment.get("DEST") or "staging") / "download.log"
        pipeline_log = download_log
        marquee = f"downloading model weights; log: {download_log}"

    click.echo(f"  {marquee}")

    kwargs["stdout"] = subprocess.PIPE
    kwargs["stderr"] = subprocess.PIPE
    kwargs["text"] = True
    kwargs["start_new_session"] = True

    stop = threading.Event()
    started = time.monotonic()
    process_holder: list[subprocess.Popen] = []

    def forward_signal(_signum, _frame):
        stop.set()
        if process_holder:
            _terminate_stage_tree(process_holder[0])

    forwarded = [
        sig for sig in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGHUP", None))
        if sig is not None
    ]
    previous = {}
    for sig in forwarded:
        try:
            previous[sig] = signal.signal(sig, forward_signal)
        except (ValueError, OSError):
            pass

    captured: dict[str, str] = {}

    def capture(proc: subprocess.Popen) -> None:
        try:
            out, err = proc.communicate()
        except BaseException:
            return
        captured["stdout"] = out or ""
        captured["stderr"] = err or ""

    reader: threading.Thread | None = None
    try:
        process = subprocess.Popen(args, **kwargs)
        process_holder.append(process)
        reader = threading.Thread(target=capture, args=(process,), daemon=True)
        reader.start()

        logs = [download_log, pipeline_log] if pipeline else [download_log]
        interval = max(float(LONG_QUIET_SECONDS), 0.01)
        while not stop.is_set():
            if process.poll() is not None and not reader.is_alive():
                break
            if stop.wait(interval):
                break
            elapsed = time.monotonic() - started
            fragment = _latest_stage_fragment(logs)
            detail = f" | {fragment}" if fragment else ""
            click.echo(f"  {marquee} ({elapsed:.0f}s elapsed){detail}")
    finally:
        if process_holder and process_holder[0].poll() is None:
            _terminate_stage_tree(process_holder[0])
        if reader is not None:
            for _ in range(6):
                if not reader.is_alive():
                    break
                reader.join(5)
        for sig, handler in previous.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass
    if stop.is_set():
        elapsed = time.monotonic() - started
        click.echo(f"  {marquee} (interrupted after {elapsed:.0f}s)")
        raise KeyboardInterrupt
    result = subprocess.CompletedProcess(
        args, process.returncode, captured.get("stdout", ""), captured.get("stderr", "")
    )
    if result.stdout and not pipeline:
        click.echo(result.stdout, nl=not result.stdout.endswith("\n"))
    return result


def _run_warp_command(
    args: list[str], *, cwd: Path | None, label: str, timeout: int
) -> None:
    environment = os.environ.copy()
    environment.pop("HF_TOKEN", None)
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"{label} failed: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        suffix = f": {detail[:500]}" if detail else ""
        raise RuntimeError(f"{label} failed (exit {result.returncode}){suffix}")


def _warp_library_name() -> str:
    if sys.platform == "darwin":
        return "libwaste.dylib"
    if sys.platform == "win32":
        return "libwaste.dll"
    return "libwaste.so"

def _installed_warp_root(prefix: Path) -> Path | None:
    """The installed WARP runtime root when it is complete and pinned."""
    root = prefix / "lib" / "warp"
    required = (
        root / "serve" / "__main__.py",
        root / _warp_library_name(),
        root / "tools" / "fetch_weights.sh",
        root / "tools" / "pipeline.sh",
    )
    if not all(path.is_file() for path in required):
        return None
    try:
        pinned = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if pinned.returncode != 0:
        return None
    return root if pinned.stdout.strip() == WARP_COMMIT else None


def install_warp(prefix: Path, ref: str = WARP_COMMIT) -> Path:
    """Build and install an exact WARP source revision, or reuse the pinned one."""
    existing = _installed_warp_root(prefix)
    if existing is not None:
        click.echo(f"  WARP already installed: {existing}")
        return existing
    for tool in ("git", "make"):
        if not shutil.which(tool):
            raise RuntimeError(
                f"'{tool}' is required to build WARP from source "
                f"(apt install {tool} / brew install {tool})"
            )

    lib_dir = prefix / "lib"
    lib_dir.mkdir(parents=True, exist_ok=True)
    dest_dir = lib_dir / "warp"

    with contextlib.ExitStack() as stage_cleanup:
        stage_dir = Path(tempfile.mkdtemp(prefix=".warp-install-", dir=lib_dir))
        stage_cleanup.callback(shutil.rmtree, stage_dir, ignore_errors=True)
        checkout = stage_dir / "checkout"

        click.echo(f"  Cloning WARP source ({ref})...")
        _run_warp_command(
            ["git", "clone", "--no-checkout", WARP_REPO, str(checkout)],
            cwd=None,
            label="git clone",
            timeout=300,
        )
        _run_warp_command(
            ["git", "fetch", "--depth", "1", "origin", ref],
            cwd=checkout,
            label=f"git fetch {ref}",
            timeout=300,
        )
        # Checkout the requested ref, never FETCH_HEAD: a fetch-by-SHA can
        # leave FETCH_HEAD naming a different branch line, and the detached
        # checkout then silently builds the wrong revision.
        _run_warp_command(
            ["git", "checkout", "--detach", ref],
            cwd=checkout,
            label=f"git checkout {ref}",
            timeout=300,
        )

        click.echo("  Building WARP...")
        _run_warp_command(
            ["make"],
            cwd=checkout,
            label="WARP build",
            timeout=3600,
        )
        click.echo("  Running WARP checks...")
        _run_warp_command(
            ["make", "check"],
            cwd=checkout,
            label="WARP checks",
            timeout=3600,
        )

        cli_name = "waste.exe" if sys.platform == "win32" else "waste"
        required = [
            checkout / cli_name,
            checkout / "serve" / "__main__.py",
            checkout / _warp_library_name(),
        ]
        missing = [str(path.relative_to(checkout)) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(
                "WARP build incomplete; missing required artifact(s): "
                + ", ".join(missing)
            )

        bin_dir = prefix / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        wrapper = bin_dir / cli_name
        if wrapper.exists() and wrapper.is_dir() and not wrapper.is_symlink():
            raise RuntimeError(f"cannot install WARP CLI: {wrapper} is a directory")

        with tempfile.TemporaryDirectory(
            prefix=".warp-launcher-", dir=bin_dir
        ) as launcher_tmpdir:
            staged_launcher = Path(launcher_tmpdir) / "waste-link"
            if sys.platform == "win32":
                shutil.copy2(checkout / cli_name, staged_launcher)
            else:
                staged_launcher.symlink_to((dest_dir / cli_name).absolute())

            backup = stage_dir / "previous"
            launcher_backup = stage_dir / "previous-launcher"
            failed_install = stage_dir / "failed-install"
            failed_launcher = stage_dir / "failed-launcher"
            had_previous = dest_dir.exists() or dest_dir.is_symlink()
            had_launcher = wrapper.exists() or wrapper.is_symlink()
            failure_message = "could not replace WARP installation"
            try:
                if had_previous:
                    os.replace(dest_dir, backup)
                if had_launcher:
                    failure_message = "could not install WARP CLI launcher"
                    os.replace(wrapper, launcher_backup)
                failure_message = "could not replace WARP installation"
                os.replace(checkout, dest_dir)
                failure_message = "could not install WARP CLI launcher"
                os.replace(staged_launcher, wrapper)
            except BaseException as exc:
                rollback_error = None
                try:
                    if launcher_backup.exists() or launcher_backup.is_symlink():
                        if wrapper.exists() or wrapper.is_symlink():
                            os.replace(wrapper, failed_launcher)
                        os.replace(launcher_backup, wrapper)
                    elif not had_launcher and (
                        wrapper.exists() or wrapper.is_symlink()
                    ):
                        os.replace(wrapper, failed_launcher)

                    if backup.exists() or backup.is_symlink():
                        if dest_dir.exists() or dest_dir.is_symlink():
                            os.replace(dest_dir, failed_install)
                        os.replace(backup, dest_dir)
                    elif not had_previous and (
                        dest_dir.exists() or dest_dir.is_symlink()
                    ):
                        os.replace(dest_dir, failed_install)
                except BaseException as rollback_exc:
                    rollback_error = rollback_exc

                if rollback_error is not None:
                    stage_cleanup.pop_all()
                if isinstance(exc, OSError):
                    message = f"{failure_message}: {exc}"
                    if rollback_error is not None:
                        message += (
                            f"; rollback also failed: {rollback_error}; "
                            f"recovery files retained at {stage_dir}"
                        )
                    raise RuntimeError(message) from exc
                if rollback_error is not None:
                    raise RuntimeError(
                        f"WARP cutover interrupted; rollback also failed: "
                        f"{rollback_error}; recovery files retained at {stage_dir}"
                    ) from exc
                raise

    click.echo(f"  WARP installed: {dest_dir}")
    return dest_dir

def _filesystem_device(path: Path) -> int:
    """Compatibility hook for WARP disk-preflight filesystem detection."""
    return _warp_models.filesystem_device(path)


def _preflight_warp_disk(
    source_dir: Path,
    output_dir: Path,
    *,
    source_bytes: int,
    output_bytes: int,
) -> None:
    """Check resumable WARP source/output requirements before conversion."""
    _warp_models.preflight_disk(
        source_dir,
        output_dir,
        source_bytes=source_bytes,
        output_bytes=output_bytes,
        device=_filesystem_device,
        disk_usage=shutil.disk_usage,
    )


def _prepare_warp_model_plan(
    model_id: str,
    *,
    staging_dir: Path,
    models_dir: Path,
) -> _warp_models.WarpInstallPlan:
    """Validate prerequisites and storage before confirmation or installation."""
    _warp_models.require_tools(
        ("git", "make", "bash", "uv", "curl"),
        which=shutil.which,
    )
    plan = _warp_models.build_plan(
        model_id,
        staging_dir=staging_dir,
        models_dir=models_dir,
    )
    _preflight_warp_disk(
        plan.source,
        plan.output,
        source_bytes=plan.source_bytes,
        output_bytes=plan.output_workspace_bytes,
    )
    return plan


def _validate_warp_container(container: Path, model_id: str) -> None:
    """Validate a completed local ``.waste`` container."""
    _warp_models.validate_container(container, model_id)


def install_warp_model(
    model_id: str,
    *,
    warp_root: Path,
    staging_dir: Path,
    models_dir: Path,
    jobs: int = 3,
    reclaim_source: bool = False,
) -> Path:
    """Install a pinned catalog model through the upstream WARP pipeline."""
    path = _warp_models.install_model(
        model_id,
        warp_root=warp_root,
        staging_dir=staging_dir,
        models_dir=models_dir,
        jobs=jobs,
        reclaim_source=reclaim_source,
        preflight=_preflight_warp_disk,
        which=shutil.which,
        run=_run_warp_stage,
        validate=_validate_warp_container,
    )
    click.echo(f"  WARP model installation complete: {path}")
    return path


def install_llamacpp(prefix: Path, variant: str = "auto", tag: str | None = None) -> Path:
    """Install llama.cpp.

    macOS: release binaries (Metal built in), source build as fallback.
    Linux: release binaries when glibc >= 2.34 (they link GLIBC_2.34 symbols),
    otherwise a source build.
    """
    variant = pick_llamacpp_variant(variant)
    if is_macos():
        try:
            return _install_llamacpp_prebuilt(prefix, variant, tag)
        except RuntimeError as e:
            click.echo(f"  Release binary unusable ({e}); building from source instead...")
            return _install_llamacpp_source(prefix)

    ver = glibc_version()
    if ver is not None and ver >= LLAMACPP_PREBUILT_MIN_GLIBC:
        return _install_llamacpp_prebuilt(prefix, variant, tag)
    click.echo(f"  Release binaries need glibc {'.'.join(map(str, LLAMACPP_PREBUILT_MIN_GLIBC))}+, "
               f"this machine has {'.'.join(map(str, ver)) if ver else 'unknown'}.")
    click.echo("  Building from source instead...")
    return _install_llamacpp_source(prefix)


def _download_to(client, url: str, dest: Path) -> None:
    with open(dest, "wb") as f:
        with client.stream("GET", url, headers={"User-Agent": "litmoe"}) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_bytes(chunk_size=1 << 20):
                f.write(chunk)


def _extract_tar(archive: Path, dest_dir: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        try:
            tar.extractall(dest_dir, filter="data")  # Python 3.12+ safe extraction
        except TypeError:
            tar.extractall(dest_dir)


def _install_llamacpp_prebuilt(prefix: Path, variant: str = "cpu", tag: str | None = None) -> Path:
    """Download llama.cpp release binaries from GitHub."""
    system = platform.system().lower()
    machine = _normalize_machine(platform.machine())
    variants = LLAMACPP_ASSETS.get((system, machine))
    if not variants:
        raise RuntimeError(f"No llama.cpp release binaries for {system}/{machine}; build from source.")
    fragment = variants.get(variant)
    if not fragment:
        raise RuntimeError(f"variant '{variant}' is not published for {system}/{machine}; "
                           f"available: {', '.join(variants)}")

    import httpx
    click.echo(f"  Resolving llama.cpp release with '{fragment}'...")
    with httpx.Client(follow_redirects=True, timeout=60.0) as client:
        release = resolve_llamacpp_release(fragment, client, tag or os.environ.get("LITMOE_LLAMACPP_TAG"))
        rel_tag = release["tag_name"]
        asset = _asset_named(release, fragment)
        assert asset is not None
        cudart = _cudart_asset(release, fragment) if "cuda" in variant else None

        dest_dir = prefix / "lib" / "llama.cpp" / "prebuilt" / f"llama-{rel_tag}"
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        for a in [asset] + ([cudart] if cudart else []):
            click.echo(f"  Downloading {a['name']} ({a['size'] / 1e6:.0f} MB)...")
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                _download_to(client, a["browser_download_url"], tmp_path)
                click.echo(f"  Extracting to {dest_dir}...")
                _extract_tar(tmp_path, dest_dir)
            finally:
                tmp_path.unlink(missing_ok=True)

    server = next((c for c in dest_dir.rglob("llama-server") if c.is_file()), None)
    if not server:
        raise RuntimeError(f"llama-server not found in extracted archive at {dest_dir}")
    # Archives nest binaries under build/bin/; move everything next to llama-server
    # so lib discovery (LD_LIBRARY_PATH = binary dir) is trivial.
    if server.parent != dest_dir:
        for item in list(server.parent.iterdir()):
            shutil.move(str(item), str(dest_dir / item.name))
        for sub in [p for p in dest_dir.iterdir() if p.is_dir() and not any(p.iterdir())]:
            sub.rmdir()
        server = dest_dir / "llama-server"

    if is_macos():
        # Strip com.apple.provenance/quarantine (blocks execve from Python) by
        # copying the binary and dylibs over themselves.
        for f in [server] + [d for d in server.parent.glob("*.dylib*") if d.is_file()]:
            clean = f.parent / f".{f.name}.clean"
            shutil.copy2(str(f), str(clean))
            shutil.move(str(clean), str(f))
    server.chmod(server.stat().st_mode | stat.S_IEXEC)

    bin_dir = prefix / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    launcher = bin_dir / "llama-server"
    if launcher.exists() or launcher.is_symlink():
        launcher.unlink()
    if is_macos():
        from litmoe.platform_utils import fix_macos_dylib_paths
        click.echo("  Fixing macOS dylib paths...")
        fix_macos_dylib_paths(server, server.parent)
        _copy_homebrew_openssl(server.parent)
        launcher.write_text(
            f"#!/bin/bash\n"
            f"export DYLD_FALLBACK_LIBRARY_PATH={server.parent}:$DYLD_FALLBACK_LIBRARY_PATH\n"
            f"export DYLD_LIBRARY_PATH={server.parent}:$DYLD_LIBRARY_PATH\n"
            f"exec {server} \"$@\"\n"
        )
        launcher.chmod(0o755)
    else:
        # A wrapper (not a symlink) so `llama-server` on PATH finds its .so files.
        launcher.write_text(
            f"#!/bin/bash\n"
            f"export LD_LIBRARY_PATH={server.parent}:$LD_LIBRARY_PATH\n"
            f"exec {server} \"$@\"\n"
        )
        launcher.chmod(0o755)

    # Only one prebuilt release is kept; remove older ones.
    for old in (prefix / "lib" / "llama.cpp" / "prebuilt").iterdir():
        if old.is_dir() and old != dest_dir:
            shutil.rmtree(old, ignore_errors=True)

    click.echo(f"  llama-server {rel_tag} ({variant}) installed: {server}")
    return dest_dir


def _copy_homebrew_openssl(dest_dir: Path) -> None:
    """macOS: copy OpenSSL dylibs next to the binary if it links against Homebrew's."""
    import glob as _glob
    for ssl_lib in ["libssl.3.dylib", "libcrypto.3.dylib", "libssl.35.dylib", "libcrypto.35.dylib"]:
        if (dest_dir / ssl_lib).exists():
            continue
        for search in ["/opt/homebrew/lib", "/usr/local/lib", "/opt/homebrew/opt/openssl@3/lib"]:
            found = _glob.glob(f"{search}/{ssl_lib}")
            if found:
                shutil.copy2(found[0], str(dest_dir / ssl_lib))
                break


def _install_llamacpp_source(prefix: Path) -> Path:
    """Build llama.cpp from source (glibc < 2.34, or no usable release binary).

    Linux: CPU build with OpenBLAS (needs libopenblas-dev) and -march=native.
    macOS: Metal + Accelerate (the default BLAS on macOS), rpath baked in and
    all @rpath references rewritten to absolute paths post-build.
    """
    from litmoe.platform_utils import fix_macos_dylib_paths

    for tool in ("git", "cmake"):
        if not shutil.which(tool):
            raise RuntimeError(f"'{tool}' is required to build llama.cpp from source "
                               f"(apt install {tool} / brew install {tool})")

    with tempfile.TemporaryDirectory() as tmpdir:
        click.echo(f"  Cloning llama.cpp to {tmpdir}...")
        clone = subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp.git", tmpdir],
            capture_output=True, text=True, timeout=300,
        )
        if clone.returncode != 0:
            raise RuntimeError(f"git clone failed: {clone.stderr.strip()[:300]}")

        dest_dir = (prefix / "lib" / "llama.cpp" / "local").resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)

        cmake_args = ["cmake", "-B", "build", "-DGGML_NATIVE=ON", "-DCMAKE_BUILD_TYPE=Release",
                      "-DLLAMA_CURL=ON"]
        if is_macos():
            cmake_args += [
                "-DGGML_METAL=ON",
                f"-DCMAKE_INSTALL_RPATH={dest_dir}",
                "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
                f"-DCMAKE_BUILD_RPATH={dest_dir}",
            ]
        else:
            cmake_args += ["-DGGML_CUDA=OFF", "-DGGML_BLAS=ON", "-DGGML_BLAS_VENDOR=OpenBLAS"]

        click.echo("  Configuring (cmake)...")
        cmake = subprocess.run(cmake_args, cwd=tmpdir, capture_output=True, text=True, timeout=300)
        if cmake.returncode != 0:
            err = cmake.stderr.strip()
            hint = ""
            if "BLAS" in err or "OpenBLAS" in err:
                hint = " (install libopenblas-dev / openblas)"
            if "CURL" in err.upper():
                hint = " (install libcurl4-openssl-dev / curl)"
            raise RuntimeError(f"cmake configure failed{hint}: {err[:400]}")

        click.echo("  Building llama-server (this takes several minutes)...")
        build = subprocess.run(
            ["cmake", "--build", "build", "--config", "Release", "-j", "--target", "llama-server"],
            cwd=tmpdir, timeout=3600,
        )
        if build.returncode != 0:
            raise RuntimeError("Build failed. See output above.")

        build_bin = Path(tmpdir) / "build" / "bin"
        server_bin = build_bin / "llama-server"
        if not server_bin.exists():
            raise RuntimeError(f"llama-server not found at {server_bin}")

        shutil.copy2(str(server_bin), str(dest_dir / "llama-server"))
        for so in list(build_bin.rglob("lib*.so*")) + list(build_bin.rglob("lib*.dylib*")):
            target = dest_dir / so.name
            if not target.exists() or so.parent == build_bin:
                shutil.copy2(str(so), str(target))

        if is_macos():
            _copy_homebrew_openssl(dest_dir)
            click.echo("  Fixing dylib paths (rewriting @rpath to absolute)...")
            if not fix_macos_dylib_paths(dest_dir / "llama-server", dest_dir):
                click.echo("  WARNING: could not fix dylib paths, will rely on env vars.")

        bin_dir = prefix / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        wrapper = bin_dir / "llama-server"
        # Never write through an existing symlink — that overwrites its target
        # (a previous install did exactly this to the release binary).
        if wrapper.exists() or wrapper.is_symlink():
            wrapper.unlink()
        if is_macos():
            wrapper.write_text(
                f"#!/bin/bash\n"
                f"export DYLD_FALLBACK_LIBRARY_PATH={dest_dir}:$DYLD_FALLBACK_LIBRARY_PATH\n"
                f"export DYLD_LIBRARY_PATH={dest_dir}:$DYLD_LIBRARY_PATH\n"
                f"exec {dest_dir}/llama-server \"$@\"\n"
            )
        else:
            wrapper.write_text(
                f"#!/bin/bash\n"
                f"export LD_LIBRARY_PATH={dest_dir}:$LD_LIBRARY_PATH\n"
                f"exec {dest_dir}/llama-server \"$@\"\n"
            )
        wrapper.chmod(0o755)

        click.echo(f"  llama-server built and installed: {dest_dir}/llama-server")
        return dest_dir


def ktransformers_wheel_supported() -> tuple[bool, str]:
    """Can `pip install ktransformers[sglang]` work here? (ok, reason)."""
    if platform.system() != "Linux" or _normalize_machine(platform.machine()) != "x86_64":
        return False, "PyPI wheels exist only for Linux x86-64"
    if sys.version_info[:2] not in ((3, 11), (3, 12)):
        return False, f"PyPI wheels exist only for Python 3.11/3.12 (running {sys.version_info.major}.{sys.version_info.minor})"
    ver = glibc_version()
    if ver is None or ver < (2, 35):
        return False, f"PyPI wheels are manylinux_2_35 (glibc 2.35+), this machine has {'.'.join(map(str, ver)) if ver else 'unknown'}"
    return True, "ok"


def install_ktransformers() -> None:
    """Install ktransformers (kt-kernel + sglang-kt) for serving.

    Serving uses `python -m sglang.launch_server --kt-method ...`; both
    kt-kernel and sglang-kt are required. Two paths:

    1. PyPI: `pip install "ktransformers[sglang]"` — Linux x86-64, Python
       3.11/3.12, glibc >= 2.35 (manylinux_2_35 wheels).
    2. Source: `git clone --recursive` + upstream `install.sh`, which builds
       sglang from third_party/ and kt-kernel from source (needs a C++
       toolchain, CMake, and the CUDA toolkit for the GPU parts).

    Not available on macOS (CUDA/triton). Serving requires an NVIDIA GPU.
    """
    if platform.system() == "Darwin":
        click.echo("  ktransformers is not available on macOS (CUDA + triton required).")
        click.echo("  Use llama.cpp instead: litmoe install --engine llamacpp (Metal backend).")
        sys.exit(1)
    if not nvidia_gpus():
        click.echo("  WARNING: no NVIDIA GPU detected. sglang-kt serving needs one (SM 8.0+); "
                   "installing anyway.")
    if not has_avx512():
        click.echo("  NOTE: CPU has no AVX-512 — only the LLAMAFILE (GGUF) expert backend will work; "
                   "FP8/BF16/RAWINT4/MXFP* kt_method values need AVX-512.")

    pip_env = os.environ.copy()
    # Some machines carry a pip.conf with an unreachable extra index (NGC) that
    # adds minutes of retries per package; force plain PyPI unless overridden.
    pip_env.setdefault("PIP_INDEX_URL", "https://pypi.org/simple/")
    pip_env["PIP_EXTRA_INDEX_URL"] = pip_env.get("LITMOE_PIP_EXTRA_INDEX_URL", "")

    ok, reason = ktransformers_wheel_supported()
    if ok:
        click.echo("  Installing ktransformers[sglang] from PyPI (kt-kernel + sglang-kt)...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", "ktransformers[sglang]"],
                                timeout=3600, env=pip_env)
        if result.returncode == 0:
            _verify_ktransformers()
            return
        click.echo("  PyPI install failed; falling back to a source build.", err=True)
    else:
        click.echo(f"  PyPI wheels not usable here ({reason}); building from source.")

    for tool in ("git", "cmake"):
        if not shutil.which(tool):
            click.echo(f"  '{tool}' is required for a source build.", err=True)
            sys.exit(1)

    src_dir = _default_prefix() / "src" / "ktransformers"
    src_dir.parent.mkdir(parents=True, exist_ok=True)
    if src_dir.exists():
        shutil.rmtree(src_dir)
    click.echo(f"  Cloning ktransformers (with submodules) to {src_dir}...")
    clone = subprocess.run(
        ["git", "clone", "--recursive", "--depth", "1", "--shallow-submodules",
         "https://github.com/kvcache-ai/ktransformers.git", str(src_dir)],
        timeout=1800,
    )
    if clone.returncode != 0:
        click.echo("  git clone failed. Manual: https://github.com/kvcache-ai/ktransformers", err=True)
        sys.exit(1)

    click.echo("  Running upstream install.sh (deps + sglang + kt-kernel; this can take 30+ minutes)...")
    result = subprocess.run(["bash", "install.sh"], cwd=str(src_dir), env=pip_env, timeout=7200)
    if result.returncode != 0:
        click.echo("  install.sh failed. See output above and "
                   "https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md", err=True)
        sys.exit(1)
    _verify_ktransformers()


def _verify_ktransformers() -> None:
    from litmoe.engines.ktransformers import missing_components
    missing = missing_components()
    if missing:
        click.echo(f"  WARNING: ktransformers install incomplete, missing: {', '.join(missing)}", err=True)
    else:
        click.echo("  ktransformers installed (kt_kernel + sglang importable).")


# ---------------------------------------------------------------------------
# Model downloader
# ---------------------------------------------------------------------------

_SKIP_BASENAMES = ("mmproj", "imatrix", "mtp")


def select_gguf_files(repo_files: list[str], quant: str) -> list[str]:
    """Pick exactly the GGUF file(s) for one quant from a repo file listing.

    Handles both layouts used on HuggingFace:
      subdir: 'UD-Q4_K_XL/Model-UD-Q4_K_XL-00001-of-00003.gguf'
      root:   'Model-Q4_K_M.gguf', 'Model.Q4_K_M.gguf', 'Model-UD-Q4_K_XL.gguf'
    A plain quant (e.g. Q4_K_M) never matches its UD- variant (UD-Q4_K_M) and
    vice versa. mmproj/imatrix/MTP files are excluded.
    """
    q = re.escape(quant)
    ud_guard = "" if quant.upper().startswith("UD-") else r"(?<!UD-)"
    pat = re.compile(rf"(?:^|[/._-]){ud_guard}{q}(?:-\d{{5}}-of-\d{{5}})?\.gguf$", re.IGNORECASE)
    out = []
    for f in repo_files:
        base = f.rsplit("/", 1)[-1].lower()
        if base.startswith(_SKIP_BASENAMES):
            continue
        if not pat.search(f):
            continue
        # Subdirectory layouts name the directory after the quant; anything in a
        # differently named directory (MTP/, dspark/, another quant) is not ours.
        parts = f.split("/")
        if len(parts) > 1 and parts[0].upper() != quant.upper():
            continue
        out.append(f)
    return sorted(out)


def select_mmproj_file(repo_files: list[str]) -> str | None:
    """Prefer mmproj-F16.gguf, then mmproj-BF16.gguf, at the repo root."""
    for name in ("mmproj-F16.gguf", "mmproj-BF16.gguf", "mmproj-f16.gguf"):
        if name in repo_files:
            return name
    return next((f for f in repo_files if f.lower().startswith("mmproj") and "/" not in f), None)


def _hf_api():
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        click.echo("  Installing huggingface_hub...", err=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub[hf_transfer]"], check=True)
        from huggingface_hub import HfApi, snapshot_download
    return HfApi(), snapshot_download


def download_model(model_name: str, quant: str | None, models_dir: Path,
                   with_mmproj: bool = True) -> tuple[Path, Path | None]:
    """Download model weights from HuggingFace.

    GGUF: returns (path to the first shard, mmproj path or None).
    safetensors (ktransformers): returns (model directory, None).
    """
    info = lookup(model_name)
    if not info:
        raise click.BadParameter(f"unknown model {model_name}")
    api, snapshot_download = _hf_api()
    repo = info["hf_repo"]

    if info["format"] == SAFETENSORS:
        dest = models_dir / model_name
        dest.mkdir(parents=True, exist_ok=True)
        click.echo(f"  Downloading {repo} (safetensors, ~{info['size_gb']} GB) -> {dest}")
        snapshot_download(repo_id=repo, local_dir=str(dest))
        if not any(dest.glob("*.safetensors")):
            raise RuntimeError(f"No .safetensors files in {dest} after download.")
        return dest, None

    quant = quant or info["default_quant"]
    if quant not in info["quants"]:
        raise click.BadParameter(
            f"{model_name} quant must be one of: {', '.join(info['quants'])}")

    click.echo(f"  Listing {repo}...")
    files_info = list(api.list_repo_tree(repo, recursive=True))
    repo_files = [f.path for f in files_info if hasattr(f, "size")]  # RepoFile only, not RepoFolder
    sizes = {f.path: (getattr(f, "size", 0) or 0) for f in files_info if hasattr(f, "size")}

    wanted = select_gguf_files(repo_files, quant)
    if not wanted:
        raise RuntimeError(f"{repo} has no GGUF files for quant '{quant}'. "
                           f"Files present: {', '.join(sorted(repo_files)[:12])}...")
    mmproj = select_mmproj_file(repo_files) if with_mmproj else None
    total_gb = sum(sizes.get(f, 0) for f in wanted + ([mmproj] if mmproj else [])) / 1e9
    click.echo(f"  {len(wanted)} file(s) for {quant}" + (f" + {mmproj}" if mmproj else "")
               + f", {total_gb:.1f} GB total")

    dest = models_dir / model_name / quant
    dest.mkdir(parents=True, exist_ok=True)
    click.echo(f"  Downloading {repo} [{quant}] -> {dest}")
    snapshot_download(repo_id=repo, allow_patterns=wanted + ([mmproj] if mmproj else []),
                      local_dir=str(dest))

    # Files that lived in a quant subdirectory land in dest/<quant>/ — move them
    # up so llama-server gets a flat directory. Never touch HF's .cache dir.
    for f in wanted:
        if "/" in f:
            src = dest / f
            if src.exists():
                shutil.move(str(src), str(dest / src.name))
    for sub in [p for p in dest.iterdir() if p.is_dir() and not p.name.startswith(".")]:
        if not any(sub.iterdir()):
            sub.rmdir()

    gguf_files = sorted(p for p in dest.glob("*.gguf") if not p.name.lower().startswith("mmproj"))
    if not gguf_files:
        raise RuntimeError(f"No .gguf files found in {dest} after download.")
    first = next((p for p in gguf_files if "-00001-of-" in p.name), gguf_files[0])
    mmproj_path = (dest / Path(mmproj).name) if mmproj and (dest / Path(mmproj).name).exists() else None
    click.echo(f"  {len(gguf_files)} GGUF file(s) downloaded ({sum(p.stat().st_size for p in gguf_files) / 1e9:.1f} GB).")
    return first, mmproj_path


# ---------------------------------------------------------------------------
# models.yaml writer
# ---------------------------------------------------------------------------

def add_model_to_config(model_name: str, engine: str, model_path: Path, n_ctx: int,
                        config_path: Path, extra_args: list[str] | None = None,
                        kt_method: str | None = None, aliases: list[str] | None = None) -> None:
    """Insert or replace a model entry in models.yaml (absolute paths)."""
    model_path = expand_path(model_path)

    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {"host": "127.0.0.1", "port": 8090, "api_key": None, "models": []}
    cfg.setdefault("host", "127.0.0.1")
    cfg.setdefault("port", 8090)
    cfg.setdefault("api_key", None)
    models = cfg.setdefault("models", [])

    entry: dict = {"id": model_name, "engine": engine, "model_path": str(model_path), "n_ctx": n_ctx}
    if engine == "llamacpp":
        entry["n_gpu_layers"] = -1
    if kt_method:
        entry["kt_method"] = kt_method
        entry["kt_num_gpu_experts"] = 0
    if extra_args:
        entry["extra_args"] = list(extra_args)
    # Keep aliases the user already had for this id; otherwise give the first
    # model in the file the Claude names so Anthropic-API clients route.
    old = next((m for m in models if m.get("id") == model_name), None)
    others = [m for m in models if m.get("id") != model_name]
    if old and old.get("aliases"):
        entry["aliases"] = old["aliases"]
    elif aliases:
        entry["aliases"] = aliases
    elif not any(m.get("aliases") for m in others):
        entry["aliases"] = list(CLAUDE_ALIASES)

    models[:] = others
    models.append(entry)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    click.echo(f"  models.yaml updated: {model_name} -> {engine} @ {model_path}")


def choose_quant(model_name: str, requested: str | None, budget_gb: float | None) -> str | None:
    """Quant to download: the user's choice, else the best that fits this machine.

    Without --quant, the catalog default is used when it fits the RAM budget;
    otherwise the largest quant that does fit (as `litmoe models` reports).
    If nothing fits, the smallest quant is chosen so the caller can warn and
    let the user decide. Non-GGUF models have no quant and return None.
    """
    info = lookup(model_name)
    if not info or info["format"] != GGUF:
        return None
    if requested:
        if requested not in info["quants"]:
            raise click.BadParameter(
                f"{model_name} quant must be one of: {', '.join(info['quants'])}")
        return requested
    default = info["default_quant"]
    if budget_gb is None:
        return default
    need = ram_needed_gb(model_name, default)
    if need is not None and need <= budget_gb:
        return default
    best = largest_quant_that_fits(model_name, budget_gb)
    if best:
        click.echo(f"  {default} needs ~{need:.0f} GB RAM, over this machine's ~{budget_gb:.0f} GB budget; "
                   f"using {best} instead (override with --quant).")
        return best
    smallest = min(info["quants"], key=lambda q: info["quants"][q])
    click.echo(f"  No {model_name} quant fits this machine's ~{budget_gb:.0f} GB budget; "
               f"smallest is {smallest} ({info['quants'][smallest]:.0f} GB).")
    return smallest


def choose_n_ctx(model_name: str, weights_gb: float | None, requested: int | None) -> int:
    """Context to write: the user's value, else the memory-fitted native context."""
    if requested:
        return requested
    info = lookup(model_name) or {}
    target = info.get("native_ctx", 131072)
    total = get_total_memory_bytes()
    if total is None or weights_gb is None:
        return target
    ctx, note = fit_context(info.get("kv_bytes_per_token", 65_536), weights_gb, total / 1e9, target,
                            macos=is_macos())
    if note:
        click.echo(f"  NOTE: {note}")
    return ctx


def print_model_table(ram_gb: float | None) -> None:
    """`litmoe models`: the catalog grouped by RAM tier, with what fits this machine."""
    budget = 0.0
    if ram_gb:
        budget = ram_gb * (0.75 if is_macos() else 1.0)
        click.echo(f"Detected {ram_gb:.0f} GB RAM" + (f" (Metal can use ~{budget:.0f} GB by default)" if is_macos() else ""))
        click.echo()
    tiers = sorted({
        info["tier"]
        for info in KNOWN_MODELS.values()
        if info["format"] != WASTE
    })
    for tier in tiers:
        click.echo(f"== {TIER_LABELS[tier]} ==")
        for mid, info in KNOWN_MODELS.items():
            if info["format"] == WASTE or info.get("tier") != tier:
                continue
            size = quant_size_gb(mid, None)
            default = info.get("default_quant", info.get("kt_method"))
            engine = "llama.cpp" if info["engine"] == "llamacpp" else "ktransformers (GPU)"
            fit = ""
            if ram_gb and info["format"] == GGUF:
                need = ram_needed_gb(mid) or 0
                if need <= budget:
                    fit = "fits"
                else:
                    q = largest_quant_that_fits(mid, budget)
                    fit = f"fits with --quant {q}" if q else "does not fit"
            click.echo(f"  {mid:32s} {engine:20s} {default:>12s} {size:5.0f} GB  {info['params']}"
                       + (f"  [{fit}]" if fit else ""))
        click.echo()

    warp_models = [(mid, info) for mid, info in KNOWN_MODELS.items() if info["format"] == WASTE]
    if warp_models:
        click.echo("== WARP conversion containers ==")
        for mid, info in warp_models:
            size = quant_size_gb(mid, None)
            unit = (
                "GiB"
                if info["output_size_bytes"] == info["size_gb"] * 1024**3
                else "GB"
            )
            click.echo(
                f"  {mid:32s} {'WARP (.waste)':20s} {info['warp_profile']:>12s} "
                f"{size:5.0f} {unit}  {info['params']}"
            )
        click.echo()
    click.echo("Install: litmoe install --model <name> [--quant <quant>]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _positive_warp_jobs(
    _ctx: click.Context,
    _param: click.Parameter,
    value: int,
) -> int:
    if value < 1:
        raise click.BadParameter("must be a positive integer")
    return value


@click.command("install")
@click.argument("targets", nargs=-1)
@click.option("--model", "model_name", type=click.Choice(sorted(KNOWN_MODELS.keys())),
              default=None, help="Model to download (see `litmoe models`)")
@click.option("--quant", default=None, help="Quantization (default: the model's default_quant if it fits this machine's RAM, else the largest that does)")
@click.option("--engine", type=click.Choice(["llamacpp", "ktransformers", "warp", "both", "none"]),
              default=None, help="Which engine(s) to install (default: the model's engine, else llamacpp)")
@click.option("--llamacpp-variant", type=click.Choice(LLAMACPP_VARIANTS), default="auto",
              help="llama.cpp release binary variant (auto = cuda if an NVIDIA GPU is visible, else cpu)")
@click.option("--llamacpp-tag", default=None, help="Pin a llama.cpp release tag (e.g. b11005)")
@click.option("--models-dir", type=click.Path(), default=None,
              help="Where to store model weights (default: ~/.litmoe/models)")
@click.option("--staging-dir", type=click.Path(), default=None,
              help="WARP source staging root (default: <models-dir>/.staging)")
@click.option("--warp-jobs", type=int, default=3, show_default=True,
              callback=_positive_warp_jobs, help="Positive parallel job count for WARP conversion")
@click.option("--reclaim-source", is_flag=True,
              help="Delete WARP source shards as conversion makes them reclaimable")
@click.option("--prefix", type=click.Path(), default=None,
              help="Install prefix for engine binaries (default: ~/.local)")
@click.option("--n-ctx", default=None, type=int,
              help="Context size written to models.yaml (default: native context, reduced to fit RAM)")
@click.option("--no-mmproj", is_flag=True, help="Skip the vision projector for multimodal models")
@click.option("--config", "-c", type=click.Path(), default=None, help="Path to models.yaml")
@click.option("--yes", is_flag=True, help="Skip confirmation prompts")
@click.pass_context
def install_cmd(
    ctx,
    targets,
    model_name,
    quant,
    engine,
    llamacpp_variant,
    llamacpp_tag,
    models_dir,
    staging_dir,
    warp_jobs,
    reclaim_source,
    prefix,
    n_ctx,
    no_mmproj,
    config,
    yes,
):
    """Install an engine and/or download model weights in one command.

    \b
    Examples:
      litmoe install                                # llama.cpp + recommendations for this RAM
      litmoe install --model gemma-4-26b-a4b        # 17 GB MoE (4B active) — laptop default
      litmoe install --model qwen3.6-35b-a3b        # 22 GB MoE (3B active)
      litmoe install --model gpt-oss-120b           # 63 GB MoE (5B active), 96 GB machines
      litmoe install --model qwen3.5-122b-a10b      # 60 GB MoE (10B active), 96 GB machines
      litmoe install --model minimax-m2.7           # 141 GB MoE, 192 GB workstation
      litmoe install --model kimi-k3                # 594 GB MoE, 768 GB server
      litmoe install --model glm-5.3-flash          # ktransformers (Linux + NVIDIA GPU)
      litmoe install glm-5.3-flash-warp              # 112 GB WARP container
      litmoe install --model deepseek-v4.1-flash-warp # 299 GB WARP container
      litmoe install --engine llamacpp --llamacpp-variant cuda
      litmoe install warp                           # pinned WARP source build
    """
    engine_from_option = (
        ctx.get_parameter_source("engine") is click.core.ParameterSource.COMMANDLINE
    )
    warp_jobs_from_option = (
        ctx.get_parameter_source("warp_jobs") is click.core.ParameterSource.COMMANDLINE
    )
    llamacpp_variant_from_option = (
        ctx.get_parameter_source("llamacpp_variant")
        is click.core.ParameterSource.COMMANDLINE
    )
    llamacpp_tag_from_option = (
        ctx.get_parameter_source("llamacpp_tag")
        is click.core.ParameterSource.COMMANDLINE
    )
    positional_engine = False
    for target in targets:
        if target in ("llamacpp", "ktransformers", "warp", "both"):
            if engine is None:
                engine = target
            positional_engine = True
        elif lookup(target):
            model_name = target if model_name is None else model_name
        else:
            raise click.BadParameter(f"unknown target: {target}")

    info = lookup(model_name) if model_name else None
    is_warp_model = bool(info and info["format"] == WASTE)
    if is_warp_model:
        incompatible = [
            ("--quant", quant is not None),
            ("--no-mmproj", no_mmproj),
            ("--n-ctx", n_ctx is not None),
            ("--engine", engine_from_option or positional_engine),
            ("--llamacpp-variant", llamacpp_variant_from_option),
            ("--llamacpp-tag", llamacpp_tag_from_option),
        ]
        for option, supplied in incompatible:
            if supplied:
                raise click.BadParameter(
                    f"{option} is not supported for WARP catalog models",
                    param_hint=option,
                )
    else:
        warp_only = [
            ("--staging-dir", staging_dir is not None),
            ("--warp-jobs", warp_jobs_from_option),
            ("--reclaim-source", reclaim_source),
        ]
        for option, supplied in warp_only:
            if supplied:
                raise click.BadParameter(
                    f"{option} is only valid for WARP catalog models",
                    param_hint=option,
                )

    if engine is None:
        engine = info["engine"] if info else "llamacpp"

    models_dir = expand_path(models_dir) if models_dir else _default_models_dir()
    staging_dir = expand_path(staging_dir) if staging_dir else models_dir / ".staging"
    prefix = expand_path(prefix) if prefix else _default_prefix()
    config_path = expand_path(config) if config else default_config_path()

    if is_warp_model:
        assert info is not None and model_name is not None
        try:
            plan = _prepare_warp_model_plan(
                model_name,
                staging_dir=staging_dir,
                models_dir=models_dir,
            )
        except Exception as exc:
            raise click.ClickException(str(exc)) from exc
        live = _warp_models._snapshot_live_processes(
            (plan.source, plan.output, plan.run_dir)
        )
        if live:
            listed = ", ".join(str(pid) for pid in live[:8])
            raise click.ClickException(
                f"a WARP install for {model_name} is already running (PID"
                f"{'' if len(live) == 1 else 's'} {listed}); wait for it to "
                "finish or stop it, then rerun."
            )

        output_unit = (
            "GiB"
            if info["output_size_bytes"] == info["size_gb"] * 1024**3
            else "GB"
        )
        click.echo("WARP conversion plan:")
        click.echo(f"  Revision: {info['hf_revision']}")
        click.echo(
            f"  Source: {info['source_size_gib']} GiB -> {plan.source}"
        )
        click.echo(
            f"  Container: {info['size_gb']} {output_unit} -> {plan.output}"
        )
        click.echo(
            f"  Conversion workspace: {info['output_workspace_gib']} GiB"
        )
        if reclaim_source:
            click.echo(
                "  WARNING: --reclaim-source is irreversible: completed source "
                "shards are removed, and a retry may need to re-download them."
            )
        if not yes:
            click.confirm("Proceed with WARP conversion?", abort=True)


    # 1. Engine install
    warp_root = None
    if engine in ("llamacpp", "both"):
        click.echo("Installing llama.cpp...")
        try:
            install_llamacpp(prefix, variant=llamacpp_variant, tag=llamacpp_tag)
        except Exception as e:
            click.echo(f"  llama.cpp install failed: {e}", err=True)
            sys.exit(1)
    if engine in ("ktransformers", "both"):
        click.echo("Installing ktransformers...")
        try:
            install_ktransformers()
        except Exception as e:
            click.echo(f"  ktransformers install failed: {e}", err=True)
            sys.exit(1)
    if engine == "warp":
        click.echo("Installing WARP...")
        try:
            if _warp_models._snapshot_live_processes(
                (prefix / "lib" / "warp",)
            ):
                raise RuntimeError(
                    "a WARP fetch or conversion is running from the runtime "
                    "tree; stop it before replacing the runtime"
                )
            warp_root = install_warp(prefix)
        except Exception as e:
            click.echo(f"  WARP install failed: {e}", err=True)
            sys.exit(1)

    # 2. Model installation
    if not model_name:
        total = get_total_memory_bytes()
        click.echo()
        if total:
            ram_gb = total / 1e9
            recs = recommended_for_ram(ram_gb * (0.75 if is_macos() else 1.0))
            click.echo(f"Detected {ram_gb:.0f} GB RAM. Models that fit, fastest first:")
            for recommendation in recs:
                recommended_info = KNOWN_MODELS[recommendation]
                click.echo(
                    f"  litmoe install --model {recommendation:32s} "
                    f"# {quant_size_gb(recommendation, None):.0f} GB, "
                    f"{recommended_info['params']}"
                )
        else:
            click.echo(f"To add a model:  litmoe install --model {DEFAULT_MODEL}")
        click.echo("Full list:  litmoe models")
        return

    assert info is not None
    total = get_total_memory_bytes()
    budget_gb = (total / 1e9) * (0.75 if is_macos() else 1.0) if total else None
    quant_val = choose_quant(model_name, quant, budget_gb)
    size_note = quant_size_gb(model_name, quant_val)
    if info["format"] != WASTE and size_note and not yes:
        need = ram_needed_gb(model_name, quant_val)
        msg = f"  {model_name} {quant_val or ''} is ~{size_note:.0f} GB on disk"
        if need and total:
            msg += f"; needs ~{need:.0f} GB RAM at 32K context (this machine: {total / 1e9:.0f} GB)"
        click.echo(msg)
        if need and budget_gb and need > budget_gb:
            click.echo("  WARNING: this will not fit in RAM; expect it to page from disk (well under 1 token/s).")
        click.confirm("Proceed with download?", abort=True)

    click.echo(f"Installing model: {model_name} [{quant_val}]" if quant_val else f"Installing model: {model_name}")
    engine_for_model = info["engine"]
    if info["format"] == WASTE:
        assert warp_root is not None
        try:
            path = install_warp_model(
                model_name,
                warp_root=warp_root,
                staging_dir=staging_dir,
                models_dir=models_dir,
                jobs=warp_jobs,
                reclaim_source=reclaim_source,
            )
        except KeyboardInterrupt:
            click.echo(
                "\n  Interrupted; the download and conversion were stopped. "
                "Partial data was preserved; rerun the same command to resume.",
                err=True,
            )
            sys.exit(130)
        except Exception as e:
            click.echo(f"  WARP model install failed: {e}", err=True)
            sys.exit(1)
        add_model_to_config(model_name, engine_for_model, path, 0, config_path)
    else:
        path, mmproj = download_model(
            model_name,
            quant_val,
            models_dir,
            with_mmproj=not no_mmproj,
        )
        if info["format"] == GGUF:
            weights_gb = None
            try:
                pattern = re.sub(r"-\d{5}-of-(\d{5})\.gguf$", r"-*-of-\1.gguf", path.name)
                files = list(path.parent.glob(pattern)) if pattern != path.name else [path]
                weights_gb = sum(file.stat().st_size for file in files) / 1e9
            except OSError:
                pass
            model_ctx = choose_n_ctx(model_name, weights_gb, n_ctx)
            extra = list(info.get("extra_args", []))
            if mmproj:
                extra += ["--mmproj", str(mmproj)]
            add_model_to_config(
                model_name,
                engine_for_model,
                path,
                model_ctx,
                config_path,
                extra_args=extra or None,
            )
        else:
            model_ctx = n_ctx or info["native_ctx"]
            add_model_to_config(
                model_name,
                engine_for_model,
                path,
                model_ctx,
                config_path,
                extra_args=list(info.get("extra_args", [])) or None,
                kt_method=info["kt_method"],
            )
            click.echo("  ktransformers entry written: needs an NVIDIA GPU at serve time "
                       "(kt_num_gpu_experts=0 keeps all experts on CPU).")
    if info.get("notes"):
        click.echo(f"  NOTE: {info['notes']}")

    from litmoe.config import load_config
    from litmoe.server import check_fits_together
    try:
        validation = check_fits_together(load_config(config_path).models)
    except Exception:  # unreadable/partial config: the serve-time check will catch it
        validation = None
    click.echo()
    crowded = validation is not None and validation.level == "no" and len(validation.per_model) > 1
    if crowded:
        click.echo(
            f"  NOTE: models.yaml now lists {len(validation.per_model)} models needing "
            f"~{validation.total_gb:.0f} GB together; this machine has "
            f"~{validation.ram_limit_gb:.0f} GB usable. `litmoe serve` loads all of them at once,"
        )
        click.echo(f"        so serve one at a time:  litmoe serve {model_name}")
        click.echo()
    click.echo("Done. Next:  litmoe serve" + (f" {model_name}" if crowded else ""))
