"""litmoe install - one-command engine + model installation.

Installs an inference engine (llama.cpp release binaries or a source build;
ktransformers via PyPI wheels or the upstream install.sh) and downloads model
weights, then writes the model entry into models.yaml.

The model catalog lives in litmoe.models (single source of truth).
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import click
import yaml

from litmoe.config import default_config_path, expand_path
from litmoe.models import (
    CLAUDE_ALIASES,
    DEFAULT_MODEL,
    GGUF,
    KNOWN_MODELS,
    SAFETENSORS,
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
        cfg = {"host": "127.0.0.1", "port": 8080, "api_key": None, "models": []}
    cfg.setdefault("host", "127.0.0.1")
    cfg.setdefault("port", 8080)
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
    ctx, note = fit_context(info.get("kv_bytes_per_token", 65_536), weights_gb, total / 1e9, target)
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
    tiers = sorted({v["tier"] for v in KNOWN_MODELS.values()})
    for tier in tiers:
        click.echo(f"== {TIER_LABELS[tier]} ==")
        for mid, info in KNOWN_MODELS.items():
            if info["tier"] != tier:
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
    click.echo("Install: litmoe install --model <name> [--quant <quant>]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command("install")
@click.argument("targets", nargs=-1)
@click.option("--model", "model_name", type=click.Choice(sorted(KNOWN_MODELS.keys())),
              default=None, help="Model to download (see `litmoe models`)")
@click.option("--quant", default=None, help="Quantization (default: the model's default_quant if it fits this machine's RAM, else the largest that does)")
@click.option("--engine", type=click.Choice(["llamacpp", "ktransformers", "both", "none"]),
              default=None, help="Which engine(s) to install (default: the model's engine, else llamacpp)")
@click.option("--llamacpp-variant", type=click.Choice(LLAMACPP_VARIANTS), default="auto",
              help="llama.cpp release binary variant (auto = cuda if an NVIDIA GPU is visible, else cpu)")
@click.option("--llamacpp-tag", default=None, help="Pin a llama.cpp release tag (e.g. b11005)")
@click.option("--models-dir", type=click.Path(), default=None,
              help="Where to store model weights (default: ~/.litmoe/models)")
@click.option("--prefix", type=click.Path(), default=None,
              help="Install prefix for engine binaries (default: ~/.local)")
@click.option("--n-ctx", default=None, type=int,
              help="Context size written to models.yaml (default: native context, reduced to fit RAM)")
@click.option("--no-mmproj", is_flag=True, help="Skip the vision projector for multimodal models")
@click.option("--config", "-c", type=click.Path(), default=None, help="Path to models.yaml")
@click.option("--yes", is_flag=True, help="Skip confirmation prompts")
def install_cmd(targets, model_name, quant, engine, llamacpp_variant, llamacpp_tag, models_dir,
                prefix, n_ctx, no_mmproj, config, yes):
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
      litmoe install --engine llamacpp --llamacpp-variant cuda
    """
    models_dir = expand_path(models_dir) if models_dir else _default_models_dir()
    prefix = expand_path(prefix) if prefix else _default_prefix()
    config_path = expand_path(config) if config else default_config_path()

    for t in targets:
        if t in ("llamacpp", "ktransformers", "both"):
            engine = t if engine is None else engine
        elif lookup(t):
            model_name = t if model_name is None else model_name
        else:
            raise click.BadParameter(f"unknown target: {t}")

    info = lookup(model_name) if model_name else None
    if engine is None:
        engine = info["engine"] if info else "llamacpp"

    # 1. Engine install
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

    # 2. Model download
    if not model_name:
        total = get_total_memory_bytes()
        click.echo()
        if total:
            ram_gb = total / 1e9
            recs = recommended_for_ram(ram_gb * (0.75 if is_macos() else 1.0))
            click.echo(f"Detected {ram_gb:.0f} GB RAM. Models that fit, fastest first:")
            for r in recs:
                ri = KNOWN_MODELS[r]
                click.echo(f"  litmoe install --model {r:32s} # {quant_size_gb(r, None):.0f} GB, {ri['params']}")
        else:
            click.echo(f"To add a model:  litmoe install --model {DEFAULT_MODEL}")
        click.echo("Full list:  litmoe models")
        return

    assert info is not None
    total = get_total_memory_bytes()
    budget_gb = (total / 1e9) * (0.75 if is_macos() else 1.0) if total else None
    quant_val = choose_quant(model_name, quant, budget_gb)
    size_note = quant_size_gb(model_name, quant_val)
    if size_note and not yes:
        need = ram_needed_gb(model_name, quant_val)
        msg = f"  {model_name} {quant_val or ''} is ~{size_note:.0f} GB on disk"
        if need and total:
            msg += f"; needs ~{need:.0f} GB RAM at 32K context (this machine: {total / 1e9:.0f} GB)"
        click.echo(msg)
        if need and budget_gb and need > budget_gb:
            click.echo("  WARNING: this will not fit in RAM; expect it to page from disk (well under 1 token/s).")
        click.confirm("Proceed with download?", abort=True)

    click.echo(f"Installing model: {model_name} [{quant_val}]" if quant_val else f"Installing model: {model_name}")
    path, mmproj = download_model(model_name, quant_val, models_dir, with_mmproj=not no_mmproj)
    engine_for_model = info["engine"]

    if info["format"] == GGUF:
        weights_gb = None
        try:
            pat = re.sub(r"-\d{5}-of-(\d{5})\.gguf$", r"-*-of-\1.gguf", path.name)
            files = list(path.parent.glob(pat)) if pat != path.name else [path]
            weights_gb = sum(f.stat().st_size for f in files) / 1e9
        except OSError:
            pass
        ctx = choose_n_ctx(model_name, weights_gb, n_ctx)
        extra = list(info.get("extra_args", []))
        if mmproj:
            extra += ["--mmproj", str(mmproj)]
        add_model_to_config(model_name, engine_for_model, path, ctx, config_path, extra_args=extra or None)
    else:
        ctx = n_ctx or info["native_ctx"]
        add_model_to_config(model_name, engine_for_model, path, ctx, config_path,
                            extra_args=list(info.get("extra_args", [])) or None,
                            kt_method=info["kt_method"])
        click.echo("  ktransformers entry written: needs an NVIDIA GPU at serve time "
                   "(kt_num_gpu_experts=0 keeps all experts on CPU).")
    if info.get("notes"):
        click.echo(f"  NOTE: {info['notes']}")

    from litmoe.config import load_config
    from litmoe.server import check_fits_together
    try:
        over = check_fits_together(load_config(config_path).models)
    except Exception:  # unreadable/partial config: the serve-time check will catch it
        over = None
    click.echo()
    if over:
        total, budget, per_model = over
        click.echo(f"  NOTE: models.yaml now lists {len(per_model)} models needing ~{total:.0f} GB together; "
                   f"this machine's budget is ~{budget:.0f} GB. `litmoe serve` loads all of them at once,")
        click.echo(f"        so serve one at a time:  litmoe serve --model {model_name}")
        click.echo()
    click.echo("Done. Next:  litmoe serve" + (f" --model {model_name}" if over else ""))
