"""litmoe CLI - main entry point."""
from __future__ import annotations

import os
import platform
import signal
import subprocess
import sys
from pathlib import Path

import click
import yaml

from litmoe import __version__
from litmoe.cli.install import install_cmd, print_model_table
from litmoe.config import default_config_path, expand_path, load_config
from litmoe.engines import kt_installed, llama_installed
from litmoe.models import (_OS_HEADROOM_GB, CLAUDE_ALIASES, DEFAULT_MODEL, KNOWN_MODELS, fit_together,
                           quant_size_gb, recommended_for_ram, smallest_gguf_model)
from litmoe.platform_utils import (
    cpu_flags,
    get_numa_nodes,
    get_physical_cores,
    get_total_memory_bytes,
    has_amx,
    has_avx512,
    is_macos,
    nvidia_gpus,
)


@click.group()
@click.version_option(version=__version__, prog_name="litmoe")
def cli():
    """litmoe - OpenAI-compatible gateway for llama.cpp and ktransformers"""
    pass


@cli.command()
def doctor():
    """Check hardware compatibility and engine availability."""
    click.echo(f"litmoe v{__version__}")
    click.echo(f"Python: {sys.executable} ({sys.version.split()[0]})")
    click.echo()

    click.echo("=== CPU ===")
    if is_macos():
        try:
            result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                    capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                click.echo(f"  {result.stdout.strip()}")
            if platform.machine() == "arm64":
                click.echo("  Architecture: Apple Silicon (ARM64) — Metal, NEON, AMX")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            click.echo("  (could not detect CPU)")
    else:
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        click.echo(f"  {line.split(':')[1].strip()}")
                        break
        except FileNotFoundError:
            click.echo("  (not Linux)")
        flags = cpu_flags()
        interesting = ["sse4_2", "avx", "avx2", "avx512f", "avx512_bf16", "avx512_vnni",
                       "amx_tile", "amx_bf16", "amx_int8"]
        click.echo(f"  Instruction sets: {', '.join(fl for fl in interesting if fl in flags) or 'unknown'}")
    click.echo(f"  Physical cores: {get_physical_cores()} (threads used by llama-server), "
               f"NUMA nodes: {get_numa_nodes()}")

    click.echo()
    click.echo("=== Memory ===")
    total_mem = get_total_memory_bytes()
    ram_gb = total_mem / 1e9 if total_mem else None
    if ram_gb:
        click.echo(f"  Total: {ram_gb:.0f} GB")
        if is_macos():
            click.echo(f"  Metal can use ~{ram_gb * 0.75:.0f} GB of it by default "
                       f"(raise with: sudo sysctl iogpu.wired_limit_mb=<MB>)")
    else:
        click.echo("  (could not detect memory)")

    click.echo()
    click.echo("=== GPU ===")
    gpus = nvidia_gpus()
    if gpus:
        for g in gpus:
            click.echo(f"  {g}")
    elif is_macos():
        click.echo("  Apple GPU (Metal) via llama.cpp")
    else:
        click.echo("  No NVIDIA GPU detected")

    click.echo()
    click.echo("=== Engines ===")
    if llama_installed():
        click.echo("  llama.cpp: installed")
    else:
        click.echo("  llama.cpp: NOT installed (litmoe install --engine llamacpp)")
    if kt_installed():
        click.echo("  ktransformers (kt-kernel + sglang-kt): installed")
    else:
        from litmoe.engines.ktransformers import missing_components
        click.echo(f"  ktransformers: NOT installed — missing {', '.join(missing_components())} "
                   f"(litmoe install --engine ktransformers)")

    click.echo()
    click.echo("=== Recommendation ===")
    if is_macos():
        click.echo("  Engine: llama.cpp (Metal). ktransformers needs Linux + NVIDIA GPU.")
    elif gpus and has_avx512():
        click.echo("  Engine: llama.cpp for GGUF models; ktransformers for native FP8/INT4 MoE "
                   f"checkpoints (AVX-512 {'+ AMX ' if has_amx() else ''}CPU backend available).")
    elif gpus:
        click.echo("  Engine: llama.cpp. ktransformers only with kt_method LLAMAFILE here "
                   "(no AVX-512: FP8/BF16/RAWINT4 CPU backends unavailable).")
    else:
        click.echo("  Engine: llama.cpp (CPU). ktransformers serving requires an NVIDIA GPU.")
    if ram_gb:
        budget = ram_gb * (0.75 if is_macos() else 1.0)
        recs = recommended_for_ram(budget)
        if recs:
            click.echo(f"  Models that fit {ram_gb:.0f} GB RAM (fastest first): "
                       + ", ".join(f"{r} ({quant_size_gb(r, None):.0f} GB)" for r in recs))
        click.echo("  Full list with fit info: litmoe models")


@cli.command("models")
def models_cmd():
    """List the model catalog by RAM tier and show what fits this machine."""
    total = get_total_memory_bytes()
    print_model_table(total / 1e9 if total else None)


@cli.command()
@click.option("--force", is_flag=True, help="Overwrite an existing models.yaml")
def init(force):
    """Create a models.yaml with fast defaults for this machine's RAM.

    Entries use HuggingFace repo specs (owner/repo:QUANT): llama-server
    downloads them on first start. Use `litmoe install --model X` to
    pre-download instead.
    """
    cfg_path = Path("models.yaml")
    if cfg_path.exists() and not force:
        click.echo(f"{cfg_path} already exists. Use --force to overwrite.")
        sys.exit(1)

    total = get_total_memory_bytes()
    ram_gb = total / 1e9 if total else None
    budget = ram_gb * (0.75 if is_macos() else 1.0) if ram_gb else None
    also_fit: list[str] = []
    if budget:
        # recommended_for_ram already leads with DEFAULT_MODEL when it fits.
        # The gateway loads every entry at once, so keep only what fits *together*.
        candidates = recommended_for_ram(budget, max_models=3)
        picks, also_fit = fit_together(candidates, budget)
        if not picks:
            picks = [candidates[0] if candidates else smallest_gguf_model()]
            also_fit = []
            click.echo(f"Warning: {ram_gb:.0f} GB RAM is below every catalog tier; "
                       f"picking the smallest model ({picks[0]}) — expect a reduced context.", err=True)
    else:
        picks = [DEFAULT_MODEL]

    models = []
    for i, mid in enumerate(picks):
        info = KNOWN_MODELS[mid]
        entry = {
            "id": mid,
            "engine": "llamacpp",
            "model_path": f"{info['hf_repo']}:{info['default_quant']}",
            "n_gpu_layers": -1,
            # 0 = "memory-aware native": the gateway raises this to the largest
            # context (up to the model's native max) that fits RAM at start-up
            # and writes the result back here.
            "n_ctx": 0,
        }
        if i == 0:
            # Anthropic-API clients (Claude Code) send Claude model names.
            entry["aliases"] = list(CLAUDE_ALIASES)
        models.append(entry)

    cfg = {"host": "127.0.0.1", "port": 8080, "api_key": None, "models": models}
    with open(cfg_path, "w") as f:
        f.write("# litmoe gateway config. Docs: https://github.com/chazhyseni/litMoE\n")
        f.write("# model_path may be a local GGUF, a HuggingFace spec owner/repo:QUANT, or a URL.\n")
        if ram_gb:
            f.write(f"# Defaults chosen for {ram_gb:.0f} GB RAM; see `litmoe models` for more.\n")
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)

    click.echo(f"Created {cfg_path} with: {', '.join(picks)}")
    if ram_gb:
        click.echo(f"  (chosen for {ram_gb:.0f} GB RAM — fastest models that fit together; edit freely)")
    if also_fit:
        click.echo(f"  Also fit on their own, not alongside the above: {', '.join(also_fit)}")
        click.echo(f"  Serve one instead with:  litmoe serve --model {also_fit[0]}")
    click.echo("Next: litmoe serve   (first start downloads the weights)")
    click.echo("Or pre-download:  litmoe install --model " + picks[0])


@cli.command()
@click.option("--config", "-c", type=click.Path(), default=None, help="Path to models.yaml")
@click.option("--log-dir", default="logs", help="Directory for engine logs")
@click.option("--model", "-m", "only", multiple=True,
              help="Serve only these model ids from models.yaml (repeatable). Default: all.")
@click.option("--force", is_flag=True, help="Start even if the selected models will not fit in RAM together")
def serve(config, log_dir, only, force):
    """Start the gateway with the configured engines.

    Every selected model is loaded at once, so together they must fit this
    machine's RAM budget. If they do not, serve refuses and shows the numbers;
    pick a subset with --model, or pass --force to start anyway.
    """
    cfg_path = str(expand_path(config)) if config else str(default_config_path())
    if not Path(cfg_path).exists():
        click.echo(f"Error: config not found: {cfg_path}", err=True)
        click.echo("Run 'litmoe init' or 'litmoe install --model X' to create one.", err=True)
        sys.exit(1)

    cfg = load_config(cfg_path)
    if only:
        known = {m.id for m in cfg.models}
        missing = [o for o in only if o not in known]
        if missing:
            click.echo(f"Error: not in {cfg_path}: {', '.join(missing)} "
                       f"(configured: {', '.join(sorted(known))})", err=True)
            sys.exit(1)
        cfg.models = [m for m in cfg.models if m.id in set(only)]

    from litmoe.server import check_fits_together, run as server_run
    over = check_fits_together(cfg.models)
    if over:
        total, budget, per_model = over
        click.echo(f"These {len(per_model)} models need ~{total:.0f} GB RAM loaded together; "
                   f"this machine's budget is ~{budget:.0f} GB"
                   + (" (75% of RAM: Metal shares unified memory)." if is_macos() else "."), err=True)
        for mid, need in per_model:
            click.echo(f"  {mid:32s} ~{need:.0f} GB", err=True)
        click.echo("The gateway starts every model at once, so this would run out of memory "
                   "(on Metal: kIOGPUCommandBufferCallbackErrorOutOfMemory, with the engines still 'running').", err=True)
        if force:
            click.echo("Continuing anyway (--force).", err=True)
        else:
            biggest_fit = max((p for p in per_model if p[1] + _OS_HEADROOM_GB <= budget),
                              key=lambda p: p[1], default=None)
            hint = biggest_fit[0] if biggest_fit else per_model[0][0]
            click.echo(f"Pick a subset:   litmoe serve --model {hint}", err=True)
            click.echo("Or edit models.yaml, or pass --force to start regardless.", err=True)
            sys.exit(1)

    click.echo(f"litmoe v{__version__} starting gateway on {cfg.host}:{cfg.port}")
    click.echo(f"Models: {[m.id for m in cfg.models]}")
    click.echo(f"Log dir: {log_dir}")
    click.echo()

    server_run(cfg, log_dir=log_dir, config_path=cfg_path)


@cli.command()
@click.option("--config", "-c", type=click.Path(), default=None)
def status(config):
    """Show running gateway and engines."""
    import httpx

    cfg_path = str(expand_path(config)) if config else str(default_config_path())
    click.echo(f"Config: {cfg_path}")
    if not Path(cfg_path).exists():
        click.echo("  (no config)")
        return

    cfg = load_config(cfg_path)
    click.echo(f"Configured models: {[m.id for m in cfg.models]}")

    host = "127.0.0.1" if cfg.host in ("0.0.0.0", "::") else cfg.host
    try:
        r = httpx.get(f"http://{host}:{cfg.port}/health", timeout=5.0)
        if r.status_code == 200:
            data = r.json()
            click.echo()
            click.echo(f"Gateway health: {data['status']}")
            for model_id, info in data.get("engines", {}).items():
                click.echo(f"  {model_id}: running={info['running']}, port={info['port']}")
    except Exception as e:
        click.echo()
        click.echo(f"Gateway not reachable: {e}")


@cli.command()
@click.option("--all", "kill_all", is_flag=True,
              help="Also SIGTERM any llama-server / sglang process not started by litmoe")
def stop(kill_all):
    """Stop the engines litmoe started (from ~/.litmoe/run/*.pid).

    Only litmoe's own engine processes are touched, so an Ollama, LM Studio or
    manually launched llama-server keeps running. Use --all to override.
    """
    from litmoe.engines.base import pid_dir

    found = False
    run_dir = pid_dir()
    for pidfile in sorted(run_dir.glob("*.pid")) if run_dir.exists() else []:
        try:
            pid = int(pidfile.read_text().strip())
        except ValueError:
            pidfile.unlink(missing_ok=True)
            continue
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            click.echo(f"  {pidfile.stem}: SIGTERM to process group of PID {pid}")
            found = True
        except ProcessLookupError:
            click.echo(f"  {pidfile.stem}: PID {pid} already gone")
        except PermissionError:
            subprocess.run(["kill", "-TERM", f"-{pid}"])
            found = True
        pidfile.unlink(missing_ok=True)

    if kill_all:
        patterns = ["llama-server", "sglang.launch_server", "kt run"]
        for pattern in patterns:
            result = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
            for pid in result.stdout.split():
                if int(pid) == os.getpid():
                    continue
                click.echo(f"  --all: SIGTERM to {pid} ({pattern})")
                try:
                    os.kill(int(pid), signal.SIGTERM)
                    found = True
                except (ProcessLookupError, PermissionError):
                    pass
    if not found:
        click.echo("  No litmoe engine processes found" + ("" if kill_all else " (use --all to match by name)"))


cli.add_command(install_cmd)


def main():
    cli()


if __name__ == "__main__":
    main()
