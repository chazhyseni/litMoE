"""Command-line entry point for paired local latency measurements."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import click
import httpx

from litmoe.benchmark import benchmark
from litmoe.config import GatewayConfig, default_config_path, load_config


@click.command("bench")
@click.argument("model", required=False)
@click.option("--config", "-c", type=click.Path(exists=True), default=None)
@click.option("--gateway", default=None, envvar="LITMOE_GATEWAY")
@click.option("--key", default=None, envvar="LITMOE_API_KEY")
@click.option("--runs", type=click.IntRange(1, 20), default=3, show_default=True)
@click.option("--max-tokens", type=click.IntRange(1, 131072), default=128, show_default=True)
@click.option("--timeout", type=click.FloatRange(min=0, max=3600, min_open=True), default=600)
@click.option("--prompt", default=None)
@click.option("--prompt-file", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def bench(ctx, model, config, gateway, key, runs, max_tokens, timeout, prompt, prompt_file, as_json):
    """Compare gateway/direct streaming on the gateway host, with other clients idle.

    Reports first generated delta separately from first visible text. Throughput
    includes prefill and transport; repeats do not prove native cache reuse.
    """
    if prompt is not None and prompt_file:
        raise click.UsageError("Choose --prompt or --prompt-file, not both.")
    try:
        path = Path(config) if config else default_config_path()
        cfg = load_config(path) if path.exists() else GatewayConfig()
        if prompt_file:
            with open(prompt_file, "rb") as source:
                data = source.read(1_000_001)
            if len(data) > 1_000_000:
                raise ValueError("prompt file exceeds one million bytes")
            prompt = data.decode("utf-8")
        if prompt is None:
            prompt = "Reply with one short sentence."
        host = "127.0.0.1" if cfg.host in ("0.0.0.0", "::") else cfg.host
        result = asyncio.run(benchmark(
            gateway or f"http://{host}:{cfg.port}", model, key or cfg.api_key,
            prompt, runs, max_tokens, timeout,
        ))
    except (OSError, ValueError, KeyError, httpx.HTTPError) as exc:
        raise click.ClickException(str(exc)) from exc
    if as_json:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(f"{result['model']} / {result['backend']} / context {result['context_window']}")
        for row in result["runs"]:
            click.echo(f"{row['order']:2d} {row['route']:7s} first_delta={row['first_generated_delta_s']}s "
                       f"first_text={row['first_visible_text_s']}s total={row['completion_s']:.3f}s "
                       f"output_tokens={row['output_tokens']} error={row['error']}")
        for note in result["limitations"]:
            click.echo(note)
    if any(not row["success"] for row in result["runs"]):
        ctx.exit(1)
