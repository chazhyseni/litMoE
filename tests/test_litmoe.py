"""Unit tests for litmoe (no network, no engines)."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from litmoe import models as M
from litmoe.config import GatewayConfig, ModelEntry, expand_model_paths, is_hf_repo_spec, load_config
from litmoe.cli import install as I
from litmoe import server as S


# ---------------------------------------------------------------------------
# catalog
# ---------------------------------------------------------------------------

def test_catalog_is_consistent():
    assert M.validate_catalog() == []


def test_default_model_is_fast_laptop_class():
    info = M.KNOWN_MODELS[M.DEFAULT_MODEL]
    assert info["tier"] == M.TIER_LAPTOP_48
    assert info["active_b"] is not None and info["active_b"] <= 5
    assert M.ram_needed_gb(M.DEFAULT_MODEL) <= 48


def test_recommendations_by_ram():
    for ram in (48, 96):
        recs = M.recommended_for_ram(ram)
        assert recs and recs[0] == M.DEFAULT_MODEL
        for r in recs:
            assert M.ram_needed_gb(r) <= ram
    # a 48 GB laptop never gets a >48 GB-tier model
    assert all(M.KNOWN_MODELS[r]["tier"] == 48 for r in M.recommended_for_ram(48))
    # big machines get the strongest tier that fits, right after the default
    assert M.KNOWN_MODELS[M.recommended_for_ram(400)[1]]["tier"] == M.TIER_SERVER_512


def test_largest_quant_that_fits():
    assert M.largest_quant_that_fits("qwen3.5-122b-a10b", 48) in ("UD-IQ1_M", "UD-IQ2_XXS")
    assert M.largest_quant_that_fits("kimi-k3", 96) is None
    q = M.largest_quant_that_fits("gemma-4-26b-a4b", 96)
    assert q == "BF16"  # everything fits, so the best precision wins


def test_legacy_alias_lookup():
    assert M.lookup("qwen3.8-2.4t") is M.KNOWN_MODELS["qwen3.8"]

def test_fit_context_shrinks_and_caps():
    # fits unchanged (20 GB weights -> 22 GB loaded, 17 GB KV, 83 GB budget)
    assert M.fit_context(65_536, 20.0, 96.0, 262144) == (262144, None)
    # must shrink: 60 GB weights, huge KV rate
    ctx, note = M.fit_context(1_000_000, 60.0, 96.0, 262144)
    assert ctx < 262144 and ctx % 4096 == 0 and ctx >= 8192 and "reduced" in note
    # weights do not fit at all: capped, never below 8192
    ctx, note = M.fit_context(65_536, 600.0, 96.0, 1048576)
    assert ctx == 32768 and "exceed" in note


def test_fit_context_uses_gpu_budget_on_macos():
    """Regression: a 73 GB model on a 103 GB Mac was written with its native 262K
    context because the fit checked RAM (90 GB), not Metal's 77 GB; -ngl -1 puts
    the KV cache on the GPU, so the engine OOMed on the first forward pass."""
    kv, weights, ram, native = 46_000, 73.0, 103.1, 262144
    linux_ctx, _ = M.fit_context(kv, weights, ram, native, macos=False)
    mac_ctx, note = M.fit_context(kv, weights, ram, native, macos=True)
    assert mac_ctx < linux_ctx
    assert mac_ctx == 32768 and "Metal" in note            # 80 GB loaded > 77 GB budget: capped
    # A model that fits the GPU budget with room keeps its native context on macOS too.
    assert M.fit_context(40_960, 17.0, 103.1, 262144, macos=True) == (262144, None)
    gpu, ram_limit = M.memory_budgets_gb(103.1, macos=True)
    assert round(gpu) == 77 and round(ram_limit) == 90
    assert M.memory_budgets_gb(103.1, macos=False) == (ram_limit, ram_limit)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ("unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL", True),
    ("unsloth/Kimi-K3-GGUF", True),
    ("zai-org/GLM-5.3-Flash", True),
    ("/models/x.gguf", False),
    ("~/models/x.gguf", False),
    ("./x.gguf", False),
    ("https://huggingface.co/x/y", False),
    ("just-a-name", False),
    ("a/b/c", False),
])
def test_is_hf_repo_spec(value, expected):
    assert is_hf_repo_spec(value) is expected


def test_expand_model_paths_leaves_hf_specs_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    d = {"model_path": "unsloth/gemma-4-12b-it-GGUF:Q4_K_M", "gguf_path": "~/m.gguf"}
    out = expand_model_paths(dict(d))
    assert out["model_path"] == d["model_path"]
    assert out["gguf_path"] == str(tmp_path / "m.gguf")


def test_duplicate_ids_and_aliases_rejected():
    with pytest.raises(ValueError):
        GatewayConfig(models=[
            ModelEntry(id="a", engine="llamacpp", model_path="x"),
            ModelEntry(id="b", engine="llamacpp", model_path="y", aliases=["a"]),
        ])
    # the same alias on the same model twice is fine
    GatewayConfig(models=[ModelEntry(id="a", engine="llamacpp", model_path="x", aliases=["c", "c"])])


def test_load_config_with_kt_fields(tmp_path):
    p = tmp_path / "models.yaml"
    p.write_text(
        "port: 8000\nmodels:\n"
        "  - id: glm\n    engine: ktransformers\n    model_path: zai-org/GLM-5.3-Flash\n"
        "    kt_method: FP8\n    kt_num_gpu_experts: 4\n"
    )
    cfg = load_config(p)
    assert cfg.models[0].kt_method == "FP8" and cfg.models[0].kt_num_gpu_experts == 4


# ---------------------------------------------------------------------------
# engines: command construction
# ---------------------------------------------------------------------------

def test_llamacpp_uses_hf_flag_for_repo_spec(monkeypatch):
    from litmoe.engines.llamacpp import LlamaCppEngine
    eng = LlamaCppEngine(ModelEntry(id="g", engine="llamacpp",
                                    model_path="unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL", n_ctx=32768))
    monkeypatch.setattr(eng, "_resolve_binary", lambda: ("/bin/llama-server", None))
    eng.set_port(8085)
    cmd = eng.build_command()
    assert cmd[:3] == ["/bin/llama-server", "-hf", "unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL"]
    assert "-m" not in cmd
    assert cmd[cmd.index("--port") + 1] == "8085"
    assert cmd[cmd.index("-c") + 1] == "32768"
    assert "-t" in cmd and int(cmd[cmd.index("-t") + 1]) >= 1


def test_llamacpp_user_threads_not_overridden(monkeypatch):
    from litmoe.engines.llamacpp import LlamaCppEngine
    eng = LlamaCppEngine(ModelEntry(id="g", engine="llamacpp", model_path="/tmp/x.gguf",
                                    extra_args=["-t", "4"]))
    monkeypatch.setattr(eng, "_resolve_binary", lambda: ("/bin/llama-server", None))
    cmd = eng.build_command()
    assert cmd.count("-t") == 1 and cmd[cmd.index("-t") + 1] == "4"


def test_llamacpp_single_slot_default_unless_user_sets_parallel(monkeypatch):
    """One server slot by default: a harness's concurrent side requests (title
    generation) otherwise prefill alongside the main conversation and halve
    its speed. -np / --parallel in extra_args is respected."""
    from litmoe.engines.llamacpp import LlamaCppEngine
    eng = LlamaCppEngine(ModelEntry(id="g", engine="llamacpp", model_path="/tmp/x.gguf"))
    monkeypatch.setattr(eng, "_resolve_binary", lambda: ("/bin/llama-server", None))
    cmd = eng.build_command()
    assert cmd[cmd.index("-np") + 1] == "1"

    eng = LlamaCppEngine(ModelEntry(id="g", engine="llamacpp", model_path="/tmp/x.gguf",
                                    extra_args=["--parallel", "4"]))
    monkeypatch.setattr(eng, "_resolve_binary", lambda: ("/bin/llama-server", None))
    cmd = eng.build_command()
    assert "-np" not in cmd and cmd[cmd.index("--parallel") + 1] == "4"



def test_llamacpp_prefers_installed_prebuilt_over_stale_source_build(tmp_path, monkeypatch):
    """A leftover local/ source build must not shadow the release `litmoe install` fetched.

    Regression: an Aug-2026 source build in local/ was chosen over a Sep-2026
    prebuilt, so a model whose architecture only the newer build knows failed
    with 'unknown model architecture' even though install had just succeeded.
    """
    from litmoe.engines.llamacpp import LlamaCppEngine
    monkeypatch.setenv("LITMOE_PREFIX", str(tmp_path))
    monkeypatch.setattr(shutil, "which", lambda *_: None)
    local = tmp_path / "lib" / "llama.cpp" / "local"
    prebuilt = tmp_path / "lib" / "llama.cpp" / "prebuilt" / "llama-b10964"
    for d in (local, prebuilt):
        d.mkdir(parents=True)
        (d / "llama-server").write_text("#!/bin/sh\n")
    eng = LlamaCppEngine(ModelEntry(id="x", engine="llamacpp", model_path="/tmp/x.gguf"))
    binary, lib_dir = eng._resolve_binary()
    assert Path(binary) == prebuilt / "llama-server"
    assert lib_dir == prebuilt

    # With no prebuilt, the source build is still found.
    shutil.rmtree(prebuilt.parent)
    binary, lib_dir = eng._resolve_binary()
    assert Path(binary) == local / "llama-server"



def test_ktransformers_command_sglang():
    from litmoe.engines.ktransformers import KtransformersEngine
    m = ModelEntry(id="glm-5.3-flash", engine="ktransformers", model_path="zai-org/GLM-5.3-Flash",
                   kt_method="FP8", kt_num_gpu_experts=2, kt_cpuinfer=32, kt_threadpool_count=2,
                   n_ctx=131072, extra_args=["--tool-call-parser", "glm47"])
    eng = KtransformersEngine(m)
    eng.set_port(8082)
    cmd = eng.build_command()
    assert cmd[1:3] == ["-m", "sglang.launch_server"]
    assert cmd[cmd.index("--model-path") + 1] == "zai-org/GLM-5.3-Flash"
    assert cmd[cmd.index("--kt-method") + 1] == "FP8"
    assert cmd[cmd.index("--kt-num-gpu-experts") + 1] == "2"
    assert cmd[cmd.index("--kt-cpuinfer") + 1] == "32"
    assert cmd[cmd.index("--served-model-name") + 1] == "glm-5.3-flash"
    assert "--trust-remote-code" in cmd and cmd[-2:] == ["--tool-call-parser", "glm47"]
    assert "ktransformers.server.main" not in cmd
    assert eng.health_url().endswith(":8082/health")


def test_ktransformers_llamafile_needs_gguf_and_bad_method_rejected():
    from litmoe.engines.ktransformers import KtransformersEngine
    eng = KtransformersEngine(ModelEntry(id="x", engine="ktransformers", model_path="/m", gguf_path="/g"))
    assert eng.kt_method() == "LLAMAFILE"
    bad = KtransformersEngine(ModelEntry(id="x", engine="ktransformers", model_path="/m", kt_method="INT9"))
    with pytest.raises(ValueError):
        bad.build_command()
    none = KtransformersEngine(ModelEntry(id="x", engine="ktransformers", model_path="/m"))
    with pytest.raises(ValueError):
        none.build_command()


def _fake_warp_root(tmp_path: Path) -> Path:
    root = tmp_path / "warp"
    (root / "serve").mkdir(parents=True)
    (root / "serve" / "__main__.py").write_text("def parse_size(value): return int(value)\n")
    (root / "serve" / "engine.py").write_text(
        "import os\n"
        "from types import SimpleNamespace\n"
        "def usable_ram(): return int(os.environ.get('TEST_WARP_RAM_GIB', 96)) * 1024**3\n"
        "def plan_memory(path, ctx):\n"
        "    required = 5 * 1024**3 + ctx * 65536\n"
        "    return SimpleNamespace(recommended_bytes=required, floor_bytes=required, vision_bytes=8*1024**3)\n"
    )
    library = "libwaste.dylib" if sys.platform == "darwin" else (
        "libwaste.dll" if sys.platform == "win32" else "libwaste.so"
    )
    (root / library).write_bytes(b"library")
    return root


def test_warp_command_uses_local_container_and_upstream_server(tmp_path, monkeypatch):
    from litmoe.engines import make_engine
    from litmoe.engines.warp import WarpEngine

    root = _fake_warp_root(tmp_path)
    model_path = tmp_path / "models" / "glm53.waste"
    model_path.mkdir(parents=True)
    monkeypatch.setenv("LITMOE_WARP_DIR", str(root))

    model = ModelEntry(
        id="glm-5.3-flash-warp",
        engine="warp",
        model_path=str(model_path),
        n_ctx=0,
        extra_args=["--threads", "8", "--no-thinking"],
    )
    eng = make_engine(model)
    assert isinstance(eng, WarpEngine)
    eng.set_port(8087)
    command = eng.build_command()
    ctx_index = command.index("--ctx")
    assert int(command[ctx_index + 1]) > 87644
    del command[ctx_index:ctx_index + 2]
    assert command == [
        sys.executable,
        str(root / "serve" / "__main__.py"),
        str(model_path),
        "--host", "127.0.0.1",
        "--port", "8087",
        "--model-id", "glm-5.3-flash-warp",
        "--threads", "8",
        "--no-thinking",
    ]
    assert eng.health_url() == "http://127.0.0.1:8087/health"


def test_warp_command_passes_explicit_context(tmp_path, monkeypatch):
    from litmoe.engines.warp import WarpEngine

    root = _fake_warp_root(tmp_path)
    model_path = tmp_path / "models" / "ds41.waste"
    model_path.mkdir(parents=True)
    monkeypatch.setenv("LITMOE_WARP_DIR", str(root))

    eng = WarpEngine(ModelEntry(
        id="deepseek-v4.1-flash-warp",
        engine="warp",
        model_path=str(model_path),
        n_ctx=32768,
    ))
    cmd = eng.build_command()
    assert cmd[cmd.index("--ctx") + 1] == "32768"


def test_warp_reports_missing_installation_and_container(tmp_path, monkeypatch):
    monkeypatch.setenv("LITMOE_PREFIX", str(tmp_path / "no-prefix"))
    from litmoe.engines.warp import WarpEngine, is_installed

    missing_root = tmp_path / "missing-warp"
    monkeypatch.setenv("LITMOE_WARP_DIR", str(missing_root))
    monkeypatch.setattr(shutil, "which", lambda *_: None)
    model = ModelEntry(id="glm-warp", engine="warp", model_path=str(tmp_path / "glm53.waste"))
    with pytest.raises(FileNotFoundError, match="litmoe install --engine warp"):
        WarpEngine(model).build_command()
    assert not is_installed()

    root = _fake_warp_root(tmp_path)
    monkeypatch.setenv("LITMOE_WARP_DIR", str(root))
    with pytest.raises(FileNotFoundError, match="WARP container not found"):
        WarpEngine(model).build_command()


def test_warp_requires_shared_library(tmp_path, monkeypatch):
    from litmoe.engines.warp import WarpEngine

    root = tmp_path / "warp"
    (root / "serve").mkdir(parents=True)
    (root / "serve" / "__main__.py").write_text("")
    model_path = tmp_path / "glm53.waste"
    model_path.mkdir()
    monkeypatch.setenv("LITMOE_WARP_DIR", str(root))
    monkeypatch.setenv("LITMOE_PREFIX", str(tmp_path / "no-prefix"))
    monkeypatch.setattr(shutil, "which", lambda *_: None)

    with pytest.raises(FileNotFoundError, match="libwaste"):
        WarpEngine(ModelEntry(
            id="glm-warp", engine="warp", model_path=str(model_path)
        )).build_command()


def test_warp_invalid_override_falls_back_to_prefix(tmp_path, monkeypatch):
    from litmoe.engines.warp import WarpEngine

    prefix = tmp_path / "prefix"
    root = _fake_warp_root(prefix / "lib")
    stale = tmp_path / "stale"
    (stale / "serve").mkdir(parents=True)
    (stale / "serve" / "__main__.py").write_text("")
    model_path = tmp_path / "glm53.waste"
    model_path.mkdir()
    (model_path / "manifest.json").write_text('{"config": {"max_position_embeddings": 131072}}')
    monkeypatch.setenv("LITMOE_WARP_DIR", str(stale))
    monkeypatch.setenv("LITMOE_PREFIX", str(prefix))
    monkeypatch.setattr(shutil, "which", lambda *_: None)

    cmd = WarpEngine(ModelEntry(
        id="glm-warp", engine="warp", model_path=str(model_path), n_ctx=0
    )).build_command()
    assert cmd[1] == str(root / "serve" / "__main__.py")

def test_warp_path_launcher_discovers_sibling_prefix_install(tmp_path, monkeypatch):
    import litmoe.engines.warp as warp

    prefix = tmp_path / "prefix"
    root = _fake_warp_root(prefix / "lib")
    launcher = prefix / "bin" / "waste.exe"
    launcher.parent.mkdir(parents=True)
    launcher.write_bytes(b"executable")
    model_path = tmp_path / "glm53.waste"
    model_path.mkdir()
    (model_path / "manifest.json").write_text('{"config": {"max_position_embeddings": 131072}}')
    monkeypatch.setenv("LITMOE_PREFIX", str(prefix))
    monkeypatch.setattr(warp.shutil, "which", lambda *_: str(launcher))

    command = warp.WarpEngine(ModelEntry(
        id="glm-warp", engine="warp", model_path=str(model_path), n_ctx=0
    )).build_command()

    assert command[1] == str(root / "serve" / "__main__.py")




def test_warp_installation_probe_never_raises(tmp_path, monkeypatch):
    from litmoe.engines.warp import is_installed
    monkeypatch.setenv("LITMOE_PREFIX", str(tmp_path / "no-prefix"))
    monkeypatch.setattr(shutil, "which", lambda *_: None)

    def no_home():
        raise RuntimeError("home is unavailable")

    monkeypatch.setattr(Path, "home", no_home)
    assert not is_installed()



def test_warp_uses_upstream_windows_library_name(monkeypatch):
    import litmoe.engines.warp as warp

    monkeypatch.setattr(warp.sys, "platform", "win32")
    assert warp._library_name() == "libwaste.dll"

def test_warp_child_environment_uses_validated_library_without_upstream_auth(
    tmp_path, monkeypatch,
):
    from litmoe.engines.warp import WarpEngine, _library_name

    root = _fake_warp_root(tmp_path)
    model_path = tmp_path / "glm53.waste"
    model_path.mkdir()
    monkeypatch.setenv("LITMOE_WARP_DIR", str(root))
    monkeypatch.setenv("WASTE_API_KEY", "ambient-secret")
    monkeypatch.setenv("WASTE_LIB", "/tmp/ambient-libwaste.so")

    engine = WarpEngine(ModelEntry(
        id="glm-warp",
        engine="warp",
        model_path=str(model_path),
        env={
            "WASTE_API_KEY": "model-secret",
            "WASTE_LIB": "/tmp/model-libwaste.so",
        },
    ))
    environment = engine.build_environment()

    assert "WASTE_API_KEY" not in environment
    assert environment["WASTE_LIB"] == str(root / _library_name())



# ---------------------------------------------------------------------------
# server
# ---------------------------------------------------------------------------

def test_port_allocation_skips_gateway_port_without_collisions():
    # probe=False: pure arithmetic, independent of what is listening on this host
    assert S.allocate_engine_ports(3, gateway_port=8082, probe=False) == [8081, 8083, 8084]
    assert S.allocate_engine_ports(2, gateway_port=8090, probe=False) == [8081, 8082]
    assert S.allocate_engine_ports(0, gateway_port=8090, probe=False) == []


def test_port_allocation_skips_ports_another_process_holds():
    """A foreign llama-server / LM Studio on the first engine port must not kill our engine."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        busy = s.getsockname()[1]
        ports = S.allocate_engine_ports(2, gateway_port=1, start=busy)
        assert busy not in ports
        assert ports == [busy + 1, busy + 2] or len(ports) == 2  # next free ones


def test_gateway_never_kills_processes_it_did_not_start(tmp_path, monkeypatch):
    """`litmoe stop` reads only ~/.litmoe/run/*.pid — no pgrep on process names."""
    from click.testing import CliRunner
    from litmoe.cli.main import cli
    import subprocess, sys

    monkeypatch.setenv("LITMOE_RUN_DIR", str(tmp_path))
    # a bystander process whose command line looks like an engine
    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)  # llama-server"])
    try:
        result = CliRunner().invoke(cli, ["stop"])
        assert result.exit_code == 0, result.output
        assert "No litmoe engine processes found" in result.output
        assert bystander.poll() is None, "stop killed a process litmoe did not start"
    finally:
        bystander.kill()
        bystander.wait()

def test_stop_all_matches_manual_warp_server(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from litmoe.cli.main import cli
    import subprocess

    monkeypatch.setenv("LITMOE_RUN_DIR", str(tmp_path))
    server = subprocess.Popen([
        sys.executable,
        "-c",
        "import time; time.sleep(30)",
        "/tmp/warp/serve/__main__.py",
        "/tmp/tiny.waste",
    ])
    run = subprocess.run

    def discover_test_process(args, **kwargs):
        assert args[:2] == ["pgrep", "-f"]
        result = run(args, **kwargs)
        # Exercise real command-line matching without exposing unrelated
        # resident engines to the CLI's deliberately broad --all cleanup.
        result.stdout = "\n".join(
            pid for pid in result.stdout.split() if pid == str(server.pid)
        )
        return result

    monkeypatch.setattr(subprocess, "run", discover_test_process)
    try:
        result = CliRunner().invoke(cli, ["stop", "--all"])
        assert result.exit_code == 0, result.output
        server.wait(timeout=5)
        assert "warp/serve/__main__" in result.output
    finally:
        if server.poll() is None:
            server.kill()
            server.wait()


def test_weights_size_from_shards(tmp_path):
    for i in (1, 2, 3):
        (tmp_path / f"m-UD-Q4_K_XL-0000{i}-of-00003.gguf").write_bytes(b"x" * 1000)
    (tmp_path / "other-00001-of-00002.gguf").write_bytes(b"x" * 5000)
    m = ModelEntry(id="m", engine="llamacpp", model_path=str(tmp_path / "m-UD-Q4_K_XL-00001-of-00003.gguf"))
    assert S.weights_size_gb(m) == pytest.approx(3000 / 1e9)


def test_weights_size_from_hf_spec_uses_catalog():
    m = ModelEntry(id="x", engine="llamacpp", model_path="unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL")
    assert S.weights_size_gb(m) == 17.0
    m2 = ModelEntry(id="x", engine="llamacpp", model_path="unsloth/gemma-4-26B-A4B-it-GGUF")
    assert S.weights_size_gb(m2) == 17.0  # default quant


def test_anthropic_translation_tool_choice_is_string():
    out = S._anthropic_to_openai({
        "model": "claude-sonnet-4-5", "max_tokens": 10,
        "system": [{"type": "text", "text": "sys"}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "dropped"},
                {"type": "tool_use", "id": "t1", "name": "f", "input": {"a": 1}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
        ],
        "tools": [{"name": "f", "description": "d", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "tool", "name": "f"},
        "stream": True,
    })
    assert out["tool_choice"] == "required"
    assert out["stream_options"] == {"include_usage": True}
    roles = [m["role"] for m in out["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]
    assert out["messages"][2]["tool_calls"][0]["function"]["arguments"] == json.dumps({"a": 1})
    assert out["tools"][0]["function"]["parameters"] == {"type": "object"}


def test_upstream_error_on_stream_is_a_real_http_error(monkeypatch):
    """Observed: llama-server answered 400 'request (53758 tokens) exceeds the
    available context size (32768 tokens)'; the gateway relayed those JSON bytes
    inside a 200 text/event-stream, and Hermes reported 'empty/malformed SSE'
    and retried nine times. The status and message must reach the client."""
    from fastapi.testclient import TestClient

    upstream = json.dumps({"error": {"code": 400, "message": "request (53758 tokens) exceeds the "
                                     "available context size (32768 tokens), try increasing it",
                                     "type": "exceed_context_size_error"}}).encode()

    class _Resp:
        status_code = 400
        async def aread(self): return upstream
        async def aclose(self): pass

    class _Client:
        async def aclose(self): pass

    async def fake_connect(url, body, timeout, headers):
        return _Client(), _Resp()
    monkeypatch.setattr(S, "_connect_stream", fake_connect)

    class _Eng:
        model = ModelEntry(id="m", engine="llamacpp", model_path="/tmp/x.gguf", aliases=["claude-sonnet-4-5"])
        base_url = "http://127.0.0.1:8081"
        process = None
        def is_running(self): return True
    gw = S.Gateway(GatewayConfig(models=[_Eng.model]))
    gw.runtime.engine = _Eng()
    gw.runtime.state = "ready"
    c = TestClient(gw.app)

    r = c.post("/v1/chat/completions", json={"model": "m", "stream": True, "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 400
    assert r.headers["content-type"].startswith("application/json")
    assert "exceeds the available context size" in r.json()["error"]["message"]

    r = c.post("/v1/messages", json={"model": "claude-sonnet-4-5", "stream": True, "max_tokens": 10,
                                     "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 400
    assert r.json()["type"] == "error" and "exceeds the available context size" in r.json()["error"]["message"]


@pytest.mark.parametrize("ending", ["complete", "consumer_close", "read_error"])
def test_anthropic_stream_real_httpx_response_lifecycle(ending):
    import asyncio
    import httpx

    class EngineStream(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            # Split an SSE record across transport chunks.
            yield b'da'
            yield b'ta: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}\n\n'
            if ending == "read_error":
                raise httpx.ReadError("upstream disconnected")
            yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":1}}\n\n'
            yield b'data: [DONE]\n\n'

        async def aclose(self):
            self.closed = True

    async def run():
        stream = EngineStream()
        client = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, stream=stream),
        ))
        response = await client.send(client.build_request("POST", "http://engine/v1/chat/completions"),
                                     stream=True)
        events = S._stream_anthropic_response(client, response, "glm-5.3-flash-warp")

        def parse(event):
            return json.loads(event.split(b"data: ", 1)[1])

        try:
            first = parse(await anext(events))
            assert first["type"] == "message_start"
            if ending == "consumer_close":
                await events.aclose()
            else:
                messages = [parse(event) async for event in events]
                text = "".join(m["delta"]["text"] for m in messages
                               if m["type"] == "content_block_delta")
                assert text == "Hi"
                if ending == "read_error":
                    assert messages[-1]["type"] == "error"
                    assert "upstream disconnected" in messages[-1]["error"]["message"]
                    assert not any(m["type"] == "message_stop" for m in messages)
                else:
                    assert [m["type"] for m in messages] == [
                        "content_block_start", "content_block_delta", "content_block_stop",
                        "message_delta", "message_stop",
                    ]
                    assert messages[-2]["delta"]["stop_reason"] == "end_turn"
                    assert messages[-2]["usage"] == {"input_tokens": 7, "output_tokens": 1}
            assert response.is_closed
            assert stream.closed
            assert client.is_closed
        finally:
            await events.aclose()
            await response.aclose()
            await client.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("code", [-15, 1])
def test_dead_engine_is_a_503_not_a_broken_200_stream(code):
    from types import SimpleNamespace
    import httpx

    model = ModelEntry(id="dead", engine="warp", model_path="/unused.waste")
    gateway = S.Gateway(GatewayConfig(models=[model]))
    gateway.runtime.engine = SimpleNamespace(
        model=model, base_url="http://127.0.0.1:1",
        process=SimpleNamespace(poll=lambda: code), is_running=lambda: False,
    )
    gateway.runtime.state = "ready"

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway.app),
                                     base_url="http://gateway") as client:
            response = await client.post("/v1/chat/completions", json={
                "model": "dead", "messages": [], "stream": True,
            })
        assert response.status_code == 503
        assert response.headers["content-type"].startswith("application/json")
        assert gateway.runtime.status()["active_model"] is None

    asyncio.run(scenario())


def test_openai_to_anthropic_response():
    resp = {"id": "chatcmpl-1", "choices": [{"finish_reason": "tool_calls", "message": {
        "content": "hello", "reasoning_content": "think",
        "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{\"a\": 1}"}}]}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4}}
    out = S._openai_to_anthropic(resp, "claude-sonnet-4-5")
    types = [b["type"] for b in out["content"]]
    assert types == ["thinking", "text", "tool_use"]
    assert out["content"][2]["input"] == {"a": 1}
    assert out["stop_reason"] == "tool_use"
    assert out["usage"] == {"input_tokens": 3, "output_tokens": 4}


# ---------------------------------------------------------------------------
# installer helpers
# ---------------------------------------------------------------------------

def _fake_warp_source(tmp_path, monkeypatch):
    import subprocess

    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=source, check=True)
    (source / "serve").mkdir()
    (source / "serve" / "__main__.py").write_text("")
    (source / "tools").mkdir()
    for script in ("fetch_weights.sh", "pipeline.sh"):
        (source / "tools" / script).write_text("#!/bin/sh\n")
    patch = tmp_path / "native.patch"
    patch.write_text(
        "diff --git a/native-prefill-ready b/native-prefill-ready\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/native-prefill-ready\n"
        "@@ -0,0 +1 @@\n"
        "+optimized\n"
    )
    monkeypatch.setattr(I, "WARP_PATCH", patch)
    (source / "Makefile").write_text(
        "all:\n"
        "\ttest -f native-prefill-ready\n"
        "\tprintf '#!/bin/sh\\nexit 0\\n' > waste\n"
        "\tchmod +x waste\n"
        "\tcp waste waste.exe\n"
        "\ttouch libwaste.so libwaste.dylib libwaste.dll\n"
        "check: all\n"
        "\ttouch check-ran\n"
    )
    subprocess.run(["git", "add", "."], cwd=source, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=source, check=True)
    ref = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    return source, ref


def test_install_warp_builds_and_verifies_pinned_source(tmp_path, monkeypatch):
    source, ref = _fake_warp_source(tmp_path, monkeypatch)
    prefix = tmp_path / "prefix"
    monkeypatch.setattr(I, "WARP_REPO", str(source))
    root = I.install_warp(prefix, ref=ref)

    assert root == prefix / "lib" / "warp"
    assert (root / "serve" / "__main__.py").is_file()
    assert (root / "check-ran").is_file()
    assert (prefix / "bin" / "waste").resolve() == root / "waste"


@pytest.mark.parametrize("marker", [None, "stale-patch"])
def test_install_warp_rebuilds_an_unpatched_or_stale_runtime(tmp_path, monkeypatch, marker):
    source, ref = _fake_warp_source(tmp_path, monkeypatch)
    monkeypatch.setattr(I, "WARP_REPO", str(source))
    monkeypatch.setattr(I, "WARP_COMMIT", ref)
    prefix = tmp_path / "prefix"
    root = I.install_warp(prefix, ref=ref)
    stamp = root / ".litmoe-patch-sha256"
    if marker is None:
        stamp.unlink()
    else:
        stamp.write_text(marker)
    (root / "native-prefill-ready").write_text("old native implementation\n")

    rebuilt = I.install_warp(prefix, ref=ref)

    assert (rebuilt / "native-prefill-ready").read_text() == "optimized\n"
    assert I._installed_warp_root(prefix) == rebuilt


def test_install_warp_patch_failure_preserves_previous_runtime(tmp_path, monkeypatch):
    source, ref = _fake_warp_source(tmp_path, monkeypatch)
    monkeypatch.setattr(I, "WARP_REPO", str(source))
    monkeypatch.setattr(I, "WARP_COMMIT", ref)
    prefix = tmp_path / "prefix"
    root = I.install_warp(prefix, ref=ref)
    stamp = (root / ".litmoe-patch-sha256").read_text()
    I.WARP_PATCH.write_text("not an applicable patch\n")

    with pytest.raises(RuntimeError, match="WARP native prefill patch failed"):
        I.install_warp(prefix, ref=ref)

    assert (root / "native-prefill-ready").read_text() == "optimized\n"
    assert (root / ".litmoe-patch-sha256").read_text() == stamp
    assert (prefix / "bin" / "waste").resolve() == root / "waste"


def test_install_warp_build_subprocesses_do_not_inherit_hf_token(
    tmp_path, monkeypatch,
):
    source, ref = _fake_warp_source(tmp_path, monkeypatch)
    prefix = tmp_path / "prefix"
    token = "runtime-build-secret"
    calls = []
    monkeypatch.setattr(I, "WARP_REPO", str(source))
    monkeypatch.setattr(I.shutil, "which", lambda tool: f"/fake/bin/{tool}")
    monkeypatch.setenv("HF_TOKEN", token)

    def fake_run(args, **kwargs):
        command = list(args)
        calls.append((command, kwargs))
        if command[:3] == ["git", "clone", "--no-checkout"]:
            shutil.copytree(source, Path(command[-1]))
        elif command == ["make"]:
            checkout = Path(kwargs["cwd"])
            (checkout / "waste").write_bytes(b"new launcher")
            (checkout / "waste.exe").write_bytes(b"new launcher")
            for library in ("libwaste.so", "libwaste.dylib", "libwaste.dll"):
                (checkout / library).write_bytes(b"new library")
        return I.subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(I.subprocess, "run", fake_run)

    I.install_warp(prefix, ref=ref)

    assert all(isinstance(kwargs.get("env"), dict) for _, kwargs in calls)
    assert all("HF_TOKEN" not in kwargs["env"] for _, kwargs in calls)


def test_install_warp_copies_discoverable_windows_launcher(tmp_path, monkeypatch):
    source, ref = _fake_warp_source(tmp_path, monkeypatch)
    prefix = tmp_path / "prefix"
    monkeypatch.setattr(I, "WARP_REPO", str(source))
    monkeypatch.setattr(I.shutil, "which", lambda tool: f"/usr/bin/{tool}")
    monkeypatch.setattr(I.sys, "platform", "win32")

    root = I.install_warp(prefix, ref=ref)

    launcher = prefix / "bin" / "waste.exe"
    assert launcher.is_file()
    assert not launcher.is_symlink()
    assert launcher.read_bytes() == (root / "waste.exe").read_bytes()


@pytest.mark.parametrize("cutover", ["tree", "launcher"])
def test_install_warp_keyboard_interrupt_restores_windows_runtime_and_launcher(
    tmp_path, monkeypatch, cutover,
):
    source, ref = _fake_warp_source(tmp_path, monkeypatch)
    prefix = tmp_path / "prefix"
    previous = prefix / "lib" / "warp"
    previous.mkdir(parents=True)
    (previous / "runtime-version").write_text("old runtime")
    launcher = prefix / "bin" / "waste.exe"
    launcher.parent.mkdir(parents=True)
    launcher.write_bytes(b"old launcher")
    real_replace = I.os.replace
    interrupted = False

    def interrupt_cutover(source_path, destination_path):
        nonlocal interrupted
        source_path = Path(source_path)
        destination_path = Path(destination_path)
        tree_cutover = source_path.name == "checkout" and destination_path == previous
        launcher_cutover = (
            source_path.name == "waste-link" and destination_path == launcher
        )
        if not interrupted and (
            (cutover == "tree" and tree_cutover)
            or (cutover == "launcher" and launcher_cutover)
        ):
            interrupted = True
            raise KeyboardInterrupt
        return real_replace(source_path, destination_path)

    monkeypatch.setattr(I, "WARP_REPO", str(source))
    monkeypatch.setattr(I.shutil, "which", lambda tool: f"/fake/bin/{tool}")
    monkeypatch.setattr(I.sys, "platform", "win32")
    monkeypatch.setattr(I.os, "replace", interrupt_cutover)

    with pytest.raises(KeyboardInterrupt):
        I.install_warp(prefix, ref=ref)

    assert interrupted
    assert previous.is_dir()
    assert (previous / "runtime-version").is_file()
    assert (previous / "runtime-version").read_text() == "old runtime"
    assert not (previous / "serve").exists()
    assert launcher.read_bytes() == b"old launcher"
    assert not launcher.is_symlink()


def test_install_warp_restores_previous_tree_when_launcher_install_fails(tmp_path, monkeypatch):
    source, ref = _fake_warp_source(tmp_path, monkeypatch)
    prefix = tmp_path / "prefix"
    previous = prefix / "lib" / "warp"
    previous.mkdir(parents=True)
    (previous / "marker").write_text("working")
    wrapper = prefix / "bin" / "waste"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text("old launcher")

    real_replace = I.os.replace

    def fail_launcher_replace(source_path, destination_path):
        if Path(source_path).name == "waste-link":
            raise OSError("simulated launcher failure")
        return real_replace(source_path, destination_path)

    monkeypatch.setattr(I, "WARP_REPO", str(source))
    monkeypatch.setattr(I.os, "replace", fail_launcher_replace)

    with pytest.raises(RuntimeError, match="could not install WARP CLI"):
        I.install_warp(prefix, ref=ref)

    assert (previous / "marker").read_text() == "working"
    assert wrapper.read_text() == "old launcher"

def test_install_warp_preserves_recovery_tree_when_rollback_fails(tmp_path, monkeypatch):
    source, ref = _fake_warp_source(tmp_path, monkeypatch)
    prefix = tmp_path / "prefix"
    previous = prefix / "lib" / "warp"
    previous.mkdir(parents=True)
    (previous / "marker").write_text("working")
    wrapper = prefix / "bin" / "waste"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text("old launcher")

    real_replace = I.os.replace

    def fail_launcher_and_restore(source_path, destination_path):
        source_name = Path(source_path).name
        if source_name == "waste-link":
            raise OSError("simulated launcher failure")
        if source_name == "previous":
            raise OSError("simulated restore failure")
        return real_replace(source_path, destination_path)

    monkeypatch.setattr(I, "WARP_REPO", str(source))
    monkeypatch.setattr(I.os, "replace", fail_launcher_and_restore)

    with pytest.raises(RuntimeError, match="rollback also failed") as exc_info:
        I.install_warp(prefix, ref=ref)

    recovery_markers = list(
        (prefix / "lib").glob(".warp-install-*/previous/marker")
    )
    assert len(recovery_markers) == 1
    assert recovery_markers[0].read_text() == "working"
    assert str(recovery_markers[0].parents[1]) in str(exc_info.value)



def test_install_cli_selects_warp_without_downloading_model(tmp_path, monkeypatch):
    from click.testing import CliRunner

    installed = []

    def unexpected_call(*args, **kwargs):
        raise AssertionError("WARP-only installation called an unrelated installer or downloader")

    monkeypatch.setattr(I, "install_warp", lambda prefix, **_: installed.append(prefix))
    monkeypatch.setattr(I, "install_llamacpp", unexpected_call)
    monkeypatch.setattr(I, "install_ktransformers", unexpected_call)
    monkeypatch.setattr(I, "download_model", unexpected_call)
    monkeypatch.setattr(I, "get_total_memory_bytes", lambda: None)

    result = CliRunner().invoke(
        I.install_cmd,
        ["warp", "--prefix", str(tmp_path / "prefix"), "--yes"],
    )
    assert result.exit_code == 0, result.output
    assert installed == [tmp_path / "prefix"]
    assert "Installing WARP" in result.output


def test_doctor_reports_warp_engine(monkeypatch):
    from click.testing import CliRunner
    import litmoe.cli.main as CM

    monkeypatch.setattr(CM, "llama_installed", lambda: False)
    monkeypatch.setattr(CM, "kt_installed", lambda: False)
    monkeypatch.setattr(CM, "warp_installed", lambda: True)
    result = CliRunner().invoke(CM.cli, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "WARP: installed" in result.output
    assert "local .waste containers" in result.output

    monkeypatch.setattr(CM, "warp_installed", lambda: False)
    result = CliRunner().invoke(CM.cli, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "WARP: NOT installed (litmoe install --engine warp)" in result.output

GEMMA_31B_FILES = [
    "gemma-4-31B-it-Q4_K_M.gguf", "gemma-4-31B-it-UD-Q4_K_M.gguf", "gemma-4-31B-it-UD-Q4_K_XL.gguf",
    "gemma-4-31B-it-Q4_K_S.gguf", "mmproj-F16.gguf", "mmproj-BF16.gguf", "imatrix_unsloth.gguf",
    "BF16/gemma-4-31B-it-BF16-00001-of-00002.gguf", "BF16/gemma-4-31B-it-BF16-00002-of-00002.gguf",
    "README.md", "config.json",
]
KIMI_FILES = [
    "UD-IQ1_S/Kimi-K3-UD-IQ1_S-00001-of-00013.gguf", "UD-IQ1_S/Kimi-K3-UD-IQ1_S-00002-of-00013.gguf",
    "UD-IQ1_M/Kimi-K3-UD-IQ1_M-00001-of-00014.gguf", "MTP/mtp-Q8_0.gguf", "dspark/Q8_0.gguf",
]
MRADERMACHER_FILES = ["Kimi-Linear-48B-A3B-Instruct.Q4_K_M.gguf", "Kimi-Linear-48B-A3B-Instruct.Q4_K_S.gguf",
                      "Kimi-Linear-48B-A3B-Instruct.IQ4_XS.gguf"]


def test_select_gguf_files_root_layout_exact_quant():
    assert I.select_gguf_files(GEMMA_31B_FILES, "Q4_K_M") == ["gemma-4-31B-it-Q4_K_M.gguf"]
    assert I.select_gguf_files(GEMMA_31B_FILES, "UD-Q4_K_M") == ["gemma-4-31B-it-UD-Q4_K_M.gguf"]
    assert I.select_gguf_files(GEMMA_31B_FILES, "UD-Q4_K_XL") == ["gemma-4-31B-it-UD-Q4_K_XL.gguf"]
    assert I.select_gguf_files(GEMMA_31B_FILES, "Q4_K_S") == ["gemma-4-31B-it-Q4_K_S.gguf"]
    assert I.select_gguf_files(GEMMA_31B_FILES, "BF16") == [
        "BF16/gemma-4-31B-it-BF16-00001-of-00002.gguf", "BF16/gemma-4-31B-it-BF16-00002-of-00002.gguf"]
    assert I.select_gguf_files(GEMMA_31B_FILES, "Q8_0") == []


def test_select_gguf_files_subdir_layout_and_exclusions():
    assert I.select_gguf_files(KIMI_FILES, "UD-IQ1_S") == [
        "UD-IQ1_S/Kimi-K3-UD-IQ1_S-00001-of-00013.gguf", "UD-IQ1_S/Kimi-K3-UD-IQ1_S-00002-of-00013.gguf"]
    assert I.select_gguf_files(KIMI_FILES, "Q8_0") == []  # MTP/ and dspark/ are not quants
    assert I.select_gguf_files(MRADERMACHER_FILES, "Q4_K_M") == ["Kimi-Linear-48B-A3B-Instruct.Q4_K_M.gguf"]


def test_select_mmproj_prefers_f16():
    assert I.select_mmproj_file(GEMMA_31B_FILES) == "mmproj-F16.gguf"
    assert I.select_mmproj_file(KIMI_FILES) is None


class _FakeResp:
    def __init__(self, data, text=""):
        self._data, self.text, self.status_code = data, text, 200

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


class _FakeClient:
    """Mimics the live GitHub state: latest=v0.4.1 with only nightly-tag.txt."""

    def __init__(self, nightly_has_asset=True):
        self.calls = []
        self.nightly_has_asset = nightly_has_asset

    def get(self, url, headers=None):
        self.calls.append(url)
        base = I.LLAMA_RELEASES_API
        if url == f"{base}/latest":
            return _FakeResp({"tag_name": "v0.4.1", "assets": [
                {"name": "nightly-tag.txt", "browser_download_url": "https://x/nightly-tag.txt"}]})
        if url == "https://x/nightly-tag.txt":
            return _FakeResp(None, text="b10964\n")
        if url == f"{base}/tags/b10964":
            assets = [{"name": "llama-b10964-bin-ubuntu-x64.tar.gz", "size": 1}] if self.nightly_has_asset else []
            return _FakeResp({"tag_name": "b10964", "assets": assets})
        if url == f"{base}/tags/b11005":
            return _FakeResp({"tag_name": "b11005", "assets": [
                {"name": "llama-b11005-bin-ubuntu-cuda-12.8-x64.tar.gz", "size": 1},
                {"name": "cudart-llama-b11005-bin-ubuntu-cuda-12.8-x64.tar.gz", "size": 1}]})
        if url.startswith(f"{base}?per_page="):
            return _FakeResp([
                {"tag_name": "b11006", "assets": [{"name": "llama-b11006-bin-macos-arm64.tar.gz"}]},  # incomplete
                {"tag_name": "b11005", "assets": [{"name": "llama-b11005-bin-ubuntu-x64.tar.gz"},
                                                  {"name": "llama-b11005-bin-ubuntu-vulkan-x64.tar.gz"}]},
            ])
        raise AssertionError(f"unexpected url {url}")


def test_resolve_release_follows_nightly_pointer():
    c = _FakeClient()
    rel = I.resolve_llamacpp_release("bin-ubuntu-x64", c)
    assert rel["tag_name"] == "b10964"


def test_resolve_release_scans_when_pointer_lacks_asset():
    c = _FakeClient(nightly_has_asset=False)
    rel = I.resolve_llamacpp_release("bin-ubuntu-x64", c)
    assert rel["tag_name"] == "b11005"  # b11006 skipped: no ubuntu-x64 asset yet


def test_resolve_release_pinned_tag_and_cudart():
    c = _FakeClient()
    rel = I.resolve_llamacpp_release("bin-ubuntu-cuda-12.8-x64", c, tag="b11005")
    assert rel["tag_name"] == "b11005"
    assert I._cudart_asset(rel, "bin-ubuntu-cuda-12.8-x64")["name"].startswith("cudart-")
    # substring must not match the vulkan variant
    assert I._asset_named({"assets": [{"name": "llama-b1-bin-ubuntu-vulkan-x64.tar.gz"}]}, "bin-ubuntu-x64") is None


def test_choose_quant_downgrades_to_what_fits():
    """`litmoe install --model X` must pick the best quant for THIS machine, not
    blindly take the tier default. Observed: the 192 GB-tier default (111 GB)
    was downloaded onto a 103 GB Mac after only a yes/no prompt."""
    info = M.KNOWN_MODELS["qwen3.8-flash-next"]
    # Plenty of RAM: the catalog default.
    assert I.choose_quant("qwen3.8-flash-next", None, 192.0) == info["default_quant"]
    # 103 GB Mac -> 77 GB Metal budget: nothing fits at 32K ctx, so the smallest quant.
    assert I.choose_quant("qwen3.8-flash-next", None, 77.0) == "UD-IQ1_S"
    # A budget where a mid quant fits: same answer `litmoe models` prints.
    assert I.choose_quant("qwen3.8-flash-next", None, 100.0) == M.largest_quant_that_fits("qwen3.8-flash-next", 100.0)
    # Explicit --quant always wins, even when it does not fit.
    assert I.choose_quant("qwen3.8-flash-next", "UD-Q4_K_XL", 77.0) == "UD-Q4_K_XL"
    # Unknown quant is rejected rather than silently substituted.
    with pytest.raises(Exception):
        I.choose_quant("qwen3.8-flash-next", "Q4_NOPE", 77.0)
    # No RAM info: fall back to the default rather than guessing.
    assert I.choose_quant("qwen3.8-flash-next", None, None) == info["default_quant"]
    # ktransformers entries have no GGUF quant.
    assert I.choose_quant("glm-5.3-flash", None, 512.0) is None


def test_add_model_to_config_writes_kt_fields(tmp_path):
    cfg = tmp_path / "models.yaml"
    I.add_model_to_config("glm-5.3-flash", "ktransformers", tmp_path / "glm", 131072, cfg,
                          extra_args=["--tool-call-parser", "glm47"], kt_method="FP8")
    loaded = load_config(cfg)
    kt = next(m for m in loaded.models if m.id == "glm-5.3-flash")
    assert kt.kt_method == "FP8" and kt.kt_num_gpu_experts == 0 and kt.extra_args == ["--tool-call-parser", "glm47"]
    # the first model in a fresh file gets the Claude aliases so Claude Code routes
    assert list(kt.aliases) == list(M.CLAUDE_ALIASES)

    I.add_model_to_config("g", "llamacpp", tmp_path / "g.gguf", 65536, cfg, aliases=["x-alias"])
    loaded = load_config(cfg)
    g = next(m for m in loaded.models if m.id == "g")
    assert g.n_gpu_layers == -1 and g.aliases == ["x-alias"]
    # a second model without explicit aliases does NOT steal them (they stay on the first)
    I.add_model_to_config("h", "llamacpp", tmp_path / "h.gguf", 65536, cfg)
    loaded = load_config(cfg)
    assert next(m for m in loaded.models if m.id == "h").aliases == []
    assert "claude-haiku-4-5" in next(m for m in loaded.models if m.id == "glm-5.3-flash").aliases
    # re-adding keeps existing aliases
    I.add_model_to_config("g", "llamacpp", tmp_path / "g2.gguf", 65536, cfg)
    assert next(m for m in load_config(cfg).models if m.id == "g").aliases == ["x-alias"]


def test_init_picks_fast_defaults_per_ram(tmp_path, monkeypatch):
    """`litmoe init` must never default a 48/96 GB laptop to a server-tier model."""
    from click.testing import CliRunner
    import litmoe.cli.main as CM

    monkeypatch.chdir(tmp_path)
    for gb, must_lead, must_not in [
        (48, "gemma-4-26b-a4b", {"kimi-k3", "qwen3.8", "minimax-m3", "deepseek-v4-flash", "gpt-oss-120b"}),
        (96, "gemma-4-26b-a4b", {"kimi-k3", "qwen3.8", "minimax-m3", "deepseek-v4-flash"}),
        (16, None, {"gemma-4-26b-a4b", "qwen3.6-35b-a3b"}),
    ]:
        monkeypatch.setattr(CM, "get_total_memory_bytes", lambda gb=gb: gb * 1024**3)
        monkeypatch.setattr(CM, "is_macos", lambda: False)
        (tmp_path / "models.yaml").unlink(missing_ok=True)
        r = CliRunner().invoke(CM.cli, ["init"])
        assert r.exit_code == 0, r.output
        cfg = load_config(tmp_path / "models.yaml")
        ids = [m.id for m in cfg.models]
        assert ids, r.output
        if must_lead:
            assert ids[0] == must_lead, (gb, ids)
        assert not (set(ids) & must_not), (gb, ids)
        assert all(m.n_ctx == 0 for m in cfg.models)          # memory-aware native ctx
        assert list(cfg.models[0].aliases) == list(M.CLAUDE_ALIASES)
        for m in cfg.models:                                    # laptop tiers must be fast: small active params
            assert (M.KNOWN_MODELS[m.id].get("active_b") or 0) <= (12 if gb <= 96 else 1e9), (gb, m.id)


def test_serve_rejects_multiple_initial_models_before_starting(monkeypatch):
    from click.testing import CliRunner
    from litmoe.cli.main import cli

    monkeypatch.setattr(S, "run", lambda *args, **kwargs: pytest.fail("multiple engines were requested"))
    result = CliRunner().invoke(cli, ["serve", "one", "two"])
    assert result.exit_code == 2


# ---------------------------------------------------------------------------
# catalog-installable WARP models
# ---------------------------------------------------------------------------

WARP_CATALOG_MODELS = {
    "glm-5.3-flash-warp": {
        "repo": "zai-org/GLM-5.3-Flash",
        "revision": "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
        "profile": "glm",
        "arch": "glm5-next",
        "source_gib": 306,
        "output_gb": 112,
        "output_bytes": 112_000_000_000,
        "output_workspace_gib": 120,
        "native_ctx": 1_048_576,
        "active_b": 17.31,
        "params": "313.89B total, 17.31B active per token, WARP container",
        "tier": M.TIER_LAPTOP_48,
    },
    "deepseek-v4.1-flash-warp": {
        "repo": "deepseek-ai/DeepSeek-V4.1-Flash",
        "revision": "dba1be0a40aa45a94ad051997016db3960a90277",
        "profile": "ds41",
        "arch": "deepseek-v41",
        "source_gib": 475,
        "output_gb": 299,
        "output_bytes": 299 * 1024**3,
        "output_workspace_gib": 310,
        "native_ctx": 1_048_576,
        "active_b": 16.62,
        "params": (
            "552.37B backbone + 197B Engram/n-gram memory, "
            "16.62B active per token, WARP container"
        ),
        "tier": M.TIER_LAPTOP_48,
    },
}


def _fake_installable_warp_root(tmp_path: Path) -> Path:
    root = _fake_warp_root(tmp_path)
    tools = root / "tools"
    tools.mkdir()
    (tools / "fetch_weights.sh").write_text("#!/usr/bin/env bash\n")
    (tools / "pipeline.sh").write_text("#!/usr/bin/env bash\n")
    return root


def _fake_waste_container(
    path: Path,
    model_id: str,
    *,
    include_chat: bool | None = None,
) -> Path:
    """Write a tiny container with the pinned WARP v0 on-disk contract."""
    expected = WARP_CATALOG_MODELS[model_id]
    layer_number = 3 if expected["arch"] == "glm5-next" else 0
    layer_file = f"experts-L{layer_number}.bin"
    trunk_data = b"\0" * 16
    layer_data = b"\1" * 32
    manifest = {
        "format_version": 0,
        "arch": expected["arch"],
        "tensor_prefix": "model.",
        "config": {"max_position_embeddings": expected["native_ctx"]},
        "expert_quant": {
            "fmt": "VQ3R",
            "stages": 3,
            "vec_dim": 128,
            "entries": 256,
            "index_block": 256,
            "index_bits": 8,
            "bits_per_weight": 0.1875,
        },
        "layers": {
            str(layer_number): {
                "file": layer_file,
                "experts": 288 if expected["arch"] == "glm5-next" else 384,
                "bytes": len(layer_data),
                "codebook_base": 0,
            },
        },
        "trunk": [{
            "name": "embed_tokens.weight",
            "fmt": 1,
            "off": 0,
            "shape": [4],
            "bytes": len(trunk_data),
        }],
    }
    path.mkdir(parents=True, exist_ok=True)
    (path / "manifest.json").write_text(json.dumps(manifest))
    (path / "trunk.bin").write_bytes(trunk_data)
    (path / layer_file).write_bytes(layer_data)
    (path / "codebooks.bin").write_bytes(b"\2" * 16)
    (path / "tokenizer.model").write_bytes(b"tiny tokenizer metadata")
    (path / "specials.json").write_text("{}")
    if include_chat is None:
        include_chat = expected["arch"] == "glm5-next"
    if include_chat:
        (path / "chat.json").write_text("{}")
    return path


def _assert_private_curl_auth(args, child_env, token: str) -> Path:
    assert "HF_TOKEN" not in child_env
    assert all(token not in str(value) for value in child_env.values())
    assert token not in " ".join(map(str, args))
    curl_home = Path(child_env["CURL_HOME"])
    curl_config = curl_home / ".curlrc"
    assert curl_home.is_dir()
    assert curl_home.stat().st_mode & 0o777 == 0o700
    assert curl_config.is_file()
    assert curl_config.stat().st_mode & 0o777 == 0o600
    return curl_home

class _FakeStageProcess:
    """Popen-shaped stand-in that runs a fake stage in a worker thread."""

    def __init__(self, args, kwargs, fake_run):
        import threading as _threading

        self.args = list(args)
        self.pid = os.getpid()
        self.returncode = None
        self.stdout = None
        self.stderr = None
        self.error = None
        self._done = _threading.Event()

        def work():
            try:
                result = fake_run(list(args), **kwargs)
                self.stdout = getattr(result, "stdout", None)
                self.stderr = getattr(result, "stderr", None)
                self.returncode = result.returncode
            except BaseException as exc:
                self.error = exc
                self.returncode = -1
            finally:
                self._done.set()

        _threading.Thread(target=work, daemon=True).start()

    def poll(self):
        if self._done.is_set() and self.error is not None:
            raise self.error
        return self.returncode

    def communicate(self):
        self._done.wait()
        if self.error is not None:
            raise self.error
        return self.stdout or "", self.stderr or ""

    def wait(self, timeout=None):
        self._done.wait(timeout)
        return self.returncode

    def terminate(self):
        pass

    def kill(self):
        pass


def _patch_stage_popen(monkeypatch, fake_run):
    """Route _run_warp_stage's Popen through *fake_run* with fast heartbeats."""
    monkeypatch.setattr(I, "LONG_QUIET_SECONDS", 0.05, raising=False)
    monkeypatch.setattr(
        I.subprocess,
        "Popen",
        lambda args, **kwargs: _FakeStageProcess(args, kwargs, fake_run),
    )


def _patch_warp_model_prerequisites(monkeypatch):
    monkeypatch.setattr(
        I.shutil,
        "which",
        lambda command: f"/fake/bin/{command}",
    )
    monkeypatch.setattr(
        I,
        "_preflight_warp_disk",
        lambda *args, **kwargs: None,
        raising=False,
    )


def test_warp_catalog_entries_are_installable_and_valid():
    assert M.WASTE == "waste"
    for model_id, expected in WARP_CATALOG_MODELS.items():
        info = M.lookup(model_id)
        assert info is M.KNOWN_MODELS[model_id]
        assert info["engine"] == "warp"
        assert info["format"] == M.WASTE
        assert info["hf_repo"] == expected["repo"]
        assert info["hf_revision"] == expected["revision"]
        assert info["warp_profile"] == expected["profile"]
        assert info["arch"] == expected["arch"]
        assert info["source_size_gib"] == expected["source_gib"]
        assert info["size_gb"] == expected["output_gb"]
        assert info["output_size_bytes"] == expected["output_bytes"]
        assert info["output_workspace_gib"] == expected["output_workspace_gib"]
        assert info["native_ctx"] == expected["native_ctx"]
        assert info["active_b"] == expected["active_b"]
        assert info["params"] == expected["params"]
        assert info["tier"] == expected["tier"]
        assert M.quant_size_gb(model_id, None) == expected["output_gb"]
        assert "quants" not in info and "default_quant" not in info
        assert model_id not in M.gguf_models()
        assert model_id not in M.kt_models()
    assert not set(WARP_CATALOG_MODELS) & set(M.recommended_for_ram(2048))
    assert M.validate_catalog() == []


@pytest.mark.parametrize("field", [
    "arch",
    "native_ctx",
    "active_b",
    "tier",
    "output_workspace_gib",
])
def test_warp_catalog_validation_rejects_missing_runtime_metadata(
    monkeypatch, field,
):
    model_id = "glm-5.3-flash-warp"
    invalid = dict(M.KNOWN_MODELS[model_id])
    invalid.pop(field)
    monkeypatch.setitem(M.KNOWN_MODELS, model_id, invalid)

    problems = "\n".join(M.validate_catalog())
    assert model_id in problems
    assert field in problems


@pytest.mark.parametrize("field,bad_value", [
    ("hf_revision", "main"),
    ("warp_profile", ""),
    ("arch", ""),
    ("source_size_gib", 0),
    ("size_gb", 0),
    ("output_size_bytes", 0),
    ("output_workspace_gib", 0),
    ("native_ctx", 0),
    ("active_b", 0),
    ("tier", -1),
])
def test_warp_catalog_validation_rejects_invalid_metadata(
    monkeypatch, field, bad_value,
):
    model_id = "glm-5.3-flash-warp"
    invalid = dict(M.KNOWN_MODELS[model_id])
    invalid[field] = bad_value
    monkeypatch.setitem(M.KNOWN_MODELS, model_id, invalid)

    problems = "\n".join(M.validate_catalog())
    assert model_id in problems
    assert field in problems


@pytest.mark.parametrize("model_id", WARP_CATALOG_MODELS)
def test_warp_container_accepts_real_v0_manifest_schema(model_id, tmp_path):
    container = _fake_waste_container(tmp_path / f"{model_id}.waste", model_id)

    I._validate_warp_container(container, model_id)

    manifest = json.loads((container / "manifest.json").read_text())
    assert manifest["format_version"] == 0
    assert manifest["arch"] == WARP_CATALOG_MODELS[model_id]["arch"]
    assert isinstance(manifest["trunk"], list) and manifest["trunk"]
    assert isinstance(manifest["layers"], dict) and manifest["layers"]
    if model_id == "deepseek-v4.1-flash-warp":
        assert not (container / "chat.json").exists()


@pytest.mark.parametrize("case,expected_detail", [
    ("wrong-format", "format"),
    ("wrong-arch", "arch"),
    ("malformed-trunk", "trunk"),
    ("empty-trunk", "trunk"),
    ("empty-layers", "layer"),
    ("nonnumeric-layer", "layer"),
    ("malformed-layer", "codebook_base"),
    ("wrong-layer-name", "experts-"),
    ("missing-layer-file", "experts-"),
])
def test_warp_container_rejects_invalid_v0_manifest(
    tmp_path, case, expected_detail,
):
    model_id = "glm-5.3-flash-warp"
    container = _fake_waste_container(tmp_path / "model.waste", model_id)
    manifest_path = container / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    layer_number = next(iter(manifest["layers"]))

    if case == "wrong-format":
        manifest["format_version"] = 1
    elif case == "wrong-arch":
        manifest["arch"] = "deepseek-v41"
    elif case == "malformed-trunk":
        manifest["trunk"] = "trunk.bin"
    elif case == "empty-trunk":
        manifest["trunk"] = []
    elif case == "empty-layers":
        manifest["layers"] = {}
    elif case == "nonnumeric-layer":
        manifest["layers"]["not-a-layer"] = manifest["layers"].pop(layer_number)
    elif case == "malformed-layer":
        manifest["layers"][layer_number].pop("codebook_base")
    elif case == "wrong-layer-name":
        old_name = manifest["layers"][layer_number]["file"]
        new_name = "renamed-expert-bank.bin"
        (container / old_name).rename(container / new_name)
        manifest["layers"][layer_number]["file"] = new_name
    elif case == "missing-layer-file":
        (container / manifest["layers"][layer_number]["file"]).unlink()
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError) as raised:
        I._validate_warp_container(container, model_id)
    assert "invalid WARP container" in str(raised.value)
    assert expected_detail in str(raised.value).lower()


@pytest.mark.parametrize("filename", [
    "trunk.bin",
    "codebooks.bin",
    "tokenizer.model",
    "specials.json",
])
def test_warp_container_rejects_missing_core_file(tmp_path, filename):
    model_id = "glm-5.3-flash-warp"
    container = _fake_waste_container(tmp_path / "model.waste", model_id)
    (container / filename).unlink()

    with pytest.raises(RuntimeError) as raised:
        I._validate_warp_container(container, model_id)
    assert filename in str(raised.value)


@pytest.mark.parametrize("missing_tool", ["uv", "curl"])
def test_install_warp_model_checks_dependencies_before_creating_paths(
    tmp_path, monkeypatch, missing_tool,
):
    root = _fake_installable_warp_root(tmp_path / "runtime")
    staging_dir = tmp_path / "staging"
    models_dir = tmp_path / "models"
    commands = []

    monkeypatch.setattr(
        I.shutil,
        "which",
        lambda command: None if command == missing_tool else f"/fake/bin/{command}",
    )
    monkeypatch.setattr(
        I.subprocess,
        "run",
        lambda *args, **kwargs: commands.append((args, kwargs)),
    )

    with pytest.raises(RuntimeError, match=rf"\b{missing_tool}\b"):
        I.install_warp_model(
            "glm-5.3-flash-warp",
            warp_root=root,
            staging_dir=staging_dir,
            models_dir=models_dir,
        )

    assert commands == []
    assert not staging_dir.exists()
    assert not models_dir.exists()


def test_install_warp_model_uses_deterministic_paths_and_pinned_pipeline_environment(
    tmp_path, monkeypatch, capsys,
):
    model_id = "deepseek-v4.1-flash-warp"
    token = "secret-test-token"
    root = _fake_installable_warp_root(tmp_path / "runtime")
    staging_dir = tmp_path / "external-staging"
    models_dir = tmp_path / "internal-models"
    ambient_curl_home = tmp_path / "ambient-curl-home"
    calls = []
    curl_homes = []
    _patch_warp_model_prerequisites(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", token)
    monkeypatch.setenv("CURL_HOME", str(ambient_curl_home))

    def fake_run(args, **kwargs):
        child_env = dict(kwargs.get("env") or {})
        curl_homes.append(_assert_private_curl_auth(args, child_env, token))
        calls.append((list(args), kwargs.get("cwd"), child_env))
        if any(Path(str(arg)).name == "pipeline.sh" for arg in args):
            _fake_waste_container(Path(child_env["OUT"]), model_id)
        return I.subprocess.CompletedProcess(args, 0, "", "")

    _patch_stage_popen(monkeypatch, fake_run)

    result = I.install_warp_model(
        model_id,
        warp_root=root,
        staging_dir=staging_dir,
        models_dir=models_dir,
        jobs=5,
        reclaim_source=True,
    )

    expected_source = (staging_dir / model_id).absolute()
    expected_output = (models_dir / f"{model_id}.waste").absolute()
    assert result == expected_output
    assert len(calls) == 2
    assert len(set(curl_homes)) == 1
    assert curl_homes[0] != ambient_curl_home

    fetch, fetch_cwd, fetch_env = calls[0]
    info = M.KNOWN_MODELS[model_id]
    assert Path(fetch[0]).name == "bash"
    assert Path(fetch[1]) == root / "tools" / "fetch_weights.sh"
    assert fetch[2:] == []
    assert fetch_env["REPO"] == info["hf_repo"]
    assert fetch_env["REVISION"] == info["hf_revision"]
    assert Path(fetch_env["DEST"]) == expected_source
    assert fetch_env["JOBS"] == "5"
    assert Path(fetch_cwd) == root

    pipeline, pipeline_cwd, pipeline_env = calls[1]
    assert Path(pipeline[0]).name == "bash"
    assert Path(pipeline[1]) == root / "tools" / "pipeline.sh"
    assert Path(pipeline_cwd) == root
    assert pipeline_env["MODEL"] == "ds41"
    assert pipeline_env["REPO"] == info["hf_repo"]
    assert pipeline_env["REVISION"] == info["hf_revision"]
    assert Path(pipeline_env["SRC"]) == expected_source
    assert Path(pipeline_env["OUT"]) == expected_output
    assert pipeline_env["JOBS"] == "5"
    assert pipeline_env["RECLAIM"] == "on"
    assert pipeline_env["MIN_FREE_GB"] == "310"
    assert "HF_TOKEN" not in fetch_env
    assert "HF_TOKEN" not in pipeline_env
    assert Path(fetch_env["CURL_HOME"]) == curl_homes[0]
    assert Path(pipeline_env["CURL_HOME"]) == curl_homes[0]
    captured = capsys.readouterr()
    assert token not in captured.out + captured.err
    assert all(not curl_home.exists() for curl_home in curl_homes)

    manifest = json.loads((result / "manifest.json").read_text())
    assert manifest["format_version"] == 0
    assert manifest["arch"] == "deepseek-v41"
    assert (result / "trunk.bin").is_file()
    assert (result / "tokenizer.model").is_file()
    assert list(result.glob("experts-L*.bin"))
    assert not (result / "chat.json").exists()


def test_install_warp_model_long_pipeline_reports_progress_and_no_timeout(
    tmp_path, monkeypatch, capsys,
):
    import time

    model_id = "glm-5.3-flash-warp"
    root = _fake_installable_warp_root(tmp_path / "runtime")
    staging_dir = tmp_path / "staging"
    models_dir = tmp_path / "models"
    source = (staging_dir / model_id).absolute()
    output = (models_dir / f"{model_id}.waste").absolute()
    run_dir = (models_dir / f"{model_id}.warp-run").absolute()
    fetch_line = "fetch preflight: 14 / 62 shards complete"
    internal_stdout = "internal curl auth setup and transfer noise"
    internal_stderr = "internal validator command noise"
    pipeline_environments = []
    _patch_warp_model_prerequisites(monkeypatch)
    monkeypatch.setattr(I, "LONG_QUIET_SECONDS", 0.1, raising=False)
    monkeypatch.setattr(
        I._warp_models, "LONG_QUIET_SECONDS", 0.1, raising=False
    )

    def fake_run(args, **kwargs):
        script = next(
            (Path(str(arg)).name for arg in args if str(arg).endswith(".sh")),
            "",
        )
        if script == "fetch_weights.sh":
            return I.subprocess.CompletedProcess(
                args, 0, stdout=fetch_line + "\n", stderr=""
            )

        pipeline_environment = dict(kwargs.get("env") or {})
        pipeline_environments.append(pipeline_environment)
        time.sleep(1.9)
        _fake_waste_container(Path(pipeline_environment["OUT"]), model_id)
        return I.subprocess.CompletedProcess(
            args,
            0,
            stdout=internal_stdout + "\n",
            stderr=internal_stderr + "\n",
        )

    _patch_stage_popen(monkeypatch, fake_run)

    result = I.install_warp_model(
        model_id,
        warp_root=root,
        staging_dir=staging_dir,
        models_dir=models_dir,
    )

    captured = capsys.readouterr()
    visible_output = captured.out + captured.err
    assert result == output
    assert captured.out.splitlines().count(fetch_line) == 1
    assert "downloading and converting" in captured.out
    assert str(source / "download.log") in captured.out
    assert str(run_dir / "pipeline.log") in captured.out
    assert internal_stdout not in visible_output
    assert internal_stderr not in visible_output


def test_warp_heartbeat_tails_download_log_progress_fragment(
    tmp_path, monkeypatch, capsys,
):
    import time

    model_id = "glm-5.3-flash-warp"
    root = _fake_installable_warp_root(tmp_path / "runtime")
    staging_dir = tmp_path / "staging"
    models_dir = tmp_path / "models"
    _patch_warp_model_prerequisites(monkeypatch)
    monkeypatch.setattr(I, "LONG_QUIET_SECONDS", 0.05, raising=False)

    def fake_run(args, **kwargs):
        environment = dict(kwargs.get("env") or {})
        script = next(
            (Path(str(arg)).name for arg in args if str(arg).endswith(".sh")),
            "",
        )
        if script == "fetch_weights.sh":
            return I.subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        log = Path(environment["SRC"]) / "download.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write("[1/62] 5.1G/12.4G (41%)\r[1/62] 5.2G/12.4G (42%)\n")
            handle.write("shards        : 25 / 62 complete\n")
        time.sleep(0.35)
        _fake_waste_container(Path(environment["OUT"]), model_id)
        return I.subprocess.CompletedProcess(
            args, 0, stdout="internal pipeline noise\n", stderr=""
        )

    _patch_stage_popen(monkeypatch, fake_run)

    I.install_warp_model(
        model_id,
        warp_root=root,
        staging_dir=staging_dir,
        models_dir=models_dir,
    )

    out = capsys.readouterr().out
    heartbeats = [line for line in out.splitlines() if "s elapsed" in line]
    assert heartbeats, out
    for line in heartbeats:
        assert "62 complete" in line, line
        assert "internal pipeline noise" not in line, line

def test_warp_non_marquee_failure_surfaces_captured_stderr(
    tmp_path, capsys,
):
    diagnostic = "validator rejected malformed fixture"

    def fail_validator(args, **kwargs):
        return I.subprocess.CompletedProcess(
            args,
            7,
            stdout="validator internal stdout\n",
            stderr=diagnostic + "\n",
        )

    with pytest.raises(RuntimeError) as raised:
        I._warp_models._run_stage(
            ["validator", "--check"],
            cwd=tmp_path,
            env={},
            stage="validator",
            source=tmp_path / "source",
            output=tmp_path / "output.waste",
            run_dir=tmp_path / "run",
            run=fail_validator,
        )

    captured = capsys.readouterr()
    assert diagnostic in str(raised.value)
    assert diagnostic not in captured.out + captured.err


def test_install_warp_model_removes_temporary_curl_auth_after_failure(
    tmp_path, monkeypatch,
):
    model_id = "glm-5.3-flash-warp"
    token = "failure-secret-token"
    root = _fake_installable_warp_root(tmp_path / "runtime")
    curl_homes = []
    _patch_warp_model_prerequisites(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", token)

    def fake_run(args, **kwargs):
        child_env = dict(kwargs.get("env") or {})
        curl_homes.append(_assert_private_curl_auth(args, child_env, token))
        failed = any(Path(str(arg)).name == "pipeline.sh" for arg in args)
        return I.subprocess.CompletedProcess(args, 19 if failed else 0)

    _patch_stage_popen(monkeypatch, fake_run)

    with pytest.raises(RuntimeError, match="pipeline"):
        I.install_warp_model(
            model_id,
            warp_root=root,
            staging_dir=tmp_path / "staging",
            models_dir=tmp_path / "models",
        )

    assert len(curl_homes) == 2
    assert all(not curl_home.exists() for curl_home in curl_homes)


def test_install_warp_model_preserves_ambient_curl_home_without_token(
    tmp_path, monkeypatch,
):
    model_id = "glm-5.3-flash-warp"
    root = _fake_installable_warp_root(tmp_path / "runtime")
    ambient_curl_home = tmp_path / "existing curl home"
    ambient_curl_home.mkdir()
    ambient_config = ambient_curl_home / ".curlrc"
    ambient_config.write_text("user-agent = fixture\n")
    calls = []
    _patch_warp_model_prerequisites(monkeypatch)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("CURL_HOME", str(ambient_curl_home))

    def fake_run(args, **kwargs):
        child_env = dict(kwargs.get("env") or {})
        calls.append(list(args))
        assert "HF_TOKEN" not in child_env
        assert child_env["CURL_HOME"] == str(ambient_curl_home)
        assert ambient_config.read_text() == "user-agent = fixture\n"
        if any(Path(str(arg)).name == "pipeline.sh" for arg in args):
            _fake_waste_container(Path(child_env["OUT"]), model_id)
        return I.subprocess.CompletedProcess(args, 0)

    _patch_stage_popen(monkeypatch, fake_run)

    result = I.install_warp_model(
        model_id,
        warp_root=root,
        staging_dir=tmp_path / "staging",
        models_dir=tmp_path / "models",
    )

    assert len(calls) == 2
    assert result.is_dir()
    assert ambient_curl_home.is_dir()
    assert ambient_config.read_text() == "user-agent = fixture\n"


def test_install_warp_model_skips_fetch_preflight_for_proven_reclaimed_source(
    tmp_path, monkeypatch,
):
    model_id = "glm-5.3-flash-warp"
    root = _fake_installable_warp_root(tmp_path / "runtime")
    staging_dir = tmp_path / "staging"
    models_dir = tmp_path / "models"
    source = staging_dir / model_id
    source.mkdir(parents=True)
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    (source / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {
            "model.layers.0.weight": shards[0],
            "model.layers.1.weight": shards[1],
        },
    }))
    (source / ".download-state").write_text("\n".join(shards) + "\n")
    (source / ".reclaimed").write_text("\n".join(shards) + "\n")
    scripts = []
    _patch_warp_model_prerequisites(monkeypatch)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    def fake_run(args, **kwargs):
        script = next(
            (Path(str(arg)).name for arg in args if str(arg).endswith(".sh")),
            "",
        )
        scripts.append(script)
        if script == "pipeline.sh":
            _fake_waste_container(Path(kwargs["env"]["OUT"]), model_id)
        return I.subprocess.CompletedProcess(args, 0)

    _patch_stage_popen(monkeypatch, fake_run)

    result = I.install_warp_model(
        model_id,
        warp_root=root,
        staging_dir=staging_dir,
        models_dir=models_dir,
        reclaim_source=True,
    )

    assert scripts == ["pipeline.sh"]
    assert result == (models_dir / f"{model_id}.waste").absolute()
    assert json.loads((result / "manifest.json").read_text())["format_version"] == 0


def test_install_warp_model_rejects_incomplete_container(tmp_path, monkeypatch):
    model_id = "glm-5.3-flash-warp"
    root = _fake_installable_warp_root(tmp_path / "runtime")
    staging_dir = tmp_path / "staging"
    models_dir = tmp_path / "models"
    _patch_warp_model_prerequisites(monkeypatch)

    def fake_run(args, **kwargs):
        if any(Path(str(arg)).name == "pipeline.sh" for arg in args):
            output = Path(kwargs["env"]["OUT"])
            _fake_waste_container(output, model_id)
            next(output.glob("experts-L*.bin")).unlink()
        return I.subprocess.CompletedProcess(args, 0, "", "")

    _patch_stage_popen(monkeypatch, fake_run)

    with pytest.raises(RuntimeError, match=r"invalid WARP container.*expert"):
        I.install_warp_model(
            model_id,
            warp_root=root,
            staging_dir=staging_dir,
            models_dir=models_dir,
        )
    assert (models_dir / f"{model_id}.waste" / "manifest.json").is_file()
    assert (models_dir / f"{model_id}.waste" / "trunk.bin").is_file()


@pytest.mark.parametrize("model_id", WARP_CATALOG_MODELS)
def test_install_warp_model_budgets_profile_workspace_and_exports_same_minimum(
    tmp_path, monkeypatch, model_id,
):
    expected = WARP_CATALOG_MODELS[model_id]
    root = _fake_installable_warp_root(tmp_path / "runtime")
    staging_dir = tmp_path / "staging"
    models_dir = tmp_path / "models"
    preflight_calls = []
    pipeline_environments = []

    monkeypatch.setattr(
        I.shutil,
        "which",
        lambda command: f"/fake/bin/{command}",
    )
    monkeypatch.setattr(
        I,
        "_preflight_warp_disk",
        lambda source, output, **sizes: preflight_calls.append(
            (Path(source), Path(output), sizes)
        ),
    )
    monkeypatch.setattr(
        I,
        "_validate_warp_container",
        lambda container, got_model_id: None,
    )

    def fake_run(args, **kwargs):
        if any(Path(str(arg)).name == "pipeline.sh" for arg in args):
            pipeline_environments.append(dict(kwargs["env"]))
        return I.subprocess.CompletedProcess(args, 0)

    _patch_stage_popen(monkeypatch, fake_run)

    I.install_warp_model(
        model_id,
        warp_root=root,
        staging_dir=staging_dir,
        models_dir=models_dir,
    )

    source = (staging_dir / model_id).absolute()
    output = (models_dir / f"{model_id}.waste").absolute()
    assert preflight_calls == [(
        source,
        output,
        {
            "source_bytes": expected["source_gib"] * 1024**3,
            "output_bytes": expected["output_workspace_gib"] * 1024**3,
        },
    )]
    assert len(pipeline_environments) == 1
    assert pipeline_environments[0]["MIN_FREE_GB"] == str(
        expected["output_workspace_gib"]
    )


@pytest.mark.parametrize(
    "failed_script,error_pattern",
    [
        ("fetch_weights.sh", "download"),
        ("pipeline.sh", "pipeline"),
    ],
)
def test_install_warp_model_preserves_partial_artifacts_on_stage_failure(
    tmp_path, monkeypatch, failed_script, error_pattern,
):
    model_id = "glm-5.3-flash-warp"
    root = _fake_installable_warp_root(tmp_path / "runtime")
    staging_dir = tmp_path / "staging"
    models_dir = tmp_path / "models"
    expected_source = staging_dir / model_id
    expected_output = models_dir / f"{model_id}.waste"
    _patch_warp_model_prerequisites(monkeypatch)

    def fake_run(args, **kwargs):
        script = next(
            (Path(str(arg)).name for arg in args if str(arg).endswith(".sh")),
            "",
        )
        expected_source.mkdir(parents=True, exist_ok=True)
        (expected_source / "partial-shard.safetensors").write_bytes(b"resumable")
        if script == "pipeline.sh":
            expected_output.mkdir(parents=True, exist_ok=True)
            (expected_output / "trunk.bin.partial").write_bytes(b"partial")
        code = 17 if script == failed_script else 0
        return I.subprocess.CompletedProcess(
            args,
            code,
            stdout="",
            stderr=f"simulated {script} failure" if code else "",
        )

    _patch_stage_popen(monkeypatch, fake_run)

    with pytest.raises(RuntimeError, match=error_pattern):
        I.install_warp_model(
            model_id,
            warp_root=root,
            staging_dir=staging_dir,
            models_dir=models_dir,
        )

    assert (expected_source / "partial-shard.safetensors").read_bytes() == b"resumable"
    if failed_script == "pipeline.sh":
        assert (expected_output / "trunk.bin.partial").read_bytes() == b"partial"


def test_warp_fetch_failure_excludes_stale_pipeline_marker(
    tmp_path, monkeypatch,
):
    model_id = "glm-5.3-flash-warp"
    root = _fake_installable_warp_root(tmp_path / "runtime")
    staging_dir = tmp_path / "staging"
    models_dir = tmp_path / "models"
    run_dir = models_dir / f"{model_id}.warp-run"
    stale_marker = "stale pipeline failure from an earlier run"
    run_dir.mkdir(parents=True)
    (run_dir / ".failed").write_text(stale_marker + "\n")
    _patch_warp_model_prerequisites(monkeypatch)

    def fail_fetch(args, **kwargs):
        is_fetch = any(
            Path(str(arg)).name == "fetch_weights.sh" for arg in args
        )
        return I.subprocess.CompletedProcess(args, 41 if is_fetch else 0)

    _patch_stage_popen(monkeypatch, fail_fetch)

    with pytest.raises(RuntimeError) as raised:
        I.install_warp_model(
            model_id,
            warp_root=root,
            staging_dir=staging_dir,
            models_dir=models_dir,
        )

    message = str(raised.value)
    assert "download failed" in message
    assert stale_marker not in message


def test_warp_pipeline_failure_reports_exact_marker_and_resumable_paths(
    tmp_path, monkeypatch,
):
    model_id = "glm-5.3-flash-warp"
    root = _fake_installable_warp_root(tmp_path / "runtime")
    staging_dir = tmp_path / "staging"
    models_dir = tmp_path / "models"
    source = (staging_dir / model_id).absolute()
    output = (models_dir / f"{model_id}.waste").absolute()
    run_dir = (models_dir / f"{model_id}.warp-run").absolute()
    stage_marker = "oracle diff (see diff.txt)"
    _patch_warp_model_prerequisites(monkeypatch)

    def fake_run(args, **kwargs):
        if any(Path(str(arg)).name == "pipeline.sh" for arg in args):
            source.mkdir(parents=True, exist_ok=True)
            (source / "partial-shard.safetensors").write_bytes(b"resumable")
            output.mkdir(parents=True, exist_ok=True)
            (output / "trunk.bin.partial").write_bytes(b"partial")
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / ".failed").write_text(stage_marker + "\n")
            return I.subprocess.CompletedProcess(args, 23)
        return I.subprocess.CompletedProcess(args, 0)

    _patch_stage_popen(monkeypatch, fake_run)

    with pytest.raises(RuntimeError) as raised:
        I.install_warp_model(
            model_id,
            warp_root=root,
            staging_dir=staging_dir,
            models_dir=models_dir,
        )

    message = str(raised.value)
    assert stage_marker in message
    assert str(source) in message
    assert str(output) in message
    assert str(run_dir) in message
    assert (source / "partial-shard.safetensors").read_bytes() == b"resumable"
    assert (output / "trunk.bin.partial").read_bytes() == b"partial"


def test_warp_disk_preflight_combines_same_filesystem_and_credits_resume(
    tmp_path, monkeypatch,
):
    staging = tmp_path / "staging"
    output = tmp_path / "model.waste"
    staging.mkdir()
    output.mkdir()
    (staging / "partial.safetensors").write_bytes(b"s" * 200)
    (output / "trunk.bin").write_bytes(b"o" * 100)
    free = {"bytes": 1000}

    monkeypatch.setattr(
        I.shutil,
        "disk_usage",
        lambda path: type("Usage", (), {"free": free["bytes"]})(),
    )
    monkeypatch.setattr(
        I,
        "_filesystem_device",
        lambda path: 7,
        raising=False,
    )

    # Remaining bytes are 800 source + 500 output. Each would fit by itself,
    # but a shared filesystem must budget for both together plus safety margin.
    with pytest.raises(RuntimeError, match="same filesystem"):
        I._preflight_warp_disk(
            staging,
            output,
            source_bytes=1000,
            output_bytes=600,
        )

    # Resume credit makes 1400 sufficient; starting from zero would not fit.
    free["bytes"] = 1400
    I._preflight_warp_disk(
        staging,
        output,
        source_bytes=1000,
        output_bytes=600,
    )


def test_warp_disk_preflight_credits_proven_reclaimed_source_only(
    tmp_path, monkeypatch,
):
    source = tmp_path / "staging"
    output = tmp_path / "model.waste"
    source.mkdir()
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    (source / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {
            "model.layers.0.weight": shards[0],
            "model.layers.1.weight": shards[1],
        },
    }))
    (source / ".download-state").write_text("\n".join(shards) + "\n")
    (source / shards[0]).write_bytes(b"verified shard")
    (source / ".reclaimed").write_text(shards[1] + "\n")

    monkeypatch.setattr(
        I,
        "_filesystem_device",
        lambda path: 7,
        raising=False,
    )
    monkeypatch.setattr(
        I.shutil,
        "disk_usage",
        lambda path: type("Usage", (), {"free": 0})(),
    )

    # The index proves the complete shard set, download-state proves each
    # finished, and the reclaim ledger explains the one no longer on disk.
    I._preflight_warp_disk(
        source,
        output,
        source_bytes=10_000,
        output_bytes=0,
    )

    # A missing shard without ledger evidence is not completion credit.
    (source / ".reclaimed").write_text("")
    with pytest.raises(RuntimeError, match="disk preflight"):
        I._preflight_warp_disk(
            source,
            output,
            source_bytes=10_000,
            output_bytes=0,
        )

    # Nor is a ledger enough when download-state never proved the shard.
    (source / ".reclaimed").write_text(shards[1] + "\n")
    (source / ".download-state").write_text(shards[0] + "\n")
    with pytest.raises(RuntimeError, match="disk preflight"):
        I._preflight_warp_disk(
            source,
            output,
            source_bytes=10_000,
            output_bytes=0,
        )


def test_warp_disk_preflight_checks_split_filesystems_independently(
    tmp_path, monkeypatch,
):
    staging = tmp_path / "staging"
    output = tmp_path / "model.waste"
    staging.mkdir()
    output.mkdir()
    (staging / "partial.safetensors").write_bytes(b"s" * 200)
    (output / "trunk.bin").write_bytes(b"o" * 100)
    free = {staging: 700, output: 10_000}

    monkeypatch.setattr(
        I,
        "_filesystem_device",
        lambda path: 1 if Path(path) == staging else 2,
        raising=False,
    )
    monkeypatch.setattr(
        I.shutil,
        "disk_usage",
        lambda path: type("Usage", (), {"free": free[Path(path)]})(),
    )

    with pytest.raises(RuntimeError, match="staging"):
        I._preflight_warp_disk(
            staging,
            output,
            source_bytes=1000,
            output_bytes=600,
        )

    free[staging] = 850
    free[output] = 500
    with pytest.raises(RuntimeError, match="output"):
        I._preflight_warp_disk(
            staging,
            output,
            source_bytes=1000,
            output_bytes=600,
        )

    free[output] = 550
    I._preflight_warp_disk(
        staging,
        output,
        source_bytes=1000,
        output_bytes=600,
    )


def test_install_help_describes_warp_model_options():
    from click.testing import CliRunner

    result = CliRunner().invoke(I.install_cmd, ["--help"])
    assert result.exit_code == 0, result.output
    assert "--staging-dir" in result.output
    assert "--warp-jobs" in result.output
    assert "--reclaim-source" in result.output
    assert "glm-5.3-flash-warp" in result.output
    assert "deepseek-v4.1-flash-warp" in result.output


@pytest.mark.parametrize("model_id", WARP_CATALOG_MODELS)
def test_warp_cli_preflights_and_explains_destructive_plan_before_confirmation(
    tmp_path, monkeypatch, model_id,
):
    from click.testing import CliRunner

    expected = WARP_CATALOG_MODELS[model_id]
    prefix = tmp_path / "prefix"
    staging_dir = tmp_path / "staging"
    models_dir = tmp_path / "models"
    config = tmp_path / "models.yaml"
    source = (staging_dir / model_id).absolute()
    output = (models_dir / f"{model_id}.waste").absolute()
    checked_tools = []
    disk_checks = []
    install_events = []

    def fake_which(command):
        checked_tools.append(command)
        return f"/fake/bin/{command}"

    def fake_disk_usage(path):
        disk_checks.append(Path(path))
        return type("Usage", (), {"free": 10**15})()

    monkeypatch.setattr(I.shutil, "which", fake_which)
    monkeypatch.setattr(I.shutil, "disk_usage", fake_disk_usage)
    monkeypatch.setattr(
        I,
        "install_warp",
        lambda *args, **kwargs: install_events.append("runtime") or tmp_path / "runtime",
    )
    monkeypatch.setattr(
        I,
        "install_warp_model",
        lambda *args, **kwargs: install_events.append("model") or output,
    )
    monkeypatch.setattr(I, "get_total_memory_bytes", lambda: None)

    result = CliRunner().invoke(I.install_cmd, [
        "--model", model_id,
        "--prefix", str(prefix),
        "--staging-dir", str(staging_dir),
        "--models-dir", str(models_dir),
        "--config", str(config),
        "--reclaim-source",
    ], input="n\n")

    assert result.exit_code != 0
    assert "Aborted" in result.output
    assert set(("git", "make", "bash", "uv", "curl")) <= set(checked_tools)
    assert disk_checks
    assert expected["revision"] in result.output
    assert f"{expected['source_gib']} GiB" in result.output
    output_unit = "GB" if model_id == "glm-5.3-flash-warp" else "GiB"
    assert f"{expected['output_gb']} {output_unit}" in result.output
    assert f"{expected['output_workspace_gib']} GiB" in result.output
    assert str(source) in result.output
    assert str(output) in result.output
    assert "irreversible" in result.output.lower()
    assert "re-download" in result.output.lower()
    prompt_at = result.output.index("Proceed with WARP conversion?")
    for detail in (expected["revision"], str(source), str(output), "irreversible"):
        assert result.output.lower().index(detail.lower()) < prompt_at
    assert install_events == []
    assert not prefix.exists()
    assert not staging_dir.exists()
    assert not models_dir.exists()
    assert not config.exists()


@pytest.mark.parametrize("option", ["--staging-dir", "--models-dir"])
@pytest.mark.parametrize("character,expected_detail", [
    pytest.param("'", "single quote", id="single-quote"),
    pytest.param("\n", "newline", id="newline"),
    pytest.param("\r", "carriage return", id="carriage-return"),
    pytest.param("\\", "backslash", id="backslash"),
])
def test_warp_cli_rejects_upstream_unsafe_paths_before_confirmation(
    tmp_path, monkeypatch, option, character, expected_detail,
):
    from click.testing import CliRunner

    bad_path = tmp_path / f"unsafe{character}path"
    staging_dir = bad_path if option == "--staging-dir" else tmp_path / "staging"
    models_dir = bad_path if option == "--models-dir" else tmp_path / "models"
    prefix = tmp_path / "prefix"
    config = tmp_path / "models.yaml"
    install_events = []

    monkeypatch.setattr(
        I.shutil,
        "which",
        lambda command: f"/fake/bin/{command}",
    )
    monkeypatch.setattr(
        I.shutil,
        "disk_usage",
        lambda path: type("Usage", (), {"free": 10**15})(),
    )
    monkeypatch.setattr(
        I,
        "install_warp",
        lambda *args, **kwargs: install_events.append("runtime") or tmp_path / "runtime",
    )
    monkeypatch.setattr(
        I,
        "install_warp_model",
        lambda *args, **kwargs: install_events.append("model"),
    )
    monkeypatch.setattr(I, "get_total_memory_bytes", lambda: None)

    result = CliRunner().invoke(I.install_cmd, [
        "--model", "glm-5.3-flash-warp",
        "--prefix", str(prefix),
        "--staging-dir", str(staging_dir),
        "--models-dir", str(models_dir),
        "--config", str(config),
    ])

    assert result.exit_code != 0
    assert expected_detail in result.output.lower()
    assert "path" in result.output.lower()
    assert "Proceed with WARP conversion?" not in result.output
    assert install_events == []
    assert not prefix.exists()
    assert not staging_dir.exists()
    assert not models_dir.exists()
    assert not config.exists()


@pytest.mark.parametrize("layout", [
    "output-under-source",
    "source-under-output",
    "source-under-run",
    "source-equals-output-via-symlink",
    "output-equals-run-via-symlink",
    "run-under-output-via-symlink",
    "source-under-output-via-ancestor-symlink",
])
def test_warp_cli_rejects_resolved_plan_path_overlap_before_preflight(
    tmp_path, monkeypatch, layout,
):
    from click.testing import CliRunner

    model_id = "glm-5.3-flash-warp"
    base = tmp_path / layout
    staging_dir = base / "staging"
    models_dir = base / "models"
    pair = ("source", "output")

    if layout == "output-under-source":
        models_dir = staging_dir / model_id
    elif layout == "source-under-output":
        staging_dir = models_dir / f"{model_id}.waste"
    elif layout == "source-under-run":
        staging_dir = models_dir / f"{model_id}.warp-run"
        pair = ("source", "run")
    elif layout == "source-equals-output-via-symlink":
        staging_dir.mkdir(parents=True)
        models_dir.mkdir(parents=True)
        output = models_dir / f"{model_id}.waste"
        output.mkdir()
        (staging_dir / model_id).symlink_to(output, target_is_directory=True)
    elif layout == "output-equals-run-via-symlink":
        models_dir.mkdir(parents=True)
        run_dir = models_dir / f"{model_id}.warp-run"
        run_dir.mkdir()
        (models_dir / f"{model_id}.waste").symlink_to(
            run_dir,
            target_is_directory=True,
        )
        pair = ("output", "run")
    elif layout == "run-under-output-via-symlink":
        models_dir.mkdir(parents=True)
        (models_dir / f"{model_id}.waste").symlink_to(
            models_dir,
            target_is_directory=True,
        )
        pair = ("output", "run")
    elif layout == "source-under-output-via-ancestor-symlink":
        models_dir.mkdir(parents=True)
        output = models_dir / f"{model_id}.waste"
        aliased_staging = output / "staging-root"
        aliased_staging.mkdir(parents=True)
        staging_dir = base / "staging-link"
        staging_dir.symlink_to(aliased_staging, target_is_directory=True)

    source = staging_dir / model_id
    output = models_dir / f"{model_id}.waste"
    run_dir = models_dir / f"{model_id}.warp-run"
    targets = (source, output, run_dir)
    before = {
        path: (path.exists(), path.is_symlink())
        for path in targets
    }
    disk_checks = []
    install_events = []
    prefix = base / "prefix"
    config = base / "models.yaml"

    monkeypatch.setattr(
        I.shutil,
        "which",
        lambda command: f"/fake/bin/{command}",
    )
    monkeypatch.setattr(
        I,
        "_preflight_warp_disk",
        lambda *args, **kwargs: disk_checks.append((args, kwargs)),
    )
    monkeypatch.setattr(
        I,
        "install_warp",
        lambda *args, **kwargs: install_events.append("runtime") or base / "runtime",
    )
    monkeypatch.setattr(
        I,
        "install_warp_model",
        lambda *args, **kwargs: install_events.append("model"),
    )
    monkeypatch.setattr(I, "get_total_memory_bytes", lambda: None)

    result = CliRunner().invoke(I.install_cmd, [
        "--model", model_id,
        "--prefix", str(prefix),
        "--staging-dir", str(staging_dir),
        "--models-dir", str(models_dir),
        "--config", str(config),
    ])

    message = result.output.lower()
    assert result.exit_code != 0
    assert "overlap" in message or "nested" in message
    planned_paths = {"source": source, "output": output, "run": run_dir}
    for name in pair:
        path = planned_paths[name]
        assert any(
            detail in message
            for detail in (name, str(path).lower(), str(path.resolve()).lower())
        )
    assert "Proceed with WARP conversion?" not in result.output
    assert disk_checks == []
    assert install_events == []
    assert {
        path: (path.exists(), path.is_symlink())
        for path in targets
    } == before
    assert not prefix.exists()
    assert not config.exists()


@pytest.mark.parametrize("requested_ctx", [None, 0, 65536, 131072])
@pytest.mark.parametrize(
    "model_id,target_args,expected_jobs,expected_reclaim",
    [
        ("glm-5.3-flash-warp", ["glm-5.3-flash-warp"], 3, False),
        (
            "deepseek-v4.1-flash-warp",
            [
                "--model", "deepseek-v4.1-flash-warp",
                "--warp-jobs", "4",
                "--reclaim-source",
            ],
            4,
            True,
        ),
    ],
)
def test_install_warp_catalog_model_dispatches_positional_and_option(
    tmp_path, monkeypatch, model_id, target_args, expected_jobs, expected_reclaim,
    requested_ctx,
):
    from click.testing import CliRunner

    prefix = tmp_path / "prefix"
    staging_dir = tmp_path / "staging dir"
    models_dir = tmp_path / "models dir"
    config = tmp_path / "models.yaml"
    old_warp = tmp_path / "old.waste"
    config.write_text(
        "host: 127.0.0.1\nport: 8090\napi_key: null\nmodels:\n"
        "  - id: existing\n    engine: llamacpp\n    model_path: /models/existing.gguf\n"
        "    n_ctx: 32768\n    aliases: [existing-alias]\n"
        f"  - id: {model_id}\n    engine: warp\n    model_path: {old_warp}\n"
        "    n_ctx: 0\n    aliases: [warp-alias]\n"
    )
    container = _fake_waste_container(
        models_dir / f"{model_id}.waste",
        model_id,
    ).absolute()
    root = _fake_installable_warp_root(tmp_path / "runtime")
    events = []

    def fake_install_runtime(got_prefix, **kwargs):
        events.append(("runtime", got_prefix))
        return root

    def fake_install_model(
        got_model,
        *,
        warp_root,
        staging_dir,
        models_dir,
        jobs,
        reclaim_source,
    ):
        events.append((
            "model",
            got_model,
            warp_root,
            staging_dir,
            models_dir,
            jobs,
            reclaim_source,
        ))
        return container

    monkeypatch.setattr(
        I.shutil,
        "which",
        lambda command: f"/fake/bin/{command}",
    )
    monkeypatch.setattr(
        I.shutil,
        "disk_usage",
        lambda path: type("Usage", (), {"free": 10**15})(),
    )
    monkeypatch.setattr(I, "install_warp", fake_install_runtime)
    monkeypatch.setattr(I, "install_warp_model", fake_install_model, raising=False)
    monkeypatch.setattr(I, "get_total_memory_bytes", lambda: None)

    result = CliRunner().invoke(I.install_cmd, [
        *target_args,
        "--prefix", str(prefix),
        "--staging-dir", str(staging_dir),
        "--models-dir", str(models_dir),
        "--config", str(config),
        *(["--n-ctx", str(requested_ctx)] if requested_ctx is not None else []),
        "--yes",
    ])

    assert result.exit_code == 0, result.output
    assert events == [
        ("runtime", prefix),
        (
            "model",
            model_id,
            root,
            staging_dir,
            models_dir,
            expected_jobs,
            expected_reclaim,
        ),
    ]
    loaded = load_config(config)
    existing = next(model for model in loaded.models if model.id == "existing")
    warp = next(model for model in loaded.models if model.id == model_id)
    assert existing.aliases == ["existing-alias"]
    assert existing.model_path == "/models/existing.gguf"
    assert warp.engine == "warp"
    assert Path(warp.model_path).is_absolute()
    assert Path(warp.model_path) == container
    assert warp.warp_auto_context is (not bool(requested_ctx))
    assert warp.n_ctx == (requested_ctx or 0)
    assert warp.aliases == ["warp-alias"]


def test_install_rejects_nonpositive_warp_jobs():
    from click.testing import CliRunner

    result = CliRunner().invoke(I.install_cmd, [
        "--model",
        "glm-5.3-flash-warp",
        "--warp-jobs",
        "0",
        "--yes",
    ])
    assert result.exit_code != 0
    assert "warp-jobs" in result.output.lower()
    assert "positive" in result.output.lower() or "at least 1" in result.output.lower()


@pytest.mark.parametrize("requested_ctx", ["-1", "2147483648"])
def test_install_rejects_invalid_warp_context_before_install(monkeypatch, requested_ctx):
    from click.testing import CliRunner

    monkeypatch.setattr(
        I, "install_warp",
        lambda *args, **kwargs: pytest.fail("invalid context must not start installation"),
    )
    result = CliRunner().invoke(I.install_cmd, [
        "--model", "glm-5.3-flash-warp", "--n-ctx", requested_ctx, "--yes",
    ])
    assert result.exit_code == 2
    assert "--n-ctx" in result.output


@pytest.mark.parametrize(
    "args,option",
    [
        (
            ["--model", "glm-5.3-flash-warp", "--quant", "Q4_K_M"],
            "--quant",
        ),
        (
            ["--model", "glm-5.3-flash-warp", "--no-mmproj"],
            "--no-mmproj",
        ),
        (
            ["--model", "glm-5.3-flash-warp", "--engine", "llamacpp"],
            "--engine",
        ),
        (
            ["--model", "glm-5.3-flash-warp", "--llamacpp-variant", "cuda"],
            "--llamacpp-variant",
        ),
        (
            ["--model", "glm-5.3-flash-warp", "--llamacpp-tag", "b11005"],
            "--llamacpp-tag",
        ),
        (
            ["--model", "gemma-4-26b-a4b", "--staging-dir", "/tmp/stage"],
            "--staging-dir",
        ),
        (
            ["--model", "gemma-4-26b-a4b", "--warp-jobs", "4"],
            "--warp-jobs",
        ),
        (
            ["--model", "gemma-4-26b-a4b", "--reclaim-source"],
            "--reclaim-source",
        ),
    ],
)
def test_install_rejects_incompatible_warp_options(monkeypatch, args, option):
    from click.testing import CliRunner

    called = []

    def unexpected(*unused_args, **unused_kwargs):
        called.append(True)
        raise AssertionError("validation must happen before installation")

    monkeypatch.setattr(I, "install_warp", unexpected)
    monkeypatch.setattr(I, "install_warp_model", unexpected, raising=False)
    monkeypatch.setattr(I, "install_llamacpp", unexpected)
    monkeypatch.setattr(I, "install_ktransformers", unexpected)
    monkeypatch.setattr(I, "download_model", unexpected)

    result = CliRunner().invoke(I.install_cmd, [*args, "--yes"])
    assert result.exit_code != 0
    assert option in result.output
    assert called == []


def test_install_warp_failure_does_not_mutate_config(tmp_path, monkeypatch):
    from click.testing import CliRunner

    prefix = tmp_path / "prefix"
    root = _fake_installable_warp_root(tmp_path / "runtime")
    staging_dir = tmp_path / "staging"
    models_dir = tmp_path / "models"
    config = tmp_path / "models.yaml"
    config.write_text(
        "host: 127.0.0.1\nport: 8090\napi_key: null\nmodels:\n"
        "  - id: existing\n    engine: llamacpp\n"
        "    model_path: /models/existing.gguf\n    n_ctx: 32768\n"
        "    aliases: [keep-me]\n"
    )
    before = config.read_bytes()
    partial = models_dir / "glm-5.3-flash-warp.waste" / "trunk.bin.partial"
    events = []

    def fail_model(*args, **kwargs):
        events.append("model")
        partial.parent.mkdir(parents=True)
        partial.write_bytes(b"resume me")
        raise RuntimeError("WARP pipeline failed at convert")

    monkeypatch.setattr(
        I.shutil,
        "which",
        lambda command: f"/fake/bin/{command}",
    )
    monkeypatch.setattr(
        I.shutil,
        "disk_usage",
        lambda path: type("Usage", (), {"free": 10**15})(),
    )
    monkeypatch.setattr(
        I,
        "install_warp",
        lambda got_prefix, **kwargs: events.append("runtime") or root,
    )
    monkeypatch.setattr(I, "install_warp_model", fail_model, raising=False)
    monkeypatch.setattr(I, "get_total_memory_bytes", lambda: None)

    result = CliRunner().invoke(I.install_cmd, [
        "--model", "glm-5.3-flash-warp",
        "--prefix", str(prefix),
        "--staging-dir", str(staging_dir),
        "--models-dir", str(models_dir),
        "--config", str(config),
        "--yes",
    ])

    assert result.exit_code != 0
    assert events == ["runtime", "model"]
    assert "convert" in result.output
    assert config.read_bytes() == before
    assert partial.read_bytes() == b"resume me"


@pytest.mark.parametrize("model_id", [
    "glm-5.3-flash-warp", "deepseek-v4.1-flash-warp",
])
@pytest.mark.parametrize("requested_ctx,auto", [
    (0, None), (65536, None), (131072, True),
    (65536, False), (4096, None), (32768, None),
])
def test_server_context_preparation_migrates_warp_defaults(
    tmp_path, monkeypatch, model_id, requested_ctx, auto,
):
    config_path = tmp_path / "models.yaml"
    container = _fake_waste_container(
        tmp_path / f"{model_id}.waste",
        model_id,
    )
    config_path.write_text(
        "host: 127.0.0.1\nport: 8090\nmodels:\n"
        f"  - id: {model_id}\n"
        "    engine: warp\n"
        f"    model_path: {container}\n"
        f"    n_ctx: {requested_ctx}\n"
        + (f"    warp_auto_context: {str(auto).lower()}\n" if auto is not None else "")
    )
    config = load_config(config_path)
    model = config.models[0]
    from litmoe.engines.warp import WarpEngine
    root = _fake_warp_root(tmp_path)
    monkeypatch.setenv("LITMOE_WARP_DIR", str(root))
    monkeypatch.setattr(WarpEngine, "start", lambda self, log_dir=None: self.build_command())
    monkeypatch.setattr(
        S,
        "compute_memory_aware_ctx",
        lambda *args, **kwargs: pytest.fail(
            "generic context sizing must not inspect WARP containers"
        ),
    )

    S.Gateway(config, config_path=str(config_path))._start_engine(model)

    if auto is True or (auto is None and requested_ctx in (0, 65536)):
        assert model.n_ctx > 87644
        assert model.warp_auto_context is True
    else:
        assert model.n_ctx == requested_ctx
        assert model.warp_auto_context is False
    saved = load_config(config_path).models[0]
    assert (saved.n_ctx, saved.warp_auto_context) == (model.n_ctx, model.warp_auto_context)


def test_warp_stage_runs_in_own_session_and_dies_with_cli(tmp_path, monkeypatch):
    """An interrupt must take down the whole stage tree, not just the leader.

    Regression: stages ran in litmoe's process group; Ctrl-C killed the CLI
    and left bash/xargs/curl running, and a rerun stacked a second fetch on
    the survivor while both wrote the same shard files.
    """
    import signal as signal_module
    import subprocess as subprocess_module
    import threading
    import time as time_module

    blocker = tmp_path / "stage-marker"
    script = tmp_path / "long-stage.sh"
    script.write_text(
        "#!/bin/sh\n"
        "trap '' TERM INT\n"
        f"while [ ! -f {blocker} ]; do sleep 0.05; done\n"
        "echo done\n"
    )
    script.chmod(0o755)
    monkeypatch.setattr(I, "LONG_QUIET_SECONDS", 0.05, raising=False)

    def stage_pgid():
        """Process group of the stage tree, or None when it is gone.

        Linux: match ``/proc/<pid>/cmdline`` directly — full argv, never
        truncated. When stdout is not a tty, some ``ps`` builds cut the
        command column at a narrow width and the long pytest tmp path no
        longer matches, so the scan would miss a live stage entirely.
        Elsewhere (macOS): fall back to ``ps`` as before.
        """
        target = str(script)
        proc_root = Path("/proc")
        if proc_root.is_dir():
            for entry in proc_root.iterdir():
                if not entry.name.isdigit():
                    continue
                try:
                    cmdline = (entry / "cmdline").read_bytes().split(b"\0")
                except OSError:
                    continue
                args = [part.decode("utf-8", "replace") for part in cmdline if part]
                if target not in args:
                    continue
                try:
                    return os.getpgid(int(entry.name))
                except OSError:
                    continue
            return None
        listings = subprocess_module.run(
            ["ps", "-axo", "pgid=,command="],
            capture_output=True,
            text=True,
        ).stdout
        for line in listings.splitlines():
            if target in line and "sleep" not in line:
                return int(line.split()[0])
        return None

    interruptible = threading.Event()

    def killer():
        deadline = time_module.monotonic() + 15
        while time_module.monotonic() < deadline:
            if interruptible.is_set():
                found = stage_pgid()
                if found is not None and found != os.getpgrp():
                    try:
                        os.kill(os.getpid(), signal_module.SIGINT)
                    except OSError:
                        pass
                    return
            time_module.sleep(0.05)
        blocker.touch()

    helper = threading.Thread(target=killer, daemon=True)
    helper.start()

    try:
        interruptible.set()
        with pytest.raises(KeyboardInterrupt):
            I._run_warp_stage(
                ["/bin/sh", str(script)],
                cwd=tmp_path,
                env={"DEST": str(tmp_path)},
            )
    finally:
        interruptible.clear()
        blocker.touch()
    helper.join(20)

    deadline = time_module.monotonic() + 10
    while time_module.monotonic() < deadline and stage_pgid() is not None:
        time_module.sleep(0.05)
    found = stage_pgid()
    assert found is None, f"stage tree survived the interrupt (pgid {found})"


def test_warp_install_refuses_concurrent_stage_processes(tmp_path, monkeypatch):
    model_id = "glm-5.3-flash-warp"
    root = _fake_installable_warp_root(tmp_path / "runtime")
    staging_dir = tmp_path / "staging"
    source = staging_dir / model_id
    source.mkdir(parents=True)
    (source / ".download-state").write_text(
        "model-00001-of-00062.safetensors\n"
    )
    calls = []

    def fake_snapshot(paths):
        calls.append(tuple(Path(path) for path in paths))
        return [4242]

    monkeypatch.setattr(
        I._warp_models, "_snapshot_live_processes", fake_snapshot
    )
    _patch_warp_model_prerequisites(monkeypatch)

    def fail_run(*args, **kwargs):
        raise AssertionError("no stage may start while another install runs")

    _patch_stage_popen(monkeypatch, fail_run)

    with pytest.raises(RuntimeError, match="already running"):
        I.install_warp_model(
            model_id,
            warp_root=root,
            staging_dir=staging_dir,
            models_dir=tmp_path / "models",
        )

    source = (staging_dir / model_id).absolute()
    output = (tmp_path / "models" / f"{model_id}.waste").absolute()
    run_dir = (tmp_path / "models" / f"{model_id}.warp-run").absolute()
    assert calls == [(source, output, run_dir)]


def test_warp_install_dedupes_download_state_ledger(tmp_path, monkeypatch):
    model_id = "glm-5.3-flash-warp"
    root = _fake_installable_warp_root(tmp_path / "runtime")
    staging_dir = tmp_path / "staging"
    source = staging_dir / model_id
    source.mkdir(parents=True)
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    (source / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {
            "model.layers.0.weight": shards[0],
            "model.layers.1.weight": shards[1],
        },
    }))
    (source / ".download-state").write_text(
        shards[0] + "\n" + shards[0] + "\n" + shards[1] + "\n" + shards[1] + "\n"
    )
    _patch_warp_model_prerequisites(monkeypatch)

    def fake_run(args, **kwargs):
        assert (source / ".download-state").read_text().splitlines() == shards
        if any(Path(str(arg)).name == "pipeline.sh" for arg in args):
            _fake_waste_container(Path(kwargs["env"]["OUT"]), model_id)
        return I.subprocess.CompletedProcess(args, 0)

    _patch_stage_popen(monkeypatch, fake_run)

    result = I.install_warp_model(
        model_id,
        warp_root=root,
        staging_dir=staging_dir,
        models_dir=tmp_path / "models",
    )

    assert result.is_dir()
    assert (source / ".download-state").read_text().splitlines() == shards

def test_stage_matcher_covers_seedless_ancestors(monkeypatch):
    """pipeline.sh carries paths only in env vars; its children carry none.

    The matcher must still count convert.py (which names the paths), its
    multiprocessing workers (which name nothing), and pipeline.sh itself
    (an ancestor of convert.py), or a rerun stacks a second conversion on
    a live one.
    """
    source = Path("/staging/glm")
    output = Path("/models/glm.waste")
    run_dir = Path("/models/glm.warp-run")
    table = {
        10: (1, "/bin/bash /warp/tools/pipeline.sh"),
        11: (10, "uv run python tools/convert.py --src /staging/glm --out /models/glm.waste"),
        12: (11, "python -c from multiprocessing.spawn import spawn_main"),
        13: (1, "python -c from multiprocessing.spawn import spawn_main"),
        14: (1, "/bin/bash /other/tools/pipeline.sh"),
    }

    monkeypatch.setattr(I._warp_models, "_stage_processes", lambda: table)

    assert I._warp_models._snapshot_live_processes((source, output, run_dir)) == [10, 11, 12]
    assert I._warp_models._snapshot_live_processes((Path("/staging/other"), Path("/models/other.waste"), Path("/models/other.run"))) == []


def test_install_warp_reuses_pinned_runtime_without_rebuilding(tmp_path, monkeypatch):
    prefix = tmp_path / "prefix"
    root = prefix / "lib" / "warp"
    (root / "serve").mkdir(parents=True)
    (root / "serve" / "__main__.py").write_text("")
    library = "libwaste.dylib" if sys.platform == "darwin" else (
        "libwaste.dll" if sys.platform == "win32" else "libwaste.so"
    )
    (root / library).write_bytes(b"library")
    (root / "tools").mkdir()
    (root / "tools" / "fetch_weights.sh").write_text("#!/usr/bin/env bash\n")
    (root / "tools" / "pipeline.sh").write_text("#!/usr/bin/env bash\n")
    import hashlib
    (root / ".litmoe-patch-sha256").write_text(
        hashlib.sha256(I.WARP_PATCH.read_bytes()).hexdigest() + "\n"
    )

    import subprocess as subprocess_module

    def fake_run(args, **kwargs):
        assert args[:3] == ["git", "rev-parse", "HEAD"], args
        return subprocess_module.CompletedProcess(args, 0, I.WARP_COMMIT + "\n", "")

    monkeypatch.setattr(I.subprocess, "run", fake_run)

    assert I.install_warp(prefix) == root


def test_warp_context_fit_uses_native_when_recommended_memory_fits():
    from litmoe.engines.warp_context import fit_native_context

    native = 1048576
    assert fit_native_context(native, 96 * 1024**3, lambda ctx: 5 * 1024**3 + ctx * 65536) == native


def test_warp_context_fit_finds_largest_safe_window():
    from litmoe.engines.warp_context import fit_native_context

    required = lambda ctx: 1024**3 + ctx * 65536
    budget = required(131072) + 4095 * 65536
    resolved = fit_native_context(1048576, budget, required)
    assert resolved == 131072
    assert required(resolved) <= budget < required(resolved + 4096)


def test_warp_context_fit_refuses_insufficient_memory():
    from litmoe.engines.warp_context import fit_native_context

    with pytest.raises(ValueError, match="memory"):
        fit_native_context(1048576, 1024, lambda ctx: 2048 + ctx * 4)


def test_warp_auto_context_refits_after_restart(tmp_path, monkeypatch):
    from litmoe.engines.warp import WarpEngine

    root = _fake_warp_root(tmp_path)
    monkeypatch.setenv("LITMOE_WARP_DIR", str(root))
    path = tmp_path / "models.yaml"
    path.write_text(
        f"models:\n  - id: glm-5.3-flash-warp\n    engine: warp\n"
        f"    model_path: {tmp_path}\n    n_ctx: 65536\n"
    )
    monkeypatch.setattr(WarpEngine, "start", lambda self, log_dir=None: self.build_command())
    first = S.Gateway(load_config(path), config_path=str(path))
    first._start_engine(first.config.models[0])
    assert first.config.models[0].n_ctx == 1048576
    second_config = load_config(path)
    second_config.models[0].env["TEST_WARP_RAM_GIB"] = "16"
    second = S.Gateway(second_config, config_path=str(path))
    second._start_engine(second.config.models[0])
    assert second.config.models[0].n_ctx == 114688
    assert load_config(path).models[0].warp_auto_context is True


@pytest.mark.parametrize("extra,expected", [
    (["--budget", str(13 * 1024**3)], 131072),
    (["--budget=0", "--budget", str(13 * 1024**3)], 131072),
    (["--budget", str(13 * 1024**3), "--budget=0"], 1048576),
    (["--budget", str(14 * 1024**3), "--vision"], 16384),
])
def test_warp_auto_context_honors_budget_and_vision(tmp_path, monkeypatch, extra, expected):
    from litmoe.engines.warp import WarpEngine

    root = _fake_warp_root(tmp_path)
    monkeypatch.setenv("LITMOE_WARP_DIR", str(root))
    engine = WarpEngine(ModelEntry(
        id="deepseek-v4.1-flash-warp", engine="warp", model_path=str(tmp_path),
        n_ctx=0, extra_args=extra,
    ))
    command = engine.build_command()
    assert int(command[command.index("--ctx") + 1]) == expected


@pytest.mark.parametrize("extra", [["--ctx", "4096"], ["--ctx=4096"], ["--ct=4096"]])
def test_warp_rejects_conflicting_context_flags(tmp_path, monkeypatch, extra):
    from litmoe.engines.warp import WarpEngine

    root = _fake_warp_root(tmp_path)
    monkeypatch.setenv("LITMOE_WARP_DIR", str(root))
    engine = WarpEngine(ModelEntry(
        id="glm-5.3-flash-warp", engine="warp", model_path=str(tmp_path),
        n_ctx=131072, extra_args=extra,
    ))
    with pytest.raises(ValueError, match="not extra_args"):
        engine.build_command()


def test_warp_planning_failure_preserves_config(tmp_path, monkeypatch):
    from litmoe.engines.warp import WarpEngine

    root = _fake_warp_root(tmp_path)
    monkeypatch.setenv("LITMOE_WARP_DIR", str(root))
    path = tmp_path / "models.yaml"
    path.write_text(
        f"models:\n  - id: glm-5.3-flash-warp\n    engine: warp\n"
        f"    model_path: {tmp_path}\n    n_ctx: 65536\n"
        "    env: {TEST_WARP_RAM_GIB: '1'}\n"
        f"  - id: explicit\n    engine: warp\n    model_path: {tmp_path}\n"
        "    n_ctx: 131072\n    warp_auto_context: false\n"
    )
    before = path.read_bytes()
    monkeypatch.setattr(WarpEngine, "start", lambda self, log_dir=None: self.build_command())
    gateway = S.Gateway(load_config(path), config_path=str(path))
    with pytest.raises(RuntimeError, match="context planning failed"):
        gateway._start_engine(gateway.config.models[0])
    assert path.read_bytes() == before


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"x-api-key": "wrong"}])
def test_runtime_control_rejects_invalid_credentials(monkeypatch, headers):
    import asyncio
    import httpx

    def unexpected_start(*args, **kwargs):
        pytest.fail("unauthorized runtime control attempted to construct an engine")

    monkeypatch.setattr(S, "make_engine", unexpected_start)
    gateway = S.Gateway(GatewayConfig(
        api_key="runtime-test-key",
        models=[ModelEntry(id="one", engine="warp", model_path="/unused.waste")],
    ))

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=gateway.app), base_url="http://gateway",
        ) as client:
            status = await client.get("/v1/runtime", headers=headers)
            switch = await client.post(
                "/v1/runtime/model", headers=headers, json={"model": "one"},
            )
            assert status.status_code == 401
            assert switch.status_code == 401

    asyncio.run(scenario())


def test_runtime_unknown_selection_does_not_start_an_engine(monkeypatch):
    import asyncio
    import httpx

    monkeypatch.setattr(
        S, "make_engine",
        lambda *args, **kwargs: pytest.fail("an unknown model triggered engine construction"),
    )
    gateway = S.Gateway(GatewayConfig(
        api_key="runtime-test-key",
        models=[ModelEntry(id="one", engine="warp", model_path="/unused.waste")],
    ))

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=gateway.app), base_url="http://gateway",
            headers={"Authorization": "Bearer runtime-test-key"},
        ) as client:
            before = (await client.get("/v1/runtime")).json()
            response = await client.post("/v1/runtime/model", json={"model": "unknown"})
            assert response.status_code == 404
            after = (await client.get("/v1/runtime")).json()
            assert after["selected_model"] == before["selected_model"]
            assert after["active_model"] == before["active_model"]
            assert after["state"] == before["state"]
            assert "runtime-test-key" not in json.dumps(after)

    asyncio.run(scenario())


@pytest.mark.parametrize("body", [[], None, {}, {"model": []}, {"model": 1}])
def test_runtime_selection_validates_model_before_starting(monkeypatch, body):
    import asyncio
    import httpx

    monkeypatch.setattr(
        S, "make_engine",
        lambda *args, **kwargs: pytest.fail("invalid selection triggered engine construction"),
    )
    gateway = S.Gateway(GatewayConfig(
        models=[ModelEntry(id="one", engine="warp", model_path="/unused.waste")],
    ))

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=gateway.app), base_url="http://gateway",
        ) as client:
            response = await client.post(
                "/v1/runtime/model", content=json.dumps(body),
                headers={"content-type": "application/json"},
            )
            assert response.status_code in (400, 422)

    asyncio.run(scenario())


def _interactive_gateway(monkeypatch, **options):
    from types import SimpleNamespace

    events, live, failures = [], set(), set()

    class Engine:
        def __init__(self, model):
            self.model, self.base_url, self._log_path = model, None, None
            self.process = SimpleNamespace(poll=lambda: None if self.is_running() else 1)

        def set_port(self, port):
            self.port = port

        def default_port(self):
            return self.port

        def start(self, log_dir=None):
            assert not live, "two engines became resident"
            live.add(self.model.id)
            self.base_url = "http://127.0.0.1:1"
            events.append(("start", self.model.id))

        def stop(self):
            live.discard(self.model.id)
            events.append(("stop", self.model.id))

        def is_running(self):
            return self.model.id in live

        async def wait_ready(self, timeout):
            return self.model.id not in failures

    monkeypatch.setattr(S, "make_engine", Engine)
    gateway = S.Gateway(GatewayConfig(models=[
        ModelEntry(id="one", engine="warp", model_path="/unused.waste", aliases=["alias-one"]),
        ModelEntry(id="two", engine="warp", model_path="/unused.waste"),
    ], **options))
    return gateway, events, live, failures


class _ConnectedRequest:
    gone = False

    async def is_disconnected(self):
        return self.gone


async def _eventually(predicate):
    import asyncio
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.001)


def test_single_resident_switch_drains_request_and_preserves_catalog(monkeypatch):
    import asyncio
    import httpx

    async def scenario():
        gateway, events, live, _ = _interactive_gateway(monkeypatch)
        await gateway.load_engines()
        lease = await gateway.runtime.acquire("alias-one", _ConnectedRequest())
        switch = asyncio.create_task(gateway.runtime.switch("two"))
        await _eventually(lambda: gateway.runtime.state == "draining")
        assert live == {"one"}
        assert events == [("start", "one")]
        await lease.close()
        assert (await switch)["active_model"] == "two"
        assert events == [("start", "one"), ("stop", "one"), ("start", "two")]
        assert [m.id for m in gateway.config.models] == ["one", "two"]
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway.app),
                                     base_url="http://gateway") as client:
            assert [m["id"] for m in (await client.get("/v1/models")).json()["data"]] == ["two"]
            result = await client.post("/v1/chat/completions", json={"model": "one", "messages": []})
            assert result.status_code == 409
        await gateway.shutdown()
        assert not live

    asyncio.run(scenario())


def test_failed_switch_never_substitutes_a_different_model(monkeypatch):
    import asyncio
    from fastapi import HTTPException

    async def scenario():
        gateway, events, live, failures = _interactive_gateway(monkeypatch)
        await gateway.load_engines()
        failures.add("two")
        with pytest.raises(HTTPException) as error:
            await gateway.runtime.switch("two")
        assert error.value.status_code == 503
        assert not live
        assert gateway.runtime.status()["active_model"] is None
        assert gateway.runtime.status()["selected_model"] == "two"
        assert gateway.runtime.status()["state"] == "failed"
        assert events == [("start", "one"), ("stop", "one"), ("start", "two"), ("stop", "two")]
        await gateway.runtime.switch("one")
        assert live == {"one"}
        await gateway.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize("ending", ["disconnect", "timeout"])
def test_bounded_queue_removes_waiters_without_dispatch(monkeypatch, ending):
    import asyncio
    from fastapi import HTTPException

    async def scenario():
        gateway, events, live, _ = _interactive_gateway(
            monkeypatch, max_queue_size=1, queue_timeout=0.1,
        )
        await gateway.load_engines()
        lease = await gateway.runtime.acquire("one", _ConnectedRequest())
        queued_request = _ConnectedRequest()
        waiter = asyncio.create_task(gateway.runtime.acquire("one", queued_request))
        await _eventually(lambda: gateway.runtime.queue_depth == 1)
        with pytest.raises(HTTPException) as full:
            await gateway.runtime.acquire("one", _ConnectedRequest())
        assert full.value.status_code == 429
        queued_request.gone = ending == "disconnect"
        with pytest.raises(HTTPException) as removed:
            await waiter
        assert removed.value.status_code == (499 if ending == "disconnect" else 429)
        assert gateway.runtime.queue_depth == 0
        assert events == [("start", "one")]
        assert live == {"one"}
        await lease.close()
        later = await gateway.runtime.acquire("one", _ConnectedRequest())
        await later.close()
        await gateway.shutdown()

    asyncio.run(scenario())


def test_stream_disconnect_stops_owned_engine_and_reloads_on_next_request(monkeypatch):
    import asyncio
    from starlette.requests import ClientDisconnect
    from litmoe.runtime import LeasedStream

    async def scenario():
        gateway, events, live, _ = _interactive_gateway(monkeypatch)
        await gateway.load_engines()
        lease = await gateway.runtime.acquire("one", _ConnectedRequest())

        async def chunks():
            yield b"data: first\n\n"
            pytest.fail("disconnected consumer requested more generation")

        class Resource:
            async def aclose(self):
                pass

        response = LeasedStream(chunks(), lease, Resource(), Resource())

        async def send(message):
            if message["type"] == "http.response.body":
                raise OSError("consumer closed socket")

        async def receive():
            await asyncio.Event().wait()

        with pytest.raises(ClientDisconnect):
            await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
        assert not live
        assert gateway.runtime.state == "stopped"
        assert not gateway.runtime.lock.locked()
        later = await gateway.runtime.acquire("one", _ConnectedRequest())
        assert live == {"one"}
        assert events == [("start", "one"), ("stop", "one"), ("start", "one")]
        await later.close()
        await gateway.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize("engine_kind,endpoint,keeps_resident", [
    ("warp", "chat/completions", True),
    ("llamacpp", "chat/completions", False),
    ("warp", "completions", False),
])
def test_stream_abort_respects_backend_contract(monkeypatch, engine_kind, endpoint, keeps_resident):
    import httpx
    from starlette.requests import ClientDisconnect

    async def scenario():
        gateway, events, live, _ = _interactive_gateway(monkeypatch)
        await gateway.runtime.switch("one")
        gateway.runtime.engine.model.engine = engine_kind
        lease = await gateway.runtime.acquire("one", _ConnectedRequest())
        upstream_closed = asyncio.Event()

        class Tokens(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'data: {"choices":[{"delta":{"reasoning_content":"Thinking"}}]}\n\n'
                await asyncio.Event().wait()

            async def aclose(self):
                upstream_closed.set()

        client = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, stream=Tokens()),
        ))

        async def connect(*args):
            response = await client.send(client.build_request("POST", "http://engine"), stream=True)
            return client, response

        monkeypatch.setattr(S, "_connect_stream", connect)
        response = await gateway._forward({"model": "one", "stream": True}, endpoint, False, lease)
        waiter = asyncio.create_task(gateway.runtime.acquire("one", _ConnectedRequest()))
        await _eventually(lambda: gateway.runtime.queue_depth == 1)
        assert not waiter.done()

        async def send(message):
            if message["type"] == "http.response.body":
                raise OSError("client disconnected")

        async def receive():
            await asyncio.Event().wait()

        with pytest.raises(ClientDisconnect):
            await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
        assert upstream_closed.is_set()
        next_lease = await waiter
        assert live == {"one"}
        assert events == ([("start", "one")] if keeps_resident else
                          [("start", "one"), ("stop", "one"), ("start", "one")])
        await next_lease.close()
        await gateway.shutdown()
        assert not live

    asyncio.run(scenario())


def test_busy_gateway_port_does_not_start_model(monkeypatch):
    import socket

    async def unexpected_load(self):
        pytest.fail("occupied gateway port started native model loading")

    monkeypatch.setattr(S.Gateway, "load_engines", unexpected_load)
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        config = GatewayConfig(host="127.0.0.1", port=occupied.getsockname()[1], models=[
            ModelEntry(id="one", engine="warp", model_path="/unused.waste"),
        ])
        with pytest.raises(SystemExit) as error:
            S.run(config)
        assert error.value.code != 0


def test_benchmark_distinguishes_role_reasoning_text_and_final_usage():
    from litmoe.benchmark import StreamMetrics

    metrics = StreamMetrics()
    metrics.event(json.dumps({"choices": [{"delta": {"role": "assistant"}}]}), 0.1)
    assert metrics.first_generated_s is None
    metrics.event(json.dumps({"choices": [{"delta": {"reasoning_content": "reason"}}]}), 0.3)
    metrics.event(json.dumps({"choices": [{"delta": {"content": "answer"}}]}), 0.8)
    metrics.event(json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}), 1.0)
    metrics.event(json.dumps({"choices": [], "usage": {"completion_tokens": 7}}), 1.1)
    metrics.event("[DONE]", 1.2)
    assert metrics.first_generated_s == 0.3
    assert metrics.first_text_s == 0.8
    assert metrics.usage["completion_tokens"] == 7
    assert metrics.done and metrics.finish_reason == "stop"


@pytest.mark.parametrize("body", [
    'data: {"error":{"message":"private-response-marker"}}\n\n',
    'data: not-json\n\n',
    'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n',
])
def test_benchmark_rejects_error_and_incomplete_streams_without_leaking_text(body):
    import asyncio
    import httpx
    from litmoe.benchmark import measure

    async def scenario():
        transport = httpx.MockTransport(lambda request: httpx.Response(
            200, content=body.encode(), headers={"content-type": "text/event-stream"},
        ))
        async with httpx.AsyncClient(transport=transport) as client:
            result = await measure(client, "http://engine/v1/chat/completions", b"{}", {}, 1)
        assert not result["success"]
        assert result["error"] is not None
        assert result["output_tokens"] is None
        assert result["output_tokens_per_response_second"] is None
        assert "private-response-marker" not in json.dumps(result)

    asyncio.run(scenario())


def test_simultaneous_admissions_cannot_overfill_the_queue(monkeypatch):
    from fastapi import HTTPException

    async def scenario():
        gateway, _, _, _ = _interactive_gateway(monkeypatch, max_queue_size=1)
        await gateway.load_engines()
        requests = [asyncio.create_task(gateway.runtime.acquire("one", _ConnectedRequest()))
                    for _ in range(10)]
        await _eventually(lambda: requests[0].done())
        lease = requests[0].result()
        assert gateway.runtime.queue_depth == 1
        assert all(task.done() and isinstance(task.exception(), HTTPException)
                   and task.exception().status_code == 429 for task in requests[2:])
        await lease.close()
        second = await requests[1]
        await second.close()
        await gateway.shutdown()

    asyncio.run(scenario())


def test_cancellation_during_native_start_recovers_process_ownership(monkeypatch):
    import threading

    async def scenario():
        gateway, events, live, _ = _interactive_gateway(monkeypatch)
        entered, release = threading.Event(), threading.Event()
        original = gateway.runtime.start_engine

        def delayed_start(model):
            entered.set()
            assert release.wait(timeout=2)
            return original(model)

        gateway.runtime.start_engine = delayed_start
        pending = asyncio.create_task(gateway.runtime.acquire("one", _ConnectedRequest()))
        assert await asyncio.to_thread(entered.wait, 1)
        pending.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert not live
        assert events == [("start", "one"), ("stop", "one")]
        assert not gateway.runtime.lock.locked()

    asyncio.run(scenario())


def test_concurrent_switches_never_overlap_resident_engines(monkeypatch):
    async def scenario():
        gateway, events, live, _ = _interactive_gateway(monkeypatch)
        await gateway.load_engines()
        first, second = await asyncio.gather(
            gateway.runtime.switch("two"), gateway.runtime.switch("one"),
        )
        assert first["active_model"] == "two"
        assert second["active_model"] == "one"
        assert live == {"one"}
        assert events == [("start", "one"), ("stop", "one"), ("start", "two"),
                          ("stop", "two"), ("start", "one")]
        assert gateway.runtime.status()["generation"] == 3
        await gateway.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize("extra", [["-np", "4"], ["--parallel=2"], ["-c", "4096"], ["--ctx-size=4096"]])
def test_gateway_rejects_flags_that_override_advertised_context_or_single_slot(extra, monkeypatch):
    model = ModelEntry(id="one", engine="llamacpp", model_path="/unused.gguf", extra_args=extra)
    gateway = S.Gateway(GatewayConfig(models=[model]))
    monkeypatch.setattr(S, "make_engine", lambda *args: pytest.fail("conflicting engine settings started"))
    with pytest.raises(ValueError):
        gateway._start_engine(model)


@pytest.mark.parametrize("action", ["cancel", "switch"])
def test_failed_stop_retains_ownership_without_reopening_admission(monkeypatch, action):
    from fastapi import HTTPException

    async def scenario():
        gateway, _, live, _ = _interactive_gateway(monkeypatch)
        await gateway.load_engines()
        engine = gateway.runtime.engine
        original_stop = engine.stop

        def failed_stop():
            raise OSError("native stop failed")

        monkeypatch.setattr(engine, "stop", failed_stop)
        if action == "cancel":
            lease = await gateway.runtime.acquire("one", _ConnectedRequest())
            with pytest.raises(OSError):
                await lease.close(abandoned=True)
        else:
            with pytest.raises(HTTPException):
                await gateway.runtime.switch("two")
        assert gateway.runtime.status()["state"] == "failed"
        assert gateway.runtime.engine is engine
        assert live == {"one"}
        with pytest.raises(HTTPException) as rejected:
            await gateway.runtime.acquire("one", _ConnectedRequest())
        assert rejected.value.status_code == 503
        for model in ("one", "two"):
            with pytest.raises(HTTPException) as rejected:
                await gateway.runtime.switch(model)
            assert rejected.value.status_code == 503
            assert live == {"one"}
        monkeypatch.setattr(engine, "stop", original_stop)
        await gateway.runtime.switch("two")
        assert live == {"two"}
        await gateway.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize("outcome", ["ready", "failed", "shutdown"])
def test_http_startup_does_not_wait_for_model_readiness(monkeypatch, outcome):
    import httpx

    async def scenario():
        gateway, events, live, _ = _interactive_gateway(monkeypatch)
        entered, release = asyncio.Event(), asyncio.Event()
        original_start = gateway.runtime.start_engine

        async def held_readiness(timeout):
            entered.set()
            await release.wait()
            return outcome == "ready"

        def start(model):
            engine = original_start(model)
            engine.wait_ready = held_readiness
            return engine

        gateway.runtime.start_engine = start
        lifespan = gateway.app.router.lifespan_context(gateway.app)
        startup = asyncio.create_task(lifespan.__aenter__())
        closed = False
        try:
            await asyncio.wait_for(entered.wait(), 2)
            # ASGI startup must finish while native readiness remains blocked.
            await asyncio.wait_for(asyncio.shield(startup), 0.5)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=gateway.app), base_url="http://gateway",
            ) as client:
                health = await client.get("/health")
                assert health.status_code == 200
                assert health.json()["runtime"]["state"] == "loading"
                assert (await client.get("/v1/models")).json()["data"] == []
                response = await client.post("/v1/chat/completions", json={
                    "model": "one", "messages": [{"role": "user", "content": "hi"}],
                })
                assert response.status_code == 503
                if outcome == "shutdown":
                    await asyncio.wait_for(lifespan.__aexit__(None, None, None), 2)
                    closed = True
                    assert not live
                    assert gateway.runtime.engine is None
                    assert events == [("start", "one"), ("stop", "one")]
                else:
                    release.set()
                    await _eventually(lambda: gateway.runtime.state == outcome)
                    discovered = (await client.get("/v1/models")).json()["data"]
                    assert {model["id"] for model in discovered} == (
                        {"one", "alias-one"} if outcome == "ready" else set()
                    )
                    assert live == ({"one"} if outcome == "ready" else set())
        finally:
            release.set()
            await startup
            if not closed:
                await lifespan.__aexit__(None, None, None)
        assert not live

    asyncio.run(scenario())


@pytest.mark.parametrize("abandoned", [False, True])
@pytest.mark.parametrize("acknowledgement", [True, False, "error"])
def test_native_lease_keeps_admission_until_idle_or_process_stop(monkeypatch, abandoned, acknowledgement):
    async def scenario():
        gateway, events, live, _ = _interactive_gateway(monkeypatch)
        await gateway.runtime.switch("one")
        engine = gateway.runtime.engine
        engine.supports_cooperative_cancel = True
        entered, finish = asyncio.Event(), asyncio.Event()

        async def wait_idle(timeout):
            entered.set()
            await finish.wait()
            if acknowledgement == "error":
                raise RuntimeError("native status unavailable")
            return acknowledgement

        engine.wait_idle = wait_idle
        lease = await gateway.runtime.acquire("one", _ConnectedRequest())
        closing = asyncio.create_task(lease.close(abandoned=abandoned))
        await asyncio.wait_for(entered.wait(), 1)
        queued = asyncio.create_task(gateway.runtime.acquire("one", _ConnectedRequest()))
        await _eventually(lambda: gateway.runtime.queue_depth == 1)
        assert not queued.done()
        assert events == [("start", "one")]
        finish.set()
        await closing
        next_lease = await queued
        assert live == {"one"}
        if acknowledgement is True:
            assert events == [("start", "one")]
        else:
            assert events == [("start", "one"), ("stop", "one"), ("start", "one")]
        await next_lease.close()
        await gateway.shutdown()

    asyncio.run(scenario())


def test_native_anthropic_transport_error_never_emits_openai_done():
    import httpx

    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'event: ping\ndata: {"type":"ping"}\n\n'
            raise httpx.ReadError("backend disconnected")

    async def scenario():
        client = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, stream=BrokenStream())))
        response = await client.send(client.build_request("POST", "http://engine"), stream=True)
        data = b"".join([chunk async for chunk in S._stream_response(client, response, anthropic=True)])
        assert b"event: error\n" in data
        assert b'"type": "error"' in data
        assert b"data: [DONE]" not in data

    asyncio.run(scenario())
