"""Unit tests for litmoe (no network, no engines)."""
from __future__ import annotations

import json
import shutil
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
    # fits unchanged
    assert M.fit_context(65_536, 20.0, 96.0, 262144) == (262144, None)
    # must shrink: 60 GB weights, 96 GB RAM -> budget 83.4, kv room 22.4 GB -> ~341K tokens @65KB... use big kv
    ctx, note = M.fit_context(1_000_000, 60.0, 96.0, 262144)
    assert ctx < 262144 and ctx % 4096 == 0 and ctx >= 8192 and "reduced" in note
    # weights do not fit at all: capped, never below 8192
    ctx, note = M.fit_context(65_536, 600.0, 96.0, 1048576)
    assert ctx == 32768 and "may not fit" in note


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


# ---------------------------------------------------------------------------
# server
# ---------------------------------------------------------------------------

def test_port_allocation_skips_gateway_port_without_collisions():
    # probe=False: pure arithmetic, independent of what is listening on this host
    assert S.allocate_engine_ports(3, gateway_port=8082, probe=False) == [8081, 8083, 8084]
    assert S.allocate_engine_ports(2, gateway_port=8080, probe=False) == [8081, 8082]
    assert S.allocate_engine_ports(0, gateway_port=8080, probe=False) == []


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


def test_dead_engine_is_a_503_not_a_broken_200_stream():
    """Observed: `litmoe stop` under a live gateway killed the engines; the next
    streaming request got HTTP 200 and then 'Stream error'. A dead engine must
    fail before the response is committed, with the reason and the log path."""
    from fastapi import HTTPException

    class _Proc:
        def __init__(self, code): self._code = code
        def poll(self): return self._code

    class _Eng:
        def __init__(self, mid, code):
            self.model = ModelEntry(id=mid, engine="llamacpp", model_path="/tmp/x.gguf")
            self.base_url = "http://127.0.0.1:8081"
            self.process = _Proc(code)
            self._log_path = Path("logs") / f"{mid}.log"

    gw = S.Gateway(GatewayConfig(models=[]))
    gw.engines = {"alive": _Eng("alive", None), "killed": _Eng("killed", -15), "crashed": _Eng("crashed", 1)}

    assert gw._resolve("alive")[0].id == "alive"
    with pytest.raises(HTTPException) as e:
        gw._resolve("killed")
    assert e.value.status_code == 503 and "was stopped" in e.value.detail and "logs/killed.log" in e.value.detail
    with pytest.raises(HTTPException) as e:
        gw._resolve("crashed")
    assert e.value.status_code == 503 and "exited with code 1" in e.value.detail


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


def test_fit_together_is_a_joint_budget_not_per_model():
    """Each of these fits a 103 GB Mac (77 GB budget) alone; loaded at once they OOM Metal."""
    picks = ["gemma-4-26b-a4b", "gpt-oss-120b", "qwen3.5-122b-a10b"]
    assert all(M.ram_needed_gb(m) <= 77 for m in picks)
    kept, dropped = M.fit_together(picks, 77.0)
    assert kept == ["gemma-4-26b-a4b"]
    assert dropped == ["gpt-oss-120b", "qwen3.5-122b-a10b"]
    # Order is preserved and a big budget keeps everything.
    assert M.fit_together(picks, 768.0) == (picks, [])
    # Headroom is counted once, not per model.
    assert M.ram_needed_together_gb([10.0, 10.0]) == 20.0 - M._OS_HEADROOM_GB


def test_init_writes_only_models_that_fit_together(tmp_path, monkeypatch):
    """Regression: init wrote 140 GB of models for a 103 GB Mac; serve then OOMed on Metal."""
    from click.testing import CliRunner
    import litmoe.cli.main as CM

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(CM, "get_total_memory_bytes", lambda: int(103e9))
    monkeypatch.setattr(CM, "is_macos", lambda: True)
    r = CliRunner().invoke(CM.cli, ["init"])
    assert r.exit_code == 0, r.output
    ids = [m.id for m in load_config(tmp_path / "models.yaml").models]
    kept, _ = M.fit_together(ids, 103 * 0.75)
    assert kept == ids, (ids, r.output)          # everything written loads together
    assert ids == ["gemma-4-26b-a4b"]
    assert "serve --model" in r.output            # and the user is told how to run the others


def test_serve_gate_two_thresholds(tmp_path, monkeypatch):
    """Over the GPU budget but within RAM -> start with a warning (partial CPU offload).
    Over RAM -> refuse with a useful hint, unless --force. Positional ids == --model."""
    from click.testing import CliRunner
    import litmoe.cli.main as CM

    cfg = tmp_path / "models.yaml"
    cfg.write_text(
        "port: 8080\nmodels:\n"
        "  - {id: gemma-4-26b-a4b, engine: llamacpp, model_path: 'unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL', n_ctx: 32768}\n"
        "  - {id: qwen3.5-122b-a10b, engine: llamacpp, model_path: 'unsloth/Qwen3.5-122B-A10B-GGUF:UD-IQ4_XS', n_ctx: 262144}\n"
        "  - {id: qwen3.8-flash-next, engine: llamacpp, model_path: 'unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL', n_ctx: 32768}\n")
    # 103 GB Mac: GPU budget ~77 GB, RAM limit ~90 GB.
    monkeypatch.setattr(S, "get_total_memory_bytes", lambda: int(103.1e9))
    monkeypatch.setattr(S, "is_macos", lambda: True)
    monkeypatch.setattr(CM, "is_macos", lambda: True)
    started = []
    monkeypatch.setattr(S, "run", lambda cfg, **kw: started.append([m.id for m in cfg.models]))
    run = lambda *args: CliRunner().invoke(CM.cli, ["serve", "-c", str(cfg), *args])

    # All three: far over RAM -> refused, and the hint names one that fits, not the same set.
    r = run()
    assert r.exit_code == 1 and not started, r.output
    assert "Serve one that fits:   litmoe serve gemma-4-26b-a4b" in r.output

    # Fits the GPU budget -> silent start. Positional form.
    r = run("gemma-4-26b-a4b")
    assert r.exit_code == 0 and started == [["gemma-4-26b-a4b"]], r.output
    assert "needs" not in r.output

    # ~78 GB: over the 77 GB Metal budget, under the 90 GB RAM limit -> starts, warns about CPU offload.
    r = run("--model", "qwen3.5-122b-a10b")
    assert r.exit_code == 0 and started[-1] == ["qwen3.5-122b-a10b"], r.output
    assert "part of it will run on the CPU" in r.output

    # ~129 GB single model: over RAM -> refused; the hint is a smaller quant, never '--model <itself>'.
    r = run("qwen3.8-flash-next")
    assert r.exit_code == 1 and started[-1] != ["qwen3.8-flash-next"], r.output
    assert "--quant" in r.output and "serve qwen3.8-flash-next" not in r.output

    r = run("qwen3.8-flash-next", "--force")
    assert r.exit_code == 0 and started[-1] == ["qwen3.8-flash-next"], r.output

    r = run("nope")
    assert r.exit_code == 1 and "not in" in r.output
