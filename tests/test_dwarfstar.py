"""Safety boundaries for managed native runtimes and cancellation acknowledgement."""
import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from litmoe.config import ModelEntry
from litmoe.engines import dwarfstar as D
from litmoe.cli import install as I


@pytest.fixture
def managed_runtime(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    root.mkdir()
    (root / "ds4-server").write_text("test executable")
    (root / "ds4-server").chmod(0o700)
    (root / "LICENSE").write_text("fixture")
    (root / "metal").mkdir()
    (root / ".litmoe-revision").write_text(D.DWARFSTAR_COMMIT)
    (root / ".litmoe-patch-sha256").write_text(hashlib.sha256(D.DWARFSTAR_PATCH.read_bytes()).hexdigest())
    monkeypatch.setenv("LITMOE_DWARFSTAR_DIR", str(root))
    monkeypatch.setenv("LITMOE_DWARFSTAR_CACHE", str(tmp_path / "cache"))
    return root


@pytest.mark.parametrize("marker", [".litmoe-revision", ".litmoe-patch-sha256"])
def test_stale_runtime_cannot_advertise_native_protocol(managed_runtime, marker):
    (managed_runtime / marker).write_text("obsolete")
    assert not D.is_installed()
    with pytest.raises(FileNotFoundError):
        D.runtime_root()


def test_replaced_model_or_runtime_never_reuses_disk_state(managed_runtime, tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"first checkpoint")
    engine = D.DwarfStarEngine(ModelEntry(id="model", engine="dwarfstar", model_path=str(model)))
    first = engine._kv_disk_dir(managed_runtime, model)
    replacement = tmp_path / "replacement.gguf"
    replacement.write_bytes(b"other checkpoint")
    replacement.replace(model)
    second = engine._kv_disk_dir(managed_runtime, model)
    assert second != first
    (managed_runtime / ".litmoe-patch-sha256").write_text("new runtime")
    assert engine._kv_disk_dir(managed_runtime, model) != second


@pytest.mark.parametrize("flag", ["--host=0.0.0.0", "-c", "--ctx", "--batched-session", "--chdir"])
def test_extra_arguments_cannot_escape_owned_serving_limits(managed_runtime, tmp_path, flag):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    engine = D.DwarfStarEngine(ModelEntry(id="model", engine="dwarfstar", model_path=str(model), extra_args=[flag]))
    with pytest.raises(ValueError):
        engine.build_command()


@pytest.mark.parametrize("stage", ["build", "publish"])
def test_failed_install_preserves_previous_runtime(tmp_path, monkeypatch, stage):
    destination = tmp_path / "lib" / "dwarfstar"
    destination.mkdir(parents=True)
    previous = destination / "previous"
    previous.write_bytes(b"keep the usable previous installation")
    monkeypatch.setattr(I, "is_macos", lambda: True)
    monkeypatch.setattr(I.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(I.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(I._warp_models, "_snapshot_live_processes", lambda paths: [])
    is_file = Path.is_file
    monkeypatch.setattr(Path, "is_file", lambda path: str(path) == "/usr/bin/clang" or is_file(path))

    def run(command, *, cwd, label, timeout):
        if label == "DwarfStar build":
            if stage == "build":
                raise RuntimeError("compiler failed")
            cwd.mkdir(parents=True)
            (cwd / "ds4-server").write_text("replacement executable")
            (cwd / "ds4-server").chmod(0o700)
            (cwd / "LICENSE").write_text("license")
            (cwd / "metal").mkdir()

    monkeypatch.setattr(I, "_run_warp_command", run)
    replace = I.os.replace

    def publish(source, target):
        if Path(source).name == "checkout":
            raise OSError("publish failed")
        return replace(source, target)

    if stage == "publish":
        monkeypatch.setattr(I.os, "replace", publish)
    with pytest.raises((RuntimeError, OSError)):
        I.install_dwarfstar(tmp_path)
    assert previous.read_bytes() == b"keep the usable previous installation"
    assert not list(destination.parent.glob(".dwarfstar-install-*"))


def _idle_engine():
    engine = D.DwarfStarEngine(ModelEntry(id="model", engine="dwarfstar", model_path="/unused.gguf"))
    engine.process = SimpleNamespace(poll=lambda: None)
    engine.base_url = "http://127.0.0.1:8081"
    return engine


@pytest.mark.parametrize("status", [[], {}, {"status": "ok", "active_requests": False, "queued_requests": 0},
                                   {"status": "ok", "active_requests": 0, "queued_requests": -1}])
def test_malformed_status_cannot_acknowledge_quiescence(monkeypatch, status):
    client = httpx.AsyncClient
    monkeypatch.setattr(D.httpx, "AsyncClient", lambda **kw: client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=status))))
    assert asyncio.run(_idle_engine().wait_idle(timeout=0.2)) is False


def test_quiescence_waits_for_both_queue_and_worker(monkeypatch):
    states = iter([(0, 1), (1, 0), (0, 0)])
    client = httpx.AsyncClient

    def respond(request):
        active, queued = next(states)
        return httpx.Response(200, json={"status": "ok", "active_requests": active, "queued_requests": queued})

    monkeypatch.setattr(D.httpx, "AsyncClient", lambda **kw: client(transport=httpx.MockTransport(respond)))
    assert asyncio.run(_idle_engine().wait_idle(timeout=1)) is True
    assert next(states, None) is None


def test_stalled_status_has_a_hard_deadline(monkeypatch):
    client = httpx.AsyncClient

    async def stalled(request):
        await asyncio.Event().wait()

    monkeypatch.setattr(D.httpx, "AsyncClient", lambda **kw: client(transport=httpx.MockTransport(stalled)))

    async def scenario():
        return await asyncio.wait_for(_idle_engine().wait_idle(timeout=0.02), timeout=1)

    assert asyncio.run(scenario()) is False
