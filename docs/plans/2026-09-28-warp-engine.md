# WARP Engine Integration Implementation Plan

> Context correction (2026-09-29): this historical plan's “0 preserves the container default” assumption is superseded. Upstream `waste_open` maps zero to 4096. litmoe now resolves WARP zero to 65536, always passes `--ctx`, and persists legacy zero repairs. Positive limits remain explicit choices; see [SETUP](../SETUP.md).

> **For Claude:** REQUIRED SUB-SKILL: Use subagent-driven development and test-driven development to implement this plan task-by-task.

**Goal:** Add a first-class local WARP engine so litMoE can serve existing `.waste` containers—especially GLM-5.3-Flash and DeepSeek-V4.1-Flash—through its OpenAI and Anthropic endpoints without a remote inference API.

**Architecture:** WARP remains the inference runtime. A new `WarpEngine` starts WARP's upstream `serve/__main__.py` on a gateway-assigned loopback port and reuses litMoE's existing process supervision, `/health` polling, OpenAI passthrough, and Anthropic translation. `litmoe install --engine warp` installs a pinned WARP source revision and builds its CLI/shared library, but model download/conversion remains explicit because a GLM container requires 306 GiB of source weights plus 112 GB output and DeepSeek requires more.

**Tech Stack:** Python 3.10+, Pydantic, Click, FastAPI/httpx, pytest, Git/Make, upstream SQLiteAI WARP.

**Repository policy:** Do not commit, push, or release without explicit user authorization. Use working-tree checkpoints instead of the commit steps normally prescribed by the planning workflow.

---

## Task 1: WARP configuration and engine adapter

**Files:**
- Modify: `litmoe/config.py:59-74`
- Create: `litmoe/engines/warp.py`
- Modify: `litmoe/engines/__init__.py:1-20`
- Modify: `tests/test_litmoe.py` near the existing engine command tests

**Step 1: Write failing behavior tests**

Add tests that:

1. construct `ModelEntry(id="glm-warp", engine="warp", model_path="~/models/glm53.waste", n_ctx=0)`;
2. create a temporary WARP root containing `serve/__main__.py` and the platform shared-library filename;
3. set `LITMOE_WARP_DIR` to that root and assert the command is:

```python
[
    sys.executable,
    str(root / "serve" / "__main__.py"),
    str(expand_path("~/models/glm53.waste")),
    "--host", "127.0.0.1",
    "--port", "8087",
    "--model-id", "glm-warp",
    *extra_args,
]
```

When `n_ctx > 0`, assert `--ctx <n>` appears before `extra_args`; when `n_ctx == 0`, omit it so WARP uses the container default. Assert `/health`, factory dispatch, and clear `FileNotFoundError` messages for a missing checkout or server/shared library.

**Step 2: Run the focused tests and observe the expected failure**

```bash
python3.12 -m pytest -q tests/test_litmoe.py -k warp
```

Expected: fail because `warp` is not a valid engine and `litmoe.engines.warp` does not exist.

**Step 3: Implement the minimal adapter**

- Extend `ModelEntry.engine` to `Literal["ktransformers", "llamacpp", "warp"]`.
- In `litmoe/engines/warp.py`, resolve the source root in this order:
  1. `LITMOE_WARP_DIR` when set;
  2. `$LITMOE_PREFIX/lib/warp` (default prefix `~/.local`);
  3. the parent of a `waste` executable on `PATH` when it contains `serve/__main__.py`.
- Require both `serve/__main__.py` and `libwaste.so` (Linux), `libwaste.dylib` (macOS), or `libwaste.dll` (Windows).
- Build the upstream server command with `sys.executable`, the expanded local `.waste` path, loopback host, assigned port, model id, optional context, and verbatim `extra_args`.
- Inherit lifecycle methods from `Engine`.
- Export `WarpEngine`/`warp_installed` and dispatch it in `make_engine()`.

**Step 4: Run the focused tests**

```bash
python3.12 -m pytest -q tests/test_litmoe.py -k warp
```

Expected: all WARP adapter tests pass.

## Task 2: Reproducible WARP source installer and doctor support

**Files:**
- Modify: `litmoe/cli/install.py`
- Modify: `litmoe/cli/main.py`
- Modify: `tests/test_litmoe.py`

**Step 1: Write failing CLI tests**

Add tests that verify:

- `litmoe install --help` accepts `warp`;
- positional `litmoe install warp` selects only the WARP installer and never enters model download code when no `--model` was supplied;
- `doctor` reports WARP installed/not installed using the adapter probe.

Use dependency seams already present in the module; do not assert incidental subprocess argument forwarding. The permanent tests must assert user-visible selection and status behavior.

**Step 2: Run the focused tests and observe failure**

```bash
python3.12 -m pytest -q tests/test_litmoe.py -k 'warp and (install or doctor)'
```

Expected: `warp` is rejected by the Click choice and doctor has no WARP line.

**Step 3: Implement source installation**

Add constants:

```python
WARP_REPO = "https://github.com/sqliteai/warp.git"
WARP_COMMIT = "09fcff352ca55223b08ee222d15054b90546c6a9"
```

Implement `install_warp(prefix: Path, ref: str = WARP_COMMIT) -> Path`:

- require `git` and `make`;
- clone/fetch the exact revision into a temporary sibling of `<prefix>/lib/warp`;
- build `waste`, the platform shared library, and run upstream `make check` (model-free synthetic test);
- verify `waste`, `serve/__main__.py`, and the shared library exist;
- replace only the installer-owned `<prefix>/lib/warp` directory after successful verification;
- install `<prefix>/bin/waste` as a symlink to the built CLI;
- return the WARP root.

Add `warp` to `--engine` choices and positional target parsing, then call this installer without invoking any model download. Keep `both` semantics unchanged to avoid an unrelated CLI compatibility break. Update `doctor` to report WARP and explain that it serves local `.waste` containers.

**Step 4: Run focused tests**

```bash
python3.12 -m pytest -q tests/test_litmoe.py -k 'warp or doctor'
```

Expected: pass.

## Task 3: Public configuration and documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/SETUP.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/METHODOLOGY.md`
- Modify: `examples/models.yaml`
- Modify: `pyproject.toml`
- Modify: `litmoe/__init__.py`

**Step 1: Update authoritative claims**

- List WARP alongside llama.cpp and ktransformers everywhere the supported engines are enumerated.
- Document `litmoe install --engine warp` and the pinned-source build.
- Add a local config example:

```yaml
- id: glm-5.3-flash-warp
  engine: warp
  model_path: ~/models/glm53.waste
  n_ctx: 0
  extra_args: ["--no-thinking"]
```

Explain that `n_ctx: 0` preserves the container default and WARP sizes its own memory budget. `extra_args` accepts upstream flags such as `--budget`, `--threads`, `--cpus`, `--cache`, `--vision`, and `--verify`.

**Step 2: State the storage and performance boundary accurately**

Document primary-source WARP figures as upstream measurements, not litMoE measurements:

- GLM-5.3-Flash: 112 GB container, 5.14 GB resident floor, 3.32 tok/s short / 3.86 tok/s long on the upstream 64 GB M5 Pro host;
- DeepSeek-V4.1-Flash: 299 GB container, 4.86 GB floor, about 3.7 tok/s;
- GLM conversion needs 306 GiB staging plus 112 GB output; litMoE does not auto-download or convert model weights.

State that internal NVMe is required for the published throughput and this repository's current persistent disk is not equivalent.

**Step 3: Update diagrams only when their source text explicitly claims two engines**

Keep SVG assets unchanged unless their visible labels would otherwise be false; do not regenerate decorative assets.

## Task 4: Verification and actual WARP smoke

**Step 1: Run the complete repository suite**

```bash
python3.12 -m pytest -q
```

Expected: all tests pass; the existing Starlette deprecation warning may remain.

**Step 2: Exercise the changed installer against upstream WARP**

```bash
rm -rf /tmp/litmoe-warp-smoke
python3.12 -m litmoe.cli.main install --engine warp --prefix /tmp/litmoe-warp-smoke
```

Expected: exact source revision checked out, `make check` passes, and these artifacts exist:

- `/tmp/litmoe-warp-smoke/lib/warp/waste`
- `/tmp/litmoe-warp-smoke/lib/warp/libwaste.so` on Linux
- `/tmp/litmoe-warp-smoke/lib/warp/serve/__main__.py`
- `/tmp/litmoe-warp-smoke/bin/waste`

**Step 3: Smoke the actual upstream server with its synthetic model**

Use the model fixture generated by upstream `make check`, start it through a real `WarpEngine`, observe `GET /health == 200`, and send one OpenAI chat-completions request through litMoE. If upstream removes the generated fixture after tests, generate it using the repository's own model-free test target; do not substitute a fake server.

**Step 4: Run static package checks and inspect user-visible commands**

```bash
python3.12 -m compileall -q litmoe
python3.12 -m litmoe.cli.main install --help
LITMOE_PREFIX=/tmp/litmoe-warp-smoke python3.12 -m litmoe.cli.main doctor
```

Expected: WARP appears in help and doctor reports it installed.

**Step 5: Clean smoke artifacts**

Remove `/tmp/litmoe-warp-smoke` and any generated logs/PID files. Do not remove or alter user model data.
