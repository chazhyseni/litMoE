# WARP Model Installation Parity Implementation Plan

> **Archived planning record.** This is retained for design history, not as current operating instructions. Shipped behavior is documented in [SETUP](../SETUP.md), [ARCHITECTURE](../ARCHITECTURE.md), and [HARNESSES](../HARNESSES.md).

> Context correction (2026-09-29): the zero-preservation tests and `--n-ctx` rejection in this historical plan are superseded. Installs default to automatic native-context fitting via WARP's memory planner; a positive `--n-ctx` selects fixed mode. Startup migrates unmarked legacy 0/65536 entries and persists the resolved window with its auto/fixed policy. Tests cover both WARP catalog models, restart sizing, and explicit overrides; see [SETUP](../SETUP.md).

**Goal:** Make GLM-5.3-Flash and DeepSeek-V4.1-Flash WARP models installable, convertible, validated, and configurable through `litmoe install --model`.

**Architecture:** Extend the single model catalog with a distinct `waste` format and immutable source-conversion metadata. Route WARP models through pinned upstream WARP download/pipeline tooling after litMoE-owned dependency and disk preflight, then register only a validated completed container. Preserve the current local-container runtime adapter and all llama.cpp/ktransformers paths.

**Tech Stack:** Python 3.10+, Click, PyYAML, Pydantic, pytest, upstream WARP shell/Python tooling, Hugging Face repositories.

---

### Task 1: Add WARP models to the catalog

**Files:**
- Modify: `litmoe/models.py:37-84`
- Modify: `litmoe/models.py:264-305`
- Modify: `litmoe/models.py:348-370`
- Modify: `litmoe/models.py:487-530`
- Test: `tests/test_litmoe.py`

**Step 1: Write failing catalog tests**

Add tests asserting:

- `lookup("glm-5.3-flash-warp")` and `lookup("deepseek-v4.1-flash-warp")` return `engine == "warp"` and `format == "waste"`;
- both entries expose immutable `hf_revision`, `warp_profile`, `source_size_gb`, and `size_gb`;
- `quant_size_gb()` returns the converted container size;
- `validate_catalog()` accepts the new format while rejecting malformed WARP metadata;
- `print_model_table()` labels the entries as WARP instead of ktransformers.

**Step 2: Run focused tests and verify RED**

Run:

```bash
python3.12 -m pytest -q tests/test_litmoe.py -k 'warp_catalog or catalog_validation'
```

Expected: failures because the WARP catalog format and entries do not exist.

**Step 3: Implement the catalog format**

Add:

```python
WASTE = "waste"
```

Add a `_w(...)` constructor that emits the standard catalog fields plus:

```python
{
    "engine": "warp",
    "format": WASTE,
    "hf_repo": ...,
    "hf_revision": ...,
    "warp_profile": ...,
    "source_size_gb": ...,
    "size_gb": ...,
}
```

Register:

- `glm-5.3-flash-warp` → `zai-org/GLM-5.3-Flash`, revision `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`, profile `glm`, 306 GiB source, 112 GB output;
- `deepseek-v4.1-flash-warp` → `deepseek-ai/DeepSeek-V4.1-Flash`, revision `dba1be0a40aa45a94ad051997016db3960a90277`, profile `ds41`, 475 GiB source, 299 GiB output.

Extend `quant_size_gb`, table labels, and validation without weakening existing GGUF/safetensors checks.

**Step 4: Run focused tests and verify GREEN**

Run the same focused command. Expected: PASS.

### Task 2: Implement safe WARP source conversion

**Files:**
- Modify: `litmoe/cli/install.py:1-35`
- Modify: `litmoe/cli/install.py:219-363`
- Add helpers near: `litmoe/cli/install.py:739-808`
- Test: `tests/test_litmoe.py`

**Step 1: Write failing installer-helper tests**

Cover public behavior for a helper such as:

```python
install_warp_model(
    model_name,
    info,
    models_dir,
    staging_root,
    prefix,
    jobs,
    reclaim_source,
)
```

Use temporary directories and fake `bash`/`uv`/WARP scripts. Assert:

- dependencies are checked before any subprocess with network semantics;
- staging and output paths are deterministic and absolute;
- same-filesystem preflight requires combined remaining bytes;
- split-filesystem preflight checks source/output independently;
- resumable existing bytes reduce remaining requirements;
- dry-run receives pinned repo/revision/destination;
- pipeline receives explicit model profile, repo, revision, source, output, jobs, reclaim, and report-directory environment;
- `HF_TOKEN` is inherited but never printed;
- nonzero subprocess exit raises a stage-specific error and preserves partial files;
- success requires `manifest.json`, `trunk.bin`, tokenizer data, and expert-bank artifacts before returning the output path.

**Step 2: Run focused tests and verify RED**

```bash
python3.12 -m pytest -q tests/test_litmoe.py -k 'warp_model_install or warp_disk_preflight'
```

Expected: failures because the conversion helper does not exist.

**Step 3: Implement minimal conversion orchestration**

Implement boring helpers for:

- positive integer validation;
- finding the complete installed WARP root;
- calculating used/resumable bytes and filesystem identities;
- preflighting combined or separate capacity with actionable errors;
- validating the output container contract;
- invoking upstream dry-run and pipeline commands with `subprocess.run(check=False)` and stage-specific exceptions.

Do not duplicate upstream download/conversion logic. Do not delete staging or partial output on failure.

**Step 4: Run focused tests and verify GREEN**

Run the focused command. Expected: PASS.

### Task 3: Wire WARP conversion into `litmoe install`

**Files:**
- Modify: `litmoe/cli/install.py:936-1083`
- Test: `tests/test_litmoe.py`

**Step 1: Write failing CLI tests**

Assert:

- positional `litmoe install glm-5.3-flash-warp` and `--model` both resolve;
- model selection automatically chooses/installs engine `warp`;
- `--staging-dir`, `--warp-jobs`, and `--reclaim-source` are visible in help;
- invalid jobs, WARP `--quant`, and WARP `--no-mmproj` fail before installation;
- confirmation includes pinned revision, source/output sizes, and paths;
- `--yes` runs the helper and writes `engine: warp`, absolute `model_path`, and `n_ctx: 0`;
- config is unchanged when conversion fails or validation rejects the output;
- a rerun replaces only the same model entry and preserves aliases/other models;
- existing llama.cpp and ktransformers CLI tests remain unchanged.

**Step 2: Run focused tests and verify RED**

```bash
python3.12 -m pytest -q tests/test_litmoe.py -k 'warp_cli_install or warp_registration'
```

Expected: failures for missing options and dispatch.

**Step 3: Implement CLI dispatch and registration**

Add Click options:

```python
@click.option("--staging-dir", type=click.Path(), default=None, ...)
@click.option("--warp-jobs", type=click.IntRange(min=1), default=3, show_default=True, ...)
@click.option("--reclaim-source", is_flag=True, ...)
```

For `format == WASTE`:

- reject inapplicable options;
- preflight and confirm before large work;
- install WARP;
- run the WARP model installer;
- call `add_model_to_config(..., engine="warp", n_ctx=0)` only after success;
- print exact resume paths on failure and a direct serve command on success.

Keep the current GGUF and safetensors branches intact.

**Step 4: Run focused tests and verify GREEN**

Run the focused command. Expected: PASS.

### Task 4: Preserve WARP runtime context semantics

**Files:**
- Modify: `litmoe/server.py:431-480`
- Test: `tests/test_litmoe.py`

**Step 1: Write a failing regression test**

Construct a WARP `ModelEntry` with `n_ctx: 0`, run the server's context-fix preparation, and assert that neither the in-memory model nor `models.yaml` is rewritten.

**Step 2: Run the focused test and verify RED**

```bash
python3.12 -m pytest -q tests/test_litmoe.py -k 'warp_context_default'
```

Expected: failure because generic context fitting substitutes the catalog native context.

**Step 3: Implement the engine exemption**

Return early from generic context fitting for non-llama.cpp engines or specifically for WARP, preserving existing ktransformers behavior as established by tests. `WarpEngine.build_command()` must continue omitting `--ctx` for zero.

**Step 4: Run the focused test and verify GREEN**

Run the focused command. Expected: PASS.

### Task 5: Update user-facing instructions

**Files:**
- Modify: `README.md`
- Modify: `docs/SETUP.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/METHODOLOGY.md`
- Modify: `examples/models.yaml`
- Modify: `litmoe/cli/install.py` command examples/help

**Step 1: Define documentation assertions**

Extend CLI/help tests to require both WARP model IDs and the `--staging-dir` conversion command. Add source-text assertions only for durable commands/identifiers where repository tests already use that pattern; do not pin prose.

**Step 2: Run assertions and verify RED**

```bash
python3.12 -m pytest -q tests/test_litmoe.py -k 'warp_help or warp_docs'
```

Expected: failure because docs still say model installation is unsupported.

**Step 3: Update documentation**

Document:

- one-command catalog workflow and resumability;
- exact source/output storage requirements and internal-NVMe recommendation;
- immutable source revisions;
- default and custom staging/output paths;
- `uv` prerequisite and `HF_TOKEN` behavior;
- opt-in destructive reclamation;
- failure/resume semantics;
- existing-container manual configuration as an alternative;
- verification limits: CI uses synthetic tools, not full model conversion.

Remove obsolete statements claiming WARP model installation is unavailable.

**Step 4: Run focused tests and verify GREEN**

Run the focused command. Expected: PASS.

### Task 6: Integrated verification and review

**Files:**
- Review all changed files

**Step 1: Run the complete test suite**

```bash
python3.12 -m pytest -q
```

Expected: all tests pass; the existing Starlette/httpx deprecation warning may remain.

**Step 2: Compile Python sources**

```bash
python3.12 -m compileall -q litmoe
```

Expected: no output and exit status 0.

**Step 3: Exercise the real CLI with fake upstream tooling**

Run a temporary-prefix smoke scenario that:

- exposes fake `git`, `make`, `bash`, and `uv`/pipeline tools;
- installs a synthetic WARP runtime;
- creates a tiny structurally valid `.waste` result;
- executes `litmoe install --model glm-5.3-flash-warp --staging-dir ... --yes`;
- loads the resulting config and constructs the WARP command;
- verifies the config points to the validated absolute container and keeps `n_ctx: 0`.

Expected: command exits 0 and the model is ready for `litmoe serve`.

**Step 4: Run specification and quality reviews**

Review against `docs/plans/2026-09-28-warp-model-install-design.md`, then review maintainability, security, subprocess/environment safety, disk arithmetic, platform behavior, and regression risk. Resolve all important findings and rerun affected checks.
