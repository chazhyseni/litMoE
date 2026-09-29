# WARP Model Installation Parity Design

> Context correction (2026-09-29): the zero-context policy below is historical and superseded. WARP installs now write `n_ctx: 0` with `warp_auto_context: true`; startup fits native context using WARP's planner and persists both the result and policy. A positive `--n-ctx` selects fixed mode. Unmarked legacy 0/65536 entries migrate to auto. WARP remains excluded from llama.cpp's full-weight fitter; see [SETUP](../SETUP.md).

## Goal

Make WARP-backed GLM-5.3-Flash and DeepSeek-V4.1-Flash installable through the same catalog command used by other litMoE models:

```bash
litmoe install --model glm-5.3-flash-warp --staging-dir /Volumes/staging
litmoe install --model deepseek-v4.1-flash-warp --staging-dir /Volumes/staging
```

A successful command installs the pinned WARP runtime, downloads a reproducibly pinned official checkpoint, converts it into a local `.waste` container, validates the container, and registers it in `models.yaml`.

## Upstream constraints

WARP v0.8.1 at commit `09fcff352ca55223b08ee222d15054b90546c6a9` supports both target architectures and provides resumable download and conversion tooling. Its release has no downloadable GLM or DeepSeek `.waste` artifacts, so local conversion is required.

| Model | Official source | Pinned revision | Source staging | Container |
|---|---|---|---:|---:|
| GLM-5.3-Flash | `zai-org/GLM-5.3-Flash` | `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a` | 306 GiB | 112 GB |
| DeepSeek-V4.1-Flash | `deepseek-ai/DeepSeek-V4.1-Flash` | `dba1be0a40aa45a94ad051997016db3960a90277` | 475 GiB | 299 GiB |

The pinned upstream `tools/pipeline.sh` expects `model.safetensors.index.json` to exist before its download loop. litMoE will first invoke `tools/fetch_weights.sh --dry-run` with the pinned source revision, which seeds and checks the index without downloading shards, then invoke the upstream pipeline.

## Catalog model

Add a third catalog format, `waste`, and two WARP catalog entries:

- `glm-5.3-flash-warp`
- `deepseek-v4.1-flash-warp`

Each entry carries:

- official Hugging Face repository and immutable revision;
- upstream WARP pipeline profile (`glm` or `ds41`);
- source-staging and output-container sizes;
- model architecture, context, active/total parameter description, RAM tier, and notes;
- WARP runtime arguments, if any.

Catalog validation rejects a WARP entry missing conversion metadata and continues enforcing the existing GGUF and safetensors invariants.

## CLI and data flow

The existing positional and `--model` forms both resolve the new catalog IDs. WARP model installation follows this sequence:

1. Resolve the model and output paths. The default output is `<models-dir>/<model-id>.waste`; the default staging root is `<models-dir>/.staging`, with one model-specific child directory.
2. Require `git`, `make`, `bash`, and `uv` before network or large writes. `HF_TOKEN` passes through to upstream tooling without logging.
3. Preflight source and output storage with `shutil.disk_usage`. When both paths share a filesystem, require their combined peak size; otherwise validate each filesystem separately. Account for resumable bytes already present.
4. Report immutable source revision, source size, output size, paths, and destructive reclaim behavior before confirmation.
5. Install or refresh the pinned WARP runtime using the existing transactional installer.
6. Run `fetch_weights.sh --dry-run --repo ... --revision ... --dest ...` to seed the source index and execute upstream download preflight.
7. Run `pipeline.sh` with an explicit environment: `MODEL`, `REPO`, `REVISION`, `SRC`, `OUT`, `JOBS`, `RECLAIM`, and a report directory beside the output.
8. Validate required container artifacts after the upstream pipeline succeeds.
9. Write or replace the model entry in `models.yaml` with `engine: warp`, an absolute container path, and `n_ctx: 0`.

New options:

- `--staging-dir PATH`: source-weight staging root; defaults under `models-dir`.
- `--warp-jobs N`: conversion worker count; positive integer, default 3.
- `--reclaim-source`: explicitly request upstream destructive source-shard reclamation.

Options that do not apply to WARP, such as `--quant` and `--no-mmproj`, fail clearly rather than being silently ignored.

## Failure and resume behavior

The model is not added to `models.yaml` until the complete pipeline and container validation pass. A failed or interrupted run preserves the source staging directory, partial container, logs, and upstream state so the same command resumes. litMoE does not delete large artifacts implicitly.

`--reclaim-source` is opt-in because it irreversibly deletes source shards as conversion consumes them. The confirmation and documentation identify that tradeoff.

Errors name the failed upstream stage and retain the exact paths needed to resume. Missing `uv`, insufficient disk space, inaccessible Hugging Face repositories, and invalid completed containers fail before configuration mutation.

## Serving behavior

WARP containers intentionally use `n_ctx: 0`, which tells the adapter to preserve the container/runtime default. The generic server context fitter must not rewrite this value for WARP models.

Existing manually configured `.waste` containers remain supported. llama.cpp and ktransformers installation behavior remains unchanged.

## Verification

Automated tests use fake upstream executables and tiny synthetic artifact trees; CI will not download hundreds of gigabytes. Tests cover:

- WARP catalog schema and table rendering;
- positional and `--model` target resolution;
- pinned source revisions and upstream command/environment construction;
- same-filesystem and split-filesystem disk preflight;
- dependency checks before network access;
- successful registration only after required artifacts exist;
- failure preservation and non-mutation of config;
- resume-compatible paths and explicit reclaim forwarding;
- `n_ctx: 0` preservation during serve preparation;
- help text, docs, and examples.

A local smoke test exercises the installer against fake WARP tooling and the real CLI. The full test suite and Python compilation run before commit and push.
