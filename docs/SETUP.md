# Setup Guide

Installing an engine, picking a model that fits your machine, and connecting
your tools. Catalog metadata comes from `litmoe/models.py`. The downloadable
model sizes and quant lists below were checked against the HuggingFace API on
2026-09-16; the WARP table records its separately pinned source revisions and
storage requirements. Run `litmoe models` for the live catalog.

## Prerequisites

- Python 3.10+
- Linux or macOS (Apple Silicon: Metal). Windows via WSL2.
- Disk: 20–70 GB for a laptop-tier catalog model; hundreds of GB for server tiers or WARP containers.

## Step 1: Install litmoe

```bash
git clone https://github.com/chazhyseni/litMoE && cd litMoE
pip install -e .
litmoe doctor          # Python, RAM, CPU cores, engines found
```

## Step 2: Install an engine

### llama.cpp (default; catalog GGUF models)

```bash
litmoe install --engine llamacpp                            # prebuilt release binary (auto: cuda if NVIDIA visible, else cpu)
litmoe install --engine llamacpp --llamacpp-variant vulkan  # or cpu | cuda | cuda13 | rocm
```

The prebuilt path downloads the current `ggml-org/llama.cpp` release asset for
your OS/arch/variant and writes a `llama-server` wrapper into `~/.local/bin/`
(the wrapper sets `LD_LIBRARY_PATH`/`DYLD_LIBRARY_PATH` so the binary finds its
`.so`/`.dylib` files). On Linux the release binaries need glibc ≥ 2.34 — older
distros fall back to a source build automatically. Metal on macOS is in the
standard macOS asset; no variant flag needed.

### ktransformers / sglang-kt (Linux + NVIDIA; CPU expert offload)

Since v0.4 the ktransformers serving stack is **SGLang + kt-kernel**: attention
runs on the GPU, routed experts on the CPU (AMX/AVX-512 fastest, AVX2 works).
This is the engine for the big MoEs on a single-GPU box with lots of RAM.

```bash
litmoe install --engine ktransformers     # PyPI wheels for kt-kernel + sglang-kt
```

PyPI wheels exist only for Linux x86-64, Python 3.11/3.12, glibc ≥ 2.35
(manylinux_2_35); elsewhere litmoe falls back to an upstream source build
(`git clone --recursive` + `install.sh`, needing a C++ toolchain, CMake, and
the CUDA toolkit). macOS is unsupported by this installer. Use a CUDA GPU
compatible with the selected model, precision, and installed backend.

### WARP (catalog conversion or an existing local `.waste` container)

WARP memory-maps local containers and pages expert weights from storage. litmoe
starts its upstream OpenAI-compatible server on a loopback port and supervises
the process; WARP performs inference locally. There is no remote inference API.

Install either catalog model directly:

```bash
litmoe install --model glm-5.3-flash-warp
litmoe install --model deepseek-v4.1-flash-warp

# Stage source weights on bulk storage, but put the runtime container on NVMe.
litmoe install --model glm-5.3-flash-warp \
  --staging-dir /mnt/bulk/warp-staging \
  --models-dir /mnt/nvme/litmoe-models

# Optional: change conversion concurrency and reclaim source shards as they finish.
litmoe install --model deepseek-v4.1-flash-warp \
  --warp-jobs 6 --reclaim-source
```

WARP catalog installation requires `git`, `make`, `bash`, `curl`, and `uv`.
Before creating data, litmoe resolves the paths and rejects source or output
paths containing a backslash, single quote, newline, or carriage return (the
pinned upstream pipeline cannot represent them safely). Source, output, and
run/report paths must not overlap or nest, including through resolved symlink
aliases. litmoe verifies the tools, checks free space for the resumable source
and output work, prints the revision, sizes, and paths, and asks for
confirmation (`--yes` skips the prompt). It then:

1. installs WARP runtime commit
   `09fcff352ca55223b08ee222d15054b90546c6a9`;
2. downloads the pinned source weights and runs the conversion pipeline,
   printing a heartbeat every minute with elapsed time and the newest
   progress line; the full logs are the staging `download.log` and the
   run-report `pipeline.log`;
3. validates the resulting WARP v0 manifest, trunk, codebooks, tokenizer,
   specials, and expert-bank artifacts; and
4. registers the absolute output path in `models.yaml` with `engine: warp`
   and `n_ctx: 0`, `warp_auto_context: true`; a positive `--n-ctx` writes
   a fixed limit with `warp_auto_context: false`.

Interrupting the install (Ctrl-C or a closed terminal) stops the download and
conversion cleanly; rerun the same command to resume — nothing already
downloaded is refetched. A second install of the same model is refused while
one is running.

When `HF_TOKEN` is set, litmoe stores it in a private temporary curl config. It
does not print the token or include it in a child process's arguments or
environment. GLM and DeepSeek have no prebuilt `.waste` release assets; litmoe
orchestrates the upstream conversion rather than implementing quantization or
inference.

| Catalog id | HuggingFace source | Pinned source revision | Source | Output workspace | Final output |
|---|---|---|---:|---:|---:|
| `glm-5.3-flash-warp` | `zai-org/GLM-5.3-Flash` | `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a` | 306 GiB | 120 GiB | 112 GB |
| `deepseek-v4.1-flash-warp` | `deepseek-ai/DeepSeek-V4.1-Flash` | `dba1be0a40aa45a94ad051997016db3960a90277` | 475 GiB | 310 GiB | 299 GiB |

The default output is `<models-dir>/<model-id>.waste`; the default staging root
is `<models-dir>/.staging`, so its source path is
`<models-dir>/.staging/<model-id>`. The run/report path is
`<models-dir>/<model-id>.warp-run`. Put `--models-dir` on fast internal NVMe
for serving. Staging can be on another filesystem. If a fetch, conversion, or
validation step fails, partial source, output, and run reports remain; rerun
the same command to resume.

The CLI heartbeat prints every minute with elapsed time and the newest
progress line from the live stage logs. Watch the full progress directly:

```bash
tail -f <models-dir>/.staging/<model-id>/download.log
tail -f <models-dir>/<model-id>.warp-run/pipeline.log
```

`--reclaim-source` is opt-in and irreversible: the pipeline deletes completed
source shards as they become reclaimable, reducing peak storage, but a retry
may need to download shards that were not proven complete. A complete
hundreds-of-GB conversion is an operator workflow, not a CI smoke test.

After conversion, serve only the new entry with:

```bash
litmoe serve glm-5.3-flash-warp
# or: litmoe serve deepseek-v4.1-flash-warp
```

For a runtime-only or manual-container setup, keep using:

```bash
litmoe install --engine warp
```

That command installs the pinned runtime under `$LITMOE_PREFIX/lib/warp`
(default `~/.local/lib/warp`) and runs upstream `make check`; it does not select
a model. Add your existing local `.waste` path to `models.yaml`. The installed
shared library is `libwaste.so` on Linux, `libwaste.dylib` on macOS, or
`libwaste.dll` on Windows.

Upstream WARP reports these figures; they are not litmoe measurements or
guarantees:

| Container | Upstream size | Upstream resident floor | Upstream throughput |
|---|---:|---:|---:|
| GLM-5.3-Flash | 112 GB | 5.14 GB | 3.32 tok/s short; 3.86 tok/s long on WARP's 64 GB M5 Pro |
| DeepSeek-V4.1-Flash | 299 GiB | 4.86 GB | about 3.7 tok/s |

## Step 3: Pick a model for your RAM

`litmoe models` prints the complete catalog and marks what fits this machine.
For llama.cpp, `litmoe install --model <id>` downloads the default quant when
it fits your RAM budget, otherwise the largest quant that does (`--quant <Q>`
overrides), and adds it to `models.yaml` with a memory-aware context size.
ktransformers entries download their native weight repositories. The two
`*-warp` entries instead describe pinned source-to-`.waste` conversions; use
the storage requirements above rather than the GGUF RAM formula below.

The RAM column below = weights × 1.10 (mmap + compute buffers) + KV cache at
32K tokens + 6 GB headroom. macOS gets 75 % of physical RAM as its budget
(unified memory shared with the OS/GPU). This formula does not describe WARP's
storage-paged containers. The llama.cpp laptop-tier choices are MoEs with
3–5 B active parameters or ≤ 31 B dense — the ones that are actually fast on
CPU/Metal.

`litmoe serve` starts the **first** configured model, or the single model named
on the command line. Other entries remain available through `litmoe switch ID`;
their RAM estimates are not summed. `litmoe init` writes alternatives that fit
individually. At startup, the selected model is checked against the GPU budget
(75% of RAM on macOS) and usable RAM (90% minus 3 GB). Over GPU budget but
under RAM it starts with a warning; over RAM it refuses unless `--force` is
explicitly selected. WARP uses its own resident-memory planner rather than
the weights-plus-KV estimate. At each startup,
`warp_auto_context: true` uses the installed WARP memory planner to fit the
native window (1,048,576 tokens for both catalog entries). Recommended resident
memory, plus vision memory when enabled, must fit 75% of WARP's usable RAM
capacity or a smaller explicit `--budget`. If native does not fit, the window
rounds down in 4096-token blocks. Planning failures stop that model rather
than silently falling back to 4096 or 65536.

The selected positive `n_ctx` is persisted with automatic mode still enabled.
Unmarked legacy `n_ctx: 0` and `65536` migrate to this policy; other positive
values stay fixed. Historical explicit 65536 cannot be distinguished from the
shipped default: set `warp_auto_context: false` and a positive `n_ctx` to retain
that limit. A positive install-time `--n-ctx N` selects fixed mode automatically.
Unknown manual model IDs need `config.max_position_embeddings` in their manifest
or a fixed context. No re-download or conversion is required; restart the gateway.

The WARP budget is based on capacity, **not current free RAM**. One litmoe
engine is resident, but other applications and independently launched engines
still consume memory. An oversized explicit `--budget` passes upstream
unchanged. `litmoe serve ID` selects one initial model; multiple IDs are rejected.

### 48 GB laptop — default tier

| Model | Type | Native ctx | Default quant | Size | RAM | Notes |
|---|---|---|---|---|---|---|
| **gemma-4-26b-a4b** (default) | MoE, 4B active | 256K | UD-Q4_K_XL | 17 GB | ~26 GB | Vision (mmproj included). `litmoe init` picks this. |
| qwen3.6-35b-a3b | MoE, 3B active | 256K | UD-Q4_K_XL | 22 GB | ~31 GB | Thinking on by default |
| nemotron-3.5-lightning-30b-a3b | hybrid MoE, 3B active | 1M | UD-Q4_K_XL | 26 GB | ~35 GB | Mamba-2 hybrid |
| gpt-oss-20b | MoE, 3.6B active | 128K | UD-Q4_K_XL | 12 GB | ~20 GB | Native MXFP4; Harmony format |
| gemma-4-12b | dense | 256K | Q4_K_M | 7 GB | ~16 GB | Vision; this default is not admitted on a 16 GB host by the serving memory gate |
| qwen3.8-9b-distill | dense | 256K | Q4_K_M | 6 GB | ~14 GB | Reasoning distill; emits `reasoning_content` |
| qwen3.8-27b | dense | 256K | UD-Q4_K_XL | 18 GB | ~28 GB | Dense architecture |
| gemma-4-31b | dense | 256K | UD-Q4_K_XL | 19 GB | ~32 GB | Vision |
| kimi-linear-48b | MoE, 3B active | 1M | Q4_K_M | 30 GB | ~39 GB | KDA linear attention |

### 96 GB laptop / desktop

| Model | Type | Native ctx | Default quant | Size | RAM |
|---|---|---|---|---|---|
| gpt-oss-120b | MoE, 5.1B active | 128K | UD-Q4_K_XL | 63 GB | ~77 GB |
| qwen3.5-122b-a10b | MoE, 10B active | 256K | UD-IQ4_XS | 60 GB | ~73 GB |
| nemotron-3-super-120b-a12b | hybrid MoE, 12B active | 1M | UD-IQ4_XS | 64 GB | ~77 GB |
| llama-4-scout | MoE, 17B active | 10M | UD-Q4_K_XL | 62 GB | ~76 GB |

### 192 GB workstation

| Model | Engine | Type | Default | Size | RAM |
|---|---|---|---|---|---|
| qwen3.8-flash-next | llama.cpp | MoE | UD-Q4_K_XL | 111 GB | ~129 GB |
| minimax-m2.7 | llama.cpp | MoE, 10B active | UD-Q4_K_XL | 141 GB | ~169 GB |
| deepseek-v4-flash | llama.cpp | MoE (MLA) | UD-Q4_K_XL | 155 GB | ~179 GB |
| deepseek-v4-flash-kt | sglang-kt | MXFP4 safetensors | — | 160 GB | ~185 GB + GPU |

### 512 GB server

| Model | Engine | Type | Default | Size | RAM |
|---|---|---|---|---|---|
| minimax-m3 | llama.cpp | 426B MoE, 23B active | UD-Q4_K_XL | 265 GB | ~302 GB |
| glm-5.3 | llama.cpp | MoE | UD-Q2_K_XL | 254 GB | ~288 GB |
| deepseek-v3.2 | llama.cpp | 671B MoE, 37B active | UD-Q2_K_XL | 247 GB | ~280 GB |
| kimi-k2.6 / kimi-k2.5 | llama.cpp | 1T MoE, 32B active | UD-Q2_K_XL / UD-IQ2_M | 340 / 345 GB | ~382 / ~388 GB |
| glm-5.3-flash | sglang-kt | MoE, 18B active, 1M ctx, multimodal | — | 328 GB | ~367 GB + GPU |
| minimax-m2.7-kt | sglang-kt | — | — | 230 GB | ~267 GB + GPU |
| minimax-m3-kt | sglang-kt | MXFP8 | — | 444 GB | ~498 GB + GPU |

### 768 GB server

| Model | Engine | Type | Default | Size | RAM |
|---|---|---|---|---|---|
| qwen3.8 (2.4T-A95B) | llama.cpp | 2.4T MoE, 95B active | UD-IQ1_S | 508 GB | ~568 GB |
| kimi-k3 | llama.cpp | 2.78T MoE, 93B active, 1M ctx | UD-IQ1_S | 594 GB | ~663 GB |
| kimi-k2-thinking | sglang-kt | INT4 safetensors | — | 594 GB | ~662 GB + GPU |
| deepseek-v3.2-kt | sglang-kt | FP8 safetensors | — | 689 GB | ~766 GB + GPU |

## Speed: what to expect

Active parameter count, memory bandwidth, compute, quantization, context,
thread count, and concurrent requests all affect throughput. Benchmark the
intended workload on the target host.

The following generation-rate ranges come from the retained local CPU logs
under [`docs/measurements/`](measurements/README.md). These are not controlled
cross-model comparisons and do not establish model quality or paging behavior.

| Model | Quant / size | Threads | Generation t/s (min–max) | Log |
|---|---|---|---|---|
| gemma-4-26b-a4b (MoE, 4B active) | UD-Q4_K_XL, 17 GB | 24 | 1.56–12.68 | `gemma-4-26b-a4b.log` |
| Qwen3.8-9B-Distill (dense) | Q4_K_M, 6 GB | 8 | 8.33–8.48 | `qwen3.8-9b-distill.log` |
| Kimi-Linear-48B-A3B (MoE, 3B active) | Q4_K_M, 30 GB | 48 | 0.03–0.58 | `kimi-linear-48b.log` |
| DeepSeek-V4-Flash | UD-IQ1_S, 83 GB | 48 | 0.11–0.34 | `deepseek-v4-flash.log` |

[`docs/measurements/README.md`](measurements/README.md) records the per-request
sequences, the run conditions, and how to read these numbers.

## Step 4: models.yaml

`litmoe install --model` writes catalog entries like the first two below and
writes a generated WARP container as an absolute `engine: warp`, `n_ctx: 0`,
`warp_auto_context: true` entry. Existing WARP containers can still be added
manually, as shown below.
`litmoe init` creates the file with the default catalog model and Claude-name
aliases. It writes `host: 127.0.0.1`; the schema default for a hand-written
file that omits `host` is `0.0.0.0`, which binds all interfaces.

```yaml
host: 127.0.0.1
port: 8090
api_key: null              # or a string → Bearer auth required

models:
  - id: gemma-4-26b-a4b
    engine: llamacpp
    model_path: ~/.litmoe/models/gemma-4-26b-a4b/UD-Q4_K_XL/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf
    n_gpu_layers: -1       # -1 all layers on GPU (no-op on CPU builds), 0 CPU only
    n_ctx: 262144          # native; lowered automatically if RAM cannot hold the KV cache
    extra_args: ["--mmproj", "~/.litmoe/models/gemma-4-26b-a4b/UD-Q4_K_XL/mmproj-F16.gguf"]
    aliases: [claude-sonnet-4-5, claude-haiku-4-5, claude-opus-4-1]

  - id: glm-5.3-flash
    engine: ktransformers
    model_path: ~/.litmoe/models/glm-5.3-flash      # safetensors dir (or the HF id zai-org/GLM-5.3-Flash)
    kt_method: FP8                                  # native precision, per the upstream tutorial; RAWINT4 for Kimi-K2.x
    kt_num_gpu_experts: 8
    kt_cpuinfer: 48
    extra_args: ["--tool-call-parser", "glm47", "--reasoning-parser", "glm45"]

  - id: glm-5.3-flash-warp
    engine: warp
    model_path: ~/models/glm53.waste
    n_ctx: 0
    warp_auto_context: true    # fit native context at each startup

  - id: deepseek-v4.1-flash-warp
    engine: warp
    model_path: ~/models/deepseek-v4.1-flash.waste
    n_ctx: 0
    warp_auto_context: true
```

Field reference (see `litmoe/config.py`):

| Field | Engine | Meaning |
|---|---|---|
| `model_path` | llama.cpp | local GGUF file/dir or `repo:QUANT` HuggingFace spec |
| `model_path` | ktransformers | local safetensors directory or HuggingFace repo id |
| `model_path` | WARP | local `.waste` container; a WARP catalog install writes the generated absolute path, and manual paths remain supported |
| `n_ctx` | llama.cpp | context; `0` = memory-aware native |
| `n_ctx` | WARP | resolved positive `--ctx N`; automatically fitted and persisted in auto mode; fixed positive window otherwise |
| `warp_auto_context` | WARP | `true`: re-fit native context each startup; `false`: preserve a positive `n_ctx`; omitted: migrate legacy 0/65536 to auto |
| `n_gpu_layers` | llama.cpp | `-ngl` |
| `extra_args` | all | engine flags; WARP rejects conflicting `--ctx`; llama.cpp rejects context/slot overrides (`-c`, `--ctx-size`, `-np`, `--parallel`, `--kv-unified-per-slot`) |
| `env` | all | extra environment for the engine process; also used by WARP's isolated memory planner |
| `kt_method` | ktransformers | CPU expert backend: `FP8`, `FP8_PERCHANNEL`, `BF16`, `RAWINT4`, `MXFP4`, `MXFP8` (AVX-512); `AMXINT4`, `AMXINT8` (Intel AMX); `LLAMAFILE` (AVX2, GGUF experts via `gguf_path`) |
| `kt_num_gpu_experts` | ktransformers | experts pinned on GPU |
| `kt_cpuinfer` / `kt_threadpool_count` | ktransformers | CPU threads for expert compute (default physical cores) / thread pools (default NUMA nodes) |
| `aliases` | all | additional model ids that route here |

## Step 5: Run

```bash
litmoe serve                            # first configured model only
litmoe serve gemma-4-26b-a4b             # one initial selection; --force skips its fit refusal
litmoe switch qwen3.6-35b-a3b            # configured alternative; drain and unload before loading
litmoe status                           # resident state, queue, effective context, capabilities
litmoe bench --runs 3 --json             # gateway vs direct engine; run here with clients idle
litmoe stop                             # stop engines litmoe started (PID files)
curl http://127.0.0.1:8090/v1/models
```

The resident engine takes a free loopback port starting at 8081, skipping the
gateway port and occupied ports. Logs append to `logs/<model-id>.log` relative
to the working directory; `litmoe serve --log-dir DIR` overrides that.
Switching waits for the active request and prevents new admission during the
transition. A failed switch leaves an explicit failed state, never the old
model masquerading as the requested one. Retry with `litmoe switch ID`.

Initial WARP startup attempts a four-token `Hello` warmup after the engine
reports ready. It bypasses admission and is not repeated by switches or
cancellation reloads; see [Engine lifecycle](ARCHITECTURE.md#engine-lifecycle).
Readiness does not establish a warm cache. A fixed WARP `--budget` is opt-in;
catalog installation does not impose 64G. GLM's pinned chat format requires
reasoning, so `--no-thinking` does not disable it.

Top-level `max_queue_size` (default 8) and `queue_timeout` (default 30 seconds)
bound admission; full/expired waits return HTTP 429. Known inactive model IDs
return 409, unknown IDs 404, and switching/failed engines 503.
Set `api_key` to protect inference and `/v1/runtime` control; null preserves
unauthenticated local use. Keep the gateway on loopback unless deliberately
exposing and securing it.

Environment variables litmoe reads (all optional; the main ones):
`LITMOE_CONFIG` (models.yaml path), `LITMOE_MODELS_DIR`, `LITMOE_PREFIX`
(engine install prefix), `LITMOE_WARP_DIR` (override the WARP source root),
`LITMOE_RUN_DIR` (PID files), `LITMOE_READY_TIMEOUT`, `LITMOE_LLAMACPP_TAG`
(pin a release), `LITMOE_GATEWAY` (CLI target), and `LITMOE_API_KEY` (CLI key).
The installer also accepts `LITMOE_PIP_EXTRA_INDEX_URL`. The prompt-cache
capability report reads `LLAMA_ARG_CACHE_PROMPT` from the process or model
`env`. Harness endpoint variables are configured by the launchers described below.

## Step 6: Connect Claude Code / Hermes / OMP / Open WebUI

See [HARNESSES.md](HARNESSES.md). Short version: use `scripts/claude-local`,
`scripts/hermes-local`, and `scripts/omp-local`; never `export
ANTHROPIC_BASE_URL` in your shell.

## Docker

`deploy/docker-compose.yml` builds a CPU llama.cpp image and runs the gateway
on `127.0.0.1:8000` plus Open WebUI on `:8080`. Edit `deploy/models.yaml`
(paths are `/models/...`, a read-only mount of `~/.litmoe/models`).

The gateway port is published loopback-only (`127.0.0.1:8000:8000`), but Open
WebUI is published as `8080:8080` on all host interfaces with
`WEBUI_AUTH=false`. Restrict that binding and enable authentication before
exposing the stack beyond localhost.
