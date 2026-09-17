# Setup Guide

Installing an engine, picking a model that fits your machine, and connecting
your tools. Model sizes and quant lists come from `litmoe/models.py`, which
was checked against the HuggingFace API on 2026-09-16; run `litmoe models`
for the live version of every table below.

## Prerequisites

- Python 3.10+
- Linux or macOS (Apple Silicon: Metal). Windows via WSL2.
- Disk: 20–70 GB for a laptop-tier model; hundreds of GB for server tiers.

> **macOS with several Pythons (Homebrew + conda):** install and run litmoe
> with the *same* interpreter, or the entry point may start under a Python
> that cannot read `~/.litmoe/models/` (TCC). `litmoe doctor` checks this.

## Step 1: Install litmoe

```bash
git clone https://github.com/chazhyseni/litMoE && cd litMoE
pip install -e .
litmoe doctor          # Python, RAM, CPU cores, engines found, config sanity
```

## Step 2: Install an engine

### llama.cpp (default; every model in the catalog has a GGUF)

```bash
litmoe install --engine llamacpp                            # prebuilt release binary (auto: cuda if NVIDIA visible, else cpu)
litmoe install --engine llamacpp --llamacpp-variant vulkan  # or cpu | cuda | cuda13 | rocm
```

The prebuilt path downloads the current `ggml-org/llama.cpp` release asset for
your OS/arch/variant, verifies it runs, and symlinks `llama-server` into
`~/.local/bin/`. On Linux the release binaries need glibc ≥ 2.34 — older
distros fall back to a source build automatically. Metal on macOS is in the
standard macOS asset; no variant flag needed.

### ktransformers / sglang-kt (Linux + NVIDIA; CPU expert offload)

Since v0.4 the ktransformers serving stack is **SGLang + kt-kernel**: attention
runs on the GPU, routed experts on the CPU (AMX/AVX-512 fastest, AVX2 works).
This is the engine for the big MoEs on a single-GPU box with lots of RAM.

```bash
litmoe install --engine ktransformers     # PyPI wheels for kt-kernel + sglang-kt
```

Not available on macOS (triton/CUDA dependency). Requires a CUDA GPU; the
upstream tutorials target SM90 (H100/H20) but SM80/SM86 work for most models.

## Step 3: Pick a model for your RAM

`litmoe models` prints the catalog grouped by tier and marks what fits this
machine. `litmoe install --model <id>` downloads the default quant when it
fits your RAM budget, otherwise the largest quant that does (`--quant <Q>`
overrides), and adds it to `models.yaml` with a memory-aware context size.

RAM column = weights × 1.08 (mmap + compute buffers) + KV cache at 32K tokens
+ 4 GB headroom. macOS gets 75 % of physical RAM as its budget (unified memory
shared with the OS/GPU). All laptop-tier models are MoEs with 3–5 B active
parameters or ≤ 31 B dense — the ones that are actually fast on CPU/Metal.

### 48 GB laptop — default tier

| Model | Type | Native ctx | Default quant | Size | RAM | Notes |
|---|---|---|---|---|---|---|
| **gemma-4-26b-a4b** (default) | MoE, 4B active | 256K | UD-Q4_K_XL | 17 GB | ~26 GB | Vision (mmproj included). `litmoe init` picks this. |
| qwen3.6-35b-a3b | MoE, 3B active | 256K | UD-Q4_K_XL | 22 GB | ~31 GB | Strong coding/agentic; thinking on by default |
| nemotron-3.5-lightning-30b-a3b | hybrid MoE, 3B active | 1M | UD-Q4_K_XL | 26 GB | ~35 GB | Mamba-2 hybrid, tiny KV |
| gpt-oss-20b | MoE, 3.6B active | 128K | UD-Q4_K_XL | 12 GB | ~20 GB | Native MXFP4; Harmony format |
| gemma-4-12b | dense | 256K | Q4_K_M | 7 GB | ~16 GB | Vision; fits 16 GB |
| qwen3.8-9b-distill | dense | 256K | Q4_K_M | 6 GB | ~14 GB | Reasoning distill; emits `reasoning_content` |
| qwen3.8-27b | dense | 256K | UD-Q4_K_XL | 18 GB | ~28 GB | Slower than the MoEs (27B active) |
| gemma-4-31b | dense | 256K | UD-Q4_K_XL | 19 GB | ~32 GB | Vision; slower than the MoEs |
| kimi-linear-48b | MoE, 3B active | 1M | Q4_K_M | 30 GB | ~39 GB | KDA linear attention: 1M ctx cheap |

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
| deepseek-v4-flash-kt | sglang-kt | FP8 safetensors | — | 160 GB | ~185 GB + GPU |

### 512 GB server

| Model | Engine | Type | Default | Size | RAM |
|---|---|---|---|---|---|
| minimax-m3 | llama.cpp | 428B MoE, 23B active | UD-Q4_K_XL | 265 GB | ~302 GB |
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

Throughput on CPU is bounded by memory bandwidth × active parameters, so a
26B MoE with 4B active runs about as fast as a 9B dense model while being a
much stronger model — that is the whole reason the default tier is
small-active MoEs. Rough rules from community numbers (not measured here):

- **3–5 B active MoE, Q4** (the 48 GB tier): tens of t/s on Apple M-series
  Max/Ultra, 10–25 t/s on a DDR5 desktop, high single digits on an AVX2-only
  DDR4 cloud VM.
- **10–17 B active** (96 GB tier): roughly a third of the above.
- **≥ 23 B active** (server tiers): needs a GPU for attention (sglang-kt) or a
  many-channel EPYC/Xeon to be interactive; otherwise batch-only.

Measured in this project — every number below is a `print_timing` line in a
log shipped under [`docs/measurements/`](measurements/README.md). Machine: AMD
EPYC 7B13, 24 physical cores, AVX2 only (no AVX-512), DDR4-3200, Google Cloud
persistent disk (~400 MB/s), no GPU, llama.cpp CPU build.

| Model | Quant / size | Threads | Generation t/s | Log |
|---|---|---|---|---|
| gemma-4-26b-a4b (MoE, 4B active) | UD-Q4_K_XL, 17 GB | 24 | 9.0–12.7 once weights are resident; 1.6–4.6 for the first requests after each (re)start | `gemma-4-26b-a4b.log`, 2026-09-16 |
| Qwen3.8-9B-Distill (dense) | Q4_K_M, 6 GB | 8 | 8.3–8.5 | `qwen3.8-9b-distill.log`, 2026-09-01 |
| Kimi-Linear-48B-A3B (MoE, 3B active) | Q4_K_M, 30 GB | 48 | 0.4–0.6 (0.03–0.05 on cold requests) — disk-bound | `kimi-linear-48b.log`, 2026-08-20 |
| DeepSeek-V4-Flash | UD-IQ1_S, 83 GB | 48 | 0.32–0.34 (0.11 cold) — disk-bound | `deepseek-v4-flash.log`, 2026-08-20 |

Reading these: the two MoEs whose experts never became resident (Kimi-Linear
at 30 GB on a box also holding other models, V4-Flash at 83 GB cold) were
paging from a ~400 MB/s disk — those are storage numbers, not model numbers.
Gemma hit double digits only after its 17 GB were paged in. The 9B dense and
the 26B-A4B MoE land in the same ~8–13 t/s band, which is the memory-bandwidth
argument in one row: same speed class, much stronger model. Earlier versions
of this file quoted 0.69 t/s for the 9B model and 0.85 t/s for Kimi-K3 from
Aug-2026 runs whose logs were overwritten before append-only logging existed;
those figures are not reproducible from the repo and are no longer cited.

## Step 4: models.yaml

`litmoe install --model` writes entries like these; `litmoe init` creates the
file with the default model and Claude-name aliases.

```yaml
host: 127.0.0.1
port: 8080
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
```

Field reference (see `litmoe/config.py`):

| Field | Engine | Meaning |
|---|---|---|
| `model_path` | both | GGUF file/dir, `repo:QUANT` HF spec, or safetensors dir |
| `n_ctx` | llama.cpp | context; `0` = memory-aware native |
| `n_gpu_layers` | llama.cpp | `-ngl` |
| `extra_args` | both | passed through verbatim to `llama-server` / `sglang.launch_server` (`-t N` overrides the physical-core thread default) |
| `env` | both | extra environment for the engine process only |
| `kt_method` | kt | CPU expert backend: `FP8`, `FP8_PERCHANNEL`, `BF16`, `RAWINT4`, `MXFP4`, `MXFP8` (AVX-512); `AMXINT4`, `AMXINT8` (Intel AMX); `LLAMAFILE` (AVX2, GGUF experts via `gguf_path`) |
| `kt_num_gpu_experts` | kt | experts pinned on GPU |
| `kt_cpuinfer` / `kt_threadpool_count` | kt | CPU threads for expert compute (default physical cores) / thread pools (default NUMA nodes) |
| `aliases` | both | additional model ids that route here |

## Step 5: Run

```bash
litmoe serve                            # gateway + engines; Ctrl-C stops everything
litmoe status                           # gateway health + per-engine state
litmoe stop                             # stop engines litmoe started (PID files)
curl http://127.0.0.1:8080/v1/models
```

Engines get ports counting up from 8081, skipping the gateway port and any
port another process already holds (so a stray llama-server on 8081 does not
kill yours). Engine logs append to `logs/<model-id>.log` with a session header
per start.

Environment variables litmoe reads (all optional, all `LITMOE_*` — it never
reads or sets `ANTHROPIC_*` / `OPENAI_*`): `LITMOE_CONFIG` (models.yaml path),
`LITMOE_MODELS_DIR`, `LITMOE_PREFIX` (engine install prefix), `LITMOE_RUN_DIR`
(PID files), `LITMOE_READY_TIMEOUT`, `LITMOE_LLAMACPP_TAG` (pin a release).

## Step 6: Connect Claude Code / Hermes / Open WebUI

See [HARNESSES.md](HARNESSES.md). Short version: use `scripts/claude-local`
and `scripts/hermes-local`; never `export ANTHROPIC_BASE_URL` in your shell.

## Docker

`deploy/docker-compose.yml` builds a CPU llama.cpp image and runs the gateway
on `127.0.0.1:8000` plus Open WebUI on `:8080`. Edit `deploy/models.yaml`
(paths are `/models/...`, a read-only mount of `~/.litmoe/models`).
