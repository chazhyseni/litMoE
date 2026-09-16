# Methodology

This document explains why litmoe is structured the way it is: a thin
Python dispatcher over ktransformers and llama.cpp, with no custom inference code.

## The problem

Open MoE models span 17 GB (Gemma-4-26B-A4B) to 600 GB+ (Kimi-K3). Two
engines cover all of them well, on different hardware:

1. **llama.cpp** — GGUF, every quant from 1.5 to 8 bit, CUDA/HIP/Metal/Vulkan/
   SYCL/CPU. The right tool from a 48 GB laptop up to a many-core server.
2. **ktransformers (sglang-kt + kt-kernel)** — attention on one GPU, routed
   experts on the CPU with AMX/AVX-512 kernels. The right tool for the
   200 GB–1 TB models on a single-GPU box with lots of RAM.

Neither engine needs help with inference. What users lack is one endpoint,
one config, sane defaults for their RAM, and a way to point their agent
harnesses at it without breaking those harnesses.

The previous version of litmoe tried to be a third engine: a custom
CPU-only C99 forward pass. It was 0.019 t/s on a 24-core EPYC. The math:

- 67 prompt tokens × 92 MoE layers × 16 experts = 98,496 expert lookups
- Each expert is 17.55 MB; ~50% dedup = ~859 GB to read from disk
- At 379 MB/s disk: 38 minutes minimum
- 24 cores × ~50 ms per expert compute: 82 minutes compute floor

No software optimization closes a 1000x gap to ktransformers and llama.cpp.
We tried AVX2 matmul, mmap, cross-layer prefetch, 2-bit quantization — all
shipped but all irrelevant. The bandwidth doesn't exist.

## What the dispatcher does instead

The dispatcher acknowledges that other people have spent years building inference
engines and uses them. Two open-source projects cover everything:

| Engine | Hardware | Strength |
|---|---|---|
| **ktransformers** (Tsinghua MADSys Lab, SOSP 2025) — served via sglang-kt | CUDA GPU + AMX / AVX-512 / AVX2 CPU | Heterogeneous CPU+GPU MoE, expert offloading, INT4/INT8/FP8/RAWINT4 experts |
| **llama.cpp** | CUDA + HIP + Metal + Vulkan + SYCL + CPU | Mature cross-platform, every quant format, every model tier |

litmoe is the front door: a Python package that:

1. Reads a `models.yaml` config; ships a catalog (`litmoe/models.py`) tiered
   by RAM so `litmoe init` picks something that is fast on *this* machine.
2. Starts the chosen engine as a subprocess (`python -m sglang.launch_server`
   or `llama-server`), supervises it, and stops it cleanly.
3. Exposes a single OpenAI + Anthropic-compatible API on 127.0.0.1:8080.
4. Routes requests to the right engine by model name or alias.
5. Connects agent harnesses (Claude Code, Hermes) **per process**, never by
   rewriting their global configuration.

That's it. No custom forward pass. No CUDA kernels. No safetensors parsing.

## Why a dispatcher is the right shape

**Inference engines are mature.** ktransformers hit SOSP 2025 with a
heterogeneous-expert scheduler; llama.cpp ships 1.5-bit to 8-bit quantization
across every GPU vendor. The optimization space is enormous and competition
between these engines is healthy. Reimplementing kernels loses to both.

**Engines already speak HTTP.** Both `sglang.launch_server` (the ktransformers
serving stack since v0.4) and `llama-server` ship OpenAI-compatible servers.
The gateway is a pass-through plus an Anthropic translation layer.

**Configuration is the hard part.** Users don't care which engine is running;
they care which model responds, and that it is fast enough on the hardware
they have. The dispatcher lets a single `models.yaml` mix engines:
gemma-4-26b-a4b → llama.cpp on a laptop, glm-5.3-flash → sglang-kt on a
GPU server. The catalog encodes what fits where so the default is never a
594 GB download on a 96 GB machine.

**Defaults must be fast, not just fit.** A 9B dense model and a 26B MoE with
4B active both ran at 8–13 t/s on an AVX2 DDR4 box — same speed class, but
the MoE is a far stronger model (and multimodal). Speed on CPU tracks *active*
parameters, so small-active MoEs are the laptop tier; dense models and big
MoEs are listed, not defaulted.

**Inference is hardware-bound, not software-bound.** The previous "optimization"
work (cross-layer prefetch, 2-bit quantization, mmap advisor, fused matmul)
was a series of single-digit-percent improvements on a fundamentally
bandwidth-limited problem. The dispatcher makes that work unnecessary: pick
the right engine for the hardware and let it do what it's good at.

## What was measured

Hardware: AMD EPYC 7B13 (24 physical cores, AVX2 only, 377–406 GB DDR4-3200,
no GPU, Google Cloud PersistentDisk at ~379 MB/s random / ~778 MB/s
sequential read). Raw logs: [`docs/measurements/`](measurements/README.md).

| Engine | Model | Threads | Tokens/sec | Notes |
|---|---|---|---|---|
| Previous C99 AVX2 forward pass | Kimi-K3 | 24 | 0.019 | 158s TTFT for 4-token prompt; thread stuck in DISK SLEEP (Aug 2026, log not retained) |
| llama.cpp | Gemma-4-26B-A4B UD-Q4_K_XL (17 GB) | 24 | 9.0–12.7 resident; 1.6–4.6 while paging in | 2026-09-16, `gemma-4-26b-a4b.log` |
| llama.cpp | Qwen3.8-9B dense Q4_K_M (6 GB) | 8 | 8.3–8.5 | 2026-09-01, `qwen3.8-9b-distill.log` |
| llama.cpp | Kimi-Linear-48B-A3B Q4_K_M (30 GB) | 48 | 0.03–0.58 | disk-bound, `kimi-linear-48b.log` |
| llama.cpp | DeepSeek-V4-Flash UD-IQ1_S (83 GB) | 48 | 0.11–0.34 | disk-bound, `deepseek-v4-flash.log` |
| ktransformers sglang-kt | large MoEs, GPU attention + CPU experts | — | 5–50 typical | upstream tutorials; needs a CUDA GPU, not measured here |

The C engine was ~45x slower than llama.cpp on the same Kimi-K3 weights
(0.019 vs the 0.85 t/s llama.cpp reached in Aug 2026, recorded in commit
`cd9e97e`/`21819c5`; that llama.cpp log was later overwritten, so the K3
figure comes from the commit history rather than a shipped log). The gap to
GPU serving is 100–1000x. There is no path from a custom C engine to
interactive inference on this VM.

## What you get

- **Laptop, 48–96 GB (Apple Silicon or x86):** llama.cpp with a 3–5B-active
  MoE from the default tier. Interactive.
- **Workstation, 192 GB:** llama.cpp with DeepSeek-V4-Flash / MiniMax-M2.7 /
  Qwen3.8-Flash-Next at Q4.
- **Server with a CUDA GPU and 512 GB+:** sglang-kt with CPU expert offload
  for GLM-5.3-Flash, Kimi-K2.x, DeepSeek-V3.2, MiniMax-M3.
- **Server, CPU only, 768 GB+:** llama.cpp with Kimi-K3 / Qwen3.8-2.4T at
  IQ1 — batch use, not chat, unless the CPU has many memory channels.

Pick the engine per model in `models.yaml`. Both engines speak OpenAI HTTP.
The dispatcher adds latency in the single-digit-millisecond range and never
touches the forward pass.

## What litmoe does NOT do

- It is not an inference engine. There are no model weights in this repo and
  no forward-pass code. The actual inference is done by subprocesses.
- It does not optimize for specific hardware. That's the engines' job.
- It does not quantize models. That's `kt quant`, `llama-quantize`, or
  third-party tools (Unsloth, MLX, etc.).
- It does not parallelize across machines. Single-node only.

## What's in this repo

```
litmoe/
├── pyproject.toml          # modern Python package
├── litmoe/
│   ├── models.py           # model catalog by RAM tier (HF-verified sizes, ctx, KV)
│   ├── config.py           # Pydantic models.yaml schema
│   ├── server.py           # FastAPI OpenAI/Anthropic gateway + engine supervision
│   ├── platform_utils.py   # RAM, physical cores, macOS quirks
│   ├── engines/
│   │   ├── base.py         # Engine abstract base, PID files, log headers
│   │   ├── ktransformers.py  # sglang-kt subprocess adapter
│   │   └── llamacpp.py     # llama-server subprocess adapter
│   └── cli/
│       ├── main.py         # litmoe doctor|init|models|install|serve|status|stop
│       └── install.py      # one-command engine + model installer
├── scripts/
│   ├── claude-local        # Claude Code → gateway, per-process env only
│   └── hermes-local        # Hermes Agent → gateway, per-process env only
├── tests/test_litmoe.py    # unit tests (no network, no engines)
├── examples/models.yaml    # tiered example config
├── deploy/                 # docker-compose: gateway (CPU llama.cpp) + Open WebUI
└── docs/
    ├── SETUP.md            # install, tiers, models.yaml reference
    ├── HARNESSES.md        # Claude Code / Hermes isolation and revert
    ├── METHODOLOGY.md      # this file
    ├── ARCHITECTURE.md     # architecture diagram
    ├── architecture.svg    # rendered diagram
    └── measurements/       # raw llama-server logs behind every t/s figure
```

## What was learned along the way

Engineering lessons from building the previous C engine (kept for honesty, not for re-use):

1. **Don't compete with mature engines.** ktransformers and llama.cpp have
   teams of PhDs. Your CPU forward pass is not a viable alternative.
2. **Hardware bottlenecks don't yield to software.** A 50x compute gap to
   llama.cpp on identical hardware means your optimization is wrong, not
   the hardware.
3. **"Measure" beats "design".** The CPU floor we calculated (82 min/response)
   was confirmed by actual wall-clock measurement (158s TTFT) only after we'd
   shipped several rounds of unmeasured "optimizations".
4. **The real product is integration.** Users want OpenAI-format APIs over
   multiple models on multiple hardware. That is what this dispatcher does.

These lessons apply generally. The repo is the artifact that follows from them.
