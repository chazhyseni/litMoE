# Methodology

This document explains why litmoe is structured as a Python dispatcher over
native inference engines. The gateway does not execute the forward pass;
the WARP installer does apply a narrowly scoped native optimization patch.

## The serving workflow

litmoe's role is to make native engines usable as one managed local service.
Model artifacts, hardware budgets, backend flags, process ownership, API
formats, and agent configuration have to agree before a coding session works.
Those are the integration responsibilities this project takes on.

The shipped adapters cover different hardware and storage arrangements:

1. **llama.cpp** — GGUF, every quant from 1.5 to 8 bit, CUDA/HIP/Metal/Vulkan/
   SYCL/CPU. The right tool from a 48 GB laptop up to a many-core server.
2. **ktransformers (sglang-kt + kt-kernel)** — attention on one GPU, routed
   experts on the CPU with AMX/AVX-512 kernels. The right tool for the
   200 GB–1 TB models on a single-GPU box with lots of RAM.
3. **WARP** — local `.waste` containers whose expert weights are memory-mapped
   and paged from fast local storage. It can run models whose complete weights
   exceed resident RAM, but that capacity does not establish interactive
   latency. The current GLM prefill and HTTP state-reset behavior are
   documented limitations.

Users need one endpoint, one config, correct protocol handling, and verified
performance with their actual agent workloads. Backend support alone does not
establish that performance; an inadequate execution path may need replacement.

The design delegates model computation to upstream engines instead of making
litmoe another inference implementation. Its contribution is the surrounding
workflow: reproducible setup, explicit lifecycle decisions, isolated client
configuration, correct supported protocol semantics, and measurements of the
actual user-facing path. This boundary also makes backend replacement
possible without asking every client to manage native processes itself.

## How litmoe uses upstream tools

| Upstream component | What it supplies | What litmoe does around it |
| --- | --- | --- |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) / GGML ecosystem | GGUF execution, quantization formats, hardware kernels, native server | Installs the runtime/artifact, fits context, supervises one server slot, routes requests |
| [KTransformers](https://github.com/kvcache-ai/ktransformers) + [SGLang](https://github.com/sgl-project/sglang) | Heterogeneous CPU/GPU inference and serving | Configures the native serving stack and owns its process lifecycle |
| [WARP](https://github.com/sqliteai/warp) | `.waste` conversion, expert paging, native inference and HTTP serving | Orchestrates pinned conversion, validates artifacts, applies its documented patch, plans context, supervises serving |
| [DwarfStar](https://github.com/antirez/ds4), by antirez and contributors | Model-specific native GPU execution, expert streaming, state handling, and server | Installs pinned runtime/GGUF, patches native discovery/count/status contracts, routes native APIs, and supervises cancellation and lifecycle |
| Model authors and artifact publishers | Trained weights, model/tokenizer metadata, quantized releases | Records source/artifact choices and installation recipes; does not claim authorship of the models |

Native engine features remain upstream work. DwarfStar's own acknowledgement
of llama.cpp and GGML is part of that attribution chain; see
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). Project identity comes from
the responsibilities litmoe implements, not from claiming those engines'
algorithms or comparing itself against them.

litmoe is the front door: a Python package that:

1. Reads a `models.yaml` config; ships RAM-tiered download choices plus pinned
   WARP conversion recipes, while `litmoe init` picks a fast default for this
   machine.
2. Starts the chosen engine as a subprocess (`llama-server`,
   `python -m sglang.launch_server`, or WARP's upstream `serve/__main__.py`),
   supervises it, and stops it cleanly.
3. Exposes a single OpenAI + Anthropic-compatible API on 127.0.0.1:8090.
4. Routes requests to the right engine by model name or alias.
5. For the two catalog WARP models, resolves pinned source and runtime
   revisions; rejects source or output paths containing a backslash, single
   quote, newline, or carriage return; requires source, output, and run/report
   paths not to overlap or nest, including through resolved symlink aliases;
   preflights `git`, `make`, `bash`, `curl`, `uv`, and storage; invokes WARP's
   upstream fetch/conversion scripts in a litmoe-owned session; validates the
   WARP v0 manifest and artifacts; and writes the absolute container path to
   the config.
6. Connects agent harnesses (Claude Code, Hermes, OMP) **per process**, never by
   rewriting their global configuration.

The gateway has no custom forward pass or quantizer. The WARP install path
orchestrates upstream conversion and applies the bundled ARM Q4/prefill patch
described in [ARCHITECTURE.md](ARCHITECTURE.md#native-prefill-optimization).
Its measured 5.7% short-prefill improvement did not fix full-harness latency.

For authenticated source fetches, litmoe puts `HF_TOKEN` in a private temporary
curl config rather than child arguments or environment.

The CLI owns every stage process group it starts: an interrupt stops the whole
download/conversion tree, and a concurrent install of the same model is
refused. Resuming the same command continues from the on-disk shard ledger —
nothing already downloaded is refetched.

While the stage runs, litmoe captures internal fetch/pipeline terminal output
and repeats a one-minute heartbeat with elapsed seconds and the staging
`download.log` plus run-report `pipeline.log` paths, so multi-hour runs stay
observable in a terminal; the CLI heartbeat is a snapshot and the live shard
counter remains in the download log.

## Why a dispatcher is the right shape

**Reuse engines, but verify the model-specific path.** ktransformers supports
heterogeneous CPU/GPU execution; llama.cpp offers broad quantization and
hardware coverage; WARP pages experts from local containers. None of that
proves fast prefill for a particular architecture. Prefer a measured upstream
implementation over recreating an entire GPU graph in the gateway project.
The [GLM replacement investigation](plans/2026-09-30-inference-redesign.md)
therefore selects DwarfStar's existing model-specific Metal graph for integration
instead of another substantial WARP rewrite. Real workload measurements remain
the test of performance, not the existence of an adapter.

**Engines already speak HTTP.** `ds4-server`, `llama-server`,
`sglang.launch_server` (the ktransformers serving stack since v0.4), and
WARP's upstream server expose local HTTP services. litmoe passes native
protocols through where supported and adapts Anthropic to OpenAI on other
backends. None of these subprocess integrations is a remote inference API.

**Configuration is the hard part.** Users don't care which engine is running;
they care which model responds, and that it is fast enough on the hardware
they have. The dispatcher lets one `models.yaml` mix engines:
gemma-4-26b-a4b → llama.cpp on a laptop, glm-5.3-flash → sglang-kt on a GPU
server, or glm-5.3-flash-warp → a local `.waste` container. The downloadable
entries encode what fits where so the default is never a 594 GB download on a
96 GB machine. The two WARP entries instead expose pinned conversion recipes
with explicit source, workspace, and output sizes; manually configured
`.waste` paths remain valid.

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


### Upstream WARP measurements

WARP upstream reports the figures below. They were not measured by litmoe and
are not litmoe performance guarantees:

| Catalog id | Pinned source revision | Source | Conversion workspace | Output | Upstream resident floor | Upstream throughput |
|---|---|---:|---:|---:|---:|---:|
| `glm-5.3-flash-warp` | `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a` | 306 GiB | 120 GiB | 112 GB | 5.14 GB | 3.32 tok/s short; 3.86 tok/s long on WARP's 64 GB M5 Pro |
| `deepseek-v4.1-flash-warp` | `dba1be0a40aa45a94ad051997016db3960a90277` | 475 GiB | 310 GiB | 299 GiB | 4.86 GB | about 3.7 tok/s |

WARP's published throughput assumes internal NVMe and is hardware-specific.
There are no prebuilt `.waste` release assets for these models. litmoe installs
WARP runtime commit `09fcff352ca55223b08ee222d15054b90546c6a9` with its bundled
native patch and orchestrates the pinned upstream pipeline; it does not supply
a quantizer.


## What you get

- **Laptop, 48–96 GB (Apple Silicon or x86):** llama.cpp with a 3–5B-active
  MoE from the default tier. Interactive.
- **Workstation, 192 GB:** llama.cpp with DeepSeek-V4-Flash / MiniMax-M2.7 /
  Qwen3.8-Flash-Next at Q4.
- **Server with a CUDA GPU and 512 GB+:** sglang-kt with CPU expert offload
  for GLM-5.3-Flash, Kimi-K2.x, DeepSeek-V3.2, MiniMax-M3.
- **Fast-internal-NVMe host:** WARP can trade storage capacity for a low
  resident floor. Install a catalog container with `litmoe install --model
  glm-5.3-flash-warp` (or `deepseek-v4.1-flash-warp`), or configure an existing
  local `.waste` container manually. Treat upstream throughput as
  hardware-specific.
- **Server, CPU only, 768 GB+:** llama.cpp with Kimi-K3 / Qwen3.8-2.4T at
  IQ1 — batch use, not chat, unless the CPU has many memory channels.

Pick the engine per model in `models.yaml`; only one is resident. All three
engines expose local OpenAI HTTP services. litmoe does not execute the forward
pass, but gateway overhead must be measured, not assumed. `litmoe bench`
alternates direct-engine and gateway requests with identical payloads and
records time to headers, generated delta, visible text, and completion.

On the target 96 GB Apple Silicon Mac, compare the same model artifact,
quantization, context, prompt, generation limit, and backend build. Record
memory pressure, swap growth, disk I/O, and native prefill/decode/cache counters.
Repeated prompts alone do not prove cache hits; alternating order alone does
not establish matched cold/warm state. Output tokens divided by total response
time is end-to-end throughput, not decode-only speed.

Retain WARP for its supported `.waste` containers until target-machine evidence
justifies replacing it. A llama.cpp/Metal or MLX comparison needs compatible
model architecture and weights, equivalent prompting/tool behavior, and actual
Mac measurements. Linux synthetic-weight protocol checks cannot select the
fastest Apple Silicon backend.

## What litmoe does NOT do

- It is not an inference engine. There are no model weights in this repo and
  no forward-pass code. The actual inference is done by local subprocesses.
- It does not call a remote inference API.
- It does not optimize for specific hardware. That's the engines' job.
- It does not implement quantization or conversion. For catalog WARP models it
  pins the source, invokes WARP's upstream conversion pipeline, validates its
  WARP v0 output, and registers the path. WARP remains the quantizer; other
  conversions use `kt quant`, `llama-quantize`, or third-party tooling such as
  Unsloth and MLX.
- It does not parallelize across machines. Single-node only.

## What's in this repo

```
litmoe/
├── pyproject.toml          # modern Python package
├── litmoe/
│   ├── models.py           # model catalog, including pinned WARP conversion recipes
│   ├── config.py           # Pydantic models.yaml schema
│   ├── server.py           # FastAPI OpenAI/Anthropic gateway + engine supervision
│   ├── platform_utils.py   # RAM, physical cores, macOS quirks
│   ├── engines/
│   │   ├── base.py         # Engine abstract base, PID files, log headers
│   │   ├── ktransformers.py  # sglang-kt subprocess adapter
│   │   ├── llamacpp.py       # llama-server subprocess adapter
│   │   └── warp.py           # upstream WARP server adapter for local .waste containers
│   └── cli/
│       ├── main.py           # litmoe doctor|init|models|install|serve|status|stop
│       └── install.py        # engine installs, model downloads, upstream WARP orchestration/validation
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

1. **Don't compete with mature engines.** llama.cpp, ktransformers, and WARP
   already specialize in compute kernels and storage-aware inference.
2. **Hardware bottlenecks don't yield to software.** A 50x compute gap to
   llama.cpp on identical hardware means your optimization is wrong, not
   the hardware.
3. **"Measure" beats "design".** The CPU floor we calculated (82 min/response)
   was confirmed by actual wall-clock measurement (158s TTFT) only after we'd
   shipped several rounds of unmeasured "optimizations".
4. **The real product is integration.** Users want OpenAI-format APIs over
   multiple models on multiple hardware. That is what this dispatcher does.

These lessons apply generally. The repo is the artifact that follows from them.
