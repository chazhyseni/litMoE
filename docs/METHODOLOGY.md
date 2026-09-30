# Methodology

This document explains why litmoe is structured as a Python dispatcher over
native inference engines. The gateway does not execute the forward pass;
it applies documented WARP optimization and DwarfStar serving-contract patches.

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
4. **DwarfStar** — the managed Apple Silicon GLM path uses its native Metal
   graph and SSD-streamed GGUF experts. Native Anthropic/OpenAI routing,
   deferred references, token counts, and acknowledged quiescence are
   integrated; real-model/client latency remains a separate verification step.

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

The gateway:

1. Reads a `models.yaml` config; ships RAM-tiered download choices, a pinned
   DwarfStar streaming GGUF, and WARP conversion recipes. `litmoe init`
   chooses a default using its RAM-fit estimate.
2. Starts the chosen engine as a subprocess (`ds4-server`, `llama-server`,
   `python -m sglang.launch_server`, or WARP's upstream `serve/__main__.py`),
   supervises it, and stops it cleanly.
3. Exposes one OpenAI- and Anthropic-compatible API. `litmoe init` writes
   `host: 127.0.0.1`; when `host` is omitted, the schema default is `0.0.0.0`.
   One model is resident at a time.
4. Resolves the active model's name or alias and rejects inactive selections
   until the operator explicitly switches models.
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

For authenticated WARP source fetches, litmoe puts `HF_TOKEN` in a private
temporary curl config rather than child arguments or environment.

The WARP conversion CLI owns every stage process group it starts: an interrupt
stops the whole download/conversion tree, and a concurrent install of the same model is
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

**Configuration spans engines.** One `models.yaml` can contain:
gemma-4-26b-a4b → llama.cpp on a laptop, glm-5.3-flash → sglang-kt on a GPU
server, or glm-5.3-flash-warp → a local `.waste` container. The downloadable
entries encode what fits where so the default is never a 594 GB download on a
96 GB machine. The two WARP entries instead expose pinned conversion recipes
with explicit source, workspace, and output sizes; manually configured
`.waste` paths remain valid.

**Defaults must fit before they are fast.** The tier list groups entries by
active parameters per token, a proxy for CPU generation speed, and records the
storage and RAM each entry needs; `litmoe init` chooses among entries that fit
this machine. Dense models and big MoEs are listed, not defaulted.

## What was measured

The local llama.cpp rows below are litmoe measurements: each value is a
generation (`eval time`) line in one of the four retained `llama-server` logs
under [`docs/measurements/`](measurements/README.md). That directory documents
the exact per-request values, the grep commands, and what the logs do and do
not record. Values are requests that ran to completion; requests canceled
mid-generation produce no timing line and are not listed. The upstream WARP
figures further down are quoted from upstream and are not in these logs.

| Engine | Model | Threads | Tokens/sec (completed records) | Log |
|---|---|---|---|---|
| llama.cpp | Gemma-4-26B-A4B-it UD-Q4_K_XL (17 GB), `--mmproj` loaded | 24 | 4.29, 1.56, 2.15, 2.02, 5.59, 9.03, 10.62, 11.62, 4.61, 10.21, 10.50, 12.68 | `gemma-4-26b-a4b.log` |
| llama.cpp | Qwen3.8-9B-Distill Q4_K_M (6 GB) | 8 | 8.33, 8.48 | `qwen3.8-9b-distill.log` |
| llama.cpp | Kimi-Linear-48B-A3B Q4_K_M (30 GB) | 48 | 0.03, 0.03, 0.05, 0.42, 0.44, 0.58, 0.45 | `kimi-linear-48b.log` |
| llama.cpp | DeepSeek-V4-Flash-0731 UD-IQ1_S (83 GB) | 48 | 0.32, 0.34, 0.33, 0.11 | `deepseek-v4-flash.log` |

These runs differ in thread count (8, 24, 48), and the servers were configured
with four slots. The logs record per-request prompt and generation timings plus
server events (model loads, session headers, cancellations); they contain no
memory-residency, page-cache, disk-throughput, or competing-process data, so
the spread between requests in one log cannot be attributed to a specific
cause. Prompt sizes varied (5–974 prompt tokens in the Gemma log), so the
`prompt eval time` lines are not a controlled prefill benchmark. There is no
measured ktransformers row; WARP's own figures are below and are not litmoe
measurements.

### Upstream WARP measurements

The two catalog WARP entries are conversion recipes, not litmoe benchmarks.
The figures below are WARP's own, published at the pinned runtime revision
`09fcff352ca55223b08ee222d15054b90546c6a9`:

- [README.md](https://github.com/sqliteai/warp/blob/09fcff352ca55223b08ee222d15054b90546c6a9/README.md) — performance table and container sizes
- [docs/GLM.md](https://github.com/sqliteai/warp/blob/09fcff352ca55223b08ee222d15054b90546c6a9/docs/GLM.md) — GLM-5.3-Flash download, trunk, expert, and floor sizes
- [docs/DS41.md](https://github.com/sqliteai/warp/blob/09fcff352ca55223b08ee222d15054b90546c6a9/docs/DS41.md) — DeepSeek-V4.1-Flash container and floor sizes

| Catalog id | Pinned source revision (HuggingFace) | Source | Conversion workspace (litmoe) | Output container | Upstream resident floor | Upstream throughput |
|---|---|---:|---:|---:|---:|---|
| `glm-5.3-flash-warp` | `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a` | 306 GiB | 120 GiB | 112 GB | 5.14 GB | 3.32 tok/s over 64 tokens; 3.86 over 200 |
| `deepseek-v4.1-flash-warp` | `dba1be0a40aa45a94ad051997016db3960a90277` | 475 GiB (510 GB as published) | 310 GiB | 299 GiB | 4.86 GB | 3.77 tok/s over 64 tokens; 3.71 over 200 |

Upstream measured these on a 64 GB MacBook Pro with an M5 Pro, container on
the internal SSD. The workspace figures are litmoe's own free-space
requirement for the conversion, not upstream numbers.

There are no prebuilt `.waste` release assets for these models. litmoe installs
WARP runtime commit `09fcff352ca55223b08ee222d15054b90546c6a9` with its bundled
native patch and orchestrates the pinned upstream pipeline; it does not supply
a quantizer.

## What you get

- **Laptop, 48–96 GB (Apple Silicon or x86):** llama.cpp with a 3–5B-active
  MoE from the default tier.
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
  IQ1.

Pick the engine per model in `models.yaml`; only one is resident. All three
engines expose local OpenAI HTTP services. litmoe does not execute the forward
pass, but gateway overhead must be measured, not assumed. `litmoe bench`
alternates direct-engine and gateway requests with identical payloads and
records time to headers, generated delta, visible text, and completion.

On a target host, compare the same model artifact, quantization, context,
prompt, generation limit, and backend build. Record memory pressure, swap
growth, disk I/O, and native prefill/decode/cache counters. Repeated prompts
alone do not prove cache hits; alternating order alone does not establish
matched cold/warm state. Output tokens divided by total response time is
end-to-end throughput, not decode-only speed.

WARP remains the path for its supported `.waste` containers; replacing it needs
measurements on the target host. A llama.cpp/Metal or MLX comparison needs
compatible model architecture and weights, equivalent prompting/tool behavior,
and actual measurements there. Linux synthetic-weight protocol checks do not
establish backend performance on Apple Silicon.

## What litmoe does NOT do

- It is not an inference engine. There are no model weights in this repo and
  no forward-pass code. The actual inference is done by local subprocesses.
- It does not call a remote inference API.
- Native engines own their hardware implementations; litmoe carries the
  documented upstream patches rather than a separate compute backend.
- It does not implement quantization or conversion. For catalog WARP models it
  pins the source, invokes WARP's upstream conversion pipeline, validates its
  WARP v0 output, and registers the path. WARP remains the quantizer; other
  conversions use `kt quant`, `llama-quantize`, or third-party tooling such as
  Unsloth and MLX.
- It does not parallelize across machines. Single-node only.

## What's in this repo

Selected paths, not a complete file inventory:

```
litmoe/
├── pyproject.toml
├── litmoe/
│   ├── models.py           # model catalog + pinned WARP/DwarfStar recipes
│   ├── config.py           # Pydantic models.yaml schema (host defaults to 0.0.0.0)
│   ├── runtime.py          # one resident engine and one inference lease, incl. streams
│   ├── server.py           # FastAPI OpenAI/Anthropic gateway
│   ├── benchmark.py        # paired direct-engine vs gateway latency measurement
│   ├── platform_utils.py   # RAM, physical cores, macOS quirks
│   ├── engines/
│   │   ├── base.py         # engine ABC: start/stop/health, PID files, log headers
│   │   ├── llamacpp.py     # llama-server adapter
│   │   ├── ktransformers.py  # sglang-kt adapter
│   │   ├── dwarfstar.py    # pinned native APIs, state identity, quiescence
│   │   ├── warp.py         # upstream WARP server adapter for local .waste containers
│   │   └── warp_context.py # fits WARP's native context via its memory planner
│   └── cli/
│       ├── main.py         # litmoe doctor|init|models|install|serve|status|stop|bench
│       ├── install.py      # engine installs, model downloads, upstream WARP orchestration
│       ├── warp_models.py  # WARP conversion planning, install locks, manifest validation
│       └── benchmark.py    # `litmoe bench` entry point
├── scripts/
│   ├── claude-local        # Claude Code → gateway, per-process env only
│   ├── hermes-local        # Hermes Agent → gateway, per-process env only
│   └── omp-local           # OMP → gateway, per-process env only
├── tests/test_litmoe.py    # unit tests (no network, no engines)
├── examples/models.yaml    # tiered example config
├── deploy/                 # docker-compose (gateway + Open WebUI), caddy/Caddyfile, gateway/
└── docs/                   # SETUP, HARNESSES, METHODOLOGY, ARCHITECTURE, plans/, measurements/
```

