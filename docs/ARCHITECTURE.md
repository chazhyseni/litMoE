# Architecture

The gateway combines OpenAI/Anthropic protocol handling with a single-resident
runtime. The four backends below are alternatives, not concurrent loads.

## Responsibility boundary

litmoe owns the serving workflow: configuration, installation orchestration,
admission, explicit model selection, owned-process lifecycle, API adaptation,
isolated client launchers, and end-to-end measurements. The selected engine
owns tensor computation, hardware kernels, expert storage/streaming, sampling,
and the actual model-state cache. The connected agent owns planning and tool
execution.

[DwarfStar (`antirez/ds4`)](https://github.com/antirez/ds4), by antirez and its
contributors, supplies the GLM Metal/SSD-streaming engine. Its inference and
state-management features belong to that project, including its acknowledged
llama.cpp/GGML foundations. litmoe's adapter and native serving patch connect
those features to the managed gateway.

See the [upstream acknowledgements](../README.md#upstream-acknowledgements)
and [license boundaries](../THIRD_PARTY_NOTICES.md).

## Serving topology

```text
Claude Code / Hermes / OMP / OpenAI and Anthropic clients
                         |
                         v
litmoe gateway: authentication, explicit model selection, bounded admission
  Messages: native DwarfStar JSON/SSE; translated OpenAI on legacy backends
  Count tokens: native rendered count; marked estimate on legacy backends
  Lifecycle: one owned process, one inference lease, drain/stop/start
                         |
             one selected adapter
              +----------+----------------+--------------+
              |          |                |              |
          DwarfStar   llama.cpp      ktransformers      WARP
          ds4-server  llama-server    SGLang/kt-kernel   WARP HTTP
          Metal/SSD  CPU/GPU GGUF     GPU/CPU experts   .waste paging

models.yaml: selected engine, local artifact, context, aliases, engine options
litmoe/models.py: download catalog and pinned native installation recipes
```


## Data flow

1. Client sends `POST /v1/chat/completions` (or `/v1/messages`) with
   `model: gemma-4-26b-a4b` — or an alias such as `claude-sonnet-4-5`.
2. Gateway resolves the id to an engine and forwards the body to its loopback
   port. DwarfStar owns the native Anthropic representation; other backends
   use Anthropic-to-OpenAI translation.
3. The selected local engine runs the forward pass (CPU, GPU, CPU experts +
   GPU attention, or WARP over a local `.waste` container).
4. Gateway relays the response; streaming responses are passed through byte
   for byte (OpenAI) or re-framed as Anthropic SSE events.

The gateway performs request validation, protocol translation, and proxying;
the selected engine executes inference. Measure gateway overhead separately
instead of assuming a fixed millisecond cost. WARP's upstream server is a
local subprocess, not a remote inference API.

The legacy Anthropic-to-OpenAI translator covers text, `tool_use`/`tool_result`,
and tools, but not every request feature. Anthropic image blocks become textual
`[image: <source type>]` placeholders rather than forwarded image data;
assistant `thinking`/`redacted_thinking` blocks are dropped from outgoing
requests; and a named `tool_choice` is sent as `required`, which forces some
tool call rather than that specific tool. DwarfStar bypasses this translator.

## WARP catalog installation

`litmoe install --model glm-5.3-flash-warp` and
`litmoe install --model deepseek-v4.1-flash-warp` are orchestration paths.
The runtime build applies the native patch described below; model conversion
and quantization remain upstream implementations. The CLI resolves deterministic
absolute source, output, and run/report paths; rejects source or output paths
containing a backslash, single quote, newline, or carriage return; and requires
the three paths not to overlap or nest, including through resolved symlink
aliases. It checks `git`, `make`, `bash`, `curl`, `uv`, and free storage,
prints the pinned revision and size plan, and confirms before writing. It
installs pinned WARP runtime commit
`09fcff352ca55223b08ee222d15054b90546c6a9` with the bundled prefill patch,
then runs WARP's upstream download and conversion pipeline. Each stage runs in
its own session and is
owned by the CLI: on interrupt or hangup the entire stage process group is
terminated, so no orphaned download or conversion survives its parent. A
second install of the same model is refused while one is running. Internal
terminal noise is captured rather than printed, and a heartbeat line with
elapsed seconds and the `download.log` / `pipeline.log` paths repeats every
minute so a multi-hour stage never looks hung.

When `HF_TOKEN` is set, litmoe places it in a private temporary curl config;
the token is not printed or passed through child arguments or environment.
After the pipeline returns, litmoe validates the WARP v0 manifest and its
referenced trunk, codebook, tokenizer, specials, and expert-bank files before
registering the absolute output path as `engine: warp`, `n_ctx: 0`, and
`warp_auto_context: true` (a positive `--n-ctx` selects fixed mode instead). Partial
source, output, and run/report data are retained so the same command can
resume; nothing already downloaded is refetched. The runtime-only
`litmoe install --engine warp` and manually configured local `.waste`
containers remain valid alternatives.

### Native prefill optimization

`litmoe/patches/warp-prefill.patch` is applied to the pinned source before
building and running upstream checks. Installation reuse requires both the
upstream commit and the patch's SHA-256 marker; an older unpatched installation
is rebuilt. A patch/build/check failure leaves the previous installation in
place. The patch ships in source distributions and wheels.

The patch pairs two ARM NEON Q4 projection rows, preserving each row's FMA
order without quantizing activations or expanding weights. The GLM/DSA
token-at-a-time prefill path also skips final normalization/output projection
for intermediate tokens whose logits are never consumed. The final token and
ordinary decode still compute the complete distribution. No model weights,
context limits, thinking settings, or harness tools are changed.

On a 96 GiB M2 Max, a 216-token repeated-text microbenchmark with a 36,000 MiB
expert cache took **97.00 s unpatched versus 91.44 s patched** (5.7% less time).
The final 154,880 logits were byte-identical and eight greedy continuation
tokens matched. This is a modest native gain, not a full-session latency
guarantee. The ARM regression covers nonzero row ranges, odd row counts,
group sizes and partial groups; upstream checks also compare prefill and
token-at-a-time output across architectures.

Reproduce the microbenchmark using upstream `test_forward` from separate
patched/unpatched builds of that commit, with the same compiler and flags:

```bash
# Tokenize this text with `waste tokenize MODEL TEXT --json`, then repeat its
# 54 returned token IDs four times as one comma-separated argument.
# TEXT: You are a coding assistant. Read the relevant source before editing,
# preserve existing behavior, and verify your changes with tests. Explain the
# purpose of a queue in a web server and describe how cancellation interacts
# with a shared inference engine. Include the important ordering guarantees
# and failure cases.
WASTE_CACHE_MB=36000 WASTE_PROFILE=1 WASTE_CHUNK=1 \
  ./test_forward MODEL IDS logits.bin 8
```

Do not run timing comparisons alongside compilation or another model process.

### Full-harness constraints and replacement evaluation

The native patch does not implement batched GLM prefill. WARP's model code
explicitly uses token-at-a-time evaluation when mHC or the DSA indexer is
present, because its chunked path lacks their state bookkeeping. Its HTTP
server resets state for every request; native save/load primitives are not
used for cross-request prefix reuse.

A fresh actual Claude launcher request contained 376 tool schemas and
84,442 native tokens. A discovery-enabled capture contained 15,482 tokens,
but the current translator drops typed tool references; the smaller capture
is not proof that deferred tool execution works. Tokenization took about
27 ms for the eager request, distinguishing prompt construction from model
prefill as a bottleneck.

DwarfStar is integrated through `engines/dwarfstar.py` and a pinned native
serving patch. It bypasses that translator, resolves typed tool references,
and uses native prefix state. Real-model and client-loop verification remains
separate from implementation and compilation. See the [redesign record](plans/2026-09-30-inference-redesign.md).


## Engine lifecycle

- DwarfStar startup requires matching source and patch markers, binds only
  loopback, and uses one native session. The configured context is neither
  fitted by llama.cpp's planner nor silently reduced.
- Its private disk-cache namespace includes model-file identity, source
  revision, and patch digest. Native strict-quant checks remain enabled.
- `/v1/litmoe/status` samples native scheduling state without the inference
  mutex. Counts include accepted clients still parsing/awaiting work and
  workers finishing checkpoint housekeeping, so closed HTTP sockets do not
  falsely acknowledge quiescence.
- Both normal EOF and aborted requests close the upstream connection and
  await native idle before releasing admission. The 10-second deadline or a
  malformed/failed status response triggers an owned-process stop. Confirmed
  cancellation preserves the resident process; switching always unloads it.
- Native `/v1/messages/count_tokens` uses the same renderer and tokenizer as
  inference, without generation. Legacy estimates carry
  `x-litmoe-token-count: estimated`.

- `litmoe serve` applies memory-aware context sizing to llama.cpp models.
  WARP's adapter invokes the installed `serve.engine.plan_memory` in an isolated
  process using the same model environment and native library as the server.
  Auto mode fits the native window against 75% of usable RAM capacity or a
  smaller explicit budget, including recommended expert-cache and vision memory.
  It persists the selected `n_ctx` with `warp_auto_context: true`, so restarts
  re-evaluate the plan. Unmarked legacy 0/65536 values migrate to auto; other
  positive values remain fixed. `warp_auto_context: false` preserves an
  intentional positive limit, including 65536. Conflicting `extra_args --ctx`
  flags are rejected. A planning failure leaves that selection unavailable,
  never silently choosing another model. WARP budgets are capacity estimates,
  not measurements of other applications' current memory pressure.
- The gateway reserves its listening socket before model startup and passes
  that same socket to uvicorn. A busy gateway port fails without spawning
  another engine; there is no bind-and-close preflight race.
- Initial load, explicit switch, cancellation, and shutdown share one asyncio
  runtime on uvicorn's event loop. The active lease covers upstream connection
  establishment and the complete downstream stream. Switches drain first;
  stop failures retain ownership and prevent another engine from starting.
- Cancelling an accepted WARP chat stream closes its upstream socket; the
  pinned server stops generation from its token callback and retains the
  resident engine/expert cache. Cancellation is not an immediate prefill
  interrupt. Non-cooperative paths (including blocking WARP calls and raw
  completions) stop the process before releasing their lease; a later request
  reloads. DwarfStar instead uses the acknowledged-quiescence path above.
  For accepted WARP streams, socket closure can release the gateway lease
  while native prefill is still running. This is a known ownership limitation,
  not a verified cooperative prefill-cancellation capability.
- WARP startup warmup shares the inference lock. It touches some experts,
  not every expert a subsequent prompt will use, and does not guarantee latency.
- Engine stdout/stderr append to `logs/<id>.log` with a per-start header.
  ASGI lifespan cleanup handles uvicorn's graceful Ctrl-C/SIGTERM shutdown.
- `litmoe stop` signals only the process groups in the PID files; `--all`
  additionally matches by name. Nothing else on the machine is touched.
- `litmoe status` polls `/health`.
- `/v1/runtime` reports the configured choices, selected/active model,
  lifecycle state, generation, queue, effective context, and capabilities.
  POST `/v1/runtime/model` performs an explicit selection using the same
  optional API-key policy as inference. Discovery lists ready models only.
- llama.cpp uses one slot; context/slot overrides in `extra_args` are rejected.
  WARP has no cross-request prompt-cache capability at the pinned revision.

## Ports and isolation

| Service | Binding | Configuration |
|---|---|---|
| Gateway, generated config | 127.0.0.1:8090 | `host`/`port` in models.yaml |
| Gateway, omitted `host` | 0.0.0.0:8090 | Set `host: 127.0.0.1` for local-only use |
| Resident engine | Loopback port starting at 8081, skipping gateway and busy ports | Allocated at startup |
| Docker gateway | 127.0.0.1:8000 (host) | `deploy/docker-compose.yml` |
| Open WebUI (Docker) | All host interfaces, port 8080; authentication disabled | Restrict binding and enable authentication before exposure |

Omitting `api_key` disables gateway authentication. Set both the host binding
and authentication policy explicitly before network deployment.

litmoe commands use `LITMOE_*` settings, including configuration, install paths,
runtime timeouts, and CLI gateway credentials. The prompt-cache capability
report also reads `LLAMA_ARG_CACHE_PROMPT` from the process or model `env`.
Engine logs default to `logs/` relative to the working directory (`--log-dir`
overrides); PID files default to `~/.litmoe/run` (`LITMOE_RUN_DIR` overrides).
Context sizing may rewrite the selected configuration file. Installers write
to the selected model/staging paths and `$LITMOE_PREFIX` (default `~/.local`).
The gateway does not configure harness credentials or global client state;
the process-scoped launchers are documented in [HARNESSES.md](HARNESSES.md).

## Source map

```
litmoe/
├── models.py          downloads + pinned WARP/DwarfStar recipes, sizes, context
├── config.py          models.yaml schema + validation
├── server.py          gateway and Anthropic↔OpenAI translation
├── runtime.py         single-resident ownership, admission, cancellation, switching
├── benchmark.py       paired HTTP/SSE timing without private payload logging
├── platform_utils.py  RAM, physical cores, macOS quirks
├── engines/
│   ├── base.py        Engine ABC: start/stop/health, PID files, log headers
│   ├── llamacpp.py    llama-server adapter (binary discovery, -hf, mmproj, threads)
│   ├── ktransformers.py  sglang-kt adapter (kt-method, GPU experts, cpuinfer)
│   ├── warp.py        upstream WARP server adapter for local .waste containers
│   └── dwarfstar.py   pinned native APIs, disk-state identity, quiescence checks
├── patches/           upstream WARP optimizations and DwarfStar serving contract
└── cli/
    ├── main.py        doctor · init · models · serve · switch · status · stop
    ├── benchmark.py   bench CLI
    └── install.py     engine installs + catalog downloads / upstream WARP orchestration and validation
scripts/claude-local   Claude Code against the gateway, per-process env only
scripts/hermes-local   Hermes Agent against the gateway, per-process env only
scripts/omp-local      OMP with an owned isolated model/config directory
tests/test_litmoe.py   catalog, config, ctx math, port allocation, Anthropic translation, stop safety
```
