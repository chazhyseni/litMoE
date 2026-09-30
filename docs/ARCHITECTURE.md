# Architecture

The gateway combines OpenAI/Anthropic protocol handling with a single-resident
runtime. The three backend boxes below are alternatives, not concurrent loads.

```
   ┌────────────────────────────────────────────────────────────────────────────────┐
   │   CLIENTS                                                                      │
   │   Claude Code (scripts/claude-local) · Hermes Agent (scripts/hermes-local)     │
   │   OMP (scripts/omp-local) · Open WebUI · aider · curl · SDK clients             │
   └─────────────────────────────────┬──────────────────────────────────────────────┘
                                     │ HTTP, 127.0.0.1:8090
                                     │ POST /v1/chat/completions   (OpenAI)
                                     │ POST /v1/completions        (OpenAI)
                                     │ POST /v1/messages           (Anthropic)
                                     │ POST /v1/messages/count_tokens
                                     │ GET  /v1/models · GET /health
                                     ▼
   ┌────────────────────────────────────────────────────────────────────────────────┐
   │                              LITMOE GATEWAY  (litmoe/server.py)                │
   │                                                                                │
   │   REQUEST ROUTER                                                               │
   │   - read `model` from the body; resolve aliases (claude-* → local id)          │
   │   - unknown model → 404; known inactive model → 409; no implicit switching     │
   │   - /v1/messages: Anthropic Messages → OpenAI chat (tools, images, thinking,   │
   │     streaming SSE re-framed as Anthropic events)                               │
   │   - stream: raw byte pass-through; non-stream: JSON relay                      │
   │                                                                                │
   │   ENGINE SUPERVISOR                                                            │
   │   - one resident subprocess, own session/pgid, PID file in ~/.litmoe/run        │
   │   - one inference lease, bounded queue, explicit drain/stop/start switching    │
   │   - llama.cpp and WARP context fitted using their own memory models            │
   │   - ASGI lifespan drains and stops the owned engine on graceful shutdown      │
   └───────────────┬──────────────────────┬───────────────────────┬────────────────┘
                   │ alternative          │ alternative           │ alternative
                   ▼                      ▼                       ▼
   ┌────────────────────────┐  ┌────────────────────────┐  ┌────────────────────────┐
   │ LLAMA.CPP ENGINE       │  │ KTRANSFORMERS ENGINE   │  │ WARP ENGINE            │
   │ engines/llamacpp.py    │  │ ktransformers.py       │  │ engines/warp.py        │
   │                        │  │                        │  │                        │
   │ spawns llama-server    │  │ spawns python -m       │  │ spawns upstream        │
   │ -m <gguf> or           │  │ sglang.launch_server   │  │ serve/__main__.py      │
   │ -hf repo:QUANT         │  │ --kt-method …          │  │ <local .waste>         │
   │                        │  │                        │  │                        │
   │ CUDA / HIP / Metal /   │  │ CUDA attention; CPU   │  │ mmap + local storage   │
   │ Vulkan / SYCL / CPU    │  │ experts via AMX /     │  │ paging; container      │
   │ GGUF 1–8 bit           │  │ AVX-512 / AVX2        │  │ owns its defaults      │
   └────────────────────────┘  └────────────────────────┘  └────────────────────────┘
                   │                      │                       │
                   └──────────────────────┴───────────┬───────────┘
                                                      ▼
                                  ┌──────────────────────────────────────┐
                                  │ models.yaml  (pydantic: config.py)  │
                                  │ host / port / api_key               │
                                  │ models:                             │
                                  │   - id, engine, model_path          │
                                  │     n_ctx, n_gpu_layers, extra_args │
                                  │     env, aliases, kt_* fields       │
                                  └──────────────────────────────────────┘
                                                      ▲
                                  ┌───────────────────┴──────────────────┐
                                  │ litmoe/models.py — install catalog   │
                                  │ downloads + pinned WARP recipes      │
                                  │ → `models`, `install`, `init`        │
                                  │ manual local paths supported         │
                                  │                                      │
                                  └──────────────────────────────────────┘

## Data flow

1. Client sends `POST /v1/chat/completions` (or `/v1/messages`) with
   `model: gemma-4-26b-a4b` — or an alias such as `claude-sonnet-4-5`.
2. Gateway resolves the id to an engine and forwards the body to that engine's
   loopback port. For `/v1/messages` it first translates Anthropic → OpenAI.
3. The selected local engine runs the forward pass (CPU, GPU, CPU experts +
   GPU attention, or WARP over a local `.waste` container).
4. Gateway relays the response; streaming responses are passed through byte
   for byte (OpenAI) or re-framed as Anthropic SSE events.

The gateway never touches the forward pass; it adds a few milliseconds and no
compute. WARP's upstream server is a local subprocess, not a remote inference
API.

## WARP catalog installation

`litmoe install --model glm-5.3-flash-warp` and
`litmoe install --model deepseek-v4.1-flash-warp` are orchestration paths, not
new inference or quantization implementations. The CLI resolves deterministic
absolute source, output, and run/report paths; rejects source or output paths
containing a backslash, single quote, newline, or carriage return; and requires
the three paths not to overlap or nest, including through resolved symlink
aliases. It checks `git`, `make`, `bash`, `curl`, `uv`, and free storage,
prints the pinned revision and size plan, and confirms before writing. It
installs pinned WARP runtime commit
`09fcff352ca55223b08ee222d15054b90546c6a9`, then runs WARP's upstream
download and conversion pipeline. Each stage runs in its own session and is
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


## Engine lifecycle

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
- Initial load, explicit switch, cancellation, and shutdown share one asyncio
  runtime on uvicorn's event loop. The active lease covers upstream connection
  establishment and the complete downstream stream. Switches drain first;
  stop failures retain ownership and prevent another engine from starting.
- Active-request cancellation stops the native process before releasing its
  lease; a later request reloads the same model. This intentionally loses
  cache state rather than assuming a closed HTTP connection stopped inference.
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

| Service | Default | Configurable |
|---|---|---|
| Gateway | 127.0.0.1:8090 | `host`/`port` in models.yaml |
| Engines | 8081, 8082, … (skips gateway port and busy ports) | `DEFAULT_ENGINE_PORT` |
| Docker gateway | 127.0.0.1:8000 (host) | `deploy/docker-compose.yml` |
| Open WebUI (Docker) | 8080 | `deploy/docker-compose.yml` |

At runtime litmoe reads only `LITMOE_*` environment variables and writes only
under `~/.litmoe/` and `models.yaml`; engine installers also write to
`$LITMOE_PREFIX` (default `~/.local`). It never sets
`ANTHROPIC_*`/`OPENAI_*` or edits harness configuration; see
[HARNESSES.md](HARNESSES.md).

## Source map

```
litmoe/
├── models.py          catalog (downloads + pinned WARP recipes, sizes, ctx, KV)
├── config.py          models.yaml schema + validation
├── server.py          gateway and Anthropic↔OpenAI translation
├── runtime.py         single-resident ownership, admission, cancellation, switching
├── benchmark.py       paired HTTP/SSE timing without private payload logging
├── platform_utils.py  RAM, physical cores, macOS quirks
├── engines/
│   ├── base.py        Engine ABC: start/stop/health, PID files, log headers
│   ├── llamacpp.py    llama-server adapter (binary discovery, -hf, mmproj, threads)
│   ├── ktransformers.py  sglang-kt adapter (kt-method, GPU experts, cpuinfer)
│   └── warp.py        upstream WARP server adapter for local .waste containers
└── cli/
    ├── main.py        doctor · init · models · serve · switch · status · stop
    ├── benchmark.py   bench CLI
    └── install.py     engine installs + catalog downloads / upstream WARP orchestration and validation
scripts/claude-local   Claude Code against the gateway, per-process env only
scripts/hermes-local   Hermes Agent against the gateway, per-process env only
scripts/omp-local      OMP with an owned isolated model/config directory
tests/test_litmoe.py   catalog, config, ctx math, port allocation, Anthropic translation, stop safety
```
