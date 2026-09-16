# Architecture

The litmoe gateway is intentionally minimal: a FastAPI pass-through plus
process supervision. Every component earns its place.

```
   ┌────────────────────────────────────────────────────────────────────────────────┐
   │   CLIENTS                                                                      │
   │   Claude Code (scripts/claude-local) · Hermes Agent (scripts/hermes-local)     │
   │   Open WebUI · aider · curl · any OpenAI / Anthropic SDK                       │
   └─────────────────────────────────┬──────────────────────────────────────────────┘
                                     │ HTTP, 127.0.0.1:8080
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
   │   - unknown model → 404 with the list of ids that ARE served                   │
   │   - /v1/messages: Anthropic Messages → OpenAI chat (tools, images, thinking,   │
   │     streaming SSE re-framed as Anthropic events)                               │
   │   - stream: raw byte pass-through; non-stream: JSON relay                      │
   │                                                                                │
   │   ENGINE SUPERVISOR                                                            │
   │   - one subprocess per model, own session/pgid, PID file in ~/.litmoe/run      │
   │   - ports 8081+ skipping the gateway port and anything already bound           │
   │   - memory-aware n_ctx: native ctx unless weights+KV exceed RAM budget         │
   │   - SIGTERM/SIGINT/SIGHUP to the gateway stops every engine (no orphans)       │
   └────────────┬───────────────────────────────────────┬───────────────────────────┘
                │ http://127.0.0.1:8082                 │ http://127.0.0.1:8081
                ▼                                       ▼
   ┌─────────────────────────────┐    ┌────────────────────────────────┐
   │   KTRANSFORMERS ENGINE      │    │   LLAMA.CPP ENGINE             │
   │   engines/ktransformers.py  │    │   engines/llamacpp.py          │
   │                             │    │                                │
   │   spawns:                   │    │   spawns: llama-server         │
   │   python -m sglang.launch_  │    │     -m <gguf> | -hf repo:QUANT │
   │     server --kt-method …    │    │     -c <ctx> -t <phys cores>   │
   │                             │    │     -ngl … --mmproj …          │
   │   attention on GPU (CUDA)   │    │                                │
   │   routed experts on CPU     │    │   CUDA / HIP / Metal / Vulkan  │
   │   kt-kernel: AMX / AVX-512  │    │   / SYCL / CPU                 │
   │   / AVX2, INT4/INT8/FP8/    │    │   GGUF 1–8 bit (Unsloth UD-*)  │
   │   RAWINT4 experts           │    │                                │
   │                             │    │   models: every tier           │
   │   models: 192 GB+ tiers     │    │                                │
   │   (GLM-5.3-Flash, DeepSeek  │    │                                │
   │   V4-Flash/V3.2, Kimi-K2.x, │    │                                │
   │   MiniMax-M2.7/M3)          │    │                                │
   └─────────────────────────────┘    └────────────────────────────────┘
                │                                       │
                └─────────────────┬─────────────────────┘
                                  ▼
                   ┌──────────────────────────────────────┐
                   │  models.yaml  (pydantic: config.py)  │
                   │  host / port / api_key               │
                   │  models:                             │
                   │    - id, engine, model_path          │
                   │      n_ctx, n_gpu_layers, extra_args │
                   │      env, aliases, kt_* fields       │
                   └──────────────────────────────────────┘
                                  ▲
                   ┌──────────────┴───────────────────────┐
                   │  litmoe/models.py — catalog           │
                   │  tiers 48 / 96 / 192 / 512 / 768 GB   │
                   │  hf_repo, quants+sizes, native ctx,   │
                   │  KV bytes/token, engine, kt flags     │
                   │  → `litmoe models`, `install`, `init` │
                   └──────────────────────────────────────┘
```

## Data flow

1. Client sends `POST /v1/chat/completions` (or `/v1/messages`) with
   `model: gemma-4-26b-a4b` — or an alias such as `claude-sonnet-4-5`.
2. Gateway resolves the id to an engine and forwards the body to that engine's
   loopback port. For `/v1/messages` it first translates Anthropic → OpenAI.
3. The engine runs the forward pass (CPU, GPU, or CPU experts + GPU attention).
4. Gateway relays the response; streaming responses are passed through byte
   for byte (OpenAI) or re-framed as Anthropic SSE events.

The gateway never touches the forward pass; it adds a few milliseconds and no
compute.

## Engine lifecycle

- `litmoe serve` reads `models.yaml`, fixes any stale `n_ctx` (memory-aware,
  written back to the file), starts each engine in its own process group,
  writes `~/.litmoe/run/<id>.pid`, waits for readiness, then serves.
- Engine stdout/stderr append to `logs/<id>.log` with a per-start header.
- Ctrl-C / SIGTERM / SIGHUP to the gateway stops all engines. (uvicorn
  re-raises the signal after its own graceful exit; litmoe installs a handler
  so that re-raise runs engine shutdown instead of killing the process.)
- `litmoe stop` signals only the process groups in the PID files; `--all`
  additionally matches by name. Nothing else on the machine is touched.
- `litmoe status` polls `/health`.

## Ports and isolation

| Service | Default | Configurable |
|---|---|---|
| Gateway | 127.0.0.1:8080 | `host`/`port` in models.yaml |
| Engines | 8081, 8082, … (skips gateway port and busy ports) | `DEFAULT_ENGINE_PORT` |
| Docker gateway | 127.0.0.1:8000 (host) | `deploy/docker-compose.yml` |
| Open WebUI (Docker) | 8080 | `deploy/docker-compose.yml` |

litmoe reads only `LITMOE_*` environment variables and writes only under
`~/.litmoe/` and `models.yaml`. It never sets `ANTHROPIC_*`/`OPENAI_*` or
edits harness configuration; see [HARNESSES.md](HARNESSES.md).

## Source map

```
litmoe/
├── models.py          catalog (tiers, quants, sizes, ctx, KV) — single source of truth
├── config.py          models.yaml schema + validation
├── server.py          gateway, Anthropic↔OpenAI translation, engine supervision
├── platform_utils.py  RAM, physical cores, macOS quirks
├── engines/
│   ├── base.py        Engine ABC: start/stop/health, PID files, log headers
│   ├── llamacpp.py    llama-server adapter (binary discovery, -hf, mmproj, threads)
│   └── ktransformers.py  sglang-kt adapter (kt-method, GPU experts, cpuinfer)
└── cli/
    ├── main.py        doctor · init · models · serve · status · stop
    └── install.py     engine install (release asset / build / kt wheels) + model download
scripts/claude-local   Claude Code against the gateway, per-process env only
scripts/hermes-local   Hermes Agent against the gateway, per-process env only
tests/test_litmoe.py   catalog, config, ctx math, port allocation, Anthropic translation, stop safety
```
