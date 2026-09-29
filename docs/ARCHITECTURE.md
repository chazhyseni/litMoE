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
   │   - catalog n_ctx is memory-aware; WARP 0 keeps its container default          │
   │   - SIGTERM/SIGINT/SIGHUP to the gateway stops every engine (no orphans)       │
   └───────────────┬──────────────────────┬───────────────────────┬────────────────┘
                   │ :8081                │ :8082                 │ :8083
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
`09fcff352ca55223b08ee222d15054b90546c6a9`, then invokes WARP's upstream fetch
dry-run and conversion pipeline. A proven reclaimed resume skips the fetch
dry-run and continues the pipeline.

When `HF_TOKEN` is set, litmoe places it in a private temporary curl config;
the token is not printed or passed through child arguments or environment.
After the pipeline returns, litmoe validates the WARP v0 manifest and its
referenced trunk, codebook, tokenizer, specials, and expert-bank files before
registering the absolute output path as `engine: warp`, `n_ctx: 0`. Partial
source, output, and run/report data are retained so the same command can
resume. The runtime-only `litmoe install --engine warp` and manually configured
local `.waste` containers remain valid alternatives.


## Engine lifecycle

- `litmoe serve` reads `models.yaml`, applies memory-aware context sizing to
  catalogued models, and preserves a WARP container's default when `n_ctx: 0`.
  It starts each engine in its own process group, writes
  `~/.litmoe/run/<id>.pid`, waits for readiness, then serves.
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
├── server.py          gateway, Anthropic↔OpenAI translation, engine supervision
├── platform_utils.py  RAM, physical cores, macOS quirks
├── engines/
│   ├── base.py        Engine ABC: start/stop/health, PID files, log headers
│   ├── llamacpp.py    llama-server adapter (binary discovery, -hf, mmproj, threads)
│   ├── ktransformers.py  sglang-kt adapter (kt-method, GPU experts, cpuinfer)
│   └── warp.py        upstream WARP server adapter for local .waste containers
└── cli/
    ├── main.py        doctor · init · models · serve · status · stop
    └── install.py     engine installs + catalog downloads / upstream WARP orchestration and validation
scripts/claude-local   Claude Code against the gateway, per-process env only
scripts/hermes-local   Hermes Agent against the gateway, per-process env only
tests/test_litmoe.py   catalog, config, ctx math, port allocation, Anthropic translation, stop safety
```
