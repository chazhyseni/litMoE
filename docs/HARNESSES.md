# Connecting agent harnesses

The launchers route individual Claude Code, Hermes, and OMP sessions to the
local gateway without changing their default provider configuration. Claude
Code and OMP use separate state directories; Hermes uses per-invocation
provider flags.

## Configuration isolation and binding

- Never rewrites `~/.claude/`, `~/.hermes/config.yaml`, `~/.hermes/.env`,
  normal OMP profiles, or shell rc files. The OMP launcher owns only its
  marked `~/.litmoe/omp/` directories.
- Never exports environment variables into your shell. `litmoe serve` sets
  variables only for the engine subprocesses it spawns.
- Generated configurations use `host: 127.0.0.1` and `port: 8090`.
  Set `host` explicitly: omitting it defaults to `0.0.0.0`, with no
  authentication unless `api_key` is set. Owned engines use loopback ports
  starting at 8081, skipping the gateway port and existing listeners.
- `litmoe stop` only signals engines litmoe started (tracked in
  `~/.litmoe/run/*.pid`). An Ollama/LM Studio/manual `llama-server` is left
  alone unless you pass `--all`.

## One resident model

`litmoe serve MODEL` loads one configured model; omitted MODEL means the first
entry. `litmoe switch OTHER` drains the current request, stops the owned native
process, then starts OTHER. Client `--model` flags do not trigger switching.
Known inactive IDs return 409 instead of substituting another model.
After switching, relaunch the wrapper so its model and context match discovery.

`litmoe status` and `/v1/runtime` expose selected/active IDs, readiness, queue,
effective context, and cache/cancellation capabilities. `/v1/models` lists only
the ready model and its aliases. The runtime endpoints use the same `api_key`
authentication as inference; keep unauthenticated gateways on loopback.

The HTTP listener starts **before** model loading finishes. During startup,
`/health` and `/v1/runtime` remain available with state `loading`, model
discovery is empty, and inference returns HTTP 503 until the engine is ready.
Use `litmoe status` to distinguish loading or failure from an unreachable
gateway; an open HTTP port does not itself mean inference is ready.

For WARP, initial startup sends a best-effort four-token warmup request under
the same inference lock used for serving. Discovery can report `ready` before
warmup finishes, but admitted requests cannot execute concurrently with it.
Switches and cancellation reloads skip this step; later prompts may still
need cold expert reads.

One inference lease lasts through the whole stream. The default queue permits
eight waiting admissions for up to 600 seconds; overflow or expiry returns 429.
Disconnecting a queued client removes its wait without dispatching inference.
For accepted WARP chat streams, cancellation closes the upstream connection:
the native token callback stops generation without reloading the engine.
This does not interrupt prefill immediately. Blocking WARP requests, raw
completions, and non-cooperative adapters terminate the owned process on active
cancellation; the next request reloads the selected model.

DwarfStar closes the upstream connection, then waits for native work and its
queue to become idle before releasing admission. This also covers apparent
normal EOF after a transport failure. Confirmed quiescence preserves the
process; failure or the 10-second deadline stops it and resets live cache.
Cancellation is checked between native prefill chunks, not instantaneously.

WARP reports no reusable prompt cache: the pinned server resets state for each
HTTP request. Its expert-weight cache is not a conversation-prefix cache.
llama.cpp may retain a prefix in its single backend slot; this is backend
capability, not a claim that a particular request hit cache. Switches and
cancellation clear llama.cpp's live state. DwarfStar owns its native live and
disk prefix cache; artifact/runtime-separated disk namespaces retain the
native strict-quant guard. No hidden prompt pruning or gateway cloud fallback.

## Claude Code

### How Claude Code decides where to send requests

| Setting | Effect | Where it can come from |
|---|---|---|
| `ANTHROPIC_BASE_URL` | Endpoint for every API call | env var, `settings.json` `env` block |
| `ANTHROPIC_AUTH_TOKEN` / `ANTHROPIC_API_KEY` | Credential; when either is set your claude.ai / Enterprise login is **not** used for that process | env, `apiKeyHelper` in settings |
| `ANTHROPIC_MODEL`, `ANTHROPIC_DEFAULT_{SONNET,OPUS,HAIKU}_MODEL`, `CLAUDE_CODE_SUBAGENT_MODEL` | Which model id the `sonnet`/`opus`/`haiku` aliases and subagents resolve to | env, settings `model` |
| `CLAUDE_CONFIG_DIR` | Where sessions, settings, and cached credentials live (default `~/.claude`) | env |
| `CLAUDE_CODE_MAX_CONTEXT_TOKENS` | Effective context discovered from the resident gateway model, rather than the unknown-model default | wrapper process env |

Environment variables override `settings.json`; both are read per process.
That is what makes clean isolation possible: set them **only** for the one
`claude` process that should talk to litmoe.

### The safe way: `scripts/claude-local`

```bash
litmoe serve                                   # gateway on :8090
./scripts/claude-local                         # Claude Code -> local model, isolated
./scripts/claude-local --model qwen3.6-35b-a3b -p "explain this repo"
claude                                         # normal Claude Code, untouched
```

`claude-local` (copy it onto your PATH if you like):

1. checks the gateway, selects its ready model (or validates `--model`), and
   passes the discovered context as `CLAUDE_CODE_MAX_CONTEXT_TOKENS`;
2. **unsets** any `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` / Bedrock /
   Vertex variables inherited from your shell, so nothing leaks either way;
3. sets `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN` (any string when the
   gateway has `api_key: null`), and all model-alias variables to the local
   model — so `sonnet`, `haiku`, subagents and background calls stay local;
4. sets `CLAUDE_CONFIG_DIR=~/.litmoe/claude-config` so local-model sessions
   and settings never mix with your real ones;
5. sets `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` (no auto-update/telemetry
   for that process);
6. `exec`s `claude --model <id> "$@"` — all of the above dies with that process.

To verify isolation on your installation, compare `claude auth status --text`
with `./scripts/claude-local auth status --text`, then run a local print-mode
request. Check that the local session uses the gateway endpoint and that plain
`claude` still uses your normal account or configured provider.

### MCP tool loading and large greetings

[Claude Code disables tool search by default for custom API endpoints](https://code.claude.com/docs/en/mcp#scale-with-mcp-tool-search)
because many proxies do not support typed tool references. In a fresh
print-mode request from this project's launcher, a greeting carried 376 tool
schemas, including 353 Claude Flow MCP tools. Existing gateway translation
and native WARP tokenization produced **84,442 tokens**; tokenization itself
took approximately 27 ms.

A request-only capture with `ENABLE_TOOL_SEARCH=true` reduced that to
**15,482 tokens**, retaining thinking and the discovery catalog. The capture
deliberately returned HTTP 400 without inference. It did **not** establish a
working search/call/result loop or DwarfStar token counts.

The DwarfStar path now forwards native Anthropic requests and SSE without
the lossy OpenAI translation. Its patch resolves `tool_reference` blocks
(`name` or `tool_name`) from the request's tool catalog, leaves deferred
schemas out of the eager tool header, and rejects unresolved references.
`claude-local` checks the ready runtime's capabilities and enables
`ENABLE_TOOL_SEARCH=true` only for this supported path. Other backends retain
`false`; this never changes global Claude configuration.

Native count requests use the backend renderer/tokenizer without generation.
Legacy counts remain estimates, marked `x-litmoe-token-count: estimated`.
Native reasoning controls remain native; forced tool choice (`any` or a
named tool) is explicitly rejected rather than silently treated as `auto`.
`auto` and `none` work. The catalog recipe does not install a vision encoder.
The legacy translator still has feature limitations; native support is not a
claim that every backend supports every Anthropic request.

OMP uses DwarfStar's native OpenAI endpoint through the gateway. Its isolated
provider enables reasoning-effort controls when native capability is present.
These protocol changes are separate from the [real-client verification record](measurements/README.md)
and the historical full-OMP timeout below.

### What NOT to do

- Do not `export ANTHROPIC_BASE_URL=...` in your shell or `.bashrc`/`.zshrc`;
  it redirects Claude Code sessions and other clients that read the variable.
- Do not put `ANTHROPIC_BASE_URL` in `~/.claude/settings.json` `env`. That is
  global for every session. If you want a *project* that always uses the local
  model, put it in that project's `.claude/settings.local.json` (gitignored)
  — it then applies only inside that directory.
- Do not set `ANTHROPIC_API_KEY` globally to a dummy value: Claude Code
  prefers it over your subscription login.

### Aliases in models.yaml

Claude Code sends Anthropic model names — the one you pick plus
`claude-haiku-4-5` for background tasks even when `--model` is set. `litmoe
init` and `litmoe install` attach the common Claude names as `aliases:` on the
first model. An alias is usable only while its model is resident; inactive
aliases return 409. This has no effect on real Anthropic traffic.
`claude-local` also sets the `ANTHROPIC_DEFAULT_*_MODEL` variables so
Claude Code itself sends the local id wherever it can.

### Undo / revert

Nothing to undo. Close the `claude-local` session; run `claude`. To also drop
the isolated state: `rm -rf ~/.litmoe/claude-config`.

## Hermes Agent

### How Hermes decides

`~/.hermes/config.yaml` → `model.provider`, `model.default`, `model.base_url`
(plus `~/.hermes/.env` for `OPENAI_BASE_URL` / keys). Per-run flags override it
without persisting: `hermes chat --provider <name> -m <model>`. Profiles
(`hermes profile create X`) give a fully separate `~/.hermes/profiles/X/` with
its own config, sessions, skills and memory.

### The safe way: `scripts/hermes-local` (one session)

```bash
./scripts/hermes-local                    # chat with the first gateway model
./scripts/hermes-local -q "one question"
hermes                                    # normal Hermes, config untouched
```

It runs `hermes chat --provider custom -m <model>` with
`CUSTOM_BASE_URL` and `OPENAI_BASE_URL` set to the gateway's `/v1` endpoint,
and `OPENAI_API_KEY` set from `--key` or `LITMOE_API_KEY` (default `litmoe`).
These settings apply only to the child process; the wrapper does not write
`config.yaml`. It sets both endpoint variables because the custom provider
uses `CUSTOM_BASE_URL`.

For an authenticated gateway, pass both `--model <active-id>` and a matching
key. The Hermes wrapper's automatic discovery currently sends no authorization
header; an explicit model skips that discovery request. `--model` does not
switch the resident model.

Even a short message includes the harness's system prompt and tool definitions.
Use the gateway's request-byte log and the engine log to distinguish
prefill from a failed request. A retry banner or `APIConnectionError` is not proof
of slow prefill: check the gateway traceback and the endpoint Hermes resolved.
Do not assume the next turn will be faster. The pinned WARP server resets its
inference state for each HTTP request; llama.cpp has different cache behavior.
Native context capacity is not a latency guarantee.

### Persistent but separate: a Hermes profile

```bash
hermes profile create litmoe --clone      # copies your config/skills, separate identity
hermes -p litmoe model                    # pick "Custom endpoint", URL http://127.0.0.1:8090/v1, model id
hermes -p litmoe                          # talks to the local model
hermes                                    # default profile, unchanged
```

### Or a model alias in your normal config

Add to `~/.hermes/config.yaml` (this does not change the default model):

```yaml
model_aliases:
  local:
    model: gemma-4-26b-a4b
    provider: custom
    base_url: "http://127.0.0.1:8090/v1"
    api_key: litmoe          # dummy only when the gateway has api_key: null
```

Select the alias with `/model local`. Use `/model` to select your normal
provider again. Give the alias its own credential: use a dummy value for an
unauthenticated gateway or the gateway's configured key when authentication
is enabled. Do not use a cloud-provider credential for the local endpoint.

### What NOT to do

`hermes config set model.provider custom` and `hermes config set model.base_url
...` change the default profile for later sessions. Use the session wrapper,
a separate profile, or a model alias when you want to keep that default.

## OMP (oh-my-pi)

```bash
litmoe serve glm-5.3-flash-warp
./scripts/omp-local
./scripts/omp-local -p "explain this repo"
omp                                  # normal OMP state, unchanged
```

Wrapper options `--gateway`, `--model`, and `--key` must precede OMP arguments.
They also accept `LITMOE_GATEWAY`, `LITMOE_MODEL`, and `LITMOE_API_KEY`.
The launcher checks authenticated runtime/discovery before starting OMP; an
inactive model, missing context, or unready engine fails without launching.

Each endpoint/model/context combination gets a marked directory under
`~/.litmoe/omp/`. Its `models.yml` defines the local OpenAI-completions provider;
`config.yml` pins all model roles to that provider and disables model fallback,
advisor calls, and context promotion. `PI_CODING_AGENT_DIR` and the explicit
config overlay isolate this invocation from ordinary OMP profiles. The API key
is an environment reference, not a credential written into those files.
Profile/provider/model/config override flags are rejected instead of silently
escaping isolation.

Title generation and automatic extension/skill/rule discovery are disabled
for this local invocation. Built-in coding tools and project context remain
available, and the launcher does not disable thinking. This avoids injecting
a machine-wide catalog into every local request; it does not remove OMP's
own system prompt or tool schemas. Set `LITMOE_OMP_DISCOVERY=1` to opt back
into automatic discovery. Explicit OMP extension paths (`-e`) remain usable.

The model advertises its actual context and text input only. Tool calling was
exercised with OMP's bash tool; vision and long-context model quality are not
established by that check. For OMP's independent prefill/generation/cache-pair
experiments, consult the installed version's `omp bench --help`; the launcher
itself is for agent sessions, not a benchmark subcommand wrapper.

**Latency limitation:** routing/tool-protocol checks are not an interactive
performance guarantee. The bundled exact-arithmetic native patch reduced a
216-token GLM-5.3-Flash microbenchmark from 97.00 to 91.44 seconds on a 96 GiB
M2 Max. The patched full-OMP request still sent 25,860 request-body bytes and
failed with an in-band stream error after 600.69 seconds, even with OMP's
`--max-time 3500`. The gateway's upstream read-inactivity timeout is 600 seconds;
OMP's CLI deadline does not override it. A native stack sample after the failure
was still inside `waste_model_prefill`. Thinking and built-in tools were retained.
This modest native gain therefore does not establish interactive full-harness
latency. Queue/cancellation fixes do not accelerate prefill, and closing the
stream does not interrupt it immediately. Do not interpret request bytes as
token counts or a large context capacity as fast prompt processing.

## Open WebUI / other OpenAI-SDK clients

Add `http://127.0.0.1:8090/v1` as an **additional** connection rather than
replacing the existing one; Open WebUI lists models from all connections.

For SDK code, pass `base_url=` to the client constructor instead of exporting
`OPENAI_BASE_URL` — an exported variable redirects every OpenAI client in the
shell, including tools you did not intend to touch.

## Streaming correctness

Claude Code uses the Anthropic Messages stream; Hermes's custom provider uses
OpenAI chat completions. Verify both paths, not only non-streaming responses.
The gateway closes the upstream response and client on stream completion,
read failure, or consumer closure.

Protocol checks with scripted responses or synthetic weights establish
transport behavior only, not real-model speed or quality.

The interactive cutover was exercised on Linux with Claude Code 2.1.283,
Hermes 0.21.5, and OMP 18.4.3 against a real HTTP gateway with scripted upstream
responses. All three completed print-mode calls; OMP executed a bash tool and
sent its result back. Separate real pinned-WARP synthetic-weight runs completed
OpenAI/Anthropic streams, cancellation/reload, explicit switching, paired
benchmark requests, and native-process cleanup after CLI SIGTERM.
These checks do **not** measure real-model speed or quality on Apple Silicon.

## Measure on the gateway host

Run with other clients idle; direct requests intentionally bypass gateway
admission. Start with a short transport check, then repeat with a representative
non-private harness-sized prompt:

```bash
litmoe bench --prompt "hi" --runs 3 --max-tokens 128 --json
litmoe bench --prompt-file /path/to/representative-prompt.txt --runs 3 --json
```

The report separates headers, first generated delta (including reasoning/tool
output), first visible text, and completion. Role-only events do not count as
tokens. Missing usage stays null; errors and incomplete streams fail the run.
Runtime identity/generation detect restarts, even when the model ID is unchanged.
Prompt hashes/sizes are recorded, not private prompt or response text.

Alternating direct/gateway order is not a matched cold/warm comparison, and
tokens divided by total response time is not decode-only speed. Record exact
model artifact/quantization and backend build separately; the report does not
fingerprint weights or binaries. Also record the Mac chip, macOS, memory
pressure, swap, disk I/O, and native prefill/decode/cache statistics before
interpreting results. DwarfStar is now an implemented integration; real-model
latency and actual client-loop success still require separate evidence.
Do not substitute its published DeepSeek throughput for GLM-5.3-Flash results.

## Troubleshooting

```bash
claude auth status --text        # should show your real login, no "Anthropic base URL" line
env | grep -E '^(ANTHROPIC|OPENAI)_'   # should be empty (or only what you knowingly set)
grep -n BASE_URL ~/.claude/settings.json ~/.hermes/.env 2>/dev/null   # should find nothing litmoe-related
litmoe status                    # gateway + engines litmoe knows about
```
