# Using litmoe with agent harnesses — without breaking their defaults

The rule litmoe follows: **pointing a harness at a local model must never
change what that harness does when you run it normally.** Claude Code should
keep using your Anthropic account, Hermes should keep using its configured
provider, and plain OMP should keep its normal profiles. Switching back requires
no cleanup.

This document lists, per harness, what it reads to decide where requests go,
what litmoe touches (nothing global), and how to verify.

## What litmoe itself never does

- Never rewrites `~/.claude/`, `~/.hermes/config.yaml`, `~/.hermes/.env`,
  normal OMP profiles, or shell rc files. The OMP launcher owns only its
  marked `~/.litmoe/omp/` directories.
- Never exports environment variables into your shell. `litmoe serve` sets
  variables only for the engine subprocesses it spawns.
- Never binds a port a harness uses by default: the gateway is `127.0.0.1:8090`
  (or whatever `port:` you set); engines take 8081+, skipping the gateway port
  and any port another process already listens on. Anthropic's real API is
  HTTPS on api.anthropic.com — no overlap.
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

One inference lease lasts through the whole stream. The default queue permits
eight waiting admissions for up to 30 seconds; overflow or expiry returns 429.
Disconnecting a queued client removes its wait without dispatching inference.
For active inference, cancellation **terminates the owned native process**:
the current adapters lack a proven request-abort acknowledgement. The next
request reloads the same selected model. This costs startup time and loses
prefix state, but does not leave abandoned decoding running in the background.

WARP reports no reusable prompt cache: the pinned server resets state for each
HTTP request. Its expert-weight cache is not a conversation-prefix cache.
llama.cpp may retain a prefix in its single backend slot; this is backend
capability, not a claim that a particular request hit cache. Switches and
cancellation clear it. No hidden prompt pruning or gateway cloud fallback.

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

Verified on 2026-09-16 against a running gateway: inside the wrapper,
`claude auth status --text` reported `Auth token: ANTHROPIC_AUTH_TOKEN` and
`Anthropic base URL: http://127.0.0.1:<gateway port>`; a `-p` prompt ran
end-to-end (`modelUsage: gemma-4-26b-a4b`, result `PONG`); immediately
afterwards plain `claude auth status` still showed the Enterprise login and the
shell had no `ANTHROPIC_*` / `CLAUDE_*` variables. Claude Code prints a
one-line `[claude-code:unrecognized_model]` notice on stderr for non-Anthropic
model ids; it is harmless.

### What NOT to do

- Do not `export ANTHROPIC_BASE_URL=...` in your shell or `.bashrc`/`.zshrc`
  — every `claude` (and every other Anthropic SDK client) in that shell
  silently goes local, and `claude auth status` will say `Auth token:` instead
  of your login. (Earlier litmoe docs suggested exactly this. Removed.)
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
`CUSTOM_BASE_URL=http://127.0.0.1:8090/v1`,
`OPENAI_BASE_URL=http://127.0.0.1:8090/v1`, and `OPENAI_API_KEY=litmoe` set
only in that process. The wrapper never writes your `config.yaml`.
Current Hermes uses `CUSTOM_BASE_URL` for the custom provider; setting only
`OPENAI_BASE_URL` can leave requests pointed at a saved endpoint instead.

Even a short message includes the harness's system prompt and tool definitions.
Use the gateway's approximate prompt-token log and the engine log to distinguish
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

`hermes profile alias litmoe` installs a `litmoe` command that always runs
that profile.

### Or a model alias in your normal config

Add to `~/.hermes/config.yaml` (this does not change the default model):

```yaml
model_aliases:
  local:
    model: gemma-4-26b-a4b
    provider: custom
    base_url: "http://127.0.0.1:8090/v1"
    api_key: litmoe          # required: without it Hermes would send your DEFAULT provider's key to this host
```

Then `/model local` inside a session, `/model` again to go back. The
`api_key` line matters even though the gateway ignores it (`api_key: null`
in models.yaml): Hermes refuses to reuse the default provider's credential for
an alias endpoint, so an alias without its own key fails instead of leaking.

### What NOT to do

`hermes config set model.provider custom` / `model.base_url ...` (the previous
README's instructions) rewrite the default profile's config, so **every** later
Hermes session uses the local model until you run the reverse commands. Use one
of the three options above instead.

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

The model advertises its actual context and text input only. Tool calling was
exercised with OMP's bash tool; vision and long-context model quality are not
established by that check. For OMP's independent prefill/generation/cache-pair
experiments, consult the installed version's `omp bench --help`; the launcher
itself is for agent sessions, not a benchmark subcommand wrapper.

## Open WebUI / other OpenAI-SDK clients

Add `http://127.0.0.1:8090/v1` as an **additional** connection rather than
replacing the existing one; Open WebUI lists models from all connections.

For SDK code, pass `base_url=` to the client constructor instead of exporting
`OPENAI_BASE_URL` — an exported variable redirects every OpenAI client in the
shell, including tools you did not intend to touch.

## Streaming correctness

Claude Code uses the Anthropic Messages stream; Hermes's custom provider uses
OpenAI chat completions. Verify both paths, not only non-streaming responses.
The gateway consumes the already-open `httpx.Response` directly and closes the
response and client on completion, read failure, or consumer closure. An
`httpx.Response` is not an asynchronous context manager: using `async with` on
it aborts the Anthropic stream immediately after `message_start`.

Protocol smoke checks with small synthetic model weights establish transport
correctness only. They do not establish responsiveness with real model weights,
long harness prompts, or the user's hardware.

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
choosing WARP versus an equivalent llama.cpp/Metal or MLX candidate.

## Checklist before you say "it's broken"

```bash
claude auth status --text        # should show your real login, no "Anthropic base URL" line
env | grep -E '^(ANTHROPIC|OPENAI)_'   # should be empty (or only what you knowingly set)
grep -n BASE_URL ~/.claude/settings.json ~/.hermes/.env 2>/dev/null   # should find nothing litmoe-related
litmoe status                    # gateway + engines litmoe knows about
```
