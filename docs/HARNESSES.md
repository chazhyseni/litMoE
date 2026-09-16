# Using litmoe with agent harnesses — without breaking their defaults

The rule litmoe follows: **pointing a harness at a local model must never
change what that harness does when you run it normally.** Claude Code should
keep using your Anthropic account, Hermes should keep using its configured
provider, and switching back must require zero cleanup.

This document lists, per harness, what it reads to decide where requests go,
what litmoe touches (nothing global), and how to verify.

## What litmoe itself never does

- Never writes to `~/.claude/`, `~/.hermes/config.yaml`, `~/.hermes/.env`,
  shell rc files, or any harness config.
- Never exports environment variables into your shell. `litmoe serve` sets
  variables only for the engine subprocesses it spawns.
- Never binds a port a harness uses by default: the gateway is `127.0.0.1:8080`
  (or whatever `port:` you set); engines take 8081+, skipping the gateway port
  and any port another process already listens on. Anthropic's real API is
  HTTPS on api.anthropic.com — no overlap.
- `litmoe stop` only signals engines litmoe started (tracked in
  `~/.litmoe/run/*.pid`). An Ollama/LM Studio/manual `llama-server` is left
  alone unless you pass `--all`.

## Claude Code

### How Claude Code decides where to send requests

| Setting | Effect | Where it can come from |
|---|---|---|
| `ANTHROPIC_BASE_URL` | Endpoint for every API call | env var, `settings.json` `env` block |
| `ANTHROPIC_AUTH_TOKEN` / `ANTHROPIC_API_KEY` | Credential; when either is set your claude.ai / Enterprise login is **not** used for that process | env, `apiKeyHelper` in settings |
| `ANTHROPIC_MODEL`, `ANTHROPIC_DEFAULT_{SONNET,OPUS,HAIKU}_MODEL`, `CLAUDE_CODE_SUBAGENT_MODEL` | Which model id the `sonnet`/`opus`/`haiku` aliases and subagents resolve to | env, settings `model` |
| `CLAUDE_CONFIG_DIR` | Where sessions, settings, and cached credentials live (default `~/.claude`) | env |

Environment variables override `settings.json`; both are read per process.
That is what makes clean isolation possible: set them **only** for the one
`claude` process that should talk to litmoe.

### The safe way: `scripts/claude-local`

```bash
litmoe serve                                   # gateway on :8080
./scripts/claude-local                         # Claude Code -> local model, isolated
./scripts/claude-local --model qwen3.6-35b-a3b -p "explain this repo"
claude                                         # normal Claude Code, untouched
```

`claude-local` (copy it onto your PATH if you like):

1. checks the gateway is up and picks the first model it serves (or `--model`);
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
first model so those requests resolve instead of 404ing. This only affects
requests that already reached litmoe; it has no effect on real Anthropic
traffic. `claude-local` also sets the `ANTHROPIC_DEFAULT_*_MODEL` variables so
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
`OPENAI_BASE_URL=http://127.0.0.1:8080/v1` and `OPENAI_API_KEY=litmoe` set
only in that process. Your `config.yaml` is never written.

### Persistent but separate: a Hermes profile

```bash
hermes profile create litmoe --clone      # copies your config/skills, separate identity
hermes -p litmoe model                    # pick "Custom endpoint", URL http://127.0.0.1:8080/v1, model id
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
    base_url: "http://127.0.0.1:8080/v1"
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

## Open WebUI / other OpenAI-SDK clients

Add `http://127.0.0.1:8080/v1` as an **additional** connection rather than
replacing the existing one; Open WebUI lists models from all connections.

For SDK code, pass `base_url=` to the client constructor instead of exporting
`OPENAI_BASE_URL` — an exported variable redirects every OpenAI client in the
shell, including tools you did not intend to touch.

## Checklist before you say "it's broken"

```bash
claude auth status --text        # should show your real login, no "Anthropic base URL" line
env | grep -E '^(ANTHROPIC|OPENAI)_'   # should be empty (or only what you knowingly set)
grep -n BASE_URL ~/.claude/settings.json ~/.hermes/.env 2>/dev/null   # should find nothing litmoe-related
litmoe status                    # gateway + engines litmoe knows about
```
