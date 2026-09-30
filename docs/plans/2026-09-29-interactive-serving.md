# Interactive Serving Implementation Plan

> **Archived planning record.** This is retained for design history, not as current operating instructions. Shipped behavior is documented in [SETUP](../SETUP.md), [ARCHITECTURE](../ARCHITECTURE.md), and [HARNESSES](../HARNESSES.md).

**Goal:** Make litmoe a single-interactive-model service shared by Claude Code, Hermes, and OMP, with observable latency and explicit model switching.

**Architecture:** One gateway owns at most one live inference engine. Configured models remain selectable, but inactive models are not advertised as ready. Requests never silently substitute models or cause background model switches. Backend-specific prefix caching is reported honestly; unsupported WARP caching is not emulated by caching answers.

**Tech stack:** Python, Click, FastAPI, httpx, existing engine subprocess adapters, shell harness launchers; OMP's existing benchmark command remains available for independent client-side checks.

## Approved policy and constraints

The user selected **one interactive model**, shared by all three clients. Switching models may unload and reload weights. Existing native-context support stays intact; no hidden prompt truncation, model substitution, cloud fallback, or blanket context-limit reduction. WARP remains supported, but no backend is assumed faster without measurements on the target Mac.

Current source evidence: WARP resets state per HTTP request; its context planner independently uses a 75% memory ceiling; the previous gateway eagerly started every configured model. Synthetic GLM smoke tests prove protocol correctness, not real-model performance on Apple Silicon.

## 1. Single-resident lifecycle and explicit switching

Files: `litmoe/server.py`, `litmoe/cli/main.py`, new `litmoe/runtime.py` if lifecycle logic needs extraction; engine adapters only where ownership requires it.

- `litmoe serve` starts the first configured model, not all models. One explicit model selects the initial active model; multiple initial selections fail clearly.
- Preserve the full configuration for later selection. Never delete inactive entries or rewrite the user's model choice implicitly.
- Add an authenticated switch operation and a `litmoe switch MODEL` command. Drain the in-flight request before unloading. Stop the old process before starting another. A failed start leaves an explicit unavailable state, never two live engines or a false healthy result.
- Expose active model, lifecycle state, configured choices, queue state, effective context, and backend cache/cancellation capabilities. `/v1/models` advertises only the ready active model and its aliases.
- Switching uses the same fit/context policy as initial loading, against a single owner budget. Do not claim that static capacity estimates measure current memory pressure.

Acceptance: two configured models cause only one process start; alias requests use the same engine; inactive-model requests fail with an actionable switch instruction; concurrent switch/request/shutdown paths never overlap live engines; failed switches and dead processes report unavailable.

## 2. Scheduling, cancellation, and prefix reuse

Files: the lifecycle module and gateway proxy; existing `litmoe/engines/base.py` and backend adapters where required.

- Serialize inference on the active engine, with a bounded queue and bounded queue wait. Reject overload explicitly rather than multiplying hidden engine requests.
- Keep the admission lease until the complete response stream ends, fails, or is cancelled.
- Cancel queued requests without dispatch. A disconnected active client must release engine work, including prefill. Where a backend has no verified cooperative cancellation contract, stop that owned engine and report that its cache was reset; reload only for a subsequent request or explicit switch.
- Reuse backend-managed prompt-prefix state only where supported. Keep llama.cpp's single-slot cache path; report WARP's reset-per-request limitation. Never cache completions as a substitute for model KV state.
- Remove unconditional claims that later requests reuse a prompt cache.

Acceptance: no engine work for a cancelled waiter; no orphan inference after active cancellation; no admission release at the first SSE event; normal completion retains the healthy engine; no cross-session prompt contamination.

## 3. OMP integration

Files: new `scripts/omp-local`; gateway model metadata; existing `docs/HARNESSES.md`.

- Match the existing launcher options: `--gateway`, `--model`, `--key`, plus OMP arguments.
- Use OMP's documented `openai-completions` custom provider in a litmoe-owned isolated configuration directory. Never rewrite normal OMP configuration, credentials, or model roles.
- Discover the active model and effective context from the gateway. Do not advertise model-native capacity as the active engine's allocation.
- Keep auxiliary model selection and fallback policy explicit; do not silently send local-session inference to a cloud provider.
- Use the existing OMP CLI, not the similarly named upstream Pi package.

Acceptance: actual OMP print-mode streaming reaches the gateway; a tool-call/response round trip works; normal OMP configuration is unchanged; errors name the selected endpoint/model.

## 4. Reproducible latency measurements

Files: new `litmoe/benchmark.py` and Click command module, registered from the CLI; existing measurement documentation.

- Support a bounded single-user streaming benchmark against the active gateway and its directly exposed engine using the same request body.
- Measure response headers, first generated delta (including reasoning), first visible text, completion, output-token usage when supplied, and derived decode throughput only when its token/time basis is valid.
- Emit machine-readable results with model/backend/context identity, endpoint route, run order, prompt hash/size, failures, and raw timing values. Do not save private prompt/response text by default.
- Repeat prompts to observe warm behavior, but do not call a repetition a cache hit without backend evidence. Avoid comparing a cold direct request with a warm gateway request as proof of gateway overhead.
- Reuse `omp bench` for independent chat, prefill, generation, and cache-pair experiments; do not duplicate its full load-testing machinery.

Acceptance: deterministic timing edge tests; HTTP and stream errors remain errors; metadata-only SSE events do not count as generated tokens; missing usage does not become fabricated token throughput; the command exercises real loopback HTTP.

## 5. Integration evidence and publication

All permanent tests stay in `tests/test_litmoe.py`. Add behavioral regressions for lifecycle boundaries, cancellation, and metric semantics rather than launcher-string or mock-forwarding assertions.

Verification:

- `python3.12 -m pytest -q`
- `python3.12 -m compileall -q litmoe tests`
- `bash -n scripts/claude-local scripts/hermes-local scripts/omp-local`
- Real HTTP SDK streaming through pinned WARP with synthetic weights, explicitly labelled as protocol-only evidence.
- Real launcher/client checks for Claude Code, Hermes, and OMP where installed, using isolated configurations and bounded executions. Report bootstrap/runtime blockers rather than replacing them with mocked success.
- Target-Mac benchmarks with real weights are required before choosing a faster backend or claiming a latency improvement.

Update existing architecture, setup, and harness documentation to match shipped behavior. Remove temporary smoke artifacts. Publish only exercised correctness claims; distinguish those from unmeasured performance outcomes.
