# LinkedIn Post: litMoE

## Primary post

Local LLM tooling has a plumbing problem, not an inference problem.

Open Mixture-of-Experts models now span 17 GB to 600 GB+, and two engines already cover that range: llama.cpp for GGUF everywhere, ktransformers (Tsinghua MADSys, SOSP 2025) for GPU attention with CPU experts on the big ones. Neither needs help with the forward pass. What's missing is one endpoint, one config, and a way to point your agent harness at it without breaking the harness.

litmoe is that layer — ~3,900 lines of Python putting both engines behind a single OpenAI *and* Anthropic-compatible API on one port.

What it adds:

→ **Anthropic Messages API.** `/v1/messages` and `count_tokens`, with a real SSE state machine translating text, thinking, and tool_use blocks event by event. Claude Code and Hermes work unchanged against a local model.

→ **A 29-model RAM-tiered catalog** (48 / 96 / 192 / 512 / 768 GB), verified against HuggingFace. `litmoe models` shows what fits *your* machine; `litmoe install --model X` handles sharded GGUFs, per-quant repo layouts, and vision projectors.

→ **Hardware-aware defaults.** Physical cores, AVX-512/AMX, NVIDIA GPUs, Apple unified-memory budget. Context is set to the model's native window and reduced only when the KV cache wouldn't fit in RAM.

→ **Supervision that minds its own business.** Engines get their own process groups and PID files; `litmoe stop` never touches your Ollama or LM Studio.

The measured result on a 24-core AVX2 CPU with **no GPU**: a 26B MoE with 4B active runs at **9–12.7 t/s** — the same band as a 9B dense model on the same box, while being far stronger and multimodal. CPU throughput tracks the parameters touched per token, not the parameters stored, which is exactly why the laptop tier is small-active MoEs. Every t/s figure in the repo ships with the raw llama-server log behind it.

The part I'd rather not have learned the expensive way: v1 of this project tried to be a third inference engine, a custom C99 CPU forward pass. It ran at **0.019 t/s**. 98,496 expert lookups × 17.55 MB means ~859 GB read per response — 38 minutes of disk floor, 82 minutes of compute floor, ~45× slower than llama.cpp on identical weights. No optimization closes that. I deleted the forward pass and shipped the integration layer, which turned out to be the actual product.

And the design rule I care most about: **pointing a harness at a local model must never change what it does when you run it normally.** litmoe never writes to `~/.claude`, `~/.hermes/config.yaml`, or your shell rc. `scripts/claude-local` sets everything per-process and execs — plain `claude` keeps using your Anthropic account, with nothing to undo.

github.com/chazhyseni/litMoE — Apache 2.0

#AI #LLM #MixtureOfExperts #LocalAI #llamacpp #OpenSource #DeveloperTools #ClaudeCode #Inference

---

## Short variant (feed-optimized, ~1,100 characters)

litmoe: one local endpoint for llama.cpp **and** ktransformers, speaking OpenAI *and* Anthropic.

A 29-model RAM-tiered catalog (48 GB laptop → 768 GB server), hardware detection that picks what actually fits, memory-aware context sizing, and per-process harness wrappers so Claude Code runs against a local model without touching your Anthropic login.

Measured on a 24-core AVX2 CPU, no GPU: a 26B MoE with 4B active does **9–12.7 t/s** — the same band as a 9B dense model, far stronger, and multimodal. CPU speed tracks active parameters, not total. Every number ships with its raw log.

v1 tried to be a third inference engine. It ran at 0.019 t/s: ~859 GB of expert weights read per response, ~45× slower than llama.cpp on the same weights. I deleted the forward pass. The integration layer was the product all along.

github.com/chazhyseni/litMoE

#AI #LLM #LocalAI #MixtureOfExperts #OpenSource

---

## Follow-up comment to post under the main post (for the technical crowd)

Implementation notes for anyone who has fought this:

• Anthropic streaming is translated event by event, not buffered — `message_start` → `content_block_*` → `message_delta` → `message_stop`, with OpenAI `tool_calls[].index` mapped onto Anthropic block indices and arguments streamed as `input_json_delta`. OpenAI streams pass through byte for byte.

• Engine ports count up from 8081 but skip the gateway port and anything already bound, verified with a real bind probe — so a stray llama-server on 8081 doesn't break startup.

• `litmoe stop` reads only `~/.litmoe/run/*.pid`. There's a test that spawns a bystander process and asserts it survives.

• RAM fit = weights × 1.08 + KV@32K + 4 GB headroom; macOS gets 75% of physical RAM, since unified memory is shared with the GPU.

• Two throughput figures were *removed* from the docs because a restart truncated their logs before append-only logging existed. Unreproducible numbers don't ship.
