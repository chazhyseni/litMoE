# Agent inference: findings and replacement decision

## Decision

Implement DwarfStar's model-specific Metal/SSD-streaming backend as a first-class litmoe integration instead of continuing small WARP optimizations as the interactive GLM strategy. Pair it with correct deferred-tool handling, prefix-state reuse, and cancellation ownership. A different executable alone does not fix the client contract.

The initial investigation did not change production configuration or the
installed WARP runtime and did not download replacement weights. Implementation
has since installed the pinned, patched DwarfStar runtime; the compatible Q4
download is in progress. No prompt has been sent to remote inference.

The remaining decisive evidence is a full-model, real-client benchmark. No additional broad backend survey is needed before that experiment.

The installer, engine adapter, native Anthropic routing, tool references,
exact token-count route, and acknowledged cancellation are implemented.
`litmoe install --engine dwarfstar --yes` built the patched Metal runtime and
native protocol checks and published it at `~/.local/lib/dwarfstar`.
`litmoe doctor` recognizes it; `litmoe models` exposes the pinned streaming
recipe. Native server checks and the Python suite pass.

The direct thinking-enabled model smoke and actual Claude/OMP benchmarks
remain required. No GLM inference success or latency is established yet.
Model validation uses pinned current upstream `main`, not the older GLM
feature branch used for the initial source/build probe.
See [upstream attribution and license notices](../../THIRD_PARTY_NOTICES.md).

## Locally observed evidence

Hardware: Apple M2 Max, 38 GPU cores, 96 GiB unified memory. Approximately 1.9 TB disk space was available during investigation. Historical swap allocation is not proof of current thrashing.

| Observation | Result | Scope |
| --- | --- | --- |
| Fresh Claude launcher request, normal configuration | 376 tool schemas; 353 Claude Flow MCP tools; tool JSON 319,103 bytes | Actual client request construction; recorder deliberately returned HTTP 400 without inference |
| Same request through existing gateway translation and native WARP prompt/tokenizer | 84,442 model tokens; tokenization approximately 27 ms | Exact tokenizer, not bytes divided by four |
| Fresh launcher request with `ENABLE_TOOL_SEARCH=true` | 12 schemas, including ToolSearch and one deferred placeholder; 15,482 native tokens | 81.7% fewer tokens; thinking remained enabled; discovery execution not verified |
| Previous native optimization | 216-token prefill: 97.00 to 91.44 seconds; matching logits and continuation | Previously measured short workload, not an 84K-token latency measurement |
| Previous full OMP run | 25,860 request bytes; failed after 600.69 seconds while native prefill was active | Previously exercised failure; not repeated in this investigation |
| DwarfStar initial compatibility probe | Older GLM feature branch: Metal-linked CLI, server, benchmark built in 12.24 seconds; all three `--help` commands succeeded | Host build/launch only; no GLM inference or GPU-kernel execution demonstrated |
| DwarfStar current-upstream compatibility probe | Current pinned `main`: same three targets built in 18.78 seconds; all three `--help` commands succeeded | Also host build/launch only; selected for the actual model smoke |

The fresh Claude capture is not the exact historical interactive request that logged 378,497 payload bytes. It used the same launcher/project in print mode; its translated body was 364,316 bytes. Keep those workloads distinct.

Diagnostic removal of schemas produced smaller prompts, but removing tools is not the proposed solution. Deferred discovery must retain access to the complete catalog and pass a real search-call-result round trip.

## Root causes and required changes

### 1. Tool loading expands a greeting into a large inference workload

[Claude Code's official MCP documentation](https://code.claude.com/docs/en/mcp#scale-with-mcp-tool-search) says tool search defaults off for non-first-party `ANTHROPIC_BASE_URL` endpoints because proxies often fail to preserve `tool_reference` blocks. This matches the captured request.

The legacy translator in `litmoe/server.py` drops deferred-tool metadata and non-text tool-result blocks. Blindly setting `ENABLE_TOOL_SEARCH=true` can reduce the first prompt while breaking the subsequent tool loop. WARP's GLM renderer also rejects deferred schemas. The new DwarfStar path bypasses that translator and resolves typed references in its native renderer; the launcher gates tool search on the ready backend capability.

Required contract:

- Resolve referenced tool definitions and render only the active definitions, while preserving discovery of the entire catalog.
- Preserve tool-call IDs, names, arguments, results, errors, and typed discovery history.
- Do not render the deferred placeholder as a callable application tool.
- Enable explicit tool search in the isolated launcher only after the complete protocol path passes. Do not rely on the default `auto` threshold: a very large advertised context can make eager loading appear acceptable.
- Treat OMP separately: its OpenAI-compatible provider is not Claude's ToolSearch protocol.

### 2. WARP's GLM prefill is serial by design

Installed WARP `src/model.c:7398-7412` explicitly falls back to one-token-at-a-time evaluation when `hc_mult` or `index_topk` is present. Its comment explains that the chunked path does not implement mHC's parallel residual streams or the DSA indexer's per-token bookkeeping. GLM uses both.

The native audit found only limited Metal matvec/expert-apply kernels, not a whole-model GPU prefill implementation. Raising a chunk-size environment variable does not supply the missing model graph. The existing patched runtime is based on [WARP v0.8.1](https://github.com/sqliteai/warp/releases/tag/v0.8.1).

A serious WARP continuation would require model-correct layer-major batching, GPU kernels for the remaining graph, expert staging, numerical parity across chunk boundaries, and serving-state integration. No speedup for that unimplemented rewrite has been measured. Do not extrapolate the short 2.36-token/s observation into a claimed measured duration for a long request.

### 3. Conversation state is discarded on each native HTTP request

WARP `serve/server.py:367-386` holds the engine lock and calls `engine.state_reset()` before building each prompt. This correctly prevents accidental cross-request contamination, but does not implement safe prefix reuse. Native state save/load primitives exist; the HTTP path does not use them.

Required design:

- Backend owns model state, exact rendered-prefix matching, and checkpoint restore.
- Persist all GLM state required for continuation, including recurrent KDA state and DSA bookkeeping, not just conventional attention KV.
- Restore a valid earlier checkpoint when tool serialization, reasoning-history policy, or another prompt segment changes; recompute the suffix.
- Namespace by model artifact, quantization, tokenizer/template version, relevant rendering options, and session/privacy boundary. Include image identity if images are supported.
- Never implement prefix reuse by caching answers or simply removing `state_reset()`.

[LM Studio's hybrid-state checkpoint design](https://lmstudio.ai/blog/mlx-engine-agentic-workloads) is a useful reference, not a GLM performance measurement. Its published example uses a different model and machine.

### 4. Client disconnect is not native cancellation

The current WARP stream path can release the gateway lease after closing the upstream connection while native prefill continues. Native callback-based cancellation is reached during generation, not reliably during prefill. The gateway also has a hard-coded 600-second upstream read timeout.

Required invariant: **one engine's admission lease remains owned until its work is quiescent**, acknowledged cancelled, or its owned process has exited. Cancelling a queued request must not dispatch it. Avoid a hidden second queue behind a released gateway lease.

Chunk-boundary cancellation is a useful primitive but does not establish a latency guarantee: a slow chunk can still delay cancellation. Measure the bound with the actual model. SSE heartbeats can report liveness; they neither accelerate prefill nor count as generated tokens. Increasing timeouts is not a performance fix.

### 5. Protocol and measurement must stop overstating support

The adapter audit found lossy handling of reasoning controls, named tool choice, image content, tool references, and usage estimates. Preserve supported semantics and explicitly reject unsupported ones rather than silently converting them into something else.

Honor the user's thinking/effort setting. Do not disable thinking to obtain attractive latency numbers. GLM documentation differs on reasoning-history clearing defaults across serving stacks; pin and test the selected backend policy rather than assuming one universal setting.

Measure queue time, render/tokenization, cache-restored tokens, newly prefilled tokens, first generated reasoning/text/tool delta, first visible answer, generation throughput, and cancellation-to-idle. HTTP 200 headers and SSE metadata are not successful inference. Report model-native capacity separately from allocated context and measured usable latency.

## Candidate comparison

| Candidate | Evidence | Decision |
| --- | --- | --- |
| Continue current WARP path | Correct model support and compact expert storage, but serial GLM prefill and reset-per-request HTTP serving | Do not continue micro-optimization as the primary interactive-serving strategy |
| DwarfStar GLM-specific Metal + SSD streaming | Actual GLM graph, 2,048-token GPU prefill chunks, streamed experts, state machinery, native Anthropic endpoint; patched runtime installed and protocol checks pass | Implemented replacement path; actual GLM latency and real-client tool loops remain to be measured |
| MLX/oMLX on larger Mac | Published 16,384-token GLM 4-bit result: 321.7 prefill tokens/s, 50.935-second TTFT, 5.2 decode tokens/s on M3 Ultra 512 GB | Larger memory alone does not establish interactive latency; benchmark before hardware purchase |
| llama.cpp GLM support | Recent upstream implementation and Metal changes identified during survey | Alternative to evaluate if needed, not an already-proven same-machine replacement |
| KTransformers on Linux/NVIDIA | Published 4x RTX 5090 result: 1,152.9 prefill tokens/s at 16,350 actual input tokens; approximately 14–17 decode tokens/s; FP8 recipe calls for at least 350 GB system RAM | Credible different-hardware route; not runnable on this Mac; no 84K-input result established |
| Z.ai hosted GLM-5.3-Flash | Official Claude-compatible endpoint and model support | Operational alternative requiring explicit privacy/cost approval and actual client verification; no workload-specific latency promise |
| Smaller model, pruning, disabled thinking/tools | Can change resource requirements by changing the task/model | Not an authorized substitute for the requested behavior |

Sources: [oMLX benchmark](https://omlx.ai/benchmarks/performance/1t786fn7), [PipeNetwork GLM MLX implementation and quantization measurements](https://github.com/PipeNetwork/glm53-flash-mlx), [KTransformers measurements](https://github.com/kvcache-ai/ktransformers/issues/2173), [official model card](https://huggingface.co/zai-org/GLM-5.3-Flash).

Published results above are third-party measurements, not local results. Quantization perplexity or tiny-model logit parity does not establish coding-agent/tool quality. A model artifact larger than RAM is not inherently impossible to run when the backend explicitly supports streaming.

## Bounded DwarfStar experiment

Reproducibility:

- Current source for model validation: [antirez/ds4](https://github.com/antirez/ds4), commit `0aaea5a238fb41a35106a551e73c8409dfb751ac` from `main`.
- Initial source/build probe: GLM feature-branch commit `b1b4ea03645434423e5cb4f39818fdc075e49825`. Source line numbers cited from that probe refer to this older revision, not automatically to current upstream.
- Weight repository: [antirez/glm-5.3-flash-gguf](https://huggingface.co/antirez/glm-5.3-flash-gguf), revision `b2fa29d7a6b410db11221c904973967b80b760f5`.
- Q4_K: 190,875,526,464 bytes, approximately 177.8 GiB. Q2: 96,505,816,384 bytes, approximately 89.9 GiB. Neither file size alone establishes runtime fit.
- The paired FP8 artifact is documented as not executable by this backend; do not download it as an inference candidate.
- Unmodified baseline build: `/tmp/litmoe-research-dwarfstar-current`; initial branch probe: `/tmp/litmoe-research-dwarfstar`. The managed patched runtime is now installed at `~/.local/lib/dwarfstar`; the existing WARP installation and weights are retained.

Prefer Q4 for the first quality-preserving candidate evaluation rather than silently choosing more aggressive Q2 compression. This is still a different quantization from the current WARP artifact, not a claim of identical model numerics.

The README's `--ctx 4096` streaming command is an example, not a total-context ceiling. The 4,096-token threshold elsewhere concerns native multi-session decode batching. The source processes longer GLM prompts in successive prefill chunks.

Important risks to measure, not explain away:

1. SSD traffic. Streaming prefill stages routed layers; repeated chunks can make storage bandwidth dominant even with fast GPU kernels. Measure bytes read and expert-cache behavior alongside TTFT.
2. Memory. Non-routed weights, recurrent/attention state, activations, and two staging layers need space in addition to the expert cache. Metal's recommended working-set size is a planner input, not a hard physical limit or performance guarantee.
3. Tool search. At the pinned source, `ds4_server.c:2527-2539` filters deferred schemas; no `tool_reference` handling was found in the inspected server/core files. Native Anthropic serving is not proof of discovery compatibility.
4. State correctness. Require same-quant checkpoint rejection (`--kv-cache-reject-different-quant`) plus artifact-specific cache storage; the documented default otherwise permits cross-quant reuse. Verify GLM restore behavior rather than generalizing DeepSeek cache documentation.
5. Cancellation. `ds4.c:64705-64825` checks cancellation between GLM prefill chunks. Verify HTTP disconnect reaches that mechanism and the scheduler does not admit conflicting work early.
6. Sampling. Leave speculative MTP off for the initial comparison. Do not silently enable approximate sampling; a later speculative experiment must explicitly preserve the required target distribution.

## Implementation sequence and acceptance gates

### A. Validate the execution engine alongside integration

Use one model process, the pinned backend/artifact, and memory-aware streamed-expert settings. Preserve the complete prompt, tools, thinking policy, and output allowance. Compare the original eager-tool workload and correctly rendered discovery workload; record exact tokens for each backend template.

Run cold prefill, repeated prefix, appended user turn, and a tool-result continuation. Record memory pressure and SSD I/O. Do not infer a cache hit merely because a repeat is faster. No production backend switch until the real workload passes.

Proposed targets, not achieved results or user-approved promises: first generated delta within 30 seconds cold and 5 seconds for a warm small continuation; cancellation-to-idle within 2 seconds. Also report first visible text and completion because hidden reasoning can make TTFT alone misleading. A 15,482-token prompt requires roughly 516 prefill tokens/s merely to fit the 30-second prefill portion; this leaves no budget for queueing or other overhead. An 84,442-token prompt requires roughly 2,815 tokens/s on the same basis.

If the candidate misses the target, use the measured bottleneck to decide between a specific backend improvement, larger hardware, and hosted inference. Do not return to blind timeout increases or unrelated kernel tweaks.

### B. Implement the DwarfStar serving contract

- `litmoe/engines/`: pinned startup, effective context, health, state/cancellation capabilities, subprocess ownership. Do not create another prompt renderer if the selected engine can correctly own the native protocol.
- `litmoe/server.py`: backend-aware protocol dispatch; lossless supported semantics; deliberate deferred-reference materialization when required; exact token counting where available; honest usage/timing.
- `litmoe/runtime.py`: single authoritative admission queue, cancellation acknowledgement/quiescence, and explicit cache-reset behavior on unavoidable restart.
- `scripts/claude-local`: capability-gated discovery override in the existing isolated environment. Keep normal Claude configuration untouched.
- `scripts/omp-local`: retain the real provider/tool contract and test its actual requests independently.
- Existing architecture, harness, and setup documentation: describe the measured selected path and remaining limitations, not an aspirational capability list.

### C. Require end-to-end proof

1. Actual Claude and OMP greetings complete with their real harness prompts and thinking enabled.
2. Claude discovers a previously deferred MCP tool, calls it, receives its result, and continues successfully. Verify the complete catalog remains reachable.
3. Multi-turn built-in and MCP tool loops preserve IDs, arguments, result errors, and reasoning policy.
4. Warm continuation demonstrably restores prefix state and recomputes only the required suffix.
5. Session A → B → A, changed tool schemas, and changed reasoning-history rendering produce no cross-session contamination or stale state.
6. Cancel while queued, during prefill, and during generation; prove the worker becomes idle and the next request runs without zombie work.
7. Exercise context/chunk boundaries and cached-versus-uncached continuations. Use model-appropriate numerical tolerances where GPU reduction order differs; never claim bit identity without measuring it.
8. Exercise model switch, backend failure, and gateway shutdown without overlapping large model processes or falsely reporting readiness.
9. Report cold/warm timing distributions from repeated real runs, not one favorable result; retain failures in the comparison.

## Hosted option, only with approval

Z.ai documents `https://api.z.ai/api/anthropic` for Claude Code in its [integration guide](https://docs.z.ai/devpack/tool/claude). Keep all model slots explicitly on the requested Flash model; the guide also contains examples selecting other GLM variants, which must not be copied blindly.

[Published pricing](https://docs.z.ai/guides/overview/pricing) is $0.15/M input tokens, $0.03/M cached input tokens, and $0.50/M output tokens for GLM-5.3-Flash. Illustratively, 84,442 input plus 20,000 output tokens costs approximately $0.022666 cold, or $0.012533 if all input qualifies for cached pricing. These are arithmetic examples, not observed bills, output-length predictions, or guaranteed cache hits. Coding-plan subscriptions have separate terms.

The service advertises the same model identity, not numerical identity with the local WARP quantization. Verify data-handling terms, account/endpoint eligibility, tool discovery, reasoning semantics, and actual latency before adoption. No automatic cloud fallback.
