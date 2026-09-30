# Measurement logs

The raw `llama-server` logs below support the local CPU throughput figures in
[SETUP.md](../SETUP.md#speed-what-to-expect) and
[METHODOLOGY.md](../METHODOLOGY.md#what-was-measured). Upstream/vendor figures
in those documents are separately attributed and are not measured by these
logs. Extract the local timing lines with:

```bash
grep -oE "eval time =.*tokens per second" docs/measurements/*.log
grep -oE "n_threads = [0-9]+" docs/measurements/*.log
```

All runs: one machine — AMD EPYC 7B13, 24 physical cores / 48 threads, AVX2
only (no AVX-512, no AMX), DDR4-3200, Google Cloud persistent disk
(~400 MB/s), no GPU, llama.cpp CPU build. Home directory paths are replaced
with `~`.

| Log | Model / quant | Date | Threads | Generation t/s (all requests, in order) |
|---|---|---|---|---|
| `gemma-4-26b-a4b.log` | Gemma-4-26B-A4B-it UD-Q4_K_XL (17 GB), `--mmproj` loaded | 2026-09-16 | 24 | 4.3, 1.6, 2.2, 2.0, 5.6, 9.0, 10.6, 11.6, 4.6, 10.2, 10.5, 12.7 |
| `qwen3.8-9b-distill.log` | Qwen3.8-9B-Distill Q4_K_M (6 GB) | 2026-09-01 | 8 | 8.3, 8.5 |
| `kimi-linear-48b.log` | Kimi-Linear-48B-A3B Q4_K_M (30 GB) | 2026-08-20 | 48 | 0.44, 0.42, 0.58, 0.05, 0.45, 0.03, … |
| `deepseek-v4-flash.log` | DeepSeek-V4-Flash-0731 UD-IQ1_S (83 GB) | 2026-08-20 | 48 | 0.33, 0.34, 0.32, 0.11 |

How to read the gemma sequence: the engine was restarted seven times during
the end-to-end test session (seven `===== litmoe session` headers), so the
dips (1.6, 2.0, 4.6) are the first requests after a restart while the 17 GB
of weights were still being paged in from disk; the plateau once resident is
9–12.7 t/s. Requests were short (5–974 prompt tokens), so prompt-eval numbers
(4–34 t/s) are dominated by per-request overhead and are not a useful
prompt-processing benchmark.

The Kimi-Linear and V4-Flash runs used 48 threads on 24 physical cores (SMT
oversubscription, since fixed — litmoe now defaults to physical cores) and,
more importantly, never had their expert weights resident: 30 GB on a machine
that was also holding other models, and 83 GB with a cold page cache. Their
numbers measure the disk, not the models.

Not included: the Aug-2026 Kimi-K3 (0.85 t/s) and Qwen3.8-9B (0.69 t/s at 48
threads) runs cited in earlier revisions of SETUP.md. Those log files were
truncated by a later restart before the append-only logging fix and cannot be
reproduced from the repository, so the figures were removed from the docs.

## Apple Silicon agent-serving investigation

The [2026-09-30 redesign report](../plans/2026-09-30-inference-redesign.md)
separately records the 96 GiB M2 Max investigation: the previous 216-token
WARP optimization, the full-OMP timeout, and actual Claude request/tokenizer
measurements of 84,442 eager-tool versus 15,482 discovery-mode tokens.
Request capture/tokenization did not run inference or prove a tool round trip.
Private request payloads are not repository benchmark fixtures.

The patched DwarfStar CLI/server/benchmark build succeeded on that Mac.
Its native server protocol suite passed, including deferred-reference
rendering, unresolved-reference rejection, and quiescence state boundaries.
The integrated Python suite passed **244 tests** (one third-party
Starlette/httpx deprecation warning); both launcher shell syntax checks passed.
These are implementation checks, not model-performance measurements.

The real `litmoe install --engine dwarfstar --yes` command built and published
the patched runtime successfully. `litmoe doctor` recognized that installation,
and `litmoe models` listed the pinned streaming GGUF recipe separately from
resident-RAM model tiers.

Client transport probes also exercised the actual launchers without model
inference. A scripted Claude `ToolSearch` response produced a continuation
with a `tool_reference` using `tool_name`; the corresponding full schema was
present in the request catalog with `defer_loading: true`. Thinking remained
adaptive/high. The probe then deliberately rejected continuation with HTTP
400; no discovered MCP tool was executed.

OMP's coding request retained `reasoning_effort: high`, 11 advertised tools,
and `max_tokens: 4096`: 25,193 bytes with the launcher's default discovery
settings, 218,858 bytes with `LITMOE_OMP_DISCOVERY=1`. Across those two runs,
140 additional skill-description compression requests were captured separately
from the coding requests. Those auxiliary calls also targeted the isolated
local model. All requests were deliberately rejected, not benchmarked.
Client startup work, foreground prefill, and decode must be measured separately.

There are not yet DwarfStar GLM model-inference timings in this measurement
set. The compatible Q4 download and subsequent native/Claude/OMP runs remain
separate verification steps; neither build success nor another model's vendor
throughput establishes that evidence.
