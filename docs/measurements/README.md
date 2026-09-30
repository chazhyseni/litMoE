# Measurement logs

Raw `llama-server` logs behind the local llama.cpp generation numbers in
[METHODOLOGY.md](../METHODOLOGY.md#what-was-measured) and
[SETUP.md](../SETUP.md#speed-what-to-expect). Every value in those local tables
is a `print_timing` line you can grep here. The upstream WARP table in both
documents is quoted from upstream and is not backed by these logs.

```bash
grep -E '\|[[:space:]]+eval time =' docs/measurements/*.log
grep -oE "n_threads = [0-9]+" docs/measurements/*.log
```

These four logs are the only retained measurement provenance in the
repository. The machine metadata used elsewhere in the docs — AMD EPYC 7B13,
24 physical cores / 48 threads, AVX2 only (no AVX-512, no AMX), DDR4-3200, a
Google Cloud persistent disk, no GPU, llama.cpp CPU build — is recorded project
setup, not something these logs establish; the logs themselves only record
that no usable GPU was found. Only the Gemma log carries its own dated session
headers (2026-09-16); the other three logs carry no date. Home directory paths
are replaced with `~`.

| Log | Model / quant | Threads | Generation t/s (completed timing records, in log order) |
|---|---|---|---|
| `gemma-4-26b-a4b.log` | Gemma-4-26B-A4B-it UD-Q4_K_XL (17 GB), `--mmproj` loaded | 24 | 4.29, 1.56, 2.15, 2.02, 5.59, 9.03, 10.62, 11.62, 4.61, 10.21, 10.50, 12.68 |
| `qwen3.8-9b-distill.log` | Qwen3.8-9B-Distill Q4_K_M (6 GB) | 8 | 8.33, 8.48 |
| `kimi-linear-48b.log` | Kimi-Linear-48B-A3B Q4_K_M (30 GB) | 48 | 0.03, 0.03, 0.05, 0.42, 0.44, 0.58, 0.45 |
| `deepseek-v4-flash.log` | DeepSeek-V4-Flash-0731 UD-IQ1_S (83 GB) | 48 | 0.32, 0.34, 0.33, 0.11 |

Each value is the `eval time` line of one request that ran to completion, in
the order the lines appear. A request canceled mid-generation logs
`stop: cancel task` and no `eval time` line, so it does not appear: the Gemma
log has no cancellations, while the Qwen, Kimi-Linear, and DeepSeek logs have
1, 4, and 4 respectively.

Thread counts differ (8, 24, 48). The Kimi-Linear and DeepSeek runs used 48
threads on the 24-physical-core machine. All four servers were configured with
four slots, and some logs show more than one request in flight, so concurrency
is not the same in every run. Each figure is a per-request generation (decode)
rate and excludes prompt processing, so it is not full-response throughput.

What the logs do not contain: resident memory, page-cache state, disk
throughput, or other processes on the machine. The spread between requests
within one log therefore cannot be attributed to a particular cause. Prompt
sizes also varied — 5–974 prompt tokens in the Gemma log — so the
`prompt eval time` lines are not a controlled prefill benchmark.

## Gemma session headers

`gemma-4-26b-a4b.log` contains seven `===== litmoe session` headers. One of
them (2026-09-16 18:02:42) failed to bind port 8081 and served no requests, and
four others loaded the model but served no request. The twelve completed
records come from the first session (17:41:08) and the last (18:22:12); the
last session used `-c 32768` where the first used `-c 262144`. Within the first
session the low values 1.56 and 2.02 are the second and fourth requests, not
first requests after a restart.

The DeepSeek log is truncated: it ends on a `prompt processing` line for a
later request, with no completion and no shutdown line.
