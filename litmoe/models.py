"""Model catalog: the single source of truth for `litmoe install --model` / `litmoe models`.

Design rule (user requirement): the *default* models must run reasonably fast on
a laptop with 96 GB — ideally 48 GB — of RAM. Speed on CPU/Metal is governed by
*active* parameters per token, so the laptop tiers are small-active MoEs
(3–12B active) plus small dense models; big MoEs are an upgrade tier for
workstations and servers. Each entry carries ``tier`` = the RAM (GB) a machine
needs to run ``default_quant`` with a 32K context and OS headroom. Any model
can drop a tier by choosing a smaller quant (``litmoe models`` shows what fits
the detected RAM).

Every value here was checked against a primary source on 2026-09-16:

- ``hf_repo`` / ``quants`` / sizes: HuggingFace API file listing (bytes / 1e9,
  rounded). Sharded quants are summed. mmproj / MTP / imatrix files excluded.
- ``arch`` / ``native_ctx``: GGUF metadata as reported by the HuggingFace API
  (``gguf.architecture``, ``gguf.context_length``) — what llama-server reads as
  ``n_ctx_train``. All GGUF archs are present in ``src/llama-arch.cpp`` on
  llama.cpp master. For safetensors models: HF ``architectures[0]`` and
  ``max_position_embeddings``.
- ``kv_bytes_per_token``: fp16 KV cache per token computed from config.json.
  GQA: 2 * layers * n_kv_heads * head_dim * 2 bytes. MLA (DeepSeek-2 style):
  layers * (kv_lora_rank + qk_rope_head_dim) * 2 bytes. Hybrid models count
  only their full-attention layers (linear-attention / Mamba / sliding-window
  layers have bounded state). Estimates for memory-fit decisions, not
  measurements.
- ``engine`` support: llama.cpp ``src/llama-arch.cpp``; ktransformers
  ``kt-kernel/python/cli/utils/model_registry.py`` (BUILTIN_MODELS) plus the
  dated tutorials in ``doc/en/kt-kernel``.

Sizes are decimal GB (1 GB = 1e9 bytes), matching HuggingFace and ``du --si``.
"""
from __future__ import annotations

import math

# Values of ModelSpec["format"]
GGUF = "gguf"
SAFETENSORS = "safetensors"

# RAM tiers (GB). A model's tier is the smallest that fits default_quant + KV@32K + headroom.
TIER_LAPTOP_48 = 48
TIER_LAPTOP_96 = 96
TIER_WORKSTATION_192 = 192
TIER_SERVER_512 = 512
TIER_SERVER_768 = 768
TIER_SERVER_1024 = 1024
TIERS = (TIER_LAPTOP_48, TIER_LAPTOP_96, TIER_WORKSTATION_192, TIER_SERVER_512,
         TIER_SERVER_768, TIER_SERVER_1024)
TIER_LABELS = {
    TIER_LAPTOP_48: "48 GB laptop",
    TIER_LAPTOP_96: "96 GB laptop / desktop",
    TIER_WORKSTATION_192: "192 GB workstation",
    TIER_SERVER_512: "512 GB server",
    TIER_SERVER_768: "768 GB server",
    TIER_SERVER_1024: "1 TB server",
}

# Headroom used when deciding whether a quant fits a machine.
_OS_HEADROOM_GB = 6.0
_FIT_CTX_TOKENS = 32768
_MODEL_OVERHEAD = 1.10  # compute buffers, embeddings, mmap slack


def _g(hf_repo: str, arch: str, params: str, active_b: float | None, quants: dict[str, int],
       default_quant: str, native_ctx: int, kv: int, tier: int, notes: str | None = None) -> dict:
    d = {
        "hf_repo": hf_repo, "engine": "llamacpp", "format": GGUF, "arch": arch,
        "params": params, "active_b": active_b, "quants": quants, "default_quant": default_quant,
        "native_ctx": native_ctx, "kv_bytes_per_token": kv, "tier": tier,
    }
    if notes:
        d["notes"] = notes
    return d


def _k(hf_repo: str, arch: str, params: str, active_b: float | None, size_gb: int, native_ctx: int,
       kv: int, kt_method: str, extra_args: list[str], tier: int, notes: str) -> dict:
    return {
        "hf_repo": hf_repo, "engine": "ktransformers", "format": SAFETENSORS, "arch": arch,
        "params": params, "active_b": active_b, "size_gb": size_gb, "native_ctx": native_ctx,
        "kv_bytes_per_token": kv, "kt_method": kt_method, "extra_args": extra_args,
        "tier": tier, "notes": notes,
    }


KNOWN_MODELS: dict[str, dict] = {
    # ==================================================================
    # 48 GB laptop tier — fast: 3–4B active MoE or ≤31B dense
    # ==================================================================
    "gemma-4-26b-a4b": _g(
        "unsloth/gemma-4-26B-A4B-it-GGUF", "gemma4", "26B total, 4B active MoE, multimodal", 4.0,
        {"UD-IQ2_XXS": 10, "UD-IQ2_M": 10, "UD-Q2_K_XL": 11, "UD-IQ3_S": 11, "UD-IQ3_XXS": 11,
         "UD-Q3_K_M": 13, "UD-Q3_K_XL": 13, "UD-IQ4_XS": 14, "UD-IQ4_NL": 14, "UD-Q4_K_S": 16,
         "MXFP4_MOE": 17, "UD-Q4_K_M": 17, "UD-Q4_K_XL": 17, "UD-Q5_K_S": 19, "UD-Q5_K_M": 21,
         "UD-Q5_K_XL": 21, "UD-Q6_K": 23, "UD-Q6_K_XL": 23, "Q8_0": 27, "UD-Q8_K_XL": 28, "BF16": 51},
        "UD-Q4_K_XL", 262144, 40_960, TIER_LAPTOP_48,  # 5 full-attention layers of 30 * 2*8*256*2; 25 SWA(1024) layers bounded
        "Recommended laptop default: MoE with 4B active params, vision input (mmproj fetched automatically with -hf)."),
    "qwen3.6-35b-a3b": _g(
        "unsloth/Qwen3.6-35B-A3B-GGUF", "qwen35moe", "35B total, 3B active MoE", 3.0,
        {"UD-IQ1_M": 10, "UD-IQ2_XXS": 11, "UD-IQ2_M": 12, "UD-Q2_K_XL": 12, "UD-IQ3_XXS": 13,
         "UD-IQ3_S": 14, "UD-Q3_K_S": 15, "UD-Q3_K_M": 17, "UD-Q3_K_XL": 17, "UD-IQ4_XS": 18,
         "UD-IQ4_NL": 18, "UD-IQ4_NL_XL": 20, "UD-Q4_K_S": 21, "MXFP4_MOE": 22, "UD-Q4_K_M": 22,
         "UD-Q4_K_XL": 22, "UD-Q5_K_S": 25, "UD-Q5_K_M": 26, "UD-Q5_K_XL": 27, "UD-Q6_K": 29,
         "UD-Q6_K_XL": 32, "Q8_0": 37, "UD-Q8_K_XL": 38, "BF16": 69},
        "UD-Q4_K_XL", 262144, 20_480, TIER_LAPTOP_48),  # 10 full-attention layers of 40 * 2*2*256*2
    "nemotron-3.5-lightning-30b-a3b": _g(
        "unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF", "nemotron_h_moe",
        "30B total, 3B active hybrid Mamba-MoE, 1M context", 3.0,
        {"UD-IQ1_M": 19, "UD-IQ2_XXS": 19, "UD-IQ2_M": 19, "UD-IQ3_XXS": 20, "UD-IQ3_S": 21,
         "UD-IQ4_NL": 21, "UD-Q3_K_XL": 21, "MXFP4_MOE": 23, "UD-Q4_K_S": 24, "UD-Q4_K_M": 25,
         "UD-Q4_K_XL": 26, "UD-Q5_K_S": 26, "UD-Q5_K_M": 30, "UD-Q5_K_XL": 30, "Q8_0": 35,
         "UD-Q6_K_XL": 35, "UD-Q8_K_XL": 39, "BF16": 66},
        "UD-Q4_K_XL", 1048576, 6_144, TIER_LAPTOP_48),  # 6 attention layers of 52 * 2*2*128*2; Mamba layers constant state
    "gpt-oss-20b": _g(
        "unsloth/gpt-oss-20b-GGUF", "gpt-oss", "21B total, 3.6B active MoE (native MXFP4)", 3.6,
        {"Q3_K_S": 11, "Q2_K": 11, "Q4_0": 12, "Q3_K_M": 12, "Q4_1": 12, "Q4_K_S": 12, "Q4_K_M": 12,
         "Q5_K_S": 12, "Q5_K_M": 12, "Q2_K_L": 12, "UD-Q4_K_XL": 12, "Q6_K": 12, "UD-Q6_K_XL": 12,
         "Q8_0": 12, "UD-Q8_K_XL": 13, "F16": 14},
        "UD-Q4_K_XL", 131072, 24_576, TIER_LAPTOP_48,  # 12 full-attention layers of 24 * 2*8*64*2; 12 SWA(128) layers bounded
        "Experts are MXFP4 in the source weights, so every quant is ~12 GB; only attention precision differs."),
    "gemma-4-12b": _g(
        "unsloth/gemma-4-12b-it-GGUF", "gemma4", "12B dense, multimodal", 12.0,
        {"UD-IQ2_M": 4, "UD-IQ3_XXS": 5, "UD-Q2_K_XL": 5, "Q3_K_S": 5, "Q3_K_M": 6, "UD-Q3_K_XL": 6,
         "IQ4_XS": 6, "IQ4_NL": 7, "Q4_0": 7, "Q4_K_S": 7, "Q4_K_M": 7, "UD-Q4_K_XL": 7, "Q4_1": 7,
         "Q5_K_S": 8, "Q5_K_M": 8, "UD-Q5_K_XL": 9, "Q6_K": 10, "UD-Q6_K_XL": 11, "Q8_0": 13,
         "UD-Q8_K_XL": 14, "BF16": 24},
        "Q4_K_M", 262144, 65_536, TIER_LAPTOP_48),  # 8 full-attention layers of 48 * 2*8*256*2
    "qwen3.8-9b-distill": _g(
        "empero-ai/Qwen3.8-9B-Distill-GGUF", "qwen35", "9B dense (distilled from Qwen3.8-2.4T)", 9.0,
        {"Q4_K_M": 6, "Q5_K_M": 7, "Q6_K": 8, "Q8_0": 10, "BF16": 18},
        "Q4_K_M", 262144, 32_768, TIER_LAPTOP_48),  # 8 full-attention layers of 32 * 2*4*256*2
    "qwen3.8-27b": _g(
        "unsloth/Qwen3.8-27B-GGUF", "qwen35", "27B dense", 27.0,
        {"UD-IQ1_S": 6, "UD-IQ1_M": 7, "UD-IQ2_XXS": 7, "UD-IQ2_S": 8, "UD-Q2_K_XL": 10, "UD-IQ3_XXS": 11,
         "UD-IQ3_S": 12, "UD-Q3_K_XL": 13, "UD-IQ4_XS": 14, "UD-Q4_K_S": 15, "Q4_0": 16, "UD-Q4_K_M": 16,
         "Q4_1": 18, "UD-Q4_K_XL": 18, "UD-Q5_K_S": 19, "UD-Q5_K_M": 20, "UD-Q5_K_XL": 21, "UD-Q6_K": 22,
         "UD-Q6_K_M": 23, "UD-Q6_K_L": 24, "UD-Q6_K_XL": 25, "UD-Q8_K_L": 28, "Q8_0": 29,
         "UD-Q8_K_XL": 31, "BF16": 55},
        "UD-Q4_K_XL", 262144, 65_536, TIER_LAPTOP_48,  # 16 full-attention layers of 64 * 2*4*256*2
        "Dense 27B: strongest small model but ~3-4x slower per token than the 3-4B-active MoEs above."),
    "gemma-4-31b": _g(
        "unsloth/gemma-4-31b-it-GGUF", "gemma4", "31B dense, multimodal", 31.0,
        {"UD-IQ2_XXS": 9, "UD-IQ2_M": 11, "UD-Q2_K_XL": 12, "UD-IQ3_XXS": 12, "Q3_K_S": 13, "Q3_K_M": 15,
         "UD-Q3_K_XL": 15, "IQ4_XS": 16, "IQ4_NL": 17, "Q4_0": 17, "Q4_K_S": 17, "Q4_K_M": 18,
         "UD-Q4_K_XL": 19, "Q4_1": 19, "Q5_K_S": 21, "Q5_K_M": 22, "UD-Q5_K_XL": 22, "Q6_K": 25,
         "UD-Q6_K_XL": 28, "Q8_0": 33, "UD-Q8_K_XL": 35, "BF16": 61},
        "UD-Q4_K_XL", 262144, 163_840, TIER_LAPTOP_48),  # 10 full-attention layers of 60 * 2*16*256*2; 50 SWA(1024) bounded
    "kimi-linear-48b": _g(
        "mradermacher/Kimi-Linear-48B-A3B-Instruct-GGUF", "kimi-linear",
        "48B total, 3B active MoE (KDA + MLA hybrid), 1M context", 3.0,
        {"Q2_K": 18, "Q3_K_S": 21, "Q3_K_M": 23, "Q3_K_L": 26, "IQ4_XS": 26, "Q4_K_S": 28, "Q4_K_M": 30,
         "Q5_K_S": 34, "Q5_K_M": 35, "Q6_K": 40, "Q8_0": 52},
        "Q4_K_M", 1048576, 8_064, TIER_LAPTOP_48),  # ~7 MLA layers of 27 * (512+64) * 2

    # ==================================================================
    # 96 GB laptop / desktop tier — 5–12B active MoE
    # ==================================================================
    "gpt-oss-120b": _g(
        "unsloth/gpt-oss-120b-GGUF", "gpt-oss", "117B total, 5.1B active MoE (native MXFP4)", 5.1,
        {"Q3_K_S": 63, "Q2_K": 63, "Q4_0": 63, "Q3_K_M": 63, "Q4_1": 63, "Q4_K_S": 63, "Q4_K_M": 63,
         "Q2_K_L": 63, "Q5_K_S": 63, "Q5_K_M": 63, "UD-Q4_K_XL": 63, "Q6_K": 63, "UD-Q6_K_XL": 63,
         "Q8_0": 63, "UD-Q8_K_XL": 64, "F16": 65},
        "UD-Q4_K_XL", 131072, 36_864, TIER_LAPTOP_96,  # 18 full-attention layers of 36 * 2*8*64*2
        "Experts are MXFP4 in the source weights, so every quant is ~63 GB."),
    "qwen3.5-122b-a10b": _g(
        "unsloth/Qwen3.5-122B-A10B-GGUF", "qwen35moe", "122B total, 10B active MoE", 10.0,
        {"UD-IQ1_M": 34, "UD-IQ2_XXS": 37, "UD-IQ2_M": 39, "UD-Q2_K_XL": 42, "UD-IQ3_XXS": 45,
         "UD-IQ3_S": 47, "Q3_K_S": 52, "Q3_K_M": 56, "UD-Q3_K_XL": 57, "UD-IQ4_XS": 60, "UD-IQ4_NL": 61,
         "Q4_K_S": 72, "MXFP4_MOE": 75, "Q4_K_M": 77, "UD-Q4_K_XL": 77, "Q5_K_S": 86, "Q5_K_M": 92,
         "UD-Q5_K_XL": 92, "Q6_K": 101, "UD-Q6_K_XL": 112, "Q8_0": 130, "UD-Q8_K_XL": 171, "BF16": 244},
        "UD-IQ4_XS", 262144, 24_576, TIER_LAPTOP_96),  # 12 full-attention layers of 48 * 2*2*256*2
    "nemotron-3-super-120b-a12b": _g(
        "unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF", "nemotron_h_moe",
        "120B total, 12B active hybrid Mamba-MoE, 1M context", 12.0,
        {"UD-IQ1_M": 53, "UD-IQ2_XXS": 53, "UD-IQ2_M": 53, "UD-Q2_K_XL": 55, "UD-IQ3_S": 57, "UD-IQ3_XXS": 57,
         "UD-Q3_K_M": 62, "UD-Q3_K_S": 62, "UD-Q3_K_XL": 63, "UD-IQ4_NL": 64, "UD-IQ4_XS": 64, "UD-Q4_K_S": 79,
         "MXFP4_MOE": 82, "UD-Q4_K_M": 83, "UD-Q4_K_XL": 84, "UD-Q5_K_S": 90, "UD-Q5_K_M": 107,
         "UD-Q5_K_XL": 108, "UD-Q6_K": 115, "UD-Q6_K_XL": 118, "Q8_0": 128, "UD-Q8_K_XL": 132, "BF16": 242},
        "UD-IQ4_XS", 1048576, 8_192, TIER_LAPTOP_96),  # 8 attention layers of 88 * 2*2*128*2
    "llama-4-scout": _g(
        "unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF", "llama4", "109B total, 17B active MoE (16 experts), 10M context", 17.0,
        {"UD-TQ1_0": 29, "UD-IQ1_S": 32, "UD-IQ1_M": 35, "UD-IQ2_XXS": 37, "UD-IQ2_M": 39, "Q2_K": 40,
         "Q2_K_L": 40, "UD-Q2_K_XL": 42, "UD-IQ3_XXS": 46, "Q3_K_S": 47, "UD-Q3_K_XL": 49, "Q3_K_M": 52,
         "IQ4_XS": 58, "IQ4_NL": 61, "Q4_0": 61, "Q4_K_S": 61, "UD-Q4_K_XL": 62, "Q4_K_M": 65, "Q4_1": 68,
         "Q5_K_S": 74, "Q5_K_M": 77, "UD-Q5_K_XL": 79, "Q6_K": 88, "UD-Q6_K_XL": 94, "Q8_0": 115,
         "UD-Q8_K_XL": 128, "BF16": 216},
        "UD-Q4_K_XL", 10485760, 49_152, TIER_LAPTOP_96),  # 12 global (NoPE) layers of 48 * 2*8*128*2; chunked layers bounded

    # ==================================================================
    # 192 GB workstation tier
    # ==================================================================
    "qwen3.8-flash-next": _g(
        "unsloth/Qwen3.8-Flash-Next-GGUF", "qwen4exp", "177B total MoE (512 experts, top-10), hybrid attention", None,
        {"UD-IQ1_S": 73, "UD-IQ1_M": 75, "UD-Q2_K_XL": 79, "UD-IQ3_XXS": 82, "UD-Q3_K_XL": 90, "UD-IQ4_XS": 94,
         "UD-Q4_K_XL": 111, "UD-Q5_K_XL": 158, "UD-Q6_K_XL": 169, "Q8_0": 188, "BF16": 354},
        "UD-Q4_K_XL", 262144, 24_576, TIER_WORKSTATION_192,  # 12 full-attention layers of 48 * 2*2*256*2
        "Released 2026-09-02; arch qwen4exp needs a llama.cpp build from September 2026 or later."),
    "minimax-m2.7": _g(
        "unsloth/MiniMax-M2.7-GGUF", "minimax-m2", "229B total, 10B active MoE", 10.0,
        {"UD-IQ1_M": 61, "UD-IQ2_XXS": 65, "UD-IQ2_M": 70, "UD-Q2_K_XL": 75, "UD-IQ3_XXS": 80, "UD-IQ3_S": 84,
         "UD-Q3_K_S": 94, "UD-Q3_K_M": 101, "UD-Q3_K_XL": 102, "UD-IQ4_XS": 108, "UD-IQ4_NL": 111,
         "UD-Q4_K_S": 131, "MXFP4_MOE": 136, "UD-Q4_K_M": 140, "UD-Q4_K_XL": 141, "UD-Q5_K_S": 159,
         "UD-Q5_K_M": 169, "UD-Q5_K_XL": 169, "UD-Q6_K": 188, "UD-Q6_K_XL": 207, "Q8_0": 243,
         "UD-Q8_K_XL": 247, "BF16": 457},
        "UD-Q4_K_XL", 196608, 253_952, TIER_WORKSTATION_192),  # 62 layers * 2*8*128*2
    "deepseek-v4-flash": _g(
        "unsloth/DeepSeek-V4-Flash-0731-GGUF", "deepseek4", "284B total MoE, 256 experts top-6", None,
        {"UD-IQ1_S": 83, "UD-IQ1_M": 87, "UD-IQ2_XXS": 91, "UD-IQ2_M": 91, "UD-Q2_K_XL": 97, "UD-IQ3_XXS": 104,
         "UD-IQ3_S": 116, "UD-Q3_K_M": 128, "UD-Q3_K_XL": 128, "UD-IQ4_NL": 137, "UD-IQ4_XS": 137,
         "UD-Q4_K_XL": 155, "UD-Q8_K_XL": 162},
        "UD-Q4_K_XL", 1048576, 88_064, TIER_WORKSTATION_192),  # 43 layers, MQA 1 kv head * 512 * 2 * 2 (upper bound)

    # ==================================================================
    # Server tiers (512 GB+)
    # ==================================================================
    "minimax-m3": _g(
        "unsloth/MiniMax-M3-GGUF", "minimax-m3", "426B total, 23B active MoE, 1M context", 23.0,
        {"UD-IQ1_M": 128, "UD-IQ2_XXS": 134, "UD-IQ2_M": 134, "UD-Q2_K_XL": 143, "UD-IQ3_XXS": 159,
         "UD-IQ3_S": 175, "UD-Q3_K_M": 195, "UD-Q3_K_XL": 195, "UD-IQ4_XS": 208, "UD-IQ4_NL": 212,
         "UD-Q4_K_S": 248, "MXFP4_MOE": 256, "UD-Q4_K_M": 264, "UD-Q4_K_XL": 265, "UD-Q5_K_S": 299,
         "UD-Q5_K_M": 318, "UD-Q5_K_XL": 318, "UD-Q6_K": 354, "UD-Q6_K_XL": 387, "Q8_0": 453,
         "UD-Q8_K_XL": 464, "BF16": 852},
        "UD-Q4_K_XL", 1048576, 122_880, TIER_SERVER_512),  # 60 layers * 2*4*128*2
    "glm-5.3": _g(
        "unsloth/GLM-5.3-GGUF", "glm-dsa", "754B total MoE, 1M context", None,
        {"UD-IQ1_S": 217, "UD-IQ1_M": 228, "UD-IQ2_M": 239, "UD-Q2_K_XL": 254, "UD-IQ3_XXS": 282,
         "UD-Q3_K_XL": 343, "UD-IQ4_XS": 365, "UD-Q4_K_XL": 467, "UD-Q5_K_XL": 562, "UD-Q6_K_XL": 684,
         "Q8_0": 801, "BF16": 1508},
        "UD-Q2_K_XL", 1048576, 89_856, TIER_SERVER_512),  # 78 MLA layers * (512+64) * 2
    "deepseek-v3.2": _g(
        "unsloth/DeepSeek-V3.2-GGUF", "deepseek2", "671B total, 37B active MoE", 37.0,
        {"UD-TQ1_0": 161, "UD-IQ1_S": 184, "UD-IQ1_M": 199, "UD-IQ2_XXS": 217, "UD-IQ2_M": 228, "Q2_K": 245,
         "Q2_K_L": 246, "UD-Q2_K_XL": 247, "UD-IQ3_XXS": 273, "Q3_K_S": 290, "Q3_K_M": 320, "UD-Q3_K_XL": 321,
         "IQ4_XS": 358, "IQ4_NL": 379, "Q4_0": 380, "Q4_K_S": 381, "Q4_K_M": 405, "UD-Q4_K_XL": 408,
         "Q4_1": 421, "Q5_K_S": 463, "Q5_K_M": 476, "UD-Q5_K_XL": 482, "Q6_K": 551, "UD-Q6_K_XL": 574,
         "Q8_0": 713, "UD-Q8_K_XL": 780, "BF16": 1342},
        "UD-Q2_K_XL", 163840, 70_272, TIER_SERVER_512),  # 61 MLA layers * (512+64) * 2
    "kimi-k2.6": _g(
        "unsloth/Kimi-K2.6-GGUF", "deepseek2", "1.03T total, 32B active MoE", 32.0,
        {"UD-Q2_K_XL": 340, "UD-Q4_K_XL": 584, "UD-Q8_K_XL": 595, "BF16": 2053},
        "UD-Q2_K_XL", 262144, 70_272, TIER_SERVER_512),
    "kimi-k2.5": _g(
        "unsloth/Kimi-K2.5-GGUF", "deepseek2", "1.03T total, 32B active MoE", 32.0,
        {"UD-TQ1_0": 240, "UD-IQ1_S": 276, "UD-IQ1_M": 301, "UD-IQ2_XXS": 327, "UD-IQ2_M": 345, "Q2_K": 374,
         "Q2_K_L": 374, "UD-Q2_K_XL": 375, "UD-IQ3_XXS": 415, "Q3_K_S": 443, "Q3_K_M": 490, "UD-Q3_K_XL": 490,
         "IQ4_XS": 547, "IQ4_NL": 579, "Q4_0": 581, "Q4_K_S": 583, "Q4_K_M": 621, "UD-Q4_K_XL": 622,
         "Q4_1": 643, "Q5_K_S": 707, "Q5_K_M": 729, "UD-Q5_K_XL": 731, "Q6_K": 843, "UD-Q6_K_XL": 878,
         "Q8_0": 1091, "UD-Q8_K_XL": 1190, "BF16": 2053},
        "UD-IQ2_M", 262144, 70_272, TIER_SERVER_512),
    "qwen3.8": _g(
        "unsloth/Qwen3.8-2.4T-A95B-GGUF", "qwen35moe", "2.4T total, 95B active MoE", 95.0,
        {"UD-Q1_0": 397, "UD-IQ1_S": 508, "UD-IQ1_M": 564, "UD-IQ2_XXS": 657, "UD-IQ2_XS": 731,
         "UD-IQ3_XXS": 956, "UD-IQ4_XS": 1311, "Q8_0": 2600, "BF16": 4893},
        "UD-IQ1_S", 262144, 94_208, TIER_SERVER_768,  # 23 full-attention layers of 92 * 2*4*256*2
        "95B active parameters: slow on CPU regardless of RAM (~1 t/s class on a 24-core AVX2 EPYC; not measured in this repo)."),
    "kimi-k3": _g(
        "unsloth/Kimi-K3-GGUF", "kimi-k3", "2.78T total, 93B active MoE, 1M context", 93.0,
        {"UD-Q1_0": 466, "UD-TQ1_0": 509, "UD-TQ2_0": 551, "UD-IQ1_S": 594, "UD-IQ1_M": 649,
         "UD-IQ2_XXS": 711, "UD-Q2_K_XL": 861, "UD-Q4_K_XL": 1509, "UD-Q8_K_XL": 1561},
        "UD-IQ1_S", 1048576, 107_136, TIER_SERVER_768,  # 93 MLA layers * (512+64) * 2
        "93B active parameters: slow on CPU regardless of RAM (0.85 t/s on a 24-core AVX2 EPYC, Aug 2026, per commit history)."),

    # ==================================================================
    # ktransformers — safetensors served by sglang-kt (python -m sglang.launch_server)
    # Linux x86-64 + NVIDIA GPU (SM 8.0+) required. kt_method = CPU expert
    # backend; FP8/BF16/RAWINT4/MXFP* need AVX-512, AMXINT4/8 need AMX.
    # ==================================================================
    "glm-5.3-flash": _k(
        "zai-org/GLM-5.3-Flash", "Glm5NextForConditionalGeneration",
        "321B total, 18B active MoE, multimodal, 1M context", 18.0, 328, 1048576,
        12_672,  # 11 DSA/MLA layers of 45 * (512+64) * 2; 34 linear layers bounded
        "FP8", ["--tool-call-parser", "glm47", "--reasoning-parser", "glm45"], TIER_SERVER_512,
        "Native FP8 in ktransformers since 2026-08-26 (doc/en/kt-kernel/GLM-5.3-Flash-Tutorial.md: "
        "NVIDIA SM89/SM120 GPU, AVX-512 FP8 CPU kernel, ~350 GB RAM). Not in released llama.cpp: arch "
        "glm5next is open PR ggml-org/llama.cpp#27754 (unsloth/GLM-5.3-Flash-GGUF needs that PR)."),
    "deepseek-v4-flash-kt": _k(
        "deepseek-ai/DeepSeek-V4-Flash", "DeepseekV4ForCausalLM", "284B total MoE, native MXFP4 experts", None,
        160, 1048576, 88_064, "MXFP4",
        ["--attention-backend", "flashinfer", "--disable-shared-experts-fusion"], TIER_WORKSTATION_192,
        "kt-kernel registry entry DeepSeek-V4-Flash (kt-method MXFP4)."),
    "minimax-m2.7-kt": _k(
        "MiniMaxAI/MiniMax-M2.7", "MiniMaxM2ForCausalLM", "229B total, 10B active MoE (FP8)", 10.0,
        230, 204800, 253_952, "FP8",
        ["--attention-backend", "flashinfer", "--disable-shared-experts-fusion",
         "--tool-call-parser", "minimax-m2", "--reasoning-parser", "minimax-append-think"], TIER_SERVER_512,
        "kt-kernel registry entry MiniMax-M2.7 (kt-method FP8)."),
    "minimax-m3-kt": _k(
        "MiniMaxAI/MiniMax-M3-MXFP8", "MiniMaxM3SparseForConditionalGeneration",
        "426B total, 23B active MoE (native MXFP8)", 23.0, 444, 1048576, 122_880, "MXFP8",
        ["--attention-backend", "flashinfer", "--disable-shared-experts-fusion",
         "--quantization", "mxfp8", "--moe-runner-backend", "triton",
         "--tool-call-parser", "minimax-m3", "--reasoning-parser", "minimax-m3"], TIER_SERVER_512,
        "kt-kernel registry entry MiniMax-M3 (kt-method MXFP8)."),
    "kimi-k2-thinking": _k(
        "moonshotai/Kimi-K2-Thinking", "DeepseekV3ForCausalLM", "1T total, 32B active MoE (native INT4)", 32.0,
        594, 262144, 70_272, "RAWINT4",
        ["--attention-backend", "flashinfer", "--disable-shared-experts-fusion"], TIER_SERVER_768,
        "kt-kernel registry entry Kimi-K2-Thinking (kt-method RAWINT4)."),
    "deepseek-v3.2-kt": _k(
        "deepseek-ai/DeepSeek-V3.2", "DeepseekV32ForCausalLM", "671B total, 37B active MoE (FP8)", 37.0,
        689, 163840, 70_272, "FP8",
        ["--attention-backend", "flashinfer", "--disable-shared-experts-fusion"], TIER_SERVER_768,
        "kt-kernel registry entry DeepSeek-V3.2 (kt-method FP8)."),
}

# Old ids still accepted in models.yaml and by the catalog lookup.
MODEL_ID_ALIASES: dict[str, str] = {
    "qwen3.8-2.4t": "qwen3.8",
}

# CPU expert backends accepted by sglang-kt --kt-method (kt-kernel README).
KT_METHODS: tuple[str, ...] = (
    "FP8", "FP8_PERCHANNEL", "BF16", "RAWINT4", "MXFP4", "MXFP8",
    "AMXINT4", "AMXINT8", "LLAMAFILE",
)

# What `litmoe init` writes when it cannot detect RAM.
DEFAULT_MODEL = "gemma-4-26b-a4b"

# Anthropic model names that Claude Code (and other Anthropic-SDK tools) send.
# `litmoe init` / `litmoe install` attach these as aliases to the FIRST model
# in a config so requests with a Claude name route somewhere instead of 404ing.
# Claude Code uses the haiku id for background/summary calls even when the
# main model is set explicitly, so all three families are needed.
CLAUDE_ALIASES = (
    "claude-sonnet-4-5", "claude-sonnet-4-5-20250929",
    "claude-opus-4-1", "claude-opus-4-1-20250805",
    "claude-haiku-4-5", "claude-haiku-4-5-20251001",
    "claude-3-7-sonnet-latest", "claude-3-5-sonnet-latest", "claude-3-5-haiku-latest",
)


def lookup(model_id: str) -> dict | None:
    """Catalog entry for a model id (or legacy alias), or None."""
    key = MODEL_ID_ALIASES.get(model_id, model_id)
    return KNOWN_MODELS.get(key)


def gguf_models() -> list[str]:
    return [k for k, v in KNOWN_MODELS.items() if v["format"] == GGUF]


def kt_models() -> list[str]:
    return [k for k, v in KNOWN_MODELS.items() if v["format"] == SAFETENSORS]


def quant_size_gb(model_id: str, quant: str | None) -> float | None:
    """Catalog size in GB for a (model, quant), or the safetensors size."""
    info = lookup(model_id)
    if not info:
        return None
    if info["format"] == SAFETENSORS:
        return float(info["size_gb"])
    q = quant or info["default_quant"]
    val = info["quants"].get(q)
    return float(val) if val is not None else None


def ram_needed_gb(model_id: str, quant: str | None = None, n_ctx: int = _FIT_CTX_TOKENS) -> float | None:
    """RAM (GB) to run a quant with an n_ctx context: weights*overhead + KV + OS headroom."""
    info = lookup(model_id)
    size = quant_size_gb(model_id, quant)
    if not info or size is None:
        return None
    kv_gb = info["kv_bytes_per_token"] * n_ctx / 1e9
    return size * _MODEL_OVERHEAD + kv_gb + _OS_HEADROOM_GB


def fit_context(kv_bytes_per_token: int, weights_gb: float, total_ram_gb: float,
                target_ctx: int, min_ctx: int = 8192) -> tuple[int, str | None]:
    """Largest context <= target_ctx whose KV cache fits next to the weights in RAM.

    Budget is 90% of RAM minus 3 GB. Returns (ctx, note); note is None when the
    target fits unchanged, otherwise a human-readable reason. Contexts are
    rounded down to a multiple of 4096 and never below min_ctx.
    """
    avail_gb = total_ram_gb * 0.9 - 3
    kv_gb = kv_bytes_per_token * target_ctx / 1e9
    if weights_gb + kv_gb <= avail_gb:
        return target_ctx, None
    max_kv_gb = avail_gb - weights_gb - 1
    if max_kv_gb <= 0:
        ctx = max(min(target_ctx, 32768), min_ctx)
        return ctx, (f"weights ({weights_gb:.0f} GB) may not fit in {total_ram_gb:.0f} GB RAM; "
                     f"context capped at {ctx}")
    ctx = int(max_kv_gb * 1e9 / kv_bytes_per_token)
    ctx = max((ctx // 4096) * 4096, min_ctx)
    return ctx, (f"reduced context from {target_ctx} to {ctx} to fit {weights_gb:.0f} GB weights + "
                 f"KV cache in {total_ram_gb:.0f} GB RAM")


def tier_for(model_id: str, quant: str | None = None) -> int | None:
    """Smallest RAM tier that fits the given quant (default_quant if None)."""
    need = ram_needed_gb(model_id, quant)
    if need is None:
        return None
    for t in TIERS:
        if need <= t:
            return t
    return TIERS[-1]


def largest_quant_that_fits(model_id: str, ram_gb: float, n_ctx: int = _FIT_CTX_TOKENS) -> str | None:
    """Highest-precision quant (by size) whose RAM need fits ram_gb, or None."""
    info = lookup(model_id)
    if not info or info["format"] != GGUF:
        return None
    fitting = [(size, q) for q, size in info["quants"].items()
               if (ram_needed_gb(model_id, q, n_ctx) or math.inf) <= ram_gb]
    if not fitting:
        return None
    return max(fitting)[1]


def smallest_gguf_model() -> str:
    """The GGUF catalog entry with the lowest RAM need at its default quant (last-resort pick)."""
    return min(gguf_models(), key=lambda m: ram_needed_gb(m) or math.inf)


def recommended_for_ram(ram_gb: float, max_models: int = 4) -> list[str]:
    """GGUF models whose default quant fits ram_gb.

    Order: the laptop default first (always fast, vision-capable), then the
    strongest tier that fits with the fastest (fewest active params) models first.
    """
    fits = [m for m in gguf_models() if (ram_needed_gb(m) or math.inf) <= ram_gb]
    if not fits:
        return []
    top_tier = max(KNOWN_MODELS[m]["tier"] for m in fits)
    by_speed = sorted(fits, key=lambda m: (KNOWN_MODELS[m].get("active_b") or 1e9))
    ordered: list[str] = [DEFAULT_MODEL] if DEFAULT_MODEL in fits else []
    ordered += [m for m in by_speed if KNOWN_MODELS[m]["tier"] == top_tier and m not in ordered]
    ordered += [m for m in by_speed if m not in ordered]
    return ordered[:max_models]


def ram_needed_together_gb(needs_gb: list[float]) -> float:
    """RAM for several models loaded at once: their needs summed, OS headroom counted once."""
    if not needs_gb:
        return 0.0
    return sum(needs_gb) - _OS_HEADROOM_GB * (len(needs_gb) - 1)


def fit_together(model_ids: list[str], ram_gb: float) -> tuple[list[str], list[str]]:
    """Split model_ids into (loadable together within ram_gb, the rest), preserving order.

    The gateway starts every configured model at once, so a config is only
    usable if the *sum* fits. Greedy in the given order: keeps adding models
    while the running total (default quant, 32K context) stays within budget.
    """
    kept: list[str] = []
    dropped: list[str] = []
    needs: list[float] = []
    for mid in model_ids:
        need = ram_needed_gb(mid)
        if need is None:
            dropped.append(mid)
            continue
        if ram_needed_together_gb(needs + [need]) <= ram_gb:
            kept.append(mid)
            needs.append(need)
        else:
            dropped.append(mid)
    return kept, dropped


def validate_catalog() -> list[str]:
    """Return a list of internal-consistency problems (empty = OK). Used by tests."""
    problems: list[str] = []
    for mid, info in KNOWN_MODELS.items():
        for key in ("hf_repo", "engine", "format", "arch", "native_ctx", "kv_bytes_per_token", "tier"):
            if key not in info:
                problems.append(f"{mid}: missing {key}")
        if info.get("engine") not in ("llamacpp", "ktransformers"):
            problems.append(f"{mid}: bad engine {info.get('engine')}")
        if info.get("tier") not in TIERS:
            problems.append(f"{mid}: bad tier {info.get('tier')}")
        if info.get("format") == GGUF:
            if info.get("engine") != "llamacpp":
                problems.append(f"{mid}: gguf must use llamacpp")
            if not info.get("quants"):
                problems.append(f"{mid}: no quants")
            elif info.get("default_quant") not in info["quants"]:
                problems.append(f"{mid}: default_quant {info.get('default_quant')} not in quants")
        elif info.get("format") == SAFETENSORS:
            if info.get("engine") != "ktransformers":
                problems.append(f"{mid}: safetensors must use ktransformers")
            if "size_gb" not in info:
                problems.append(f"{mid}: missing size_gb")
            if info.get("kt_method") not in KT_METHODS:
                problems.append(f"{mid}: bad kt_method {info.get('kt_method')}")
        else:
            problems.append(f"{mid}: bad format {info.get('format')}")
        if not isinstance(info.get("native_ctx"), int) or info["native_ctx"] < 4096:
            problems.append(f"{mid}: implausible native_ctx")
        if not isinstance(info.get("kv_bytes_per_token"), int) or info["kv_bytes_per_token"] <= 0:
            problems.append(f"{mid}: implausible kv_bytes_per_token")
        # The declared tier must actually hold the default quant.
        computed = tier_for(mid)
        if computed is not None and computed != info.get("tier"):
            problems.append(f"{mid}: declared tier {info.get('tier')} but default quant needs tier {computed}"
                            f" ({ram_needed_gb(mid):.0f} GB)")
    for alias, target in MODEL_ID_ALIASES.items():
        if target not in KNOWN_MODELS:
            problems.append(f"alias {alias} -> unknown {target}")
        if alias in KNOWN_MODELS:
            problems.append(f"alias {alias} shadows a real entry")
    if DEFAULT_MODEL not in KNOWN_MODELS or KNOWN_MODELS[DEFAULT_MODEL]["tier"] != TIER_LAPTOP_48:
        problems.append("DEFAULT_MODEL must be a 48 GB-tier model")
    return problems
