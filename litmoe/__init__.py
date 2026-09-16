"""litmoe - OpenAI-compatible gateway for llama.cpp and ktransformers.

This is a thin orchestration layer. It does NOT contain inference code.
For inference, it dispatches to:
- llama.cpp (https://github.com/ggml-org/llama.cpp) - GGUF models on
  CPU/CUDA/Metal/Vulkan/ROCm; the default engine (laptops through servers)
- ktransformers (https://github.com/kvcache-ai/ktransformers) - sglang-kt with
  kt-kernel CPU expert offload for native FP8/INT4 MoE checkpoints
  (Linux + NVIDIA GPU)
"""

__version__ = "0.2.0"
