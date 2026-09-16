# litmoe - OpenAI-compatible gateway
# No inference code in this image. Install ktransformers/llama.cpp as needed.

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install
COPY pyproject.toml README.md ./
COPY litmoe ./litmoe
RUN pip install --no-cache-dir -e .

# Optional: install engines
# llama.cpp (CPU-only with BLAS + native CPU features):
#   RUN apt-get install -y build-essential cmake pkg-config libopenblas-dev libopenblas0
#   RUN git clone https://github.com/ggml-org/llama.cpp && \
#       cd llama.cpp && \
#       cmake -B build -DGGML_CUDA=OFF -DGGML_BLAS=ON \
#             -DGGML_BLAS_VENDOR=OpenBLAS -DGGML_NATIVE=ON \
#             -DCMAKE_BUILD_TYPE=Release && \
#       cmake --build build --config Release -j --target llama-server && \
#       cp build/bin/llama-server /usr/local/bin/ && \
#       cp build/bin/lib*.so* /usr/local/lib/ && ldconfig
#
# ktransformers (Linux + NVIDIA GPU only; wheels need glibc >= 2.35, Python 3.11/3.12):
#   RUN pip install "ktransformers[sglang]"
#   # or, when the wheels do not match this image:
#   RUN git clone --recursive https://github.com/kvcache-ai/ktransformers.git && \
#       cd ktransformers && bash install.sh

# Config
ENV LITMOE_CONFIG=/config/models.yaml
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["litmoe", "serve"]