# Upstream acknowledgements and license boundaries

litmoe is a local-serving integration layer. Its configuration, lifecycle,
API adaptation, harness launchers, and measurement tools rely on independently
developed inference engines and model artifacts. Native inference algorithms,
kernels, quantization formats, and model weights remain the work of their
respective authors.

litmoe's [Apache 2.0 license](LICENSE) covers its own code; it does not relicense
upstream software or model weights. This document acknowledges the principal
runtime/artifact projects, not an exhaustive inventory of transitive package
dependencies. The exact downloaded revision's license and any component-level
notices remain authoritative. Preserve them with redistributed source or
binaries, including adapted code, and identify local modifications separately.

## DwarfStar (`antirez/ds4`)

- Project: <https://github.com/antirez/ds4>
- Created by Salvatore Sanfilippo (antirez), with the ds4.c contributors.
- Relationship: a native engine integrated for GLM-5.3-Flash serving through
  litmoe's pinned installer and adapter. Real-model verification is recorded
  separately from implementation.
- litmoe's local modification: `litmoe/patches/dwarfstar-serving.patch`
  implements typed tool-reference resolution, rendered token counting,
  native quiescence reporting, and protocol regressions. It does not replace
  or claim authorship of the upstream inference engine.
- Upstream contribution: model-specific native execution, Metal/CUDA/ROCm
  kernels, expert streaming, model-state handling, prompt/server machinery,
  and the project's model-conversion and validation tools. These are not
  litmoe inventions.
- DwarfStar explicitly credits llama.cpp/GGML. Its current root license also
  retains DeepSeek's copyright notice; that attribution chain is preserved
  below rather than collapsed into a litmoe-only credit.
- Source pinned by the integration:
  `0aaea5a238fb41a35106a551e73c8409dfb751ac`.
  The initial build-only probe used the older GLM feature-branch revision
  `b1b4ea03645434423e5cb4f39818fdc075e49825`.

The following is the root [MIT license at the pinned revision](https://github.com/antirez/ds4/blob/0aaea5a238fb41a35106a551e73c8409dfb751ac/LICENSE).
Other files or bundled components may carry additional notices, which must
also be retained when those components are redistributed.

```text
MIT License

Copyright (c) 2026 The ds4.c authors
Copyright (c) 2023-2026 The ggml authors
Copyright (c) 2023 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Other inference and serving projects

| Project | Relationship to litmoe |
| --- | --- |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) and [GGML](https://github.com/ggml-org/ggml) | Native runtime, GGUF/quantization ecosystem, and hardware kernels used by the llama.cpp integration; also foundational work acknowledged by DwarfStar |
| [KTransformers](https://github.com/kvcache-ai/ktransformers) | Heterogeneous CPU/GPU execution and expert-offload implementation used by the ktransformers adapter |
| [SGLang](https://github.com/sgl-project/sglang) | Serving stack used with kt-kernel/KTransformers |
| [WARP](https://github.com/sqliteai/warp) | `.waste` conversion, paging, inference, and local HTTP server; installed at a pinned revision with litmoe's separately documented native patch |

These engines are fetched/installed separately rather than presented as
litmoe-authored implementations. WARP's inspected pinned revision uses Apache
2.0; retain its original notices and identify the bundled patch as a local
modification. Consult each other project's selected revision for its complete
license and incorporated-component notices.

## Models, quantized artifacts, and distribution

The model catalog references weights created by their original model authors,
including Z.ai for GLM. Quantized artifacts may be published separately by
projects such as [Unsloth](https://huggingface.co/unsloth) or DwarfStar. Credit
both the original model and the artifact publisher when identifying a deployed
model. [Hugging Face](https://huggingface.co/) provides repository/distribution
infrastructure; hosting a file does not make Hugging Face its model author.

Engine software licenses and model-weight licenses are separate. Check the
exact model card, source revision, and artifact terms before redistribution or
use. litmoe does not bundle model weights or claim authorship of downloaded
models or quantizations.

## Connected agent clients

Claude Code, Hermes, and OMP remain independent clients. Their agent planning,
tool execution, and user interfaces belong to those projects. litmoe supplies
isolated configuration and local API access; it does not claim ownership of
the clients or imply their endorsement.
