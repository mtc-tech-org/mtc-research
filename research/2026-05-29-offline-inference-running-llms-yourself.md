---
title: "Offline inference: running LLMs yourself (batch-focused, deep dive)"
date: 2026-05-29
tags: [llm, inference, offline, batch, quantization, vllm, sglang, mlx, gpu, apple-silicon]
status: draft
scope: "Self-hosted/offline LLM inference across NVIDIA GPUs and Apple Silicon, optimized for batch/throughput workloads. Deep technical dive."
---

# Offline inference: running LLMs yourself

> **Focus of this note:** *offline / batch* inference — large jobs (dataset processing, evals, bulk generation) where end-to-end **throughput (tokens/sec, $/M tokens)** matters and per-request latency does not. Covers both **NVIDIA GPU servers** and **Apple Silicon**, with the underlying physics so the tool choices make sense rather than being cargo-culted.

## TL;DR

- Batch inference is a **throughput** problem, and throughput is mostly a **memory-bandwidth** problem, not a compute problem. The decode phase is memory-bound; your job is to keep the GPU/accelerator fed.
- The three levers that matter most for batch: **(1) quantization** (shrinks weights → more room for batch + less bandwidth per token), **(2) continuous batching** (keeps the device saturated), **(3) large batch size** (amortizes weight reads across many sequences). The ceiling on batch size is set by **KV-cache memory**, not weight memory.
- **NVIDIA, default pick:** `vLLM` (offline `LLM` class) — best balance of throughput, flexibility, fast startup. Step up to **SGLang** for prefix-heavy work (shared system prompts, RAG) or **TensorRT-LLM** if you can eat a long compile for a fixed model and want max throughput.
- **Apple Silicon, default pick:** `mlx-lm` (Apple's MLX) for best perf-per-watt on M-series; `llama.cpp` (Metal) for max model/format coverage; `Ollama` for convenience. Batch throughput on Mac is improving fast (`vllm-mlx` continuous batching) but still an order of magnitude below a datacenter GPU.

---

## 1. The mental model: offline batch ≠ interactive serving

| | Interactive serving | **Offline / batch (this note)** |
|---|---|---|
| Metric that matters | TTFT, inter-token latency (per user) | **Total throughput**, total wall-clock, **$/M tokens** |
| Batch size | Small, dictated by arrivals | **As large as memory allows** |
| Tolerable latency | Low (human in loop) | High (no human waiting) |
| Best framework knob | Low-latency scheduling | Max batch, max GPU saturation |

Because nobody is waiting, you trade latency for throughput aggressively: huge batches, longer queues, speculative decoding tuned for throughput, multi-GPU sharding for capacity rather than speed.

---

## 2. The physics: prefill vs decode, and why decode is the bottleneck

Every generation has two phases with **opposite** performance characteristics:

- **Prefill** — process the whole prompt in one pass. All prompt tokens go through as one big matmul → **high arithmetic intensity → compute-bound**. Scales well, uses the GPU's FLOPs.
- **Decode** — generate one token at a time. Each step must **read the entire model weights and the entire KV cache from memory** to produce *one* token → **low arithmetic intensity → memory-bandwidth-bound**.

**Arithmetic intensity** = FLOPs ÷ bytes moved from memory. The roofline model says: if your AI is below the hardware's "ridge point," you're memory-bound. Decode sits at roughly **60–80 ops/byte**, far below an H100's ridge (~hundreds of FLOP/byte), so during decode GPU compute utilization collapses to **20–40%**. ([roofline survey](https://arxiv.org/html/2402.16363v4), [TDS: prefill compute-bound, decode memory-bound](https://towardsdatascience.com/prefill-is-compute-bound-decode-is-memory-bound-why-your-gpu-shouldnt-do-both/))

**Consequence for batch:** the single most effective throughput lever is **batching the decode step**. One weight read amortized over N sequences turns a memory-bound, 30%-utilized kernel into something approaching compute-bound. This is *why* continuous batching exists and why batch jobs want the largest batch the memory budget allows.

---

## 3. Memory math: how to size the hardware

Three consumers of accelerator memory. You must fit all three.

### 3.1 Model weights
```
weight_bytes ≈ num_params × bytes_per_param
  FP16/BF16 = 2     INT8/FP8 = 1     INT4 = 0.5
```
- Llama-3 **8B**: 16 GB (FP16) / 8 GB (INT8) / **~4 GB (INT4)**
- Llama-3 **70B**: 140 GB (FP16) / 70 GB (INT8) / **~35 GB (INT4)**

Weights are read **once per decode step for the whole batch**, so quantizing weights directly cuts the per-token bandwidth bill — the dominant cost in decode.

### 3.2 KV cache — the real ceiling on batch size
```
kv_bytes = 2 × num_layers × num_kv_heads × head_dim × seq_len × bytes_per_elem × batch_size
           ^K and V
```
([formula source](https://www.morphllm.com/kv-cache-explained))

KV cache grows **linearly with both context length and batch size**, and it is *separate* from weights. Worked examples (FP16 KV, GQA models):

- **Llama-3 8B** (32 layers, 8 KV heads, head_dim 128): `2×32×8×128×2 = 128 KB/token` → **1 GB per 8K-token sequence**. A batch of 64 × 8K = **64 GB** of KV cache alone.
- **Llama-3 70B** (80 layers, 8 KV heads, head_dim 128): `2×80×8×128×2 = 320 KB/token` → **~2.5 GB per 8K sequence**. Batch of 32 × 8K = **~80 GB** of KV — more than the (INT4) weights.

**Takeaways:**
- At batch scale, **KV cache usually dominates**, not weights. Long-context batch jobs are KV-bound.
- Grouped-Query Attention (GQA, small `num_kv_heads`) is the reason modern models are batchable at all — it shrinks this term ~4–8×.
- KV cache can also be quantized (FP8/INT8 KV) to roughly halve this — a big lever for long-context batch.

### 3.3 Activations / overhead
Transient, smaller, but non-trivial at large batch × long sequence. Budget ~10–20% headroom. Frameworks also reserve a chunk of VRAM as the paged KV pool (vLLM `gpu_memory_utilization`, default ~0.9).

**Sizing rule of thumb:** `usable_VRAM − weights − overhead = KV budget`, then `max_batch ≈ KV_budget ÷ (bytes_per_token × seq_len)`.

---

## 4. Quantization (the highest-leverage knob for self-hosting)

Quantization shrinks weights (and optionally KV/activations), which simultaneously (a) lets the model fit on cheaper hardware, (b) enlarges the KV budget → bigger batches, (c) cuts per-token memory bandwidth → faster decode. The cost is accuracy, which varies by method and task.

| Format | Bits | Where it runs | Notes / when to use |
|---|---|---|---|
| **FP8** | 8 | NVIDIA Hopper/Blackwell (native) | Near-FP16 quality, **fastest on modern GPUs**; the default for high-end GPU batch jobs. |
| **AWQ** | 4 | GPU (Marlin kernels) | Activation-aware; protects salient weights → **holds up best on reasoning/coding** among 4-bit. ~**741 tok/s on H200** w/ Marlin. |
| **GPTQ** | 4 | GPU (Marlin kernels) | Calibration-based INT4; excellent tooling, ~**712 tok/s H200**; but can lose **~10 pp on coding** vs AWQ/GGUF. |
| **GGUF** (Q4_K_M etc.) | mixed 2–8 | CPU, **Apple Metal**, mixed CPU+GPU (llama.cpp) | Most universal/portable; great for Mac & CPU offload; Q4_K_M is the quality/size sweet spot. |
| **NVFP4 / MXFP4** | 4 | Blackwell (native FP4) | New micro-scaled FP4; strong throughput, but quality can drop (e.g. **80–92%** on hard non-English reasoning). |
| **INT8** | 8 | Broad | Safe, ~near-lossless; good when INT4 hurts quality. |
| **MLX 4-bit** | 4 | Apple Silicon | MLX-native quantization; the default for batchable models on Mac. |

Rules of thumb ([ai.rs](https://ai.rs/ai-developer/quantization-methods-compared), [Sesame Disk](https://sesamedisk.com/quantization-techniques-ai-inference-2026/), [VRLA](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/)):
- **GPU batch, modern card:** FP8 if you have the VRAM, else **AWQ-4bit** for the best accuracy-per-byte.
- **Quality-sensitive (code, math, non-English):** prefer AWQ or INT8 over GPTQ/FP4; verify on *your* eval, since degradation is task-specific.
- **Mac / CPU:** GGUF Q4_K_M (llama.cpp) or MLX 4-bit.
- Always **measure accuracy on your task** — generic benchmark deltas don't transfer to your data.

---

## 5. Batching internals (what the frameworks actually do)

- **Static batching** (naive): wait, group N requests, run them lockstep until the *longest* finishes. Short sequences idle while long ones run → poor utilization. Avoid.
- **Continuous / in-flight batching**: the scheduler adds and evicts sequences from the running batch **every step**, so a finished sequence is immediately replaced. This is the single biggest GPU-utilization unlock and is table-stakes in 2026. ([Runpod](https://www.runpod.io/articles/guides/vllm-pagedattention-continuous-batching), [TianPan](https://tianpan.co/blog/2026-04-09-continuous-batching-llm-inference))
- **PagedAttention** (vLLM): manage KV cache like OS virtual memory — fixed-size "pages" instead of one contiguous buffer per sequence. Near-zero fragmentation, KV sharing for common prefixes, **up to ~90% less KV overhead**; enabled vLLM's original ~24× over HF Transformers / ~3.5× over TGI. ([vLLM](https://docs.vllm.ai/en/latest/), [futureagi](https://futureagi.com/blog/what-is-vllm-2026/))
- **RadixAttention** (SGLang): a radix tree over KV pages for **automatic prefix sharing/reuse** across requests — why SGLang wins on shared-system-prompt / RAG / multi-turn workloads (up to **6.4×** on prefix-heavy). ([particula](https://particula.tech/blog/sglang-vs-vllm-inference-engine-comparison))
- **Chunked prefill (Sarathi)**: split a long prefill into chunks and **piggyback decode steps** into the same batch so the compute-bound prefill fills the bubbles left by memory-bound decode. Up to **10× decode throughput**, ~**1.33× e2e**. ([Sarathi paper](https://arxiv.org/pdf/2308.16369))
- **Prefill/decode disaggregation**: run prefill and decode on *separate* GPU pools since they have opposite bottlenecks — a 2025–2026 production pattern for large fleets. ([O'Reilly ch.17](https://www.oreilly.com/library/view/ai-systems-performance/9798341627772/ch17.html))

---

## 6. Throughput techniques for batch jobs

- **Max out batch size** — the #1 lever (Section 2). Push `gpu_memory_utilization` high and batch until KV-bound.
- **Speculative decoding**: a small draft model proposes K tokens, the target verifies them in one pass → **2–3×** in many cases; now built into vLLM, SGLang, TensorRT-LLM, and `mlx-lm` (0.21+). Historically thought to lose its edge at very large batch (compute-bound regime), but recent findings show large-batch decode is *still memory-bound*, so it can help there too — tune draft length `γ` (high `γ` can spike latency at 40+ concurrency). ([premai](https://blog.premai.io/speculative-decoding-2-3x-faster-llm-inference-2026/), [openreview](https://openreview.net/forum?id=59OJOgKLzN))
- **Multi-GPU parallelism** (for models too big for one card, or to raise aggregate throughput):
  - **Tensor parallelism (TP):** shard each layer across GPUs; needs fast interconnect (NVLink). Lowers per-step latency, enables bigger models/batches. TP=2/4 pairs well with speculative decoding.
  - **Pipeline parallelism (PP):** split layers across GPUs/stages; better over slower interconnect, adds pipeline bubbles.
  - **Expert parallelism (EP):** shard MoE experts — relevant for MoE models (e.g. A3B-style).
- **KV-cache quantization (FP8/INT8 KV):** roughly doubles batch/context headroom for long-context jobs.

---

## 7. The tool landscape

### NVIDIA / server-class GPUs

| Engine | Best for | Throughput (Llama-8B, H100, ballpark) | Tradeoff |
|---|---|---|---|
| **vLLM** (offline `LLM` class) | **Default**; flexible, fast start, broad model/quant support | ~12,500 tok/s | Slightly behind on raw peak; easiest to operate. |
| **SGLang** | Prefix-heavy (RAG, shared prompts, multi-turn), high concurrency | ~16,200 tok/s (**~29% > vLLM**) | RadixAttention shines on shared prefixes; otherwise comparable. |
| **TensorRT-LLM** | Fixed model, max throughput/latency | ~15–30% > vLLM in some cases; best single-request | **~28-min compile**, less flexible, NVIDIA-only. |
| **TGI** (HF) | HF-ecosystem serving | baseline-ish | Fine, generally trails the above on peak throughput. |

Sources: [Spheron H100 bench](https://www.spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks/), [particula](https://particula.tech/blog/sglang-vs-vllm-inference-engine-comparison), [yottalabs](https://www.yottalabs.ai/post/best-llm-inference-engines-in-2026-vllm-tensorrt-llm-tgi-and-sglang-compared).

**For pure offline batch:** vLLM's offline `LLM.generate()` over a list of prompts is the path of least resistance — it applies continuous batching + PagedAttention internally and saturates the GPU without you running a server. ([vLLM offline inference docs](https://docs.vllm.ai/en/latest/serving/offline_inference/))

### Apple Silicon (M-series)

| Engine | Best for | Notes |
|---|---|---|
| **MLX / `mlx-lm`** | Best perf-per-watt on M-series; the native stack | Apple's framework; unified memory; spec-decoding in 0.21+. M5 is hardware-tuned for MLX (Qwen3-14B-4bit: **TTFT 4.06×**, gen **1.19×** vs M4). |
| **llama.cpp (Metal)** | Max model/format coverage (GGUF), CPU+GPU offload | Most portable; slightly slower than MLX on equivalent models. |
| **Ollama** | Convenience / one-command runs | Wraps llama.cpp/MLX; MLX backend lifts e.g. Qwen-3.5-35B-A3B from ~45 → **70–80 tok/s** on M4 Max 32GB. |
| **vllm-mlx** | Emerging **batch** on Mac | Continuous batching: **3.4×** at 5 concurrent, **4.3×** at 16; up to **525 tok/s** (Qwen3-0.6B, M4 Max). |

Sources: [contracollective](https://contracollective.com/blog/llama-cpp-vs-mlx-ollama-vllm-apple-silicon-2026), [yage.ai](https://yage.ai/share/mlx-apple-silicon-en-20260331.html), [willitrunai](https://willitrunai.com/blog/mlx-vs-ollama-apple-silicon-benchmarks), [Apple M5/MLX research (arXiv)](https://arxiv.org/pdf/2601.19139).

**Reality check for batch on Mac:** unified memory lets a Mac *hold* large models cheaply, but memory **bandwidth** (and lack of datacenter-class batching maturity) caps throughput. A Mac is excellent for *fitting* a 70B model and for low-volume/private batch; a single H100/H200 will still crush it on $/M-token at scale. (Your `ane-inference` project is the relevant local angle — ANE adds another execution path worth tracking.)

---

## 8. Decision guide for an offline batch job

1. **Pick the model & quant** to fit your accelerator with KV headroom (Sections 3–4). Start at INT4/AWQ (GPU) or 4-bit MLX/GGUF (Mac); fall back to INT8/FP8 if quality regresses on your eval.
2. **Pick the engine:** vLLM offline by default → SGLang if prefixes are shared → TensorRT-LLM if model is fixed and you need peak. On Mac: mlx-lm (or llama.cpp for coverage).
3. **Maximize batch:** raise `gpu_memory_utilization`, batch until KV-bound; quantize KV cache for long contexts.
4. **Add speculative decoding** with a small draft model if the target is large and the task is predictable (summarization/translation/code).
5. **Scale out** with TP (fast interconnect) or PP (slower links) only when one device can't hold it or you need more aggregate throughput.
6. **Measure** tok/s and $/M-token end-to-end, and accuracy on a held-out slice — not vendor benchmarks.

---

## 9. Worked sizing example (sanity-check the math)

*Job: summarize 1M docs, ~4K-token prompts, ~512-token outputs, Llama-3 70B, single 80 GB H100.*
- Weights INT4: ~35 GB → leaves ~37 GB (at 0.9 util) for KV + overhead.
- KV per 4.5K-token sequence (70B): ~1.4 GB (FP16 KV) → **~26 concurrent sequences**; ~52 with FP8 KV.
- Decode is memory-bound → throughput rises ~linearly with batch until ~compute-bound or KV-full.
- Verdict: 70B INT4 on one H100 is feasible but KV-tight; FP8-KV or a second GPU (TP=2) materially raises the achievable batch and throughput. Spec-decoding likely +2× on this summarization-style workload.

---

## 10. Open questions / next research

- Concrete vLLM offline-batch recipe + measured tok/s on the hardware we actually have.
- FP8 vs AWQ-INT4 accuracy on *our* tasks (run a small eval).
- ANE (`ane-inference`) as a third execution path on Mac — where does it beat MLX/Metal for batch?
- Prefill/decode disaggregation: worth it below fleet scale?
- KV-cache quantization quality impact at long context.

---

## Sources

- vLLM: [docs](https://docs.vllm.ai/en/latest/), [offline inference](https://docs.vllm.ai/en/latest/serving/offline_inference/), [what is vLLM 2026](https://futureagi.com/blog/what-is-vllm-2026/), [PagedAttention/continuous batching](https://www.runpod.io/articles/guides/vllm-pagedattention-continuous-batching)
- Engine comparisons: [Spheron H100 benchmarks](https://www.spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks/), [SGLang vs vLLM (particula)](https://particula.tech/blog/sglang-vs-vllm-inference-engine-comparison), [best engines 2026 (Yotta)](https://www.yottalabs.ai/post/best-llm-inference-engines-in-2026-vllm-tensorrt-llm-tgi-and-sglang-compared)
- Apple Silicon: [MLX inference engine (yage.ai)](https://yage.ai/share/mlx-apple-silicon-en-20260331.html), [llama.cpp vs MLX vs Ollama vs vLLM](https://contracollective.com/blog/llama-cpp-vs-mlx-ollama-vllm-apple-silicon-2026), [MLX vs Ollama benchmarks](https://willitrunai.com/blog/mlx-vs-ollama-apple-silicon-benchmarks), [native LLM inference at scale on Apple Silicon (arXiv)](https://arxiv.org/pdf/2601.19139)
- Quantization: [methods compared (ai.rs)](https://ai.rs/ai-developer/quantization-methods-compared), [techniques 2026 (Sesame Disk)](https://sesamedisk.com/quantization-techniques-ai-inference-2026/), [INT4/INT8/FP8/AWQ/GPTQ (VRLA)](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/)
- Memory/roofline: [roofline survey (arXiv)](https://arxiv.org/html/2402.16363v4), [prefill vs decode (TDS)](https://towardsdatascience.com/prefill-is-compute-bound-decode-is-memory-bound-why-your-gpu-shouldnt-do-both/), [KV cache explained (Morph)](https://www.morphllm.com/kv-cache-explained), [memory wall (Spheron)](https://www.spheron.network/blog/ai-memory-wall-inference-latency-guide-2026/)
- Batching/decoding: [Sarathi chunked prefill (arXiv)](https://arxiv.org/pdf/2308.16369), [speculative decoding 2026 (premai)](https://blog.premai.io/speculative-decoding-2-3x-faster-llm-inference-2026/), [high-throughput spec decoding (OpenReview)](https://openreview.net/forum?id=59OJOgKLzN), [continuous batching (TianPan)](https://tianpan.co/blog/2026-04-09-continuous-batching-llm-inference)
