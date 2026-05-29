---
title: "Runbook: offline/batch LLM inference on an NVIDIA B200 (Blackwell)"
date: 2026-05-29
tags: [llm, inference, offline, batch, b200, blackwell, nvidia, vllm, nvfp4, docker, runbook]
status: published
scope: "Step-by-step, Docker-first runbook for running offline/batch inference on a B200 GPU. Companion to the offline-inference deep-dive note."
related: ["2026-05-29-offline-inference-running-llms-yourself.md"]
---

# Runbook: offline/batch inference on a B200 (Blackwell)

> Companion to [[2026-05-29-offline-inference-running-llms-yourself]]. Goal: take a list of prompts and run them through a model on a B200 at high throughput, the simplest reliable way. **Docker-first** (also the recommended Blackwell path, since NVIDIA's container ships the PyTorch build that supports the B200).

> **On the numbers & citations:** every spec and quantitative claim carries an inline `[n]` citation to [References](#references). Vendor/third-party figures are 2026 results that vary by model, build, and configuration — verify on your own box. Memory-sizing figures marked *(computed)* are arithmetic from the formula in the companion note, not measured.

## Why the B200 changes the recipe

| Spec | B200 | Why it matters for batch |
|---|---|---|
| Memory | **192 GB HBM3e** [1][3] | Fit a 70B in FP4 with a *huge* KV budget → very large batches. |
| Bandwidth | **8.0 TB/s** [1][3] | Decode is memory-bound; ~2.4× H100's bandwidth → ~that much more decode throughput. |
| Tensor cores | 5th-gen, **native FP4** [1] | NVFP4 runs at ~2× FP8 throughput; up to **18 PFLOPS sparse FP4** [1]. |
| Interconnect | NVLink 5, **1.8 TB/s/GPU** [1] | Cheap tensor-parallel scaling across 8 GPUs in an HGX/DGX B200 [4]. |
| Net result | ~**4× H100** inference throughput; ~**17,500 tok/s** on Llama-70B [1] | |

**The B200-specific lever is NVFP4** — a 4-bit float with native hardware support; use it [2].

## Hard requirements (the part people get wrong)

- **Compute capability `sm_100`** (B200/GB200; B300 is `sm_103`) [5].
- **CUDA ≥ 12.8** and a recent driver (R570+) — Blackwell will not run on older CUDA [5][6].
- **Use NVIDIA's PyTorch build** (in the NGC/vLLM containers). Stock PyPI PyTorch often lacks `sm_100` kernels → `CUDA error: no kernel image is available for execution on the device`. **Do not `pip install` a different torch over the container's build** [7].
- vLLM ≥ v0.20 uses **FlashAttention 4** by default on Blackwell (SM100/SM103) — good, leave it [5].

---

## Step-by-step

### Step 0 — Get a B200 host
B200s are datacenter cards; you rent them (single-GPU instances on RunPod/Lambda/Jarvis/CoreWeave, or a full 8×GPU HGX/DGX B200 [4]). One B200 = 192 GB [1], which already covers most single-model batch jobs. SSH in.

### Step 1 — Verify GPU, driver, CUDA
```bash
nvidia-smi                      # confirm "B200", note driver version
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
```
Driver must support CUDA ≥ 12.8 (R570+); if not, fix the host image before anything else [5][6].

### Step 2 — Verify Docker + NVIDIA Container Toolkit
Most B200 cloud images ship these. Confirm the GPU is visible *inside* a container:
```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```
If this fails, install the NVIDIA Container Toolkit and restart Docker.

### Step 3 — Pick model + quantization
Default for the B200: a **pre-quantized NVFP4** checkpoint (e.g. `nvidia/Llama-3.1-8B-Instruct-NVFP4`, or a 70B NVFP4 build); vLLM auto-detects the quantization from the checkpoint config [9][11].
- **Max quality / simplicity:** FP8 (a 70B FP8 ≈ 70 GB fits easily in 192 GB [1]).
- **Max throughput:** NVFP4 *(computed: 70B ≈ ~35 GB → ~138 GB left for KV → enormous batches)* [2].
Sizing math is in the companion note (§3). On 192 GB you are rarely weight-bound; you're KV-bound — exactly the regime where the B200's memory shines.

### Step 4 — Offline batch inference (primary path)
Feed a *list* of prompts to vLLM's offline `LLM` class — it applies continuous batching + PagedAttention internally and saturates the GPU [8].

`batch_infer.py`:
```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="nvidia/Llama-3.1-8B-Instruct-NVFP4",  # NVFP4 auto-detected
    gpu_memory_utilization=0.90,
    max_model_len=8192,
    # tensor_parallel_size=1,   # raise for multi-GPU (Step 6)
)
sp = SamplingParams(temperature=0.0, max_tokens=512)

prompts = [line.strip() for line in open("prompts.txt")]  # your batch
for out in llm.generate(prompts, sp):                      # batched automatically
    print(out.prompt[:40], "->", out.outputs[0].text[:80])
```

Run it inside the vLLM container (mount HF cache + workdir) [8]:
```bash
docker run --rm --gpus all --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v "$PWD":/work -w /work \
  --entrypoint python3 \
  vllm/vllm-openai:latest batch_infer.py
```
- `--ipc=host` (or `--shm-size=16g`) is required — vLLM shares memory across processes [8].
- For an MoE NVFP4 model, add `-e VLLM_USE_FLASHINFER_MOE_FP4=1` [9].

### Step 5 — (Alternative) OpenAI-compatible server
To stream a batch through an API instead [8]:
```bash
docker run -d --name vllm --gpus all --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model nvidia/Llama-3.1-8B-Instruct-NVFP4 \
  --gpu-memory-utilization 0.90 --max-model-len 8192
```
Then POST prompts concurrently to `http://localhost:8000/v1/completions`; continuous batching merges them. For pure offline jobs, Step 4 is simpler and faster.

### Step 6 — Scale across multiple B200s (only if needed)
For a model too big for 192 GB, or to raise aggregate throughput, use tensor parallelism over NVLink 5 [1]:
```bash
# in the LLM(...) call:  tensor_parallel_size=4
# server flag:           --tensor-parallel-size 4   (+ --shm-size=16g)
```
TP shards each layer; NVLink 5 (1.8 TB/s) [1] makes this efficient on an 8×B200 box. Use pipeline parallelism only across slower links/nodes.

### Step 7 — Maximize throughput
- Push batch size until **KV-bound** (raise `gpu_memory_utilization`, lengthen the prompt list).
- Quantize the **KV cache** (FP8 KV) for long-context jobs to roughly double the batch.
- Add **speculative decoding** with a small draft model for predictable tasks (summarization/translation/code).
- Measure **tok/s and $/M-token** end-to-end, plus accuracy on a held-out slice. Don't trust vendor numbers.

### Step 8 — (Peak performance) TensorRT-LLM + ModelOpt
For the absolute max on a *fixed* dense NVFP4 model, the production path is **NVIDIA ModelOpt (PTQ calibration) → TensorRT-LLM engine build → serve**: TensorRT-LLM (1.x) has native FP4 and the best Blackwell throughput, at the cost of a long per-model compile and less flexibility [9][10]. Worth it for stable, high-volume pipelines; overkill for one-off batch jobs.

---

## Worked sizing on one B200 (192 GB)
*70B NVFP4, 4K-token prompts — all figures (computed) from the companion-note formula:*
- Weights ≈ 35 GB → ~138 GB usable for KV at 0.9 util.
- 70B KV ≈ 320 KB/token (FP16 KV) → ~430K tokens of KV → e.g. **batch ≈ 100 sequences × 4K tokens**, or far more with FP8 KV.
- Decode throughput scales ~linearly with batch until compute-bound — the B200's 8 TB/s + FP4 cores keep that ceiling very high [1].

## Gotchas
- Stock PyPI PyTorch over the container build → "no kernel image" error; keep the container's torch [7].
- Forgetting `--ipc=host`/`--shm-size` → crashes/hangs under tensor parallel [8].
- NVFP4 accuracy is task-specific (can dip on hard reasoning / non-English); validate on your data and fall back to FP8 if it regresses [2].

## References
1. *NVIDIA B200 Specs and Benchmarks: 192GB HBM3e, FP4, 17,500 tok/s on Llama 70B (2026)* (Spheron) — https://www.spheron.network/blog/nvidia-b200-complete-guide/
2. *FP4 Quantization on Blackwell GPUs: Throughput, Cost, and When It's Worth It* (Spheron) — https://www.spheron.network/blog/fp4-quantization-blackwell-gpu-cost/
3. *NVIDIA B200 Specs — 192GB Blackwell Datacenter GPU* (Jarvis Labs) — https://jarvislabs.ai/gpu/nvidia-b200
4. *DGX B200* (NVIDIA) — https://www.nvidia.com/en-us/data-center/dgx-b200/
5. vLLM GPU installation docs — https://docs.vllm.ai/en/stable/getting_started/installation/gpu/
6. *NVIDIA Blackwell Tuning Guide* (NVIDIA docs) — https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html
7. *Fix for vLLM Incompatibility on B200: Keep NVIDIA PyTorch, Use Newer vLLM* (Medium, I. Selmi) — https://medium.com/@imen.selmi/fix-for-vllm-incompatibility-on-b200-keep-nvidia-pytorch-use-newer-vllm-8e0b3a0a4d16
8. vLLM Docker deployment docs — https://docs.vllm.ai/en/stable/deployment/docker/
9. *Accelerating Large Language Models with NVFP4 Quantization* (Red Hat Developer) — https://developers.redhat.com/articles/2026/02/04/accelerating-large-language-models-nvfp4-quantization
10. *NVIDIA Model-Optimizer* (GitHub) — https://github.com/NVIDIA/Model-Optimizer
11. *fp4 Quantization with NVFP4* (vLLM llm-compressor docs) — https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w4a4_fp4/
