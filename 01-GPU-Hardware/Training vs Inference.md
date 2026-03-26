---
tags: [nca-aiio, nca-genl, nvidia/gpu-hardware, nvidia/llm-genai, fundamentals]
aliases: [Training vs Inference, Training, Inference, Forward pass, Backpropagation, Model serving]
---

# Training vs Inference

> **Domain**: GPU Hardware / LLM Foundations
> **Cert Relevance**: Both — NCA-AIIO (AI Foundations 38%) + NCA-GENL (LLM Foundations ~25%)
> **Related**: [[GPU Architecture]], [[NVIDIA DGX Systems]], [[CUDA]], [[NVIDIA NIM]], [[Kubernetes for AI]], [[RAG Architecture]], [[LLM Evaluation Metrics]]

## Overview

Every AI model's life has exactly two phases: **training**, where the model learns by adjusting its weights across billions of examples, and **inference**, where the trained model applies what it learned to generate responses on new inputs. These two phases have radically different computational profiles — different GPU requirements, memory footprints, precision formats, latency constraints, and infrastructure patterns. Understanding this distinction is the mental model that makes everything else in the NVIDIA stack make sense.

---

## Training — Building the Model

Training teaches a neural network by iteratively adjusting its weights to minimise prediction error across a large dataset.

### The Training Loop

```
For each batch of training data:
  1. Forward pass  — input flows through the network → prediction generated
  2. Loss calculation — measure how wrong the prediction was
  3. Backward pass (backpropagation) — calculate each weight's contribution to the error
  4. Weight update (gradient descent) — nudge weights to reduce error
  5. Repeat — billions of times across the entire dataset
```

This loop runs continuously for **weeks to months** on thousands of GPUs for large models.

### Memory Requirements During Training

Training requires storing far more than just the model weights:

| Component | Memory Per Parameter | Notes |
|---|---|---|
| **Weights** (FP16) | 2 bytes | The model itself |
| **Gradients** | 2 bytes | Direction of weight updates |
| **Optimiser state** (Adam) | 8 bytes | Momentum + variance per weight |
| **Activations** | Variable | Intermediate layer outputs |
| **Total (approx)** | **~16–18 bytes** | Per parameter |

For a 70B parameter model: 70B × 16 bytes ≈ **~1.1 TB** — requires multi-node training on [[NVIDIA DGX Systems]].

### Training Infrastructure

| Requirement | Why |
|---|---|
| Maximum VRAM | Weights + gradients + optimiser state all in GPU memory |
| High compute throughput | Billions of weight updates |
| Fast inter-GPU comms | Gradient synchronisation via all-reduce across GPUs |
| [[NVLink and NVSwitch]] | Intra-node gradient sync at 900 GB/s |
| [[InfiniBand]] | Inter-node gradient sync at 400 Gb/s |
| Batch workload | Finite job — runs to completion |

**Preferred hardware**: A100, H100 — maximum VRAM and Tensor Core throughput
**K8s resource type**: `PyTorchJob` / `MPIJob` — finite batch job

---

## Inference — Using the Model

Inference applies a trained, frozen model to new inputs to generate predictions or outputs.

### What Happens During Inference

```
Per request:
  1. Input tokenised and embedded
  2. Forward pass only — data flows through the network once
  3. Output generated — token by token (for LLMs) or single prediction
  4. Response returned to user
```

No backpropagation. No weight updates. Weights are **frozen** — read-only.

### Memory Requirements During Inference

Only the model weights are needed — no gradients, no optimiser state:

| Component | Memory Per Parameter | Notes |
|---|---|---|
| **Weights** (FP16) | 2 bytes | Full precision |
| **Weights** (INT8) | 1 byte | Quantised — half the memory |
| **Weights** (FP8) | 0.5 bytes | H100 Transformer Engine |
| **KV Cache** | Variable | Key/value cache for attention — grows with context length |

For a 70B model at FP16: ~140 GB — fits on 2× H100 80GB
For a 70B model at INT8: ~70 GB — fits on 1× H100 80GB

### Inference Infrastructure

| Requirement | Why |
|---|---|
| Low latency | Users are waiting for responses |
| High throughput | Serve many simultaneous requests |
| VRAM for model weights | Entire model must fit in GPU memory |
| Dynamic batching | Group multiple requests for GPU efficiency |
| Horizontal scaling | Add more inference pods as demand grows |

**Preferred hardware**: L4 (edge), T4, L40S, H100 — depending on model size and latency needs
**K8s resource type**: `Deployment` + `HPA` — long-running service, scales with demand

---

## Head-to-Head Comparison

| Dimension | Training | Inference |
|---|---|---|
| **Compute** | Forward + backward pass | Forward pass only |
| **Frequency** | Once (or for fine-tuning) | Per user request |
| **Duration** | Weeks to months | Milliseconds |
| **GPU** | A100, H100 — max compute | L4, T4, L40S — cost-optimised |
| **Memory per param** | ~16 bytes | ~1–2 bytes |
| **Precision** | FP16 / BF16 mixed | INT8 / FP8 quantised |
| **Latency req** | Not time-sensitive | Critical (< 1s target) |
| **Interconnect need** | Critical — gradient sync | Low — stateless requests |
| **Parallelism** | Tensor + pipeline + data | Tensor + request batching |
| **K8s pattern** | Finite Job | Long-running Deployment + HPA |
| **Failure mode** | Job restarts from checkpoint | Pod replaced, request retried |

---

## The Pipeline: Training → Inference

```
Training cluster (DGX SuperPOD, A100/H100)
        ↓
Trained weights saved as checkpoint
        ↓
Optional: Fine-tuning (LoRA, PEFT) on domain-specific data   → [[Fine-Tuning]]
        ↓
Optional: Quantisation (FP16 → INT8/FP8) for efficiency
        ↓
Model packaged for serving
        ↓ Triton Inference Server / [[NVIDIA NIM]] / TensorRT-LLM
        ↓
Inference serving (L40S / H100 data center, L4 edge)
        ↓
User responses via REST API
```

---

## Precision Formats: Training vs Inference

Precision (the number of bits used to represent each number) affects both accuracy and memory/compute requirements:

| Precision | Bits | Typical Use | Notes |
|---|---|---|---|
| FP32 | 32 | Training baseline | Most accurate, most memory |
| TF32 | 19 | Ampere+ training default | Near-FP32 accuracy, ~8× faster |
| FP16 / BF16 | 16 | Mixed-precision training | Standard modern training approach |
| FP8 | 8 | H100 inference / training | Transformer Engine — LLM-optimised |
| INT8 | 8 | Inference quantisation | Fastest inference, smallest memory footprint |
| INT4 | 4 | Extreme quantisation | Maximum compression, some accuracy loss |

**Mixed-precision training**: uses FP16/BF16 for most operations (Tensor Core-optimised) while keeping a master copy of weights in FP32 (accurate). This is the standard approach for A100/H100 training.

**Quantisation for inference**: converting a trained FP16 model to INT8 halves memory requirements and increases throughput — without retraining the model. The accuracy trade-off is usually acceptable for inference.

> [!warning]
> Quantisation is an **inference optimisation** — you quantise after training, not during it. Training in INT8 from scratch typically degrades model quality significantly. The workflow is: train in FP16/BF16, then quantise to INT8/FP8 for deployment.

---

## KV Cache — The Inference-Specific Memory Challenge

During LLM inference, the attention mechanism needs to reference all previous tokens' key and value vectors. These are cached in **KV (Key-Value) cache** — which grows with context length:

```
KV cache size ≈ 2 × num_layers × num_heads × head_dim × context_length × batch_size × bytes_per_element
```

For long contexts (32K, 128K tokens) with large batch sizes, KV cache can consume more GPU memory than the model weights themselves. This is why large-context inference requires careful memory management — and why H100 with 80 GB HBM3 is preferred over smaller GPUs for high-throughput LLM serving.

---

> [!note] Cybersecurity Connection
> Training vs inference maps directly to **offline vs online security processing**. Training is like building a threat intelligence database or training a detection model offline — expensive, slow, done once or periodically, produces a static artefact (model weights / rule set). Inference is like running that detection model against live traffic — fast, repeated, latency-sensitive, happens millions of times per day. You size your offline analysis infrastructure (SIEM correlation engine, malware sandbox) differently from your real-time enforcement infrastructure (firewall, IDS/IPS in-line). Same compute-profile reasoning applies: different tools, different hardware, different failure tolerances.

---

## Related Notes

- [[GPU Architecture]] – Tensor Cores and HBM memory are the hardware that makes both phases efficient
- [[NVIDIA DGX Systems]] – the training platform; DGX SuperPOD for large training runs
- [[CUDA]] – the programming model that executes both training and inference compute on GPU
- [[NVIDIA NIM]] – packages trained models as inference microservices with OpenAI-compatible API
- [[Kubernetes for AI]] – training = PyTorchJob; inference = Deployment + HPA
- [[RAG Architecture]] – a RAG pipeline is inference-only; retrieval augments inference, not training
- [[LLM Evaluation Metrics]] – metrics like Truthfulness Score are measured at inference time
- [[InfiniBand]] – critical for training (gradient sync); less critical for stateless inference
- [[Multi-Instance GPU]] – MIG partitions a GPU for shared inference serving across tenants

---

## Key Mental Model

Training is **building the weapon** — slow, expensive, done in a controlled environment, produces a static artefact you then carry into the field. Inference is **firing the weapon** — fast, repeated, must work under pressure in real-time. You wouldn't design a weapons factory the same way you design a soldier's kit. Same principle: design your training cluster (DGX SuperPOD, A100/H100, InfiniBand) for throughput and scale; design your inference cluster (L4, T4, Kubernetes HPA) for latency and cost per request.

> [!tip] Exam Tip
> Two distinctions are tested on both NCA-AIIO and NCA-GENL: **(1) training requires backpropagation and weight updates; inference is forward-pass only with frozen weights** — this drives every hardware and infrastructure difference; **(2) training memory = ~16 bytes/param (weights + gradients + optimiser); inference memory = ~1–2 bytes/param (weights only)**. If a scenario mentions selecting a GPU for inference cost-optimisation, the answer is L4 or T4, not H100.
