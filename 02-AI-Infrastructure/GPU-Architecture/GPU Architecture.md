---
tags: [nca-aiio, nca-aiio/infrastructure, gpu, hardware]
aliases: [GPU Architecture, NVIDIA GPU Internals, SM, Streaming Multiprocessor, Tensor Core, CUDA Core]
---

# GPU Architecture

> **Exam Domain**: AI Infrastructure (40%)
> **Related**: [[CUDA]], [[GPU vs CPU]], [[Multi-Instance GPU]], [[NVLink and NVSwitch]], [[NVIDIA DGX Systems]], [[DCGM]], [[Training vs Inference]]

## Overview

A GPU (Graphics Processing Unit) is a processor built for massively parallel computation. Where a CPU has a small number of powerful, general-purpose cores optimised for sequential logic, a GPU contains thousands of smaller cores designed to execute the same operation across huge datasets simultaneously. NVIDIA GPUs are the dominant compute platform for AI because the core math of deep learning — matrix multiplication — maps perfectly onto this parallel architecture. Understanding GPU internals is essential for sizing, managing, and troubleshooting AI infrastructure.

---

## CPU vs GPU: The Core Distinction

| Characteristic | CPU | GPU |
|---|---|---|
| Core count | Few (8–128) | Thousands |
| Core design | Complex, general-purpose | Simple, specialised |
| Optimised for | Sequential, branchy logic | Parallel, repetitive math |
| Memory | System RAM (DDR) | VRAM / HBM (on-card) |
| AI role | Orchestration, pre/post-processing | Training and inference compute |

> [!note]
> CPUs and GPUs work together — the CPU manages the overall program flow and feeds work to the GPU. Neither replaces the other in an AI system.

---

## NVIDIA GPU Compute Hierarchy

NVIDIA GPUs are structured as a hierarchy from the chip level down to individual threads:

```
GPU Chip
└── Streaming Multiprocessors (SMs)   ← many per GPU (e.g. H100: 132 SMs)
    ├── CUDA Cores                     ← general float/int arithmetic
    ├── Tensor Cores                   ← specialised matrix math for AI
    ├── Shared Memory / L1 Cache       ← fast on-chip memory per SM
    └── Warps (groups of 32 threads)   ← basic scheduling unit
```

### Streaming Multiprocessor (SM)

The SM is the fundamental compute unit. Each SM is a self-contained mini-processor with its own:
- CUDA Cores and Tensor Cores
- Registers (private per-thread scratch space)
- Shared Memory (fast, shared across threads in a block)
- Warp schedulers (issue instructions to groups of 32 threads)

The total number of SMs determines the GPU's raw parallelism. A100: 108 SMs. H100: 132 SMs.

### CUDA Cores

General-purpose arithmetic units — handle standard floating-point (FP32, FP64) and integer operations. Every [[CUDA]] kernel runs on CUDA Cores.

### Tensor Cores

Specialised matrix multiply-accumulate (MMA) units introduced with the Volta architecture. They perform matrix operations in a single clock cycle that would take many cycles on standard CUDA Cores. **Tensor Cores are the reason modern NVIDIA GPUs are so dominant for AI training and inference.**

| Generation | Architecture | Key GPUs |
|---|---|---|
| 1st gen | Volta | V100 |
| 2nd gen | Turing | T4 |
| 3rd gen | Ampere | A100 |
| 4th gen | Hopper | H100 |
| 5th gen | Blackwell | B100, B200 |

### Warp

A warp is a group of **32 threads** that execute the same instruction at the same time (SIMT — Single Instruction, Multiple Threads). This is the GPU scheduler's atomic unit. If threads in a warp diverge (e.g. an if/else branch), performance degrades — this is called **warp divergence**.

> [!warning]
> Warp divergence is a common performance trap. When threads in a warp take different code paths, the GPU must serialise them — eliminating the parallelism benefit. Well-written GPU code minimises branching.

---

## GPU Memory Hierarchy

GPUs have a multi-level memory hierarchy, entirely separate from system RAM:

| Level | Scope | Speed | Capacity | Notes |
|---|---|---|---|---|
| **Registers** | Per-thread | Fastest | ~255 per thread | Private to each thread |
| **Shared Memory / L1** | Per-SM (on-chip) | Very fast | 48–228 KB | Shared within a thread block |
| **L2 Cache** | Whole GPU (on-chip) | Fast | 6–50 MB | Shared across all SMs |
| **HBM / GDDR (VRAM)** | Whole GPU (on-card) | High bandwidth | 16–80+ GB | Main device memory — model weights live here |
| **System RAM** | Host (CPU side) | Slow for GPU | Hundreds of GB | Must cross PCIe bus to reach GPU |

### HBM — High Bandwidth Memory

Data center GPUs (A100, H100) use **HBM (High Bandwidth Memory)** instead of GDDR. HBM is stacked directly onto the GPU package, delivering dramatically higher memory bandwidth:

| GPU | Memory Type | Bandwidth |
|---|---|---|
| H100 SXM | HBM3 | ~3.35 TB/s |
| A100 SXM | HBM2e | ~2.0 TB/s |
| RTX 4090 (consumer) | GDDR6X | ~1.0 TB/s |

High memory bandwidth is critical for large model [[Training vs Inference]] — model weights must be fed to Tensor Cores faster than they can compute.

> [!note] Cybersecurity Connection
> GPU memory isolation maps directly to multi-tenant security. Each SM's shared memory is inaccessible to threads running on other SMs — hardware-enforced isolation analogous to VM memory separation. [[Multi-Instance GPU]] (MIG) extends this to full hardware partitioning: separate memory, compute, and cache per instance — the GPU equivalent of dedicated tenants on shared infrastructure.

---

## Key NVIDIA GPU Generations

| Architecture | Year | Key GPUs | Landmark Feature |
|---|---|---|---|
| **Volta** | 2017 | V100 | First Tensor Cores — dedicated AI math |
| **Turing** | 2018 | T4, RTX 20xx | 2nd-gen Tensor Cores, RT Cores |
| **Ampere** | 2020 | A100, RTX 30xx | 3rd-gen Tensor Cores, [[Multi-Instance GPU]], NVLink 3.0 |
| **Hopper** | 2022 | H100 | 4th-gen Tensor Cores, Transformer Engine, NVLink 4.0 |
| **Blackwell** | 2024 | B100, B200, GB200 | 5th-gen Tensor Cores, NVLink 5.0, 192 GB HBM3e |

> [!tip] Exam Tip
> The **A100** and **H100** are the most exam-relevant GPUs. Know that: A100 introduced MIG; H100 introduced the Transformer Engine (FP8 precision for LLMs); both use HBM and are the foundation of [[NVIDIA DGX Systems]].

---

## Precision Formats and AI Workloads

GPUs support multiple numeric precision levels. Lower precision = faster compute + less memory, but less accuracy. The exam tests awareness of these trade-offs:

| Precision | Bits | Use Case |
|---|---|---|
| FP64 | 64-bit | Scientific computing, HPC |
| FP32 | 32-bit | Standard training baseline |
| TF32 | 19-bit | Ampere default for training — faster with minimal accuracy loss |
| FP16 / BF16 | 16-bit | Mixed-precision training — standard modern approach |
| FP8 | 8-bit | H100 Transformer Engine — LLM training and inference |
| INT8 | 8-bit integer | Inference quantisation — fastest, smallest models |

> [!note]
> **Mixed-precision training** uses FP16 for most operations (fast, Tensor Core-optimised) while keeping a master copy of weights in FP32 (accurate). This is the standard approach for training large models on A100/H100.

---

## How This Connects to the Full Stack

```
AI Framework (PyTorch / TensorFlow)
        ↓
[[CUDA]] + cuDNN / cuBLAS
        ↓
Tensor Cores / CUDA Cores (SMs)
        ↓
HBM — model weights and activations
        ↓
[[NVLink and NVSwitch]] — GPU-to-GPU communication (multi-GPU)
        ↓
[[NVIDIA DGX Systems]] — full server platform built around these GPUs
```

---

## Related Notes

- [[CUDA]] – the programming model that targets CUDA Cores and Tensor Cores
- [[GPU vs CPU]] – conceptual comparison of parallel vs sequential architectures
- [[Multi-Instance GPU]] – hardware partitioning of a single GPU into isolated instances
- [[NVLink and NVSwitch]] – high-bandwidth GPU-to-GPU interconnect
- [[NVIDIA DGX Systems]] – server platforms built around A100/H100 GPUs
- [[DCGM]] – tool for monitoring GPU health, utilisation, and errors at the SM level
- [[Training vs Inference]] – different GPU precision and memory requirements per phase

---

## Key Mental Model

A GPU is a **warehouse of identical workers**. The warehouse is the GPU chip. Each floor is an SM. Each worker is a CUDA Core or Tensor Core. The foreman (warp scheduler) hands out identical tasks to 32 workers at a time. The faster you can get materials (memory bandwidth) to the workers, the more productive the warehouse. HBM is the high-speed conveyor belt that feeds them. Tensor Cores are specialist stations that do one type of job (matrix math) at superhuman speed.

> [!tip] Exam Tip
> Tensor Cores ≠ CUDA Cores. Tensor Cores are specialised for matrix multiply-accumulate (the core operation in neural networks) and are far faster for AI workloads. Every modern NVIDIA data center GPU (V100 and newer) has both. Knowing this distinction — and that Tensor Cores were introduced with Volta — is frequently tested.
