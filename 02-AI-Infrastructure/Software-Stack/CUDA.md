---
tags: [nca-aiio, nca-aiio/infrastructure, software, cuda, gpu]
aliases: [CUDA, Compute Unified Device Architecture, CUDA Toolkit, CUDA Runtime]
---

# CUDA

> **Exam Domain**: AI Infrastructure (40%)
> **Related**: [[GPU Architecture]], [[GPU vs CPU]], [[AI Containers]], [[NGC Catalog]], [[NVIDIA GPU Operator]], [[Training vs Inference]]

## Overview

CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform and programming model that enables software to run directly on NVIDIA GPUs. It extends C/C++ with special syntax for launching massively parallel workloads across thousands of GPU cores simultaneously. CUDA is the foundational layer beneath every major AI framework — without it, GPUs would be inaccessible to software like PyTorch or TensorFlow. For AI infrastructure, CUDA is the bridge between application code and GPU hardware.

---

## How CUDA Works

CUDA exposes the GPU as a massively parallel processor. A developer writes a **kernel** — a function that runs on the GPU — and launches it across thousands of threads at once. Those threads are organised into a three-level hierarchy:

| Unit | Description |
|------|-------------|
| **Thread** | Smallest unit — one GPU core executing one instance of the kernel |
| **Block** | Group of threads that share fast on-chip memory and can synchronise |
| **Grid** | All blocks combined for a single kernel launch |

```python
# Conceptual example: adding two arrays in parallel on the GPU
import cupy as cp          # CuPy = NumPy-like library built on CUDA

a = cp.array([1, 2, 3, 4])
b = cp.array([10, 20, 30, 40])
c = a + b                  # This runs on the GPU via CUDA
print(c)                   # [11, 22, 33, 44]
```

> [!note]
> In practice, AI engineers rarely write raw CUDA code. They use frameworks (PyTorch, TensorFlow) that call CUDA libraries automatically. Understanding the stack matters for the exam — not writing kernels.

---

## The CUDA Software Stack

CUDA is a layered stack. AI frameworks sit at the top; the GPU hardware is at the bottom.

```
┌─────────────────────────────────┐
│  AI Frameworks (PyTorch, TF)    │  ← Developer-facing
├─────────────────────────────────┤
│  cuDNN / cuBLAS / cuSPARSE      │  ← Optimised math libraries
├─────────────────────────────────┤
│  CUDA Runtime API               │  ← Memory mgmt, kernel launch
├─────────────────────────────────┤
│  CUDA Driver API                │  ← Low-level hardware interface
├─────────────────────────────────┤
│  GPU Hardware                   │  ← Physical silicon
└─────────────────────────────────┘
```

| Component | Role |
|-----------|------|
| **CUDA Toolkit** | Compiler (`nvcc`), libraries, profiler, debugger — the dev environment |
| **CUDA Runtime** | API for allocating GPU memory, copying data, launching kernels |
| **CUDA Driver** | Low-level interface; installed with the GPU driver |
| **cuDNN** | Deep Neural Network library — optimised primitives for conv layers, pooling, etc. |
| **cuBLAS** | Basic Linear Algebra Subroutines — GPU-accelerated matrix math |
| **cuSPARSE** | Sparse matrix operations for certain model types |

> [!warning]
> CUDA version and GPU driver version must be compatible. A CUDA Toolkit version mismatch with the installed driver is a common real-world (and exam-relevant) failure mode. Always check compatibility before deploying.

---

## CUDA and AI Frameworks

No AI engineer writes raw CUDA kernels for training models. The stack works like this:

```
Your Python Code (PyTorch / TensorFlow)
        ↓
cuDNN / cuBLAS  (NVIDIA-optimised math primitives)
        ↓
CUDA Runtime    (memory management, kernel scheduling)
        ↓
GPU Hardware    (thousands of cores doing parallel math)
```

When you call `model.to("cuda")` in PyTorch, you are telling it to move data to GPU memory and use CUDA for all subsequent operations. PyTorch handles everything below that call automatically.

---

## Host vs Device Memory

A critical CUDA concept: CPU and GPU have **separate memory spaces**.

| Term | Meaning |
|------|---------|
| **Host** | The CPU and its RAM |
| **Device** | The GPU and its VRAM (video RAM) |
| **Host → Device** | Copying data from RAM to GPU memory before compute |
| **Device → Host** | Copying results back to RAM after compute |

This data transfer is a performance bottleneck — minimising host-device transfers is a key optimisation in real AI workloads.

> [!note] Cybersecurity Connection
> Think of host vs device memory like two air-gapped environments. Data must be explicitly transferred between them — it doesn't flow automatically. In security terms, this is like a controlled data diode: you must initiate the transfer, and the movement itself has a cost (latency + bandwidth). Misconfigured or excessive transfers are a performance "vulnerability" in AI workloads.

---

## CUDA in the Container/Deployment World

In production AI deployments, CUDA is packaged inside [[AI Containers]]. NVIDIA provides pre-built containers via the [[NGC Catalog]] that include the correct CUDA version, cuDNN, and framework — eliminating version compatibility issues.

The [[NVIDIA GPU Operator]] automates CUDA driver installation across Kubernetes nodes, so operators don't have to manually manage CUDA on every GPU server.

---

## Related Notes

- [[GPU Architecture]] – the physical hardware CUDA programs execute on
- [[GPU vs CPU]] – why parallel GPU execution via CUDA beats sequential CPU for AI
- [[AI Containers]] – how CUDA is packaged and deployed in containerised environments
- [[NGC Catalog]] – NVIDIA's registry of CUDA-ready container images
- [[NVIDIA GPU Operator]] – automates CUDA driver deployment on Kubernetes
- [[Training vs Inference]] – both phases depend on CUDA, but with different performance profiles

---

## Key Mental Model

CUDA is the **operating system interface for the GPU** — just as a syscall lets your application talk to the Linux kernel, a CUDA kernel call lets your AI code talk to the GPU hardware. Everything above it (PyTorch, TensorFlow, cuDNN) is userspace. Everything below it is hardware. CUDA is the privileged boundary in between.

> [!tip] Exam Tip
> CUDA is **NVIDIA-exclusive** — it only runs on NVIDIA GPUs. This is the single most important architectural distinction. AMD and Intel have their own equivalents (ROCm, oneAPI), but for NCA-AIIO, CUDA = NVIDIA = the standard for AI compute.
