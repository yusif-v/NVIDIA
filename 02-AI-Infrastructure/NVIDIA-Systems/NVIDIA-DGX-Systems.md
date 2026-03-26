---
tags: [nca-aiio, nca-aiio/infrastructure, hardware, gpu]
aliases: [DGX, NVIDIA DGX, DGX Systems, DGX A100, DGX H100, DGX H200, DGX B200, DGX SuperPOD, DGX Pod]
---

# NVIDIA DGX Systems

> **Exam Domain**: AI Infrastructure (40%)
> **Related**: [[GPU Architecture]], [[NVLink and NVSwitch]], [[InfiniBand]], [[NVIDIA Base Command]], [[DCGM]], [[CUDA]], [[Multi-Instance GPU]], [[AI Containers]], [[NGC Catalog]], [[Kubernetes-for-AI]], [[Slurm]]

## Overview

NVIDIA DGX Systems are purpose-built, fully integrated AI supercomputers — hardware, interconnects, and software designed and validated together for AI training and inference workloads. Unlike commodity GPU servers assembled from parts, DGX systems use NVSwitch to create a non-blocking, all-to-all GPU interconnect within a node, eliminating PCIe bandwidth bottlenecks during multi-GPU training. Each DGX ships with a pre-validated software stack (DGX OS, CUDA, DCGM, NGC access) and scales from a single node to a DGX SuperPOD cluster of 32+ nodes connected by InfiniBand. For NCA-AIIO, DGX is the canonical reference architecture for enterprise AI infrastructure.

---

## DGX Product Line

### DGX H100 — Current Flagship

| Spec | Value |
|---|---|
| **GPUs** | 8× H100 SXM 80GB |
| **Total GPU memory** | 640 GB HBM3 |
| **GPU interconnect** | NVLink 4.0 + NVSwitch |
| **GPU-to-GPU bandwidth** | 900 GB/s (bidirectional) |
| **Network ports** | 8× NDR InfiniBand (400 Gb/s each) |
| **System RAM** | 2 TB DDR5 |
| **CPUs** | 2× Intel Xeon Sapphire Rapids |
| **Storage** | 30 TB NVMe SSD |
| **Power** | ~10.2 kW |

### DGX H200

| Spec | Value |
|---|---|
| **GPUs** | 8× H200 SXM 141GB |
| **Total GPU memory** | 1.1 TB HBM3e |
| **GPU interconnect** | NVLink 4.0 + NVSwitch |
| **Key advantage** | 76% more VRAM than H100 — fits larger models in memory |

### DGX A100 — Previous Generation

| Spec | Value |
|---|---|
| **GPUs** | 8× A100 SXM 80GB |
| **Total GPU memory** | 640 GB HBM2e |
| **GPU interconnect** | NVLink 3.0 + NVSwitch |
| **GPU-to-GPU bandwidth** | 600 GB/s |
| **Network ports** | 8× HDR InfiniBand (200 Gb/s each) |
| **MIG support** | ✅ Yes — up to 56 MIG instances per node |

### DGX B200 — Blackwell Generation

| Spec | Value |
|---|---|
| **GPUs** | 8× B200 SXM |
| **GPU interconnect** | NVLink 5.0 + NVSwitch |
| **Tensor Cores** | 5th generation |

### DGX Spark / DGX Station — Workstation Form Factor

| Product | Form Factor | Target User |
|---|---|---|
| **DGX Spark** | Desktop | Individual researchers, developers |
| **DGX Station** | Desktop tower | Team-level AI development |

> [!warning]
> DGX Spark and DGX Station are **not rack-mounted data center systems** — they are workstation-class products for individual use. The exam focuses on rack-scale DGX (A100, H100, H200, B200) for data center AI infrastructure.

---

## The Key Differentiator: NVSwitch Interconnect

The most important architectural distinction between DGX and a generic multi-GPU server is **NVSwitch** — the all-to-all, non-blocking GPU interconnect fabric.

### Generic GPU Server (PCIe)
```
GPU0 ─┐
GPU1 ─┤── PCIe switch ── CPU ── System RAM
GPU2 ─┤   (shared bus, ~64 GB/s, serialised)
GPU3 ─┘
```
PCIe is a shared bus — GPUs compete for bandwidth. GPU-to-GPU communication routes through the CPU, adding latency.

### DGX Node (NVSwitch)
```
        NVSwitch fabric
GPU0 ───────────────── GPU1
 │  ╲               ╱  │
 │    ╲           ╱    │
GPU3 ───────────────── GPU2
     (all pairs connected, non-blocking, full bandwidth simultaneously)
```

Every GPU can communicate with every other GPU at **full [[NVLink and NVSwitch]] bandwidth simultaneously** — no contention, no CPU routing. This is critical for gradient all-reduce operations during distributed training.

| Metric | PCIe (generic server) | NVSwitch (DGX H100) |
|---|---|---|
| GPU-to-GPU BW | ~64 GB/s | 900 GB/s |
| Topology | Shared bus via CPU | Non-blocking all-to-all |
| Latency | Higher (CPU hop) | Lower (direct) |
| Contention | Yes | No |

---

## Scaling: DGX Node → Pod → SuperPOD

DGX systems are designed to scale in defined building blocks:

```
┌─────────────────────────────────────────────────────┐
│  DGX SuperPOD  (32+ DGX nodes)                      │
│  ┌─────────────────────────────────────────────┐    │
│  │  DGX Pod  (8–32 DGX nodes)                  │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐   │    │
│  │  │ DGX Node │  │ DGX Node │  │ DGX Node │   │    │
│  │  │ 8× H100  │  │ 8× H100  │  │ 8× H100  │   │    │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘   │    │
│  │       └─────────────┴─────────────┘         │    │
│  │            InfiniBand fabric                │    │
│  └─────────────────────────────────────────────┘    │
│      NVIDIA Quantum-2 InfiniBand spine switches     │
│      NVIDIA Base Command (cluster management)       │
└─────────────────────────────────────────────────────┘
```

| Scale | Units | GPUs | Use Case |
|---|---|---|---|
| **DGX Node** | 1 system | 8 GPUs | Single large model, experimentation |
| **DGX Pod** | 8–32 nodes | 64–256 GPUs | Medium training runs, teams |
| **DGX SuperPOD** | 32+ nodes | 256–1000s GPUs | Frontier model training, hyperscale |

### Inter-Node Communication
Within a DGX node: [[NVLink and NVSwitch]] — 900 GB/s, low latency
Between DGX nodes: [[InfiniBand]] — NDR 400 Gb/s per port, 8 ports per node
This is the fundamental bandwidth hierarchy every DGX deployment lives within.

---

## DGX Software Stack

Every DGX ships with a pre-validated, AI-optimised software stack:

| Layer | Component | Role |
|---|---|---|
| **OS** | DGX OS (Ubuntu-based) | Optimised Linux for AI workloads |
| **GPU Driver** | NVIDIA Driver | Pre-installed, validated for hardware |
| **Container Runtime** | Docker + NVIDIA Container Toolkit | GPU-visible containers |
| **Compute** | [[CUDA]], cuDNN, NCCL | GPU programming and comms primitives |
| **Monitoring** | [[DCGM]] | GPU health, telemetry, diagnostics |
| **Management** | [[NVIDIA Base Command]] agent | Cluster-level job and resource management |
| **Registry Access** | [[NGC Catalog]] | Pre-configured access to NVIDIA containers |

> [!note]
> The "validated together" value of DGX is critical for the exam. The DGX software stack is not assembled by the operator — it ships as a tested, certified baseline. Driver/CUDA/cuDNN compatibility issues that plague generic server deployments are eliminated by design.

---

## DGX and MIG

On DGX A100 and H100 nodes, [[Multi-Instance GPU]] partitioning is supported — enabling a single DGX node to serve multiple isolated workloads simultaneously:

| DGX Model | GPUs | Max MIG Instances (1g profile) |
|---|---|---|
| DGX A100 | 8× A100 | 56 (7 per GPU) |
| DGX H100 | 8× H100 | 56 (7 per GPU) |

In a [[Kubernetes for AI]] deployment, the [[NVIDIA GPU Operator]]'s MIG Manager configures MIG partitions across all DGX nodes in the cluster from a single ConfigMap — no per-node manual configuration.

---

## DGX Management: Base Command and Orchestration

DGX systems are managed through two complementary tools:

| Tool | Scope | Role |
|---|---|---|
| [[NVIDIA Base Command]] | Cluster-wide | Job scheduling, resource management, experiment tracking across DGX nodes |
| [[Kubernetes for AI]] | Cluster-wide | Container orchestration — inference services, MLOps pipelines |
| [[Slurm]] | Cluster-wide | HPC-style batch job scheduling — common in SuperPOD deployments |
| [[DCGM]] | Per-node | GPU health monitoring, telemetry, diagnostics |

Large DGX SuperPOD deployments often run **both** Slurm (for batch training jobs) and Kubernetes (for inference and MLOps services) on the same hardware — different workload types, different schedulers.

---

## DGX vs. HGX vs. Generic GPU Server

| Platform | What It Is | Target |
|---|---|---|
| **DGX** | Complete integrated AI supercomputer — GPU + NVSwitch + software | Enterprises, research labs — turnkey AI |
| **HGX** | GPU baseboard only — OEMs build servers around it | Server vendors building AI servers |
| **Generic GPU server** | Standard server + PCIe GPU cards | Cost-optimised, flexible — no NVSwitch |

> [!warning]
> **HGX ≠ DGX.** HGX is the GPU baseboard component that OEMs (Dell, HP, Supermicro) use to build their own AI servers. DGX is NVIDIA's own complete server product built around HGX. Both use NVSwitch; only DGX ships with the full validated software stack and NVIDIA support. This distinction is exam-tested.

> [!note] Cybersecurity Connection
> A DGX system maps closely to the concept of a **hardened appliance** in security architecture. Rather than a general-purpose server you harden yourself (installing drivers, configuring software, validating compatibility), DGX ships with a vendor-validated, known-good baseline — analogous to how a hardware security module (HSM) or next-gen firewall ships with hardened, vendor-certified firmware. The DGX OS is the GPU equivalent of that baseline: stripped, optimised, and tested. [[NVIDIA Base Command]] and [[DCGM]] provide the management and telemetry plane, equivalent to the management interfaces and SNMP/syslog on a security appliance. In a zero-trust model, a DGX is an easier asset to baseline and monitor than a self-assembled GPU server precisely because its software state is more predictable.

---

## Related Notes

- [[GPU Architecture]] – the H100/A100 GPU hardware inside every DGX node
- [[NVLink and NVSwitch]] – the all-to-all interconnect that makes DGX's intra-node bandwidth possible
- [[InfiniBand]] – the inter-node fabric connecting DGX nodes in Pod and SuperPOD deployments
- [[Multi-Instance GPU]] – MIG partitioning supported on DGX A100 and H100
- [[NVIDIA Base Command]] – the cluster management platform for DGX Pod and SuperPOD
- [[DCGM]] – GPU monitoring pre-installed on every DGX node
- [[CUDA]] – the compute layer of the DGX software stack
- [[NGC Catalog]] – pre-configured access to NVIDIA containers on every DGX
- [[Kubernetes for AI]] – DGX nodes are commonly managed as K8s GPU nodes via GPU Operator
- [[Slurm]] – HPC-style batch scheduler commonly deployed on DGX SuperPOD clusters
- [[AI Containers]] – the workload format that runs on DGX via the pre-installed container runtime

---

## Key Mental Model

A DGX system is an **AI supercomputer in a box** — the same way a network appliance is a firewall in a box. The individual components (GPUs, NVLink, InfiniBand, software) are not novel by themselves — what DGX adds is their integration, validation, and support as a single unit. The NVSwitch fabric is what makes it a supercomputer rather than just a server with GPUs: every GPU talks to every other GPU at full speed simultaneously, with no bottlenecks. Scale that to a SuperPOD and you have a petaflop-scale AI factory built from 32 of these validated units.

> [!tip] Exam Tip
> Three DGX facts that are most frequently tested: **(1)** DGX uses [[NVLink and NVSwitch]] for intra-node GPU communication — not PCIe; **(2)** DGX H100 has 8× H100 GPUs with 640 GB total GPU memory; **(3)** DGX scales to SuperPOD via [[InfiniBand]] between nodes, managed by [[NVIDIA Base Command]]. Know the distinction between DGX (complete system), HGX (GPU baseboard for OEMs), and generic GPU servers (PCIe, no NVSwitch).
