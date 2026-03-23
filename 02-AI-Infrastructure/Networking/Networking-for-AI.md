---
tags: [nca-aiio, nca-aiio/infrastructure, networking, hardware]
aliases: [Networking for AI, AI Networking, AI Data Center Networking, RDMA, NCCL, RoCE]
---

# Networking for AI

> **Exam Domain**: AI Infrastructure (40%)
> **Related**: [[NVLink and NVSwitch]], [[InfiniBand]], [[BlueField DPU]], [[NVIDIA DGX Systems]], [[GPU Architecture]], [[Kubernetes-for-AI]], [[AI Security and Compliance]], [[Training vs Inference]]

## Overview

AI networking — particularly for distributed training — has fundamentally different requirements from standard enterprise networking. Distributed training workloads perform constant collective communication operations (AllReduce, AllGather) that transfer hundreds of gigabytes of gradient data between GPUs every second. Standard TCP/IP Ethernet cannot sustain the bandwidth and latency requirements of these workloads at scale. AI data centers therefore use a layered networking architecture: **NVLink/NVSwitch** for ultra-fast intra-node GPU communication, **InfiniBand or RoCE** with RDMA for low-latency inter-node communication, and standard Ethernet for storage and management traffic. Understanding this layered model — and the role of each technology — is core NCA-AIIO exam content.

---

## The Three Network Layers in an AI Cluster

AI clusters use distinct networks for distinct purposes — each sized for its workload:

```
┌──────────────────────────────────────────────────────────┐
│  Layer 1: Intra-Node (within server)                     │
│  [[NVLink and NVSwitch]]                                 │
│  H100: ~900 GB/s total  |  Latency: sub-microsecond      │
│  Purpose: GPU-to-GPU gradient sync within a DGX node     │
├──────────────────────────────────────────────────────────┤
│  Layer 2: Inter-Node (between servers)                   │
│  [[InfiniBand]] (HDR/NDR) or RoCE v2                     │
│  200–400 Gb/s per port  |  Latency: ~1–2 μs              │
│  Purpose: Cross-node gradient sync, RDMA GPU comms       │
├──────────────────────────────────────────────────────────┤
│  Layer 3: Storage & Management                           │
│  Standard Ethernet (100GbE / 400GbE)                     │
│  Purpose: Dataset loading, checkpointing, orchestration  │
└──────────────────────────────────────────────────────────┘
```

> [!tip] Exam Tip
> **NVLink = intra-node only** (within one server). **InfiniBand = inter-node** (between servers). This scope distinction is the single most tested networking fact in NCA-AIIO. NVLink cannot connect GPUs across servers — that requires InfiniBand or RoCE.

---

## Why AI Networking Demands More Than Ethernet

Distributed training relies on **collective communication operations** — all GPUs must exchange gradient data after every training step:

| Operation | What It Does |
|---|---|
| **AllReduce** | Every GPU sends its gradients; all GPUs receive the sum — the core operation in data-parallel training |
| **AllGather** | Each GPU gathers data from all others — used in model-parallel training |
| **Broadcast** | One GPU sends data to all others — parameter initialisation |
| **Reduce-Scatter** | Complement to AllGather — used in ZeRO optimiser for LLM training |

For a 70B parameter model with 32-bit gradients: a single AllReduce transfers ~280 GB of gradient data. This happens thousands of times per training run. Standard 10/25GbE cannot sustain this — it creates **network-bound GPU idle time**, the most expensive bottleneck in AI training.

---

## NVLink and NVSwitch — Intra-Node Fabric

**[[NVLink and NVSwitch]]** is NVIDIA's proprietary high-speed GPU interconnect, connecting all GPUs within a single server node:

| Generation | Architecture | Bandwidth per GPU | Total node bandwidth |
|---|---|---|---|
| NVLink 3.0 | Ampere (A100) | 600 GB/s | ~4.8 TB/s (8-GPU node) |
| NVLink 4.0 | Hopper (H100) | 900 GB/s | ~7.2 TB/s (8-GPU node) |
| NVLink 5.0 | Blackwell (B200) | 1,800 GB/s | ~14.4 TB/s (8-GPU node) |

**NVSwitch** is the crossbar switch that provides full all-to-all connectivity between all GPUs in a node — every GPU can communicate with every other GPU simultaneously at full bandwidth.

> [!warning]
> NVLink connects GPUs **within** a single server. It does not extend across servers. If an exam question describes connecting GPUs across multiple nodes, NVLink is not the answer — [[InfiniBand]] or RoCE is.

---

## InfiniBand — Inter-Node Fabric

**[[InfiniBand]]** is the dominant high-performance interconnect for AI training clusters. NVIDIA acquired Mellanox (the leading IB vendor) in 2020 — making InfiniBand a first-party NVIDIA technology.

| Generation | Speed | Bandwidth | Typical Use |
|---|---|---|---|
| EDR | 100 Gb/s | 12.5 GB/s | Older clusters |
| HDR | 200 Gb/s | 25 GB/s | A100-era clusters |
| NDR | 400 Gb/s | 50 GB/s | H100-era clusters |
| XDR | 800 Gb/s | 100 GB/s | Emerging / Blackwell |

### RDMA — The Critical Capability

InfiniBand's most important feature for AI is **RDMA (Remote Direct Memory Access)**:

```
Standard TCP/IP path (slow):
GPU → CPU → NIC → network → NIC → CPU → GPU
       ↑ CPU involved twice, adds latency and burns CPU cycles

RDMA path (fast):
GPU → NIC → network → NIC → GPU
       ↑ CPU completely bypassed
```

RDMA allows a GPU on one server to read or write directly into the GPU memory of another server — no CPU involvement. This is essential for NCCL collective operations at scale.

> [!note] Cybersecurity Connection
> RDMA is a privileged network capability — it allows direct memory access across server boundaries. In a multi-tenant AI cluster, uncontrolled RDMA is a security risk: a compromised workload with RDMA access could potentially read memory belonging to another tenant's GPU workload. The [[BlueField DPU]] addresses this by enforcing RDMA access control policies at the hardware level — only authorised workloads on authorised ports can initiate RDMA operations, enforced independently of the host OS. This is zero-trust networking applied to the memory fabric.

---

## RoCE — RDMA over Converged Ethernet

**RoCE** (RDMA over Converged Ethernet, pronounced "rocky") brings RDMA capabilities to standard Ethernet infrastructure — a lower-cost alternative to dedicated InfiniBand fabric.

| Feature | InfiniBand (NDR) | RoCE v2 |
|---|---|---|
| **Bandwidth** | 400 Gb/s | 100–400 Gb/s |
| **Latency** | ~1 μs | ~1–2 μs |
| **RDMA** | ✅ Native | ✅ via UDP/IP |
| **Transport** | Dedicated IB fabric | Standard Ethernet switches |
| **Lossless requirement** | Built-in | Requires PFC + ECN (lossless Ethernet config) |
| **Cost** | Higher | Lower (reuses existing Ethernet infra) |
| **Typical use** | Flagship training clusters | Cost-optimised or cloud GPU clusters |

> [!warning]
> RoCE v2 requires **lossless Ethernet** — configured via Priority Flow Control (PFC) and Explicit Congestion Notification (ECN). Standard lossy Ethernet causes RoCE performance to degrade dramatically. This is a common real-world misconfiguration and an exam-relevant distinction.

---

## NCCL — The Collective Communication Library

**NCCL** (NVIDIA Collective Communications Library, pronounced "nickel") is the software library that performs AllReduce, AllGather, Broadcast, and Reduce-Scatter operations across GPUs during distributed training. It runs on top of the network fabric and automatically selects the optimal communication path:

```
NCCL
├── Intra-node:  uses [[NVLink and NVSwitch]] (fastest)
├── Inter-node:  uses [[InfiniBand]] RDMA (preferred)
│               or RoCE RDMA (alternative)
│               or TCP/IP Ethernet (fallback — slowest)
└── Auto-detects available transport at runtime
```

```python
# NCCL is used transparently by PyTorch distributed training
import torch.distributed as dist

dist.init_process_group(
    backend="nccl",      # Use NCCL as the communication backend
    init_method="env://"
)
# All subsequent distributed operations use NCCL automatically
```

NCCL is why InfiniBand and NVLink deliver their performance benefits to AI frameworks — it's the bridge between the hardware fabric and the training code.

---

## BlueField DPU — Programmable Network Security

The **[[BlueField DPU]]** (Data Processing Unit) is a programmable SmartNIC with an integrated ARM CPU that runs its own OS, independent of the host server. It sits between the network and the server:

```
External Network
        ↓
BlueField DPU  ←── ARM CPU running its own OS
│   ├── Network offload (encryption, compression, packet processing)
│   ├── Zero-trust policy enforcement
│   ├── RDMA access control
│   └── Storage acceleration (NVMe-oF)
        ↓
Host CPU + GPU
```

### AI Infrastructure Use Cases

| Use Case | How BlueField Helps |
|---|---|
| **Multi-tenant isolation** | Enforces network segmentation between tenant workloads at hardware level |
| **RDMA access control** | Controls which workloads can use RDMA — prevents cross-tenant memory access |
| **Storage acceleration** | Accelerates NVMe-oF (network storage) — offloads storage I/O from GPU server CPU |
| **Security offload** | Runs firewall, IDS/IPS functions without consuming host CPU cycles |
| **Infrastructure isolation** | Management plane runs on DPU OS, isolated from tenant workloads on host |

> [!note] Cybersecurity Connection
> The BlueField DPU is the clearest hardware security parallel in the NVIDIA stack. It implements **zero-trust networking at silicon** — enforcing policies that cannot be bypassed by the host OS or a compromised container. Think of it as a hardware security module (HSM) for the network interface: the host can be fully compromised, and the DPU still enforces its policies independently. In multi-tenant GPU clusters, this is the boundary between tenant A's RDMA traffic and tenant B's GPU memory — enforced not by K8s NetworkPolicy (software, bypassable) but by the DPU's hardware policy engine.

---

## Bandwidth Comparison: All AI Network Technologies

Understanding relative bandwidth helps frame the hierarchy:

| Technology | Bandwidth | Scope | Protocol |
|---|---|---|---|
| **NVLink 4.0** (H100) | 900 GB/s per GPU | Intra-node | Proprietary |
| **PCIe Gen5** | ~128 GB/s | CPU↔GPU | PCIe |
| **InfiniBand NDR** | 50 GB/s per port | Inter-node | IB |
| **InfiniBand HDR** | 25 GB/s per port | Inter-node | IB |
| **RoCE v2 (400GbE)** | 50 GB/s per port | Inter-node | Ethernet/UDP |
| **400GbE Ethernet** | 50 GB/s | Storage/mgmt | Ethernet |
| **100GbE Ethernet** | 12.5 GB/s | Storage/mgmt | Ethernet |

The 18× gap between NVLink (900 GB/s) and InfiniBand NDR (50 GB/s) explains why topology-aware scheduling in [[Kubernetes-for-AI]] matters — keeping distributed training within a single NVLink domain is dramatically faster than crossing the InfiniBand fabric.

---

## AI Network Topology

A typical AI training cluster has a **spine-leaf topology** with dedicated networks:

```
                    ┌─────────────────┐
                    │  Spine switches │  (InfiniBand or Ethernet)
                    └────────┬────────┘
               ┌─────────────┼─────────────┐
        ┌──────┴──────┐      │      ┌──────┴──────┐
        │ Leaf switch │      │      │ Leaf switch │
        └──────┬──────┘      │      └──────┬──────┘
      ┌────────┼────────┐    │    ┌────────┼────────┐
   DGX-H100  DGX-H100   │   ...   DGX-H100   DGX-H100
   (8×H100)  (8×H100)   │         (8×H100)   (8×H100)
                        │
              ┌─────────┴───────────┐
              │   Storage network   │  (separate Ethernet fabric)
              │   (NFS, Lustre,     │
              │    NVMe-oF)         │
              └─────────────────────┘
```

Separate networks for:
- **Compute fabric**: InfiniBand / RoCE — gradient sync, NCCL traffic
- **Storage fabric**: Ethernet — dataset loading, checkpoint writes
- **Management/OOB**: Ethernet — BMC, IPMI, orchestration

---

## Related Notes

- [[NVLink and NVSwitch]] – intra-node GPU-to-GPU interconnect; full detail on bandwidth and generations
- [[InfiniBand]] – inter-node high-performance fabric; RDMA, HDR/NDR specifications
- [[BlueField DPU]] – programmable SmartNIC for network offload and zero-trust security enforcement
- [[NVIDIA DGX Systems]] – DGX nodes use NVLink internally and InfiniBand externally
- [[GPU Architecture]] – NVLink bandwidth is a GPU architecture property (generation-specific)
- [[Kubernetes-for-AI]] – topology-aware scheduling places pods within NVLink domains for maximum bandwidth
- [[Training vs Inference]] – distributed training drives the extreme networking requirements; inference is less demanding
- [[AI Security and Compliance]] – RDMA access control, network segmentation, DPU-enforced isolation

---

## Key Mental Model

AI networking is a **bandwidth hierarchy**: NVLink at the top (intra-node, blazing fast), InfiniBand in the middle (inter-node, very fast), Ethernet at the bottom (storage and management, adequate). The further gradient data has to travel in this hierarchy, the slower training becomes. Every architectural decision in an AI cluster — node sizing, rack placement, topology-aware scheduling — is an attempt to keep as much communication as possible at the top of this hierarchy. The BlueField DPU is the security guardian sitting at the boundary between the node and the fabric, enforcing who can cross between layers.

> [!tip] Exam Tip
> The three most tested networking distinctions: **(1) NVLink = intra-node only, InfiniBand = inter-node** — never confuse their scope. **(2) RDMA** = CPU-bypassed direct GPU-to-GPU memory access — this is what makes InfiniBand and RoCE fast for AI. **(3) RoCE requires lossless Ethernet** (PFC + ECN) — standard lossy Ethernet degrades RoCE performance severely.
