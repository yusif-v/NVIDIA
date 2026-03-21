---
tags: [nca-aiio, nca-aiio/infrastructure, software, orchestration, gpu]
aliases: [GPU Operator, NVIDIA GPU Operator, gpu-operator]
---

# NVIDIA GPU Operator

> **Exam Domain**: AI Infrastructure (40%)
> **Related**: [[Kubernetes for AI]], [[CUDA]], [[DCGM]], [[NGC Catalog]], [[AI Containers]], [[Multi-Instance GPU]], [[GPU Architecture]]

## Overview

The NVIDIA GPU Operator is a Kubernetes Operator that automates the deployment, configuration, and lifecycle management of the entire NVIDIA GPU software stack on Kubernetes worker nodes. Without it, every GPU node in a cluster requires manual installation of the driver, CUDA toolkit, container runtime integration, and device plugin — a fragile, time-consuming process that drifts over time. The GPU Operator replaces all of this with a single declarative deployment that enforces a consistent GPU software state across the entire cluster, including new nodes as they are added.

---

## The Problem: GPU Nodes in Kubernetes Are Not Plug-and-Play

A standard Kubernetes node is ready to run CPU workloads immediately. GPU nodes are not — they require a precisely layered software stack before Kubernetes can schedule GPU workloads onto them:

```
AI Framework (PyTorch, TensorFlow)
        ↓
[[CUDA]] Runtime + cuDNN
        ↓
NVIDIA Container Toolkit     ← makes GPU visible inside containers
        ↓
NVIDIA Driver                ← kernel module, talks to hardware
        ↓
GPU Hardware
        +
Kubernetes Device Plugin     ← tells K8s scheduler "this node has GPUs"
DCGM Exporter                ← exposes GPU metrics to monitoring
MIG Manager                  ← configures MIG partitions if needed
```

Manually installing and maintaining this stack on dozens or hundreds of nodes creates version drift, configuration inconsistency, and significant operational overhead.

---

## What the GPU Operator Manages

The GPU Operator installs and manages the following components as Kubernetes **DaemonSets** — one pod per GPU node, automatically:

| Component | Role |
|---|---|
| **NVIDIA Driver** | Kernel module enabling OS-to-GPU communication |
| **CUDA Toolkit** | Runtime libraries required by GPU applications |
| **NVIDIA Container Toolkit** | Integrates GPU access into containerd/Docker |
| **Kubernetes Device Plugin** | Advertises GPU resources to the K8s scheduler |
| **[[DCGM]] Exporter** | Exports GPU telemetry (temp, utilisation, errors) to Prometheus |
| **MIG Manager** | Configures [[Multi-Instance GPU]] partitions on A100/H100 |
| **Node Feature Discovery (NFD)** | Labels nodes with GPU capabilities for scheduling |
| **GPU Feature Discovery (GFD)** | Discovers and labels GPU-specific features (architecture, VRAM, etc.) |

> [!note]
> All components run as DaemonSets managed by Kubernetes. This means the GPU Operator itself is self-healing — if a component crashes, Kubernetes restarts it automatically.

---

## The Operator Pattern

The GPU Operator follows the Kubernetes **Operator pattern**: you declare a desired state, and the Operator's control loop continuously reconciles the actual state to match it.

```yaml
# Example: Installing the GPU Operator via Helm
# This single command provisions the entire GPU stack across all GPU nodes
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace
```

After installation, when a new GPU node joins the cluster, the GPU Operator **automatically** detects it and installs the full software stack — no manual intervention required.

> [!warning]
> The GPU Operator manages the software stack on nodes — it does not schedule GPU workloads. Workload scheduling is still handled by the Kubernetes scheduler using the resource limits set by the Device Plugin (e.g. `nvidia.com/gpu: 1` in a pod spec).

---

## Requesting a GPU in a Pod

Once the GPU Operator is deployed, workloads request GPU resources through standard Kubernetes resource limits:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-workload
spec:
  containers:
  - name: training-job
    image: nvcr.io/nvidia/pytorch:24.01-py3
    resources:
      limits:
        nvidia.com/gpu: 1   # Request 1 GPU from the node
```

The Device Plugin (managed by the GPU Operator) makes `nvidia.com/gpu` a schedulable resource — the same way CPU and memory are natively schedulable.

---

## MIG Integration

On A100 and H100 GPUs, the GPU Operator's **MIG Manager** handles [[Multi-Instance GPU]] configuration. You declare the desired MIG partition profile in a ConfigMap, and the MIG Manager applies it across all eligible nodes:

```yaml
# Example MIG strategy configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: mig-parted-config
data:
  config.yaml: |
    version: v1
    mig-configs:
      all-1g.10gb:           # 7x MIG instances per A100 80GB
        - devices: all
          mig-enabled: true
          mig-devices:
            1g.10gb: 7
```

> [!note]
> MIG Manager only activates on MIG-capable hardware (A100, H100). On other GPUs it is a no-op. The GPU Operator detects hardware capabilities automatically via Node Feature Discovery.

---

## Component Sources: NGC Catalog

All GPU Operator components are pulled as containers from the [[NGC Catalog]] — NVIDIA's signed, verified container registry. This ensures:
- Images are cryptographically verified (not tampered with)
- Version compatibility is guaranteed between components
- Air-gapped deployments can mirror the catalog internally

> [!note] Cybersecurity Connection
> The GPU Operator's reliance on the [[NGC Catalog]] for all components is directly analogous to a **trusted software supply chain**. Just as you'd only install packages from a signed, audited repository in a hardened Linux environment, the GPU Operator only pulls from NVIDIA's verified registry. This is infrastructure-as-code applied to GPU software — consistent declared state across all nodes, no configuration drift, and a clear audit trail of what is running where. Compare this to the GPU Operator's role as an **automated hardening baseline**: it eliminates the manual, node-by-node installation that creates the inconsistency and drift that attackers exploit.

---

## GPU Operator vs. Manual Installation

| Aspect | Manual Installation | GPU Operator |
|---|---|---|
| **Setup** | SSH into each node, install packages | `helm install` once |
| **New nodes** | Manual reinstall required | Automatic on node join |
| **Version consistency** | Prone to drift | Enforced across all nodes |
| **Driver updates** | Manual, disruptive | Rolling update via Operator |
| **MIG configuration** | Script per node | Declarative ConfigMap |
| **Monitoring integration** | Manual DCGM setup | Auto-deployed DaemonSet |
| **Source of truth** | Ad hoc | Git-tracked Helm values |

---

## Where GPU Operator Fits in the Stack

```
[[Kubernetes for AI]]          ← orchestration layer
        ↓
NVIDIA GPU Operator            ← THIS NOTE — manages GPU software on K8s nodes
        ↓
[[CUDA]] + Container Toolkit   ← GPU software stack (managed by Operator)
        ↓
[[GPU Architecture]]           ← physical GPU hardware
        +
[[DCGM]]                       ← monitoring (deployed by Operator as DaemonSet)
[[NGC Catalog]]                ← source of all Operator container images
```

---

## Related Notes

- [[Kubernetes for AI]] – the orchestration platform the GPU Operator runs on top of
- [[CUDA]] – the GPU software stack the Operator installs and manages
- [[DCGM]] – GPU monitoring tool deployed automatically by the Operator as a DaemonSet
- [[Multi-Instance GPU]] – MIG partitioning is configured cluster-wide by the Operator's MIG Manager
- [[NGC Catalog]] – source registry for all GPU Operator component images
- [[AI Containers]] – the containerised workloads that consume GPU resources the Operator provisions

---

## Key Mental Model

The GPU Operator is to GPU nodes what a **configuration management tool** (Ansible, Chef, Puppet) is to servers — but Kubernetes-native. It declares "every GPU node in this cluster should have exactly this software stack at exactly this version" and continuously enforces that state. You write it once, it applies everywhere, and it self-heals. The GPU itself is just hardware until the Operator installs the stack — after that, it's a first-class Kubernetes resource.

> [!tip] Exam Tip
> The GPU Operator's core value proposition is **automating the GPU software stack on Kubernetes at scale** — driver, CUDA toolkit, container runtime integration, device plugin, and monitoring, all deployed as DaemonSets from a single Helm chart. The exam will test whether you understand *what it manages* (the whole stack) and *how it integrates with Kubernetes* (via the Device Plugin making `nvidia.com/gpu` a schedulable resource).
