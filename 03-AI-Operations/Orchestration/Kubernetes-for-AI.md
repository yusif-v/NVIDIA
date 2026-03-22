---
tags: [nca-aiio, nca-aiio/operations, orchestration, gpu, software]
aliases: [Kubernetes for AI, K8s for AI, GPU Kubernetes, AI orchestration]
---

# Kubernetes for AI

> **Exam Domain**: AI Operations (22%)
> **Related**: [[NVIDIA GPU Operator]], [[DCGM]], [[Multi-Instance GPU]], [[Slurm]], [[NVLink and NVSwitch]], [[InfiniBand]], [[AI Containers]], [[NGC Catalog]], [[MLOps]], [[AI Security and Compliance]], [[Training vs Inference]]

## Overview

Kubernetes is the dominant container orchestration platform for AI workloads — managing scheduling, scaling, and lifecycle of both training jobs and inference services across GPU clusters. Standard Kubernetes requires AI-specific extensions to handle GPUs: the [[NVIDIA GPU Operator]] automates the GPU software stack on nodes, the Device Plugin registers GPUs as schedulable resources, and Kubeflow operators add distributed training primitives that vanilla K8s lacks. For NCA-AIIO, the key is understanding how K8s is extended for AI — not K8s fundamentals themselves.

---

## Making GPUs Schedulable: The Device Plugin

Standard K8s only knows about CPU and memory. GPUs become schedulable through the **NVIDIA Device Plugin**, deployed automatically by the [[NVIDIA GPU Operator]]:

```
GPU hardware
        ↓
NVIDIA Driver          (installed by GPU Operator)
        ↓
Container Toolkit      (GPU visibility inside containers)
        ↓
Device Plugin          (registers nvidia.com/gpu with K8s scheduler)
        ↓
K8s scheduler          (can now place GPU-requesting pods)
```

Once deployed, GPU capacity appears on node descriptions:

```bash
kubectl describe node gpu-node-01
# Capacity:
#   cpu:              96
#   memory:           1056Gi
#   nvidia.com/gpu:   8        ← 8 GPUs registered
```

Pods request GPUs via standard resource limits:

```yaml
resources:
  limits:
    nvidia.com/gpu: 2    # Request 2 GPUs — integer only, no fractional GPU
```

> [!warning]
> Standard Kubernetes GPU allocation is **all-or-nothing per GPU** — you cannot request 0.5 GPUs. Fractional GPU sharing requires [[Multi-Instance GPU]] (MIG) or time-slicing configuration. A pod requesting `nvidia.com/gpu: 1` gets exclusive access to one full GPU.

---

## Training vs. Inference: Different K8s Patterns

The most important architectural distinction for AI on K8s:

| Aspect | Training | Inference |
|---|---|---|
| **K8s resource type** | Job / PyTorchJob / MPIJob | Deployment + Service |
| **Lifecycle** | Finite — runs to completion | Long-running — continuous |
| **Scaling** | Fixed replicas (distributed training) | Horizontal (HPA on request volume) |
| **GPU type** | High VRAM — A100, H100 | Lower power — L4, T4, L40S |
| **Inter-pod comms** | Required (NCCL, MPI) | None — stateless replicas |
| **Failure handling** | Checkpoint + restart | Pod replacement, no state lost |
| **Priority** | Batch — can be preempted | Latency-sensitive — high priority |

---

## Distributed Training: Kubeflow Operators

Vanilla K8s Jobs don't understand "launch N pods that must all start together and communicate via NCCL." Kubeflow adds CRDs for AI-aware distributed training:

| Operator | Framework | Use Case |
|---|---|---|
| **PyTorchJob** | PyTorch DDP | Most common — LLM fine-tuning, vision models |
| **MPIJob** | MPI / Horovod | HPC-style distributed training |
| **TFJob** | TensorFlow | TF parameter server / MirroredStrategy |
| **PaddleJob** | PaddlePaddle | Baidu's framework |

```yaml
# PyTorchJob example — 8-node distributed training
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: llm-finetune
  namespace: ml-team-a
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: trainer
            image: nvcr.io/nvidia/pytorch:24.01-py3
            resources:
              limits:
                nvidia.com/gpu: 8    # 8 GPUs on master node
    Worker:
      replicas: 7                    # 7 workers × 8 GPUs = 56 worker GPUs
      restartPolicy: OnFailure       # 64 GPUs total
      template:
        spec:
          containers:
          - name: trainer
            image: nvcr.io/nvidia/pytorch:24.01-py3
            resources:
              limits:
                nvidia.com/gpu: 8
```

---

## Gang Scheduling

Standard K8s schedules pods one at a time — catastrophic for distributed training where all pods must start simultaneously:

```
Problem: Need 8 pods running concurrently for distributed training
K8s schedules 7/8 → all 7 sit idle waiting for pod 8
→ GPU resources wasted, job never progresses
```

**Gang scheduling** ensures all-or-nothing pod group placement — either all 8 pods start together or none start:

| Tool | Role |
|---|---|
| **Volcano** | K8s batch scheduler with native gang scheduling |
| **Koordinator** | Gang scheduling + colocation optimisation |
| **Coscheduling plugin** | K8s scheduler plugin for gang scheduling |

> [!tip] Exam Tip
> Gang scheduling is the solution to the "pods waiting indefinitely" problem in distributed AI training on Kubernetes. The exam tests whether you recognise this as an AI-specific scheduling requirement that standard K8s does not provide natively.

---

## Topology-Aware Scheduling

GPU-to-GPU bandwidth varies dramatically by physical placement:

| Communication Path | Bandwidth | Latency |
|---|---|---|
| Same GPU (SM to SM) | ~3+ TB/s | Nanoseconds |
| Same node via [[NVLink and NVSwitch]] | ~600 GB/s | Microseconds |
| Cross-node via [[InfiniBand]] | ~200 GB/s | ~1–2 μs |
| Cross-node via Ethernet | ~25–100 GB/s | Higher |

For distributed training, placing all pods on nodes within the same NVLink domain dramatically reduces communication overhead. The **NVIDIA Network Operator** applies topology labels to nodes, enabling the scheduler to make topology-aware placement decisions.

---

## MIG in Kubernetes

On A100/H100 GPUs with [[Multi-Instance GPU]] enabled, MIG partitions appear as separate schedulable extended resources:

```bash
kubectl describe node h100-node
# Capacity:
#   nvidia.com/mig-1g.10gb: 7    ← 7 × 1/7th H100 partitions
#   nvidia.com/mig-3g.40gb: 2    ← 2 × 3/7th H100 partitions
```

Pods request MIG slices exactly like full GPUs:

```yaml
resources:
  limits:
    nvidia.com/mig-1g.10gb: 1    # One small MIG slice
```

The [[NVIDIA GPU Operator]]'s MIG Manager handles all partition configuration automatically via ConfigMap — no manual per-node `nvidia-smi mig` commands.

```yaml
# MIG strategy ConfigMap — applied cluster-wide by MIG Manager
apiVersion: v1
kind: ConfigMap
metadata:
  name: mig-parted-config
  namespace: gpu-operator
data:
  config.yaml: |
    version: v1
    mig-configs:
      all-balanced:
        - devices: all
          mig-enabled: true
          mig-devices:
            3g.40gb: 2    # 2 large slices per H100
            1g.10gb: 3    # 3 small slices per H100
```

---

## Inference Serving on Kubernetes

Inference workloads are long-running HTTP services, deployed as standard K8s Deployments:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-inference
spec:
  replicas: 3           # 3 inference replicas
  template:
    spec:
      containers:
      - name: nim
        image: nvcr.io/nim/meta/llama3-8b-instruct:latest
        resources:
          limits:
            nvidia.com/gpu: 1
        ports:
        - containerPort: 8000
---
# HorizontalPodAutoscaler — scale on GPU utilisation or request rate
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: HorizontalPodAutoscaler
    name: llm-inference
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: nvidia.com/gpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

## GPU Resource Management

### ResourceQuotas — GPU Budget per Namespace

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: ml-team-gpu-quota
  namespace: ml-team-a
spec:
  hard:
    requests.nvidia.com/gpu: "16"   # Team A gets max 16 GPUs
    limits.nvidia.com/gpu: "16"
```

### LimitRange — Per-Pod GPU Constraints

```yaml
apiVersion: v1
kind: LimitRange
metadata:
  name: gpu-limit-range
  namespace: ml-team-a
spec:
  limits:
  - type: Container
    max:
      nvidia.com/gpu: "8"     # No single container gets more than 8 GPUs
    default:
      nvidia.com/gpu: "1"
```

---

## Key NVIDIA K8s Components Summary

| Component | Deployed By | Role |
|---|---|---|
| **Device Plugin** | [[NVIDIA GPU Operator]] | Registers `nvidia.com/gpu` with K8s scheduler |
| **Container Toolkit** | [[NVIDIA GPU Operator]] | GPU visibility inside containers |
| **MIG Manager** | [[NVIDIA GPU Operator]] | Configures MIG partitions cluster-wide |
| **[[DCGM]] Exporter** | [[NVIDIA GPU Operator]] | GPU telemetry → Prometheus |
| **Network Operator** | NVIDIA Network Operator | InfiniBand / RoCE config, topology labels |
| **PyTorchJob / MPIJob** | Kubeflow | Distributed training CRDs |
| **Volcano / Koordinator** | Third-party | Gang scheduling, batch AI scheduling |

> [!note] Cybersecurity Connection
> Kubernetes RBAC in AI clusters is familiar territory — but with GPU-specific dimensions. **ResourceQuotas on `nvidia.com/gpu`** enforce GPU budget per team/namespace, preventing a single team from monopolising cluster resources (analogous to rate-limiting in security). **PodSecurityAdmission** is actively contested in AI environments — CUDA containers often default to root, and enforcing non-root requires careful image validation. **NetworkPolicy** for distributed training: training pods need intra-job communication (NCCL traffic on specific ports) but should be isolated from other namespaces — define ingress/egress rules that allow only the training collective communication patterns. This is the GPU equivalent of microsegmentation.

---

## Kubernetes vs. Slurm for AI

| Aspect | Kubernetes | [[Slurm]] |
|---|---|---|
| **Primary use** | Mixed workloads — training + inference + services | Batch HPC/AI training jobs |
| **Workload type** | Containers (microservices + jobs) | Scripts / MPI jobs (bare metal or containers) |
| **Scheduling model** | Continuous, declarative | Queue-based, reservation |
| **GPU sharing** | MIG, time-slicing via Device Plugin | Native GPU binding |
| **Ecosystem** | Cloud-native — Helm, CI/CD, GitOps | HPC-native — job arrays, accounting |
| **Typical environment** | Cloud, hybrid, enterprise K8s clusters | Supercomputers, national labs, large HPC centers |

> [!warning]
> The exam tests **Kubernetes AND Slurm** as complementary orchestrators — not competing ones. Many large AI environments run both: Slurm for batch training at scale, Kubernetes for inference serving and MLOps pipelines.

---

## Related Notes

- [[NVIDIA GPU Operator]] – automates the entire GPU software stack on K8s nodes; prerequisite for GPU workloads
- [[DCGM]] – GPU monitoring deployed by GPU Operator as a DaemonSet; integrates with Prometheus on K8s
- [[Multi-Instance GPU]] – MIG partitions exposed as K8s schedulable resources via Device Plugin
- [[Slurm]] – the HPC alternative to K8s for batch AI training; both are exam topics
- [[NVLink and NVSwitch]] – topology-aware scheduling places pods within same NVLink domain for fast comms
- [[InfiniBand]] – cross-node GPU communication in distributed training; Network Operator configures it on K8s
- [[AI Containers]] – the packaging format for all AI workloads running on K8s
- [[NGC Catalog]] – source for all training and inference container images used in K8s workloads
- [[MLOps]] – K8s is the platform layer that MLOps pipelines (Kubeflow Pipelines, Argo) run on
- [[AI Security and Compliance]] – RBAC, ResourceQuotas, NetworkPolicy, PodSecurityAdmission for GPU clusters
- [[Training vs Inference]] – training → Job/PyTorchJob; inference → Deployment + HPA; fundamentally different K8s patterns

---

## Key Mental Model

Kubernetes for AI is **standard K8s with three additions**: GPUs as schedulable resources (Device Plugin), AI-aware scheduling primitives (gang scheduling, topology awareness), and distributed training operators (PyTorchJob, MPIJob). Everything else — RBAC, namespaces, Services, PVCs, Helm — works identically to what you know from CKA. The GPU Operator handles the hard part (driver lifecycle, MIG config, monitoring deployment) so the platform team doesn't have to. The application team just writes a PyTorchJob manifest and requests `nvidia.com/gpu: 8`.

> [!tip] Exam Tip
> The two most exam-tested K8s for AI concepts are: **(1) the Device Plugin** — it is what makes `nvidia.com/gpu` a schedulable resource (deployed by [[NVIDIA GPU Operator]]), and **(2) training vs. inference workload patterns** — training is a finite Job (PyTorchJob), inference is a long-running Deployment with HPA. If a scenario describes a GPU job that never starts despite available GPUs, the answer is almost always **gang scheduling** — all pods must be schedulable simultaneously.
