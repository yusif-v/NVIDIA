---
tags: [nca-aiio, nca-aiio/operations, monitoring, gpu, software]
aliases: [nvidia-smi, NVIDIA System Management Interface, nvsmi]
---

# nvidia-smi

> **Exam Domain**: AI Operations (22%) — also relevant to AI Infrastructure (40%)
> **Related**: [[DCGM]], [[GPU Architecture]], [[CUDA]], [[Kubernetes-for-AI]], [[NVIDIA GPU Operator]]

## Overview

`nvidia-smi` (NVIDIA System Management Interface) is a command-line tool bundled with the NVIDIA GPU driver that provides real-time visibility into GPU hardware state. It reports utilisation, VRAM usage, temperature, power draw, performance state, ECC error counts, and per-process GPU consumption — all from a single command. It is the first diagnostic tool an operator reaches for when investigating GPU health or performance on a single node, and the baseline CLI for any GPU-equipped system running NVIDIA hardware.

---

## Core Output Fields

Running `nvidia-smi` produces a summary table. Every field is exam-relevant:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.54.03    Driver Version: 535.54.03    CUDA Version: 12.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA H100 80GB  Off   | 00000000:01:00.0 Off |                    0  |
| N/A   42C    P0   312W / 700W |  45231MiB / 81920MiB |     87%      Default |
+-----------------------------------------------------------------------------+
```

| Field | Description | What to Watch For |
|---|---|---|
| `Driver Version` | Installed NVIDIA driver | Must match CUDA Toolkit compatibility |
| `CUDA Version` | Max supported CUDA version | Not the installed Toolkit — the driver's ceiling |
| `Temp` | GPU die temperature (°C) | H100 throttles around 83°C |
| `Perf` | P-state: P0 = max, P8 = idle | P0 during training is expected |
| `Pwr: Usage/Cap` | Current draw / TDP (W) | Near-cap = fully utilised; far below = idle or throttling |
| `Memory-Usage` | VRAM used / total (MiB) | OOM kills happen when this maxes out |
| `GPU-Util` | % SMs active last sample | Low during training = CPU bottleneck or data starvation |
| `ECC` | Uncorrectable memory errors | Any non-zero value = hardware fault, investigate immediately |

> [!warning]
> `CUDA Version` in `nvidia-smi` output is the **maximum** CUDA version the installed driver supports — not the version of the CUDA Toolkit installed on the system. These are different things. A driver reporting CUDA 12.2 does not mean CUDA 12.2 Toolkit is installed.

---

## Key Commands

### Basic Snapshot
```bash
nvidia-smi
```
Instantaneous view of all GPUs on the system.

### Continuous Monitoring
```bash
# Refresh every 2 seconds
nvidia-smi dmon -s u -d 2

# Loop with custom metrics every 5 seconds
nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,\
memory.used,memory.total,power.draw \
--format=csv --loop=5
```

### Per-Process GPU Memory
```bash
# Which processes are using GPU memory, and how much?
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
--format=csv,noheader
```

### Full Detail Output
```bash
# All GPU properties — verbose, useful for inventory and debugging
nvidia-smi -q

# Driver and CUDA version only
nvidia-smi --version
```

### Multi-GPU Systems
```bash
# Target a specific GPU by index
nvidia-smi -i 0        # GPU 0 only
nvidia-smi -i 0,1,2    # GPUs 0, 1, and 2
```

---

## Performance States (P-States)

NVIDIA GPUs dynamically shift between power/performance states:

| P-State | Description |
|---|---|
| **P0** | Maximum performance — full clock speeds, full power |
| **P1–P7** | Intermediate states — reduced clocks/power |
| **P8** | Near-idle — minimal power draw |
| **P12** | Deep idle |

During active AI training, GPUs should be in **P0**. If you see P2 or higher during an expected training job, the GPU may be throttling due to thermal limits, power limits, or lack of work.

---

## ECC Errors — What They Mean

ECC (Error Correcting Code) is a memory error detection and correction mechanism built into data center GPUs (A100, H100):

| Error Type | Meaning | Action |
|---|---|---|
| **Single-bit (correctable)** | Memory bit flipped, hardware corrected it | Monitor; accumulation is a warning sign |
| **Double-bit (uncorrectable)** | Memory corrupted, hardware cannot fix it | Immediate investigation; can cause job crashes or silent data corruption |

```bash
# Check ECC error counts
nvidia-smi --query-gpu=ecc.errors.corrected.volatile.total,\
ecc.errors.uncorrected.volatile.total \
--format=csv
```

> [!warning]
> Uncorrectable ECC errors in a training job mean the model weights may be silently corrupted. This is not just a hardware problem — it is a data integrity problem. Jobs should be checkpointed and restarted when uncorrectable errors are detected.

> [!note] Cybersecurity Connection
> ECC error monitoring via `nvidia-smi` parallels integrity checking in security. Just as you'd monitor filesystem integrity with tools like Tripwire or AIDE to catch unexpected bit-level changes, monitoring ECC errors catches hardware-level memory corruption that could silently corrupt AI model weights. From a security standpoint, `nvidia-smi --query-compute-apps` also functions like `ps aux` for the GPU — revealing exactly which PIDs are consuming GPU memory, useful for detecting unauthorised compute usage in a multi-tenant environment.

---

## nvidia-smi vs. DCGM

Understanding when to use each tool is an exam-tested distinction:

| Aspect | nvidia-smi | [[DCGM]] |
|---|---|---|
| **Scope** | Single node, interactive | Multi-node, programmatic |
| **Primary use** | Ad-hoc debugging, quick checks | Production monitoring at scale |
| **Integration** | Terminal / shell scripts | Prometheus, Grafana, Kubernetes |
| **Metrics depth** | Basic (util, temp, power, memory, ECC) | Deep (PCIe errors, NVLink health, XID codes, profiling) |
| **Alerting** | None built-in | Full alerting via exporters |
| **Deployment** | Pre-installed with driver | Deployed separately (or via [[NVIDIA GPU Operator]]) |

> [!tip] Exam Tip
> `nvidia-smi` = **single-node, interactive, immediate**. [[DCGM]] = **multi-node, automated, production**. If an exam scenario asks about monitoring a fleet of GPU servers in a data center, the answer is DCGM. If it asks about checking a single GPU's health or which process is consuming VRAM, the answer is nvidia-smi.

---

## nvidia-smi in Containerised / Kubernetes Environments

In containerised AI workloads managed by [[Kubernetes-for-AI]], `nvidia-smi` can be run inside a pod to see GPU state from the container's perspective:

```bash
# Run nvidia-smi inside a running GPU pod
kubectl exec -it <pod-name> -- nvidia-smi
```

The [[NVIDIA GPU Operator]] ensures the driver and container toolkit are installed so GPU visibility works correctly inside containers. Without the Operator, `nvidia-smi` inside a container may fail or show no GPUs.

---

## Related Notes

- [[DCGM]] – production-grade GPU monitoring; complements nvidia-smi at scale
- [[GPU Architecture]] – understanding what nvidia-smi metrics represent (SMs, VRAM, TDP)
- [[CUDA]] – `CUDA Version` in nvidia-smi output refers to the driver's max supported version
- [[NVIDIA GPU Operator]] – deploys the driver that makes nvidia-smi available on K8s nodes
- [[Kubernetes-for-AI]] – nvidia-smi can be exec'd inside pods for container-level GPU visibility

---

## Key Mental Model

`nvidia-smi` is the **stethoscope** of GPU operations — a lightweight, always-available diagnostic tool that tells you immediately whether the GPU is healthy, busy, or in trouble. It doesn't replace a full monitoring system ([[DCGM]]), just like a stethoscope doesn't replace an ICU monitoring system. But it's the first thing you pick up when something feels wrong, and it's always there because it ships with the driver.

> [!tip] Exam Tip
> Know the four key metrics `nvidia-smi` surfaces and what anomalies in each indicate: **GPU-Util** (low = CPU/data bottleneck; high = GPU is working), **Memory-Usage** (near-max = OOM risk), **Temp** (high = thermal throttle risk), **ECC errors** (any uncorrectable = hardware fault requiring investigation). These map directly to AI Operations troubleshooting scenarios on the exam.
