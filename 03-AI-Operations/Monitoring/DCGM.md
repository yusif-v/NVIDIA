---
tags: [nca-aiio, nca-aiio/operations, monitoring, gpu, software]
aliases: [DCGM, Data Center GPU Manager, dcgmi, DCGM Exporter, nv-hostengine]
---

# DCGM

> **Exam Domain**: AI Operations (22%)
> **Related**: [[nvidia-smi]], [[NVIDIA GPU Operator]], [[Kubernetes for AI]], [[GPU Architecture]], [[NVLink and NVSwitch]], [[Multi-Instance GPU]], [[AI Security and Compliance]]

## Overview

DCGM (Data Center GPU Manager) is NVIDIA's production monitoring and management library for GPU-accelerated data centers. It runs as a persistent daemon (`nv-hostengine`) on each GPU node, continuously collecting deep telemetry — utilisation, memory, temperature, power, ECC errors, XID fault codes, PCIe health, and NVLink link status. In Kubernetes environments, the **DCGM Exporter** DaemonSet (deployed automatically by the [[NVIDIA GPU Operator]]) exposes all metrics in Prometheus format, feeding them into Grafana for dashboards and alerting. DCGM is the difference between reactive firefighting and proactive GPU fleet management.

---

## Architecture: How DCGM Fits in the Stack

```
Grafana (dashboards + alerts)
        ↑
Prometheus (metric aggregation)
        ↑
DCGM Exporter (DaemonSet, one per GPU node)   ← exposes /metrics endpoint
        ↑
DCGM Daemon / nv-hostengine (per node)        ← raw GPU telemetry collection
        ↑
GPU Hardware (all GPUs on the node)
```

| Component | Role |
|---|---|
| **nv-hostengine** | Background daemon; polls GPU hardware continuously |
| **dcgmi** | CLI tool for querying the daemon interactively |
| **DCGM Exporter** | Translates DCGM metrics into Prometheus format |
| **Prometheus** | Scrapes all exporters across the cluster |
| **Grafana** | Visualises metrics, defines alerting thresholds |

> [!note]
> In Kubernetes, the DCGM Exporter is deployed automatically by the [[NVIDIA GPU Operator]] as a DaemonSet — one pod per GPU node. No manual installation required when the Operator is present.

---

## What DCGM Monitors

DCGM provides significantly deeper telemetry than [[nvidia-smi]]:

| Category | Key Metrics |
|---|---|
| **Utilisation** | GPU-Util %, SM active %, Tensor Core active %, memory BW utilisation |
| **Memory** | VRAM used/free (MiB), framebuffer utilisation % |
| **Thermal** | GPU temperature (°C), slowdown threshold, shutdown threshold |
| **Power** | Power draw (W), power limit (W), total energy (J) |
| **ECC Errors** | Correctable (single-bit), uncorrectable (double-bit) — volatile + aggregate |
| **XID Errors** | Hardware/driver fault event codes — specific fault identification |
| **PCIe** | TX/RX throughput (MB/s), replay error count |
| **NVLink** | Per-link bandwidth (GB/s), error counts per link |
| **Clocks** | SM clock (MHz), memory clock (MHz), throttle reasons |
| **Profiling** | Fine-grained SM pipeline utilisation (requires Profiling API) |

---

## XID Error Codes

XID errors are NVIDIA's internal fault codes logged when a GPU hardware or driver event occurs. DCGM captures them in real time. Each code maps to a specific fault type:

| XID | Meaning | Severity |
|---|---|---|
| 13 | Graphics engine exception | Medium — often software/driver |
| 31 | GPU memory page fault | Medium |
| 43 | GPU stopped processing | High — hardware fault |
| 45 | Preemptive cleanup (job killed due to error) | High |
| 74 | NVLink error | High — interconnect fault |
| 79 | GPU has fallen off the PCIe bus | Critical — catastrophic failure |
| 92 | High single-bit ECC error rate | Warning — hardware degradation |

> [!warning]
> XID 79 ("GPU fallen off bus") means the GPU is no longer visible to the system — it requires physical hardware investigation. Any workload on that node at the time will have lost data. Automated alerting on critical XIDs is a core DCGM use case.

---

## DCGM Health Check Framework

Beyond passive monitoring, DCGM can run **active health checks** — targeted diagnostic tests against GPU subsystems:

```bash
# Quick health check — all GPUs (seconds)
dcgmi diag -r 1

# Medium diagnostic (minutes)
dcgmi diag -r 2

# Full burn-in diagnostic (run offline, takes ~10 min)
dcgmi diag -r 3

# Health check output example
dcgmi health -g 0 -c    # Check group 0, show results
```

| Check Level | Duration | Use Case |
|---|---|---|
| Level 1 | Seconds | Pre-job sanity check |
| Level 2 | Minutes | New node admission |
| Level 3 | ~10 min | Full burn-in / RMA decision |

Health checks are used to **validate GPU nodes before admitting them to a cluster** and periodically to catch degrading hardware before it causes silent job failures.

---

## Key DCGM CLI Commands

```bash
# List all GPUs detected by DCGM
dcgmi discovery -l

# View live metrics for all GPUs
dcgmi dmon -e 203,204,155,150,156

# Query a specific field (e.g. GPU temp = field 150)
dcgmi dmon -e 150 -d 2    # poll every 2 seconds

# Check GPU health status
dcgmi health -g 0 -c

# Run quick diagnostic
dcgmi diag -r 1

# View XID error events
dcgmi stats -g 0 --enable
dcgmi stats -g 0 --print
```

---

## DCGM Exporter: Prometheus Integration

In Kubernetes, the DCGM Exporter exposes GPU metrics as a Prometheus scrape endpoint:

```yaml
# Example: DCGM Exporter DaemonSet (deployed by GPU Operator)
# Exposes metrics at http://<node-ip>:9400/metrics

# Example Prometheus scrape config
scrape_configs:
  - job_name: 'dcgm-exporter'
    kubernetes_sd_configs:
      - role: endpoints
    relabel_configs:
      - source_labels: [__meta_kubernetes_endpoints_name]
        regex: 'dcgm-exporter'
        action: keep
```

```bash
# Example Prometheus alert rule — GPU temperature
groups:
- name: gpu-alerts
  rules:
  - alert: GPUHighTemperature
    expr: DCGM_FI_DEV_GPU_TEMP > 80
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "GPU temperature exceeded 80°C for 5 minutes"
```

---

## DCGM vs. nvidia-smi

| Aspect | [[nvidia-smi]] | DCGM |
|---|---|---|
| **Scope** | Single node, interactive | Multi-node, programmatic |
| **Deployment** | Bundled with driver | Separate install / GPU Operator |
| **Metrics depth** | Basic (util, temp, power, memory) | Deep (XID, PCIe, NVLink, profiling) |
| **Integration** | Terminal only | Prometheus, Grafana, REST API |
| **Alerting** | None | Full alerting via Prometheus rules |
| **Health checks** | None | Active diagnostic framework |
| **Use case** | Ad-hoc node debugging | Production fleet monitoring 24/7 |

> [!tip] Exam Tip
> The exam distinction is clear: **nvidia-smi = single node, interactive, immediate**. **DCGM = multi-node, automated, production-scale**. When a scenario describes monitoring a GPU data center fleet, alerting on faults, or integrating with Prometheus/Grafana — the answer is always DCGM.

---

## DCGM in MIG Environments

On A100 and H100 GPUs with [[Multi-Instance GPU]] enabled, DCGM can monitor **per-MIG-instance metrics** — treating each MIG partition as an independent monitoring target:

```bash
# List MIG instances visible to DCGM
dcgmi discovery -l    # shows individual MIG compute instances

# Monitor per-instance utilisation
dcgmi dmon -e 203 -g <mig-group-id>
```

This enables per-tenant GPU monitoring in shared multi-tenant clusters — each MIG instance's utilisation, memory, and errors are tracked independently.

> [!note] Cybersecurity Connection
> DCGM is the closest AI infrastructure parallel to a **SIEM** in security engineering. The daemon on each node = endpoint agents. XID error collection = log ingestion and event normalisation. Prometheus scraping = central log aggregation. Grafana alerting rules = SIEM correlation rules and alert policies. Automated node isolation on critical XID = automated incident response playbooks. The architecture is isomorphic — if you understand a SIEM deployment, you already understand how DCGM fits into a GPU data center. The key difference: DCGM monitors hardware health, not user behaviour — but the pipeline is identical.

---

## Related Notes

- [[nvidia-smi]] – single-node GPU inspection tool; DCGM's lightweight counterpart
- [[NVIDIA GPU Operator]] – automatically deploys DCGM Exporter as a DaemonSet on K8s
- [[Kubernetes for AI]] – the orchestration layer DCGM Exporter runs on
- [[GPU Architecture]] – the hardware internals that DCGM metrics describe (SMs, ECC, NVLink)
- [[NVLink and NVSwitch]] – DCGM monitors per-link NVLink health and bandwidth
- [[Multi-Instance GPU]] – DCGM can monitor individual MIG instances independently
- [[AI Security and Compliance]] – DCGM health checks and XID alerting as a GPU fleet security control

---

## Key Mental Model

DCGM is the **SIEM for your GPU fleet**. Just as a SIEM ingests logs from every endpoint, normalises events, feeds a central aggregator, and triggers alerts when signatures match — DCGM ingests GPU telemetry from every node, normalises it into structured metrics and XID codes, feeds Prometheus, and triggers Grafana alerts when thresholds are breached. The pipeline is identical. The domain is different. If you'd never run a 500-node environment without a SIEM, you'd never run a 500-GPU cluster without DCGM.

> [!tip] Exam Tip
> DCGM's three core capabilities for the exam: **(1) continuous GPU telemetry collection** including XID errors and NVLink health, **(2) Prometheus/Grafana integration** via the DCGM Exporter DaemonSet, and **(3) active health check framework** for node validation and diagnostics. Know all three — exam scenarios test each one independently.
