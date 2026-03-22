---
tags: [nvidia, software, rapids, data-science, general-knowledge]
aliases: [RAPIDS, NVIDIA RAPIDS, Rapids AI, GPU data science]
---

# RAPIDS

> **Exam Relevance**: ⚠️ Not on NCA-AIIO blueprint — general NVIDIA ecosystem knowledge
> **Related**: [[CUDA]], [[NGC Catalog]], [[AI Containers]], [[GPU vs CPU]], [[GPU Architecture]], [[cuML]], [[cuDF]], [[cuGraph]], [[cuSpatial]], [[cuSignal]], [[RAFT]]

## Overview

RAPIDS is NVIDIA's open-source suite of GPU-accelerated data science libraries. It reimplements the standard Python data science stack — DataFrames, machine learning, graph analytics, signal processing — to execute entirely on NVIDIA GPUs via [[CUDA]], eliminating the CPU bottleneck on large-scale data workloads. The defining design principle is **end-to-end GPU residency**: data is loaded once into GPU memory and stays there across the full pipeline — ETL, feature engineering, model training, and inference — removing the costly CPU↔GPU transfers that fragment traditional workflows. Each RAPIDS library is API-compatible with its CPU counterpart, making migration from existing Python workflows a matter of changing import statements.

> [!warning]
> RAPIDS is **not covered by the NCA-AIIO exam**. This note is general NVIDIA ecosystem knowledge — valuable context for understanding GPU acceleration beyond deep learning, but not a study priority for the cert.

---

## The RAPIDS Library Suite

| Library | Role | CPU Equivalent | Key Use Case |
|---|---|---|---|
| **[[cuDF]]** | GPU DataFrame operations | Pandas | ETL, filtering, joins, aggregations at scale |
| **[[cuML]]** | GPU classical ML algorithms | scikit-learn | Training classical models on large tabular datasets |
| **[[cuGraph]]** | GPU graph analytics | NetworkX | PageRank, community detection, BFS, graph neural networks |
| **[[cuSpatial]]** | GPU geospatial analytics | GeoPandas | Spatial joins, trajectory analysis, point-in-polygon |
| **[[cuSignal]]** | GPU signal processing | SciPy signal | Time-series, FFT, filtering pipelines |
| **[[RAFT]]** | Shared GPU primitives | — | Nearest neighbours, clustering, linear algebra — used internally by cuML/cuGraph |
| **[[cuCIM]]** | GPU image I/O and medical imaging | Pillow / OpenCV | Medical image processing, pathology pipelines |

---

## The Core Design Principle: End-to-End GPU Residency

Traditional data science pipelines bounce data between CPU and GPU at every stage:

```
# Traditional pipeline — data crosses the PCIe bus repeatedly
CSV on disk → Pandas (CPU) → sklearn preprocessing (CPU) → GPU training → CPU evaluation
                  ↑ slow              ↑ slow                   ↑ fast         ↑ slow
```

RAPIDS keeps data on GPU throughout:

```
# RAPIDS pipeline — data stays in GPU VRAM
CSV on disk → cuDF (GPU) → cuML preprocessing (GPU) → cuML training (GPU) → GPU evaluation
                 ↑ fast            ↑ fast                    ↑ fast              ↑ fast
```

The PCIe bus (CPU↔GPU transfer) is a major bottleneck — bandwidth is orders of magnitude lower than GPU memory bandwidth. RAPIDS eliminates this bottleneck by design.

---

## cuDF — GPU DataFrames

cuDF is the foundation of RAPIDS — the GPU equivalent of Pandas. It implements the Pandas API on GPU memory (VRAM), enabling fast ETL on datasets that would be slow on CPU:

```python
import cudf

# Load directly into GPU memory
df = cudf.read_csv("billion_row_dataset.csv")   # stays on GPU

# Pandas-identical API
df["new_col"] = df["col_a"] / df["col_b"]
filtered = df[df["value"] > 100]
grouped = df.groupby("category")["sales"].sum()

# Convert to/from Pandas if needed
pandas_df = df.to_pandas()       # GPU → CPU
cudf_df = cudf.from_pandas(pdf)  # CPU → GPU
```

---

## cuGraph — GPU Graph Analytics

cuGraph provides GPU-accelerated implementations of standard graph algorithms. Particularly relevant for **network analysis and security workloads**:

```python
import cugraph
import cudf

# Build an edge list (e.g. network connections)
edges = cudf.DataFrame({
    "src": [0, 1, 2, 3, 4],
    "dst": [1, 2, 3, 4, 0],
    "weight": [1.0, 1.0, 1.0, 1.0, 1.0]
})

G = cugraph.Graph()
G.from_cudf_edgelist(edges, source="src", destination="dst")

# PageRank — identifies high-influence nodes
pagerank = cugraph.pagerank(G)

# Louvain community detection — finds clusters
parts, modularity = cugraph.louvain(G)

# Connected components — finds isolated subgraphs
components = cugraph.connected_components(G)
```

> [!note] Cybersecurity Connection
> cuGraph's algorithms map directly to network threat detection use cases. **Louvain community detection** identifies clusters of nodes with dense internal communication — in a network flow graph, these clusters can reveal botnet C2 infrastructure or lateral movement groups. **PageRank** on a network graph identifies the most connected nodes — in a threat context, those are pivot points or high-value targets. **Connected components** finds isolated subgraphs — useful for detecting air-gapped or anomalous network segments. Running these on GPU via cuGraph enables real-time analysis at enterprise scale (millions of nodes and edges) that would be impractically slow on CPU-based NetworkX. The GPU infrastructure covered in NCA-AIIO is exactly what runs these security analytics workloads.

---

## RAPIDS in the NVIDIA Ecosystem

RAPIDS sits at the application layer, consuming the GPU infrastructure stack:

```
RAPIDS (cuDF, cuML, cuGraph, ...)     ← application layer (this note)
        ↓
CUDA + cuBLAS + cuSolver + RAFT       ← GPU compute primitives
        ↓
GPU Architecture.                     ← hardware (any CUDA-capable GPU)
        ↓
NVIDIA DGX Systems.    / cloud GPUs   ← system platform
```

All RAPIDS containers are distributed via the [[NGC Catalog]] at `nvcr.io/nvidia/rapidsai/` and run as [[AI Containers]] on any NVIDIA GPU system.

```bash
# Pull the RAPIDS base container
docker pull nvcr.io/nvidia/rapidsai/base:24.02-cuda12.0-py3.11

# Run with GPU access
docker run --gpus all -it --rm \
  nvcr.io/nvidia/rapidsai/base:24.02-cuda12.0-py3.11 \
  jupyter lab --ip 0.0.0.0 --allow-root
```

---

## Hardware Requirements

RAPIDS is less demanding than deep learning — mid-range data center GPUs are sufficient for most workloads:

| GPU | VRAM | Suitable For |
|---|---|---|
| T4 | 16 GB | Medium datasets, cloud notebooks |
| A10G | 24 GB | Large datasets, production ETL |
| L4 | 24 GB | Edge analytics, cost-optimised |
| A100 40/80 GB | 40–80 GB | Very large datasets, full pipeline |

Unlike deep learning training, RAPIDS does not require [[NVLink and NVSwitch]] for most workloads — a single GPU handles the majority of data science use cases.

---

## RAPIDS vs. Deep Learning Frameworks

| Aspect | RAPIDS | PyTorch / TensorFlow |
|---|---|---|
| **Primary workload** | Classical ML + data engineering | Deep learning (neural networks) |
| **Data type** | Tabular, graph, geospatial, signal | Images, text, audio, tabular |
| **scikit-learn compat** | ✅ Drop-in replacement | ❌ Different paradigm |
| **GPU memory need** | Low–moderate | High (especially LLMs) |
| **Pipeline integration** | End-to-end GPU via cuDF | Data usually preprocessed on CPU |

> [!note]
> RAPIDS and deep learning frameworks are **complementary**. A production AI pipeline might use cuDF for data ingestion and feature engineering, [[cuML]] for classical model baselines and feature selection, and PyTorch for the deep learning components — all running on the same GPU infrastructure.

---

## Related Notes

- [[cuDF]] – GPU DataFrame library — the ETL foundation of the RAPIDS pipeline
- [[cuML]] – GPU classical ML algorithms — GPU scikit-learn
- [[cuGraph]] – GPU graph analytics — network analysis, community detection, PageRank
- [[cuSpatial]] – GPU geospatial analytics — spatial joins and trajectory analysis
- [[cuSignal]] – GPU signal processing — time-series, FFT, filtering pipelines
- [[RAFT]] – shared GPU primitives used internally across cuML and cuGraph
- [[cuCIM]] – GPU image and medical imaging library
- [[CUDA]] – the compute layer all RAPIDS libraries run on
- [[NGC Catalog]] – source for all RAPIDS container images
- [[AI Containers]] – how RAPIDS is packaged and deployed
- [[GPU vs CPU]] – RAPIDS is the clearest demonstration of GPU parallelism benefits outside deep learning
- [[GPU Architecture]] – T4, A10G, and A100 are the typical RAPIDS hardware targets

---

## Key Mental Model

RAPIDS is the **GPU-native data science workbench**. Think of it as replacing the entire CPU-based Python data science stack — not just the model training step, but every step from raw data to final prediction — with GPU equivalents that share the same API. The goal isn't to replace deep learning frameworks; it's to ensure that everything *around* the model (data loading, preprocessing, feature engineering, classical baselines, graph analytics) gets the same GPU acceleration as the model itself. One GPU, zero wasted cycles on CPU-bound data wrangling.
