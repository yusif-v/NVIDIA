---
tags: [nvidia, software, rapids, machine-learning, general-knowledge]
aliases: [cuML, RAPIDS cuML, GPU machine learning, RAPIDS, cuDF, cuGraph]
---

# cuML

> **Exam Relevance**: ⚠️ Not on NCA-AIIO blueprint — general NVIDIA ecosystem knowledge
> **Related**: [[CUDA]], [[NGC Catalog]], [[AI Containers]], [[GPU vs CPU]], [[Training vs Inference]], [[GPU Architecture]]

## Overview

cuML is NVIDIA's GPU-accelerated machine learning library — part of the **RAPIDS** open-source ecosystem. It reimplements classical ML algorithms (linear regression, random forests, k-means, PCA, UMAP, SVM, DBSCAN, and more) to execute on NVIDIA GPUs via [[CUDA]], delivering speedups of 10–100× over CPU-based scikit-learn on large datasets. The API is **intentionally identical to scikit-learn** — switching from CPU to GPU requires only changing the import statement. cuML keeps data resident on GPU across the full pipeline when combined with cuDF (GPU DataFrames), eliminating costly CPU↔GPU transfers between preprocessing and training.

> [!warning]
> cuML is **not covered by the NCA-AIIO exam**. This note is general NVIDIA ecosystem knowledge — useful for understanding GPU acceleration beyond deep learning, but not a study priority for the cert.

---

## The RAPIDS Ecosystem

cuML is one library in NVIDIA's **RAPIDS** suite — a collection of GPU-accelerated data science libraries designed as drop-in replacements for standard CPU tools:

| RAPIDS Library | Role | CPU Equivalent |
|---|---|---|
| **cuDF** | GPU DataFrame operations — load, filter, join, aggregate | Pandas |
| **cuML** | GPU classical ML algorithms | scikit-learn |
| **cuGraph** | GPU graph analytics — PageRank, community detection, BFS | NetworkX |
| **cuSpatial** | GPU geospatial analytics | GeoPandas |
| **cuSignal** | GPU signal processing | SciPy signal |
| **RAFT** | Shared GPU primitives — nearest neighbours, clustering, linear algebra | — |

The key design principle: **data stays on GPU across the entire pipeline**. cuDF handles ingestion and feature engineering; cuML trains the model — no host↔device transfers between stages.

---

## Drop-in scikit-learn Replacement

cuML's defining feature is API compatibility with scikit-learn. The import changes; the rest of the code does not:

```python
# ── CPU baseline (scikit-learn) ──────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

rf = RandomForestClassifier(n_estimators=500, max_depth=10)
rf.fit(X_train, y_train)

# ── GPU equivalent (cuML) ────────────────────────────────────────────
from cuml.ensemble import RandomForestClassifier
from cuml.cluster import KMeans
from cuml.decomposition import PCA

rf = RandomForestClassifier(n_estimators=500, max_depth=10)  # identical
rf.fit(X_train, y_train)                                      # runs on GPU
```

This zero-friction migration path is intentional — NVIDIA's goal is to eliminate the barrier between existing scikit-learn workflows and GPU acceleration.

---

## Supported Algorithms

cuML covers all major classical ML algorithm families:

### Supervised Learning
| Algorithm | cuML Class |
|---|---|
| Linear Regression | `cuml.linear_model.LinearRegression` |
| Logistic Regression | `cuml.linear_model.LogisticRegression` |
| Ridge / Lasso | `cuml.linear_model.Ridge` / `Lasso` |
| Random Forest (classifier + regressor) | `cuml.ensemble.RandomForestClassifier` |
| Support Vector Machine | `cuml.svm.SVC` / `SVR` |
| K-Nearest Neighbours | `cuml.neighbors.KNeighborsClassifier` |

### Unsupervised Learning
| Algorithm | cuML Class |
|---|---|
| K-Means | `cuml.cluster.KMeans` |
| DBSCAN | `cuml.cluster.DBSCAN` |
| PCA | `cuml.decomposition.PCA` |
| UMAP | `cuml.manifold.UMAP` |
| t-SNE | `cuml.manifold.TSNE` |
| Isolation Forest | `cuml.ensemble.IsolationForest` |

---

## End-to-End GPU Pipeline with cuDF + cuML

The full power of RAPIDS comes from combining cuDF and cuML so data never leaves the GPU:

```python
import cudf                                    # GPU DataFrame (like Pandas)
import cuml
from cuml.ensemble import RandomForestClassifier
from cuml.model_selection import train_test_split

# Load data directly into GPU memory
df = cudf.read_csv("dataset.csv")             # stays on GPU

# Feature engineering on GPU
df["feature_ratio"] = df["col_a"] / df["col_b"]
X = df.drop("label", axis=1)
y = df["label"]

# Split on GPU
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train on GPU — no CPU round-trip
model = RandomForestClassifier(n_estimators=500)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"Accuracy: {score:.4f}")
```

---

## Performance Characteristics

cuML speedups are dataset-size dependent — the larger the dataset, the greater the GPU advantage:

| Dataset Size | scikit-learn (CPU) | cuML (GPU) | Approx. Speedup |
|---|---|---|---|
| 100K rows | Fast (seconds) | Fast (seconds) | ~2–5× |
| 1M rows | Minutes | Seconds | ~10–50× |
| 10M rows | Hours | Minutes | ~50–100× |
| 100M rows | Impractical | Minutes–hours | Transformative |

> [!note]
> For small datasets (< 100K rows), scikit-learn on CPU is often faster due to GPU kernel launch overhead. cuML shines at scale — it's designed for datasets where CPU becomes a bottleneck, not small tabular experiments.

---

## Deployment and Distribution

cuML is available as:

```bash
# Via conda (recommended)
conda install -c rapidsai -c conda-forge -c nvidia \
  cuml=24.02 python=3.11 cuda-version=12.0

# Via pip
pip install cuml-cu12

# Via NGC container (recommended for reproducibility)
docker pull nvcr.io/nvidia/rapidsai/base:24.02-cuda12.0-py3.11
```

All RAPIDS containers are distributed via the [[NGC Catalog]] at `nvcr.io/nvidia/rapidsai/` and run as [[AI Containers]] on any NVIDIA GPU system.

---

## Hardware Requirements

cuML is less demanding than deep learning frameworks — it runs well on mid-range data center GPUs:

| GPU | VRAM | Suitable For |
|---|---|---|
| T4 | 16 GB | Medium datasets, edge inference |
| A10G | 24 GB | Large datasets, cloud ML |
| A100 40/80 GB | 40–80 GB | Very large datasets, production |
| L4 | 24 GB | Edge / cost-optimised inference |

Unlike deep learning, cuML does not require [[NVLink and NVSwitch]] or multi-GPU for most workloads — a single GPU is sufficient for most classical ML use cases.

> [!note] Cybersecurity Connection
> **cuGraph** — cuML's graph analytics sibling in RAPIDS — has direct security applications. Algorithms like PageRank, connected components, Louvain community detection, and breadth-first search are core tools in network threat analysis: mapping lateral movement paths, detecting C2 communication clusters, identifying anomalous graph structures in network flow data. Running these via cuGraph on a GPU enables real-time analysis of enterprise-scale network graphs (millions of nodes and edges) that would be impractically slow on CPU-based NetworkX. The GPU infrastructure you're studying for NCA-AIIO is the same infrastructure that runs these security analytics workloads.

---

## cuML vs. Deep Learning Frameworks

| Aspect | cuML (RAPIDS) | PyTorch / TensorFlow |
|---|---|---|
| **Algorithm type** | Classical ML (trees, clustering, regression, SVM) | Deep learning (neural networks) |
| **Model size** | Small–medium (fits in VRAM easily) | Can be very large (100B+ params) |
| **Data type** | Tabular (structured, numeric, categorical) | Images, text, audio, tabular |
| **Training time** | Seconds to minutes | Hours to days |
| **GPU memory need** | Low–moderate | High (especially LLMs) |
| **scikit-learn compat** | ✅ Drop-in | ❌ Different API |

> [!note]
> cuML and deep learning frameworks are **complementary, not competing**. A real AI pipeline often uses both: cuML for feature engineering and classical model baselines, PyTorch/TensorFlow for deep learning components.

---

## Related Notes

- [[CUDA]] – the compute layer all cuML operations run on
- [[NGC Catalog]] – source for RAPIDS container images (`nvcr.io/nvidia/rapidsai/`)
- [[AI Containers]] – how cuML is packaged and deployed in production
- [[GPU vs CPU]] – cuML is the clearest demonstration of GPU parallelism advantages for non-DL workloads
- [[Training vs Inference]] – cuML covers training of classical models; inference is typically CPU or cuML predict()
- [[GPU Architecture]] – T4 and A10G are the typical hardware targets for cuML workloads

---

## Key Mental Model

cuML is **scikit-learn with a GPU engine swap**. The car looks identical from the driver's seat — same steering wheel (API), same pedals (method calls), same dashboard (output format). Under the hood, the CPU engine has been replaced with a GPU. You don't need to learn a new car to drive it faster — you just need to understand why the engine swap matters: parallelism at scale. For datasets where the CPU was the bottleneck, the speedup is transformative. For small datasets, you won't notice the difference.

> [!note]
> The three things worth remembering about cuML for any NVIDIA role: **(1)** it's scikit-learn-compatible — zero migration friction, **(2)** it's part of the **RAPIDS** ecosystem alongside cuDF and cuGraph — the full data science pipeline runs on GPU, and **(3)** speedups are dataset-size dependent — it's designed for scale, not toy datasets.
