---
tags: [nvidia, software, rapids, machine-learning, general-knowledge]
aliases: [cuML, RAPIDS cuML, GPU machine learning, GPU scikit-learn]
---

# cuML

> **Exam Relevance**: ⚠️ Not on NCA-AIIO blueprint — general NVIDIA ecosystem knowledge
> **Related**: [[RAPIDS]], [[CUDA]], [[NGC Catalog]], [[AI Containers]], [[GPU vs CPU]], [[Training vs Inference]]

## Overview

cuML is the machine learning library within the **[[RAPIDS]]** ecosystem — a GPU-accelerated drop-in replacement for scikit-learn. It reimplements classical ML algorithms (linear regression, random forests, k-means, PCA, UMAP, SVM, DBSCAN, and more) to execute entirely on NVIDIA GPUs via [[CUDA]], delivering 10–100× speedups over CPU-based scikit-learn on large datasets. The API is intentionally identical to scikit-learn — switching from CPU to GPU requires only changing the import statement. When combined with cuDF (GPU DataFrames from the [[RAPIDS]] suite), cuML enables a fully GPU-resident ML pipeline with no CPU↔GPU transfers between preprocessing and training.

> [!warning]
> cuML is **not covered by the NCA-AIIO exam**. This note is general NVIDIA ecosystem knowledge. See [[RAPIDS]] for the full ecosystem context.

---

## Drop-in scikit-learn Replacement

cuML's defining feature is API compatibility with scikit-learn:

```python
# ── CPU baseline (scikit-learn) ──────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

model = RandomForestClassifier(n_estimators=500, max_depth=10)
model.fit(X_train, y_train)

# ── GPU equivalent (cuML) — only the import changes ──────────────────
from cuml.ensemble import RandomForestClassifier
from cuml.cluster import KMeans
from cuml.decomposition import PCA

model = RandomForestClassifier(n_estimators=500, max_depth=10)
model.fit(X_train, y_train)   # runs on GPU
```

---

## Supported Algorithms

### Supervised Learning

| Algorithm | cuML Class |
|---|---|
| Linear Regression | `cuml.linear_model.LinearRegression` |
| Logistic Regression | `cuml.linear_model.LogisticRegression` |
| Ridge / Lasso | `cuml.linear_model.Ridge` / `Lasso` |
| Random Forest | `cuml.ensemble.RandomForestClassifier` / `Regressor` |
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

## End-to-End GPU Pipeline

Combined with cuDF from [[RAPIDS]], data never leaves GPU memory:

```python
import cudf                                      # GPU DataFrames
from cuml.ensemble import RandomForestClassifier
from cuml.model_selection import train_test_split

# Load and transform entirely on GPU
df = cudf.read_csv("large_dataset.csv")
X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train on GPU — no CPU round-trip at any step
model = RandomForestClassifier(n_estimators=500)
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.4f}")
```

---

## Performance at Scale

cuML speedups grow with dataset size — negligible on small data, transformative at scale:

| Dataset Size | scikit-learn (CPU) | cuML (GPU) | Approx. Speedup |
|---|---|---|---|
| 100K rows | Seconds | Seconds | ~2–5× |
| 1M rows | Minutes | Seconds | ~10–50× |
| 10M rows | Hours | Minutes | ~50–100× |
| 100M rows | Impractical | Minutes–hours | Transformative |

> [!note]
> For datasets under ~100K rows, scikit-learn on CPU is often faster due to GPU kernel launch overhead. cuML is designed for the scale where CPU becomes the bottleneck.

---

## Deployment

cuML is part of the [[RAPIDS]] container stack, distributed via the [[NGC Catalog]]:

```bash
# Via conda
conda install -c rapidsai -c nvidia cuml=24.02 python=3.11 cuda-version=12.0

# Via pip
pip install cuml-cu12

# Via NGC container (recommended)
docker run --gpus all -it --rm \
  nvcr.io/nvidia/rapidsai/base:24.02-cuda12.0-py3.11
```

---

## Related Notes

- [[RAPIDS]] – the parent ecosystem; cuML is one library within it alongside cuDF, cuGraph, and others
- [[CUDA]] – the compute layer cuML runs on
- [[NGC Catalog]] – source for RAPIDS/cuML container images
- [[GPU vs CPU]] – cuML demonstrates GPU parallelism benefits for classical ML, not just deep learning
- [[Training vs Inference]] – cuML handles training of classical models; `model.predict()` handles inference

---

## Key Mental Model

cuML is **scikit-learn with the CPU engine swapped for a GPU**. The dashboard is identical — same API, same method names, same output formats. Under the hood, every algorithm has been rewritten to exploit thousands of GPU cores running in parallel. The swap is invisible to your code; the speedup is very visible on large datasets.

> [!note]
> The three things worth remembering about cuML: **(1)** drop-in scikit-learn replacement — change the import, keep the code; **(2)** part of [[RAPIDS]] — pairs with cuDF for a fully GPU-resident pipeline; **(3)** dataset-size dependent speedup — designed for scale, not toy datasets.
