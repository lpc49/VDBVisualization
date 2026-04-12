# VDBVisualization

> Interactive visualization of vector database graphs built with HNSW (Hierarchical Navigable Small World), designed for pedagogy, dataset inspection, and algorithm comparison.

---

## Overview

This tool renders the internal graph structure of an HNSW index across its layers, using three complementary layouts — Laplacian (intrinsic), MDS (shortest-path), and UMAP (embedding). Nodes are colored and sized by **search importance** (how frequently they are visited during greedy search) and the plots are fully interactive.

**Current limitations and planned work:**
- Only normalized angular (cosine) datasets are supported. Euclidean distance support is planned for a future commit.
- The codebase is currently a single file. Splitting into modules with unit tests is planned.
- The tool is designed for pedagogy around HNSW specifically. A future version will support comparison with other ANN algorithms (e.g., IVF-PQ, NSG).
- Not optimized for large-scale datasets. Python was chosen deliberately to leverage standard scientific packages. The MVP targets GloVe-25 (dimension 25, 50k vectors).

---

## Setup

### 1. Create environment

```bash
conda create -n hnsw-py311 python=3.11
conda activate hnsw-py311
pip install h5py matplotlib networkx scipy scikit-learn umap-learn
```

### 2. Build hnswlib (vendored)

hnswlib is vendored locally with a custom binding (`get_links`) that exposes per-layer neighbor lists. See [File Notes](#file-notes) for details.

```bash
cd hnswlib
pip install .
```

### 3. Fetch data

```bash
mkdir data
cd data
wget http://ann-benchmarks.com/glove-25-angular.hdf5
cd ..
python build_index_subset.py
```

This builds and saves an HNSW index over the first 50k GloVe-25 vectors.

---

## Usage

### View the top (highest) layer

```bash
python graph_visualization.py
```

### View a specific layer (e.g., layer 1)

```bash
python graph_visualization.py --layer=1
```

### Zoom into a neighborhood around a specific node

With UMAP (embedding) distance — default:
```bash
python graph_visualization.py --layer=1 --node_id=123
```

With graph (BFS) distance:
```bash
python graph_visualization.py --layer=1 --node_id=123 --distance_mode=graph
```

The selected node is highlighted with a **green circle**. The neighborhood size can be controlled:

```bash
python graph_visualization.py --layer=1 --node_id=123 --neighborhood_size=300
```

### Interactive features

Clicking on any node in the plot prints its metadata to the console and highlights it with a **red circle**:

```
Clicked node: 4821 | importance=0.7312 | density=0.5140 | graph distance=3
```

---

## Key Concepts

### Graph layouts

| Layout | Method | Interpretation |
|---|---|---|
| **Intrinsic (Laplacian)** | Spectral embedding via normalized graph Laplacian eigenvectors | Reveals intrinsic graph topology — entry points appear as outliers |
| **MDS (shortest path)** | Classical MDS on pairwise BFS distances | Reflects navigation distances; high-importance nodes cluster centrally |
| **UMAP** | Nonlinear dimensionality reduction on raw embedding vectors | Most faithful to the original vector space geometry |

### Scores

**Search importance** (`node_importance`, `edge_importance`)  
Estimated by simulating `n_queries=500` random greedy searches from the top-layer entry point. A node/edge's importance score is proportional to how many search traces pass through it. High-importance nodes are the "hubs" of the HNSW graph — skipping them would significantly degrade recall.

**Embedding density** (`density`)  
Computed as `1 / mean_distance_to_knn`, where distances are inner-product distances in the original vector space. High-density nodes lie in crowded regions of the embedding space, where many semantically similar vectors cluster together.

### Distance modes (for neighborhood selection)

| Mode | Definition | Use case |
|---|---|---|
| `graph` | BFS hop count on the HNSW layer graph | Explore the graph neighborhood as a search would traverse it |
| `umap` | L2 distance in raw embedding space | Explore which vectors are semantically closest |

**Graph distance** (printed on node click) counts the number of edges in the shortest path between the selected node and the clicked node in the full HNSW graph.

---

## Results

### Full layer 1

*(Insert screenshot here)*

- **Intrinsic layout:** Nodes cluster in the center, with entry-point nodes forming clear spurs at the periphery. This reflects the HNSW design: upper-layer entry points have long-range connections that place them structurally distant from the densely connected base cluster.
- **MDS layout:** Star-shaped structure, with the highest-importance nodes clustering near the center. This is consistent with HNSW's greedy routing — central hubs are visited by nearly every search path regardless of query direction.
- **UMAP layout:** Reveals the underlying embedding geometry — arms of densely packed vectors corresponding to semantic clusters in the GloVe space.

### Neighborhood around a node (graph distance)

*(Insert screenshot here)*

Zooming into a node's graph neighborhood preserves the structural patterns seen at the full-graph level — a self-similarity consistent with HNSW's hierarchical small-world construction.

### Neighborhood around a node (UMAP/embedding distance)

*(Insert screenshot here)*

Some high-importance nodes that are close in embedding space are not directly connected by an edge — they require a path of 2 or more hops. This reveals a key HNSW property: the graph is not a nearest-neighbor graph. Connectivity is determined during index construction by the order of insertion and the `M` parameter, not purely by embedding proximity. Nodes that appear close in UMAP may only be reachable via intermediaries, reflecting the approximate (rather than exact) nature of HNSW search.

---

## File Notes

`hnswlib` is vendored from commit [`d9b3608`](https://github.com/nmslib/hnswlib/commit/d9b3608c83d83b46c96e25088cb1d729b29dcfe9) of the official repository.

**Local modification:** `bindings.cpp` — added `get_links()`, which exposes per-layer neighbor lists for a given node. This is not available in the upstream library and is required for layer extraction.

---

## Other datasets

Other ANN benchmark datasets (e.g., SIFT-128) are available at:  
[https://github.com/erikbern/ann-benchmarks/](https://github.com/erikbern/ann-benchmarks/?tab=readme-ov-file)

> ⚠️ Only normalized angular datasets are currently supported.

---

## Read more

See the accompanying [LaTeX article](link to ArXiV) for a detailed treatment of HNSW, the visualization methodology, and interpretation of results.