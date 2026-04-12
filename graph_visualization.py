import argparse
import random
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Set

import hnswlib
import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import umap
from scipy.sparse.csgraph import laplacian
from sklearn.manifold import MDS
import heapq
import logging


# =========================
# Constants
# =========================
EDGE_ALPHA_MIN = 0.01
EDGE_ALPHA_SCALE = 0.1
EDGE_WIDTH_MIN = 0.25
EDGE_WIDTH_SCALE = 0.75


# =========================
# Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

logger = logging.getLogger(__name__)

def log(step, message):
    step_upper = step.upper()

    if step_upper in ("WARN", "WARNING"):
        logger.warning(f"[{step_upper}] {message}")
    elif step_upper in ("ERR", "ERROR"):
        logger.error(f"[{step_upper}] {message}")
    elif step_upper in ("DEBUG",):
        logger.debug(f"[{step_upper}] {message}")
    else:
        logger.info(f"[{step_upper}] {message}")

# =========================
# Data Loading
# =========================
def load_data(index_path: str, hdf5_path: str, n_nodes: int):
    """Load HNSW index and vector dataset."""
    idx = hnswlib.Index(space='ip', dim=25)
    idx.load_index(index_path, max_elements=n_nodes)

    with h5py.File(hdf5_path, 'r') as f:
        data = f['train'][:n_nodes].astype(np.float32)

    return idx, data


# =========================
# Graph Construction
# =========================
def extract_layers(idx, n_nodes: int):
    """Extract nodes and edges per HNSW layer."""
    layer_edges: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)
    layer_nodes: Dict[int, Set[int]] = defaultdict(set)

    for node in range(n_nodes):
        links = idx.get_links(node)
        for layer, neighbors in links.items():
            layer_nodes[layer].add(node)
            for nb in neighbors:
                edge = (min(node, nb), max(node, nb))
                layer_edges[layer].add(edge)

    return layer_nodes, layer_edges


def build_graph(layer_edges): 
    """Build a NetworkX graph from edge list."""
    G = nx.Graph() 
    for edges in layer_edges.values(): 
        G.add_edges_from(edges) 
    return G

# =========================
# Layouts
# =========================
def compute_intrinsic_layout(nodes, edges):
    """Spectral layout using Laplacian eigenvectors."""
    n = len(nodes)
    if n < 3 or not edges:
        return np.random.randn(n, 2)

    node_idx = {node: i for i, node in enumerate(nodes)}
    rows, cols = [], []
    for u, v in edges:
        i, j = node_idx[u], node_idx[v]
        rows += [i, j]
        cols += [j, i]

    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    L_sparse = laplacian(A, normed=True)

    k = min(3, n - 1)
    _, eigvecs = spla.eigsh(L_sparse, k=k, which='SM')
    return eigvecs[:, 1:3] if k >= 3 else eigvecs[:, 0:2]


def landmark_mds(G, n_landmarks=200):
    """Compute a scalable approximation of MDS using landmark nodes."""
    nodes_g = list(G.nodes)
    n_g = len(nodes_g)
    n_landmarks = min(n_landmarks, n_g)
    node_idx = {node: i for i, node in enumerate(nodes_g)}

    landmarks = random.sample(nodes_g, n_landmarks)
    landmark_idx = [node_idx[l] for l in landmarks]

    lengths = dict(nx.all_pairs_shortest_path_length(G))
    D_landmarks = np.zeros((n_landmarks, n_g))
    for i, l in enumerate(landmarks):
        for j, v in enumerate(nodes_g):
            D_landmarks[i, j] = lengths[l].get(v, n_g)

    mds = MDS(
        n_components=2,
        metric='precomputed',
        normalized_stress='auto',
        random_state=42,
        n_init=1,
        init='classical_mds'
    )
    pos_landmarks = mds.fit_transform(D_landmarks[:, landmark_idx])

    pos_all = np.zeros((n_g, 2))
    for i, node in enumerate(nodes_g):
        if node in landmarks:
            pos_all[i] = pos_landmarks[landmark_idx.index(i)]
        else:
            dists = D_landmarks[:, i]
            weights = 1 / (dists + 1e-6)
            pos_all[i] = np.average(pos_landmarks, axis=0, weights=weights)

    return pos_all


def compute_mds_layout(nodes, edges, landmark_cutoff=200, n_landmarks=200):
    """MDS layout based on shortest path distances."""
    n = len(nodes)
    if n <= 2:
        return np.random.randn(n, 2)

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    lengths = dict(nx.all_pairs_shortest_path_length(G))
    D = np.array([[lengths[u].get(v, n) for v in nodes] for u in nodes], dtype=float)

    if n <= landmark_cutoff:
        return MDS(
            n_components=2,
            metric='precomputed',
            normalized_stress='auto',
            random_state=42,
            n_init=1,
            init='classical_mds'
        ).fit_transform(D)
    else:
        return landmark_mds(G, n_landmarks=n_landmarks)


def compute_umap_layout(nodes, data):
    """UMAP embedding of node vectors."""
    n = len(nodes)
    if n <= 2:
        return np.random.randn(n, 2), 2

    n_neighbors = min(15, max(2, n // 3))
    n_neighbors = min(n_neighbors, n - 1)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric='cosine',
        random_state=42,
        n_jobs=1
    )
    return reducer.fit_transform(data[nodes]), n_neighbors

# =========================
# Scores (search importance and embedding densityy)
# =========================

def normalize_scores(score_dict):
    """helper for score normalization."""
    values = np.array(list(score_dict.values()), dtype=float)
    if len(values) == 0:
        return score_dict

    min_v = values.min()
    max_v = values.max()

    if max_v - min_v < 1e-9:
        return {k: 0.0 for k in score_dict}

    return {
        k: (v - min_v) / (max_v - min_v)
        for k, v in score_dict.items()
    }

def print_top_3_scores(scores):
    """helper for printing top 3 nodes sorted by score."""
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True )[:3]

    for k, v in sorted_scores:
        print(f"-- {k}: {v:.4f}")



def hnsw_search_trace(G: nx.Graph, data: np.ndarray, query_vec, entry_node: int, ef: int):
    """Used for estimating search importance."""
    visited = set()
    visited_edges = defaultdict(int)

    def dist(u):
        return -np.dot(data[u], query_vec)  # inner product search

    candidates = [(dist(entry_node), entry_node)]
    heapq.heapify(candidates)

    while candidates and len(visited) < ef:
        d, node = heapq.heappop(candidates)

        if node in visited:
            continue

        visited.add(node)

        for nb in G[node]:
            visited_edges[(min(node, nb), max(node, nb))] += 1

            if nb not in visited:
                heapq.heappush(candidates, (dist(nb), nb))

    return visited, visited_edges


def estimate_query_importance(G: nx.Graph, data: np.ndarray, layer_nodes, n_queries=500, ef=50):
    """Estimate node/edge importance via simulated queries."""
    node_counts = defaultdict(int)
    edge_counts = defaultdict(int)

    # entry point = any node in highest layer
    top_layer = max(layer_nodes.keys())
    entry_candidates = list(layer_nodes[top_layer])

    for _ in range(n_queries):
        q = data[np.random.randint(0, len(data))]
        entry = random.choice(entry_candidates)

        visited_nodes, visited_edges = hnsw_search_trace(
            G, data, q, entry, ef=ef
        )

        for n in visited_nodes:
            node_counts[n] += 1

        for e, c in visited_edges.items():
            edge_counts[e] += c

    return node_counts, edge_counts


def estimate_global_density(idx, data: np.ndarray, nodes: List[int], k: int = 100):
    """Calculating the nodes' densities = 1/d with d the embedding distance (similarity) to knn."""
    vecs = data[nodes]

    labels, distances = idx.knn_query(vecs, k=k + 1)

    densities = {}
    for i, node in enumerate(nodes):
        dists = distances[i][1:k+1]
        avg_dist = np.mean(dists)
        densities[node] = 1.0 / (avg_dist + 1e-9)

    return densities

# =========================
# Node neighborhoods
# =========================
def get_k_closest_nodes_graph(G, source, k=100):
    """Retrieve the k closest nodes to a source node using graph distance."""
    visited = set([source])
    queue = deque([source])

    result = []

    while queue and len(result) < k:
        node = queue.popleft()
        result.append(node)

        for nb in G[node]:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)

    return result

def get_k_closest_nodes_umap(all_nodes, data, source, k=100):
    """Retrieve the k closest nodes to a source node in embedding space."""
    vecs = data[all_nodes]
    source_vec = data[source]

    dists = np.linalg.norm(vecs - source_vec, axis=1)
    idx = np.argsort(dists)[:k]

    return [all_nodes[i] for i in idx]


def select_k_neighborhood(
    layer: int,
    layer_nodes,
    layer_edges,
    data: np.ndarray,
    node_id: int | None = None,
    distance_mode: str = "graph",
    neighborhood_size: int = 500,
):
    """
    Select a subset of nodes and edges from a given HNSW layer.

    If node_id is provided, returns the k closest nodes to that node
    according to the selected distance mode. Otherwise, returns all nodes in the layer.

    Args:
        layer (int): Layer index.
        layer_nodes (dict): Mapping layer -> set of nodes.
        layer_edges (dict): Mapping layer -> set of edges.
        data (np.ndarray): Node embeddings.
        node_id (int | None): Center node for neighborhood selection.
        distance_mode (str): 'graph' (BFS distance) or 'umap' (vector distance).
        neighborhood_size (int): Number of nodes to select.

    Returns:
        tuple:
            nodes (List[int]): Selected node IDs.
            edges (List[Tuple[int, int]]): Filtered edges among selected nodes.

    Notes:
        - 'graph' mode uses BFS → respects graph structure.
        - 'umap' mode uses vector similarity → respects embedding topology.
    """
    # Build layer graph once
    G_layer = nx.Graph()
    G_layer.add_edges_from(layer_edges[layer])

    # --- Node selection ---
    if node_id is None:
        nodes = list(layer_nodes[layer])
    else:
        if distance_mode == "graph":
            nodes = get_k_closest_nodes_graph(G_layer, node_id, k=neighborhood_size)
        elif distance_mode == "umap":
            all_nodes = list(layer_nodes[layer])
            nodes = get_k_closest_nodes_umap(all_nodes, data, node_id, k=neighborhood_size)
        else:
            raise ValueError(f"Unknown distance mode: {distance_mode}")

    node_set = set(nodes)

    # --- Edge filtering ---
    edges = [
        (u, v)
        for (u, v) in layer_edges[layer]
        if u in node_set and v in node_set
    ]

    return nodes, edges

# =========================
# Plotting
# =========================
def compute_node_styles(nodes, node_scores):
    vals = np.array([node_scores.get(n, 0) for n in nodes], dtype=float)

    vals = np.log1p(vals)
    vals /= (vals.max() + 1e-9)

    cmap = plt.get_cmap('plasma')
    colors = cmap(vals)
    sizes = 20 + 300 * vals

    return vals, colors, sizes, cmap


def compute_edge_strength(edges, edge_importance):
    vals = np.array([
        edge_importance.get((min(u, v), max(u, v)), 0)
        for (u, v) in edges
    ], dtype=float)

    vals = np.log1p(vals)
    vals /= (vals.max() + 1e-9)

    return vals


def compute_edge_alpha_scale(n_edges):
    # reference ~20k edges → scale = 1
    ref = 20000
    scale = np.sqrt(ref / max(n_edges, 1))

    # clamp to avoid extremes
    return np.clip(scale, 0.5, 10.0)


def plot_graph(
    layer,
    nodes,
    edges,
    data,
    node_importance_global,
    edge_importance,
    node_importance_local=None,
    density_global=None,
    title_suffix="",
    highlight_node=None,
    distance_map=None
):
    """Main visualization function."""
    edge_alpha_boost = compute_edge_alpha_scale(len(edges))
    node_idx = {node: i for i, node in enumerate(nodes)}

    pos_intrinsic = compute_intrinsic_layout(nodes, edges)
    pos_mds = compute_mds_layout(nodes, edges)
    pos_umap, n_neighbors = compute_umap_layout(nodes, data)

    plots = [(pos_intrinsic, "Intrinsic (Laplacian)")]
    if pos_mds is not None:
        plots.append((pos_mds, "MDS (shortest path)"))
    plots.append((pos_umap, f"UMAP (n_neighbors={n_neighbors})"))

    if node_importance_local is None:
        node_importance_local = node_importance_global

    node_vals, colors, sizes, cmap = compute_node_styles(nodes, node_importance_local)
    edge_vals = compute_edge_strength(edges, edge_importance)

    fig, axes = plt.subplots(
        1, 3,
        figsize=(15, 5),
        constrained_layout=True
    )

    scatters = []

    for ax, (pos, title) in zip(axes, plots):
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        for (u, v), strength in zip(edges, edge_vals):
            i, j = node_idx[u], node_idx[v]
            ax.plot(
                [pos[i, 0], pos[j, 0]],
                [pos[i, 1], pos[j, 1]],
                alpha = np.clip(
                    (EDGE_ALPHA_MIN + EDGE_ALPHA_SCALE * strength) * edge_alpha_boost,
                    0.0,
                    1.0
                ),
                lw=EDGE_WIDTH_MIN + EDGE_WIDTH_SCALE * strength,
                color=cm.inferno(strength)
            )

        sc = ax.scatter(
            pos[:, 0], pos[:, 1],
            s=sizes,
            c=colors,
            alpha=0.9,
            edgecolors='white',
            linewidths=0.5,
            picker=True
        )

        scatters.append(sc)

        # --- highlight input node in green ---
        if highlight_node is not None and highlight_node in node_idx:
            idx_highlight = node_idx[highlight_node]
            x, y = pos[idx_highlight]

            ax.scatter(
                [x], [y],
                s=500,
                facecolors='none',
                edgecolors='green',
                linewidths=2.5,
                zorder=11
            )

    highlight_artists = []
    def on_pick(event):
        if hasattr(event, "ind") and len(event.ind) > 0:
            idx_clicked = event.ind[0]
            node_id = nodes[idx_clicked]
            importance = node_importance_global.get(node_id, 0)
            density = density_global.get(node_id, 0) if density_global else 0

            if distance_map is not None:
                graph_distance = distance_map.get(node_id, -1)
                print(
                    f"Clicked node: {node_id} | "
                    f"importance={importance:.4f} | "
                    f"density={density:.4f} | "
                    f"graph distance={graph_distance}"
                )
            else:
                print(
                    f"Clicked node: {node_id} | "
                    f"importance={importance:.4f} | "
                    f"density={density:.4f}"
                )

            # --- remove previous highlights ---
            for artist in highlight_artists:
                artist.remove()
            highlight_artists.clear()

            # --- add new highlights on all subplots ---
            for ax, (pos, _) in zip(axes, plots):
                x, y = pos[idx_clicked]
                circle = ax.scatter(
                    [x], [y],
                    s=400,
                    facecolors='none',
                    edgecolors='red',
                    linewidths=2,
                    zorder=10
                )
                highlight_artists.append(circle)

            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('pick_event', on_pick)

    sm = cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("Node importance")

    if highlight_node is None:
        context = "Full graph"
    else:
        context = f"Neighborhood of node {highlight_node}"

    plt.suptitle(
        f"HNSW Layer {layer} — {context} — {len(nodes)} nodes, {len(edges)} edges"
    )

    plt.show()


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int,
                        help='Layer to display (view layer L)')
    parser.add_argument('--node_id', type=int, default=None,
                        help='Node ID to focus on')
    parser.add_argument('--distance_mode', type=str, default='umap',
                        choices=['umap', 'graph'],
                        help='Distance used to select neighborhood of node: "umap" or "graph"')
    parser.add_argument('--neighborhood_size', type=int, default=500,   
                        help='Number of closest nodes to keep')
    parser.add_argument('--n_nodes', type=int, default=50000)
    args = parser.parse_args()

    # fixing seed for reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    log("LOAD", "Loading data...")
    idx, data = load_data(
        'data/glove25_50k.bin',
        'data/glove-25-angular.hdf5',
        args.n_nodes
    )
    log("LOAD", f"Loaded data: {len(data)} vectors")
    
    log("LAYERS", "Extracting layers...")
    layer_nodes, layer_edges = extract_layers(idx, n_nodes=args.n_nodes)

    for l in sorted(layer_nodes):
        print(f"[LAYERS] layer {l}: {len(layer_nodes[l])} nodes, {len(layer_edges[l])} edges")

    if args.layer is None:
        args.layer = max(layer_nodes.keys())
        log("ARGS", f"No layer provided: using max layer {args.layer}")

    if args.layer not in layer_nodes:
        log("WARN", f"Layer {args.layer} does not exist")
        args.layer = max(layer_nodes.keys())
        log("ARGS", f"Fallback: using max layer {args.layer}")
    
    if args.node_id is not None and args.node_id not in layer_nodes[args.layer]:
        log("WARN", f"Node {args.node_id} not in layer {args.layer}")
        args.node_id = None

    if args.node_id is not None:
        log("SELECT", f"Selecting neighborhood (size={args.neighborhood_size}, distance mode={args.distance_mode})")
    nodes, edges = select_k_neighborhood(
        layer=args.layer,
        layer_nodes=layer_nodes,
        layer_edges=layer_edges,
        data=data,
        node_id=args.node_id,
        distance_mode=args.distance_mode,
        neighborhood_size=args.neighborhood_size
    )
    
    log("GRAPH", "Building graph...")
    graph = build_graph(layer_edges)

    log("IMPORTANCE", "Estimating search importance...")
    node_importance_raw, edge_importance = estimate_query_importance(
        graph,
        data,
        layer_nodes
    )

    # --- restrict to current layer only ---
    layer_node_set = set(layer_nodes[args.layer])

    node_importance_raw = {
        n: c for n, c in node_importance_raw.items()
        if n in layer_node_set
    }


    log("IMPORTANCE", f"Top 3 nodes for search importance:")
    # --- normalize importance ---
    node_importance_global = normalize_scores(node_importance_raw)

    # --- calculate local importance if a node id is provided ---
    if args.node_id is not None:
        local_importance = {
            n: node_importance_global.get(n, 0)
            for n in nodes
        }
        local_importance = normalize_scores(local_importance)
    else:
        local_importance = node_importance_global

    print_top_3_scores(node_importance_global)

    log("DENSITY", "Estimating global density...")
    density_raw = estimate_global_density(
        idx,
        data,
        list(layer_node_set)
    )

    log("DENSITY", "Top 3 nodes for embedding density:")
    density_global = normalize_scores(density_raw)
    print_top_3_scores(density_global)


    distance_map = None
    if args.node_id is not None:
        log("DISTANCE MAP", "Mapping node distances...")
        distance_map = nx.single_source_shortest_path_length(graph, args.node_id)

    log("PLOT", f"Plotting layer {args.layer}...")
    plot_graph(
        args.layer,
        nodes,
        edges,
        data,
        node_importance_global,   # for click
        edge_importance,
        node_importance_local=local_importance,
        density_global=density_global,
        highlight_node=args.node_id,
        distance_map=distance_map
    )

if __name__ == '__main__':
    main()