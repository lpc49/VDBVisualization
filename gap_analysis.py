import argparse
import random
from collections import defaultdict

import hnswlib
import h5py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import umap
from scipy.sparse.csgraph import laplacian
from sklearn.manifold import MDS


def load_data(index_path='data/glove25_50k.bin', hdf5_path='data/glove-25-angular.hdf5', n_nodes=50000):
    idx = hnswlib.Index(space='ip', dim=25)
    idx.load_index(index_path, max_elements=n_nodes)
    with h5py.File(hdf5_path, 'r') as f:
        data = f['train'][:n_nodes].astype(np.float32)
    return idx, data


def extract_layers(idx, n_nodes=50000):
    layer_edges = defaultdict(set)
    layer_nodes = defaultdict(set)

    for node in range(n_nodes):
        links = idx.get_links(node)
        for layer, neighbors in links.items():
            layer_nodes[layer].add(node)
            for nb in neighbors:
                layer_edges[layer].add((min(node, nb), max(node, nb)))

    for l in sorted(layer_nodes):
        print(f"layer {l}: {len(layer_nodes[l])} nodes, {len(layer_edges[l])} edges")

    return layer_nodes, layer_edges


def get_layer_subgraph(layer_nodes, layer_edges, view_layer, node_id=None, anchor_layer=None):
    if node_id is None:
        return sorted(layer_nodes[view_layer]), sorted(layer_edges[view_layer])

    if anchor_layer is None:
        candidates = [l for l in layer_nodes if node_id in layer_nodes[l]]
        if not candidates:
            raise ValueError(f"node_id={node_id} not found in any layer")
        anchor_layer = max(candidates)

    if node_id not in layer_nodes[anchor_layer]:
        raise ValueError(f"node_id={node_id} not found in anchor_layer={anchor_layer}")

    if view_layer > anchor_layer:
        raise ValueError("view_layer must be <= anchor_layer")

    frontier = {node_id}
    for l in range(anchor_layer, view_layer, -1):
        next_frontier = set()
        for u, v in layer_edges[l]:
            if u in frontier:
                next_frontier.add(v)
            elif v in frontier:
                next_frontier.add(u)
        frontier |= next_frontier

    nodes = sorted(frontier)
    node_set = set(nodes)
    edges = [(u, v) for (u, v) in layer_edges[view_layer] if u in node_set and v in node_set]
    return nodes, edges


def compute_intrinsic_layout(nodes, edges):
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


def compute_mds_layout(nodes, edges, landmark_cutoff=200, n_landmarks=200):
    n = len(nodes)
    if n <= 2:
        return np.random.randn(n, 2)

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    def landmark_mds(G, n_landmarks=200):
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
            dissimilarity='precomputed',
            normalized_stress='auto',
            random_state=42
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

    lengths = dict(nx.all_pairs_shortest_path_length(G))
    D = np.array([[lengths[u].get(v, n) for v in nodes] for u in nodes], dtype=float)

    if n <= landmark_cutoff:
        return MDS(
            n_components=2,
            dissimilarity='precomputed',
            normalized_stress='auto',
            random_state=42
        ).fit_transform(D)
    else:
        return landmark_mds(G, n_landmarks=n_landmarks)


def compute_umap_layout(nodes, data):
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
        random_state=42
    )
    return reducer.fit_transform(data[nodes]), n_neighbors


def plot_graph(layer, nodes, edges, data, title_suffix="", interactive=True):
    n = len(nodes)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    node_idx = {node: i for i, node in enumerate(nodes)}

    pos_intrinsic = compute_intrinsic_layout(nodes, edges)
    pos_mds = compute_mds_layout(nodes, edges)
    pos_umap, n_neighbors = compute_umap_layout(nodes, data)

    plots = [(pos_intrinsic, "Intrinsic: Laplacian (sparse)")]
    if pos_mds is not None:
        plots.append((pos_mds, "MDS (shortest path)"))
    plots.append((pos_umap, f"UMAP (n_neighbors={n_neighbors})"))

    fig, axes = plt.subplots(1, len(plots), figsize=(5 * len(plots), 5))
    if len(plots) == 1:
        axes = [axes]

    scatter_refs = []

    def on_pick(event):
        artist = event.artist
        for sc, ax in scatter_refs:
            if artist == sc and event.mouseevent.inaxes == ax:
                ind = event.ind[0]
                print(f"Clicked node id: {nodes[ind]}")
                break

    for ax, (pos, title) in zip(axes, plots):
        for u, v in edges:
            i, j = node_idx[u], node_idx[v]
            ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], alpha=0.15, lw=1.0)

        degrees = np.array([G.degree(node) for node in nodes], dtype=float)
        sc = ax.scatter(
            pos[:, 0], pos[:, 1],
            s=50 + 20 * degrees,
            alpha=0.9,
            edgecolors='white',
            linewidths=0.8,
            picker=True
        )
        scatter_refs.append((sc, ax))

        if n <= 50:
            for i, node in enumerate(nodes):
                ax.annotate(str(node), (pos[i, 0], pos[i, 1]), fontsize=6, ha='center', va='center')

        ax.set_title(title)
        ax.axis('off')

    if interactive:
        fig.canvas.mpl_connect('pick_event', on_pick)

    plt.suptitle(f"HNSW layer {layer} {title_suffix} — {n} nodes, {len(edges)} edges")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, required=True,
                        help='Layer to display (view layer k)')
    parser.add_argument('--node_id', type=int, default=None,
                        help='Anchor node id to focus on')
    parser.add_argument('--anchor_layer', type=int, default=None,
                        help='Layer where node_id should be interpreted (p)')
    parser.add_argument('--n_nodes', type=int, default=50000)
    parser.add_argument('--no_interactive', action='store_true')
    args = parser.parse_args()

    idx, data = load_data(n_nodes=args.n_nodes)
    layer_nodes, layer_edges = extract_layers(idx, n_nodes=args.n_nodes)

    # If no anchor layer is given, infer the highest layer containing node_id.
    anchor_layer = args.anchor_layer
    if args.node_id is not None and anchor_layer is None:
        candidates = [l for l in layer_nodes if args.node_id in layer_nodes[l]]
        if not candidates:
            raise ValueError(f"node_id={args.node_id} not found in any layer")
        anchor_layer = max(candidates)

    nodes, edges = get_layer_subgraph(
        layer_nodes,
        layer_edges,
        view_layer=args.layer,
        node_id=args.node_id,
        anchor_layer=anchor_layer,
    )

    title_suffix = ""
    if args.node_id is not None:
        if anchor_layer is not None:
            title_suffix = f"(anchor node {args.node_id} from layer {anchor_layer})"
        else:
            title_suffix = f"(focused on node {args.node_id})"

    plot_graph(
        args.layer,
        nodes,
        edges,
        data,
        title_suffix=title_suffix,
        interactive=not args.no_interactive
    )


if __name__ == '__main__':
    main()