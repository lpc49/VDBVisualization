import hnswlib
import numpy as np

dim, n = 25, 500
data = np.random.randn(n, dim).astype(np.float32)

idx = hnswlib.Index(space='l2', dim=dim)
idx.init_index(max_elements=n, ef_construction=200, M=16)
idx.add_items(data, np.arange(n))

# the critical call — this is what ChromaDB hides
links = idx.get_links(0)
print("layer structure for node 0:", links)
# expect: {0: [...], 1: [...]} or {0: [...]} if not promoted

max_layer = idx.element_count  # sanity
layers = {i: [] for i in range(4)}
for node in range(n):
    lk = idx.get_links(node)
    for l in lk:
        layers[l].append(node)
print("nodes per layer:", {l: len(v) for l,v in layers.items() if v})
