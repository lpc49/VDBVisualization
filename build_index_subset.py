import hnswlib, numpy as np, h5py, time

# TODO: replace hardcoded seed and max_elements with a json config file used by both build_index_subset.py and graph_visualization.py

# memory allocation for the index when building idx.init_index()
# must match the value used in graph_visualization.py
MAX_ELEMENTS = 50000

# fixing seed for reproducibility
# must match the value used in graph_visualization.py
SEED=420

with h5py.File('data/glove-25-angular.hdf5', 'r') as f:
    data = f['train'][:MAX_ELEMENTS].astype(np.float32)

# inner product space for unit-normalized vectors
idx = hnswlib.Index(space='ip', dim=25)
idx.init_index(max_elements=MAX_ELEMENTS, ef_construction=200, M=16, random_seed=SEED)

t = time.time()
idx.add_items(data, np.arange(len(data)))
print(f"built in {time.time()-t:.1f}s")

# extract layer membership
layers = {}
for node in range(len(data)):
    for l in idx.get_links(node):
        layers.setdefault(l, []).append(node)

for l in sorted(layers):
    print(f"layer {l}: {len(layers[l])} nodes")

idx.save_index('data/glove25_50k.bin')
print("index saved")
