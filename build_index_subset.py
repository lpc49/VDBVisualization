import hnswlib, numpy as np, h5py, time

max_elements = 50000 #subset size

with h5py.File('data/glove-25-angular.hdf5', 'r') as f:
    data = f['train'][:max_elements].astype(np.float32)

# inner product space for unit-normalized vectors
idx = hnswlib.Index(space='ip', dim=25)
idx.init_index(max_elements=max_elements, ef_construction=200, M=16)

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
