import h5py, numpy as np

with h5py.File('data/glove-25-angular.hdf5', 'r') as f:
    print("keys:", list(f.keys()))
    train = f['train'][:]   # (1183514, 25) — use this
    test  = f['test'][:]    # (10000, 25)  — ignore for now

print(f"train: {train.shape}, dtype: {train.dtype}")
print(f"l2 norms (should be ~1): {np.linalg.norm(train[:5], axis=1)}")
# glove-25-angular is unit-normalized — use cosine/inner product