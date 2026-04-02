# VDBVisualization
Visualization of vector data bases wih HNSW


## Setting up
- create env
conda create -n hnsw-py311 python=3.11
conda activate hnsw-py311
pip install h5py matplotlib

- build hnswlib (see below)
cd hnswlib
pip install .

- fetch data
mkdir data
cd data
wget http://ann-benchmarks.com/glove-25-angular.hdf5
cd ..
python build_index_subset.py 

## Run
python gap_analysis.py --layer=1


## Info on files 
hnswlib was vendored based on: https://github.com/nmslib/hnswlib/commit/d9b3608c83d83b46c96e25088cb1d729b29dcfe9
Local modification: bindings.cpp: def get_links
