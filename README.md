# ScACAGCL

Official implementation of **ScACAGCL: Adaptive Curriculum Adversarial Graph Contrastive Learning for scRNA-seq Clustering**.

## Overview

ScACAGCL is a graph-based contrastive learning framework for single-cell RNA sequencing (scRNA-seq) clustering. It adaptively calibrates adversarial training dynamics to learn robust cell representations.

## Requirements



Python== 3.13.15 \
h5py== 3.14.0 \
keras== 3.13.2\
networkx== 3.7\
numpy== 2.1.3\
pandas== 3.0.6\
scikit-learn== 1.6.1\
scipy== 1.16.3\
torch== 2.11.0+cu130\
torch-geometric== 2.8.1\
torch-scatter== 2.1.2\
torchaudio== 2.11.0+cu130\
torchvision== 0.26.0+cu130





## Usage
+ Create two folders named 'data', and 'result' in the project folder. Put dataset file (e.g, yan.h5) into the 'data' folder. Output of the program are stored in the 'result' folder
 
+ Run the model using the following command:

python scACAGCL.py \
    --data_file "PATH_TO_DATASET.h5" \
    --num_cluster NUMBER_OF_CLUSTERS \
    --random_seed RANDOM_SEED

Example

python scACAGCL.py \
    --data_file "/path/to/Pollen.h5" \
    --num_cluster 11 \
    --random_seed 12345

Arguments:

--data_file: Path to the input scRNA-seq dataset in HDF5 (.h5) format.

--num_cluster: Expected number of clusters in the dataset.

--random_seed: Random seed for reproducibility.



## Datasets
14 datasets used in this work can be directly downloaded here [Datasets](https://zenodo.org/records/15272789)

## Citation

If you use this code, please cite the associated research paper.

## Authors
Author email: md.zobaerzihad99@gmail.com
