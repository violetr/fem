# A flexible EM-like clustering algorithm for noisy data

This repository contains the code that implements the FREM clustering algorithm corresponding to the paper https://arxiv.org/pdf/1907.01660.pdf. This algorithm follows a EM scheme focused on robustess to noise. 

## Algorithm

The clustering algorithm is implemented as a function in `EM_CG.py` and it can called in the following way

```python 
theta_estimated, cond_prob = FREM(K, dataset)
```
where `K` is the number of clusters.

## Datasets

To download the datasets used to compare the different clustering algorithms in these notebooks:

- MNIST (https://www.kaggle.com/oddrationale/mnist-in-csv)
- smallNORB (https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/)
- 20newsgroup (`fetch_20newsgroups` from sklearn library)


