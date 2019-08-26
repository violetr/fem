# A flexible EM-like clustering algorithm for noisy data

This repository contains the code that implements the FREM clustering algorithm corresponding to the paper https://arxiv.org/pdf/1907.01660.pdf. This algorithm follows a EM scheme focused on robustess to noise. 

## Algorithm

The clustering algorithm described in Section 2 is implemented as a function in `EM_CG.py`. It can be called in the following way

```python 
theta_estimated, cond_prob = FREM(K, dataset)
```
where `K` is the number of clusters.

## Datasets

To download the datasets used to compare the different clustering algorithms in the notebooks:

- MNIST (https://www.kaggle.com/oddrationale/mnist-in-csv)
- smallNORB (https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/)
- 20newsgroup (`fetch_20newsgroups` from sklearn library)

## Notebooks

The notebooks 

- experiments-MNIST.ipynb
- experiments-NORBand20newsgroup.ipynb
- experiments-simulations.ipynb

contain the experiments and comparisons described in Section 3 of the the paper.

## Copyright

Authors

- Violeta Roizman (violeta.roizman@l2s.centralesupelec.fr)
- Matthieu Jonckheere 
- Frédéric Pascal 

Copyright 2019 @CentraleSupelec and UBA

The python wrapper used to read smallNORB data is available [here](https://github.com/ndrplz/small_norb) (Copyright (c) 2017 Andrea Palazzi).
