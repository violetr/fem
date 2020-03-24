# A flexible EM-like Clustering Algorithm for Noisy Data

This repository contains the code that implements the F-EM clustering algorithm corresponding to https://arxiv.org/pdf/1907.01660.pdf. This algorithm follows a two-step EM scheme focused on robustness to noise.

## Algorithm

The clustering algorithm described in Section 2 is implemented as a class in the file `_fem.py`. It can be called in the following way

```python
fem = FEM(K)
fem.fit(data)
```
where `K` is the number of clusters. 

To get the clustering labels you just need to access the labels attribute of the method object like this:

```python
fem.labels_
```

You can specify a number between 0 (reject everything) and 1 (reject nothing) for the outlier rejection:

```python
fem = FEM(K, thres = b)
fem.fit(dataset)
```

To classify new data points after the model is fitted you can run the following line:

```python
classif_labels = fem.predict(new_data)
```

External libraries required to run it:

- `numpy`
- `scipy`
- `scikit-learn`
- `math`
- `random`

## Datasets

You can download the datasets used to compare the different clustering algorithms:

- MNIST (https://www.kaggle.com/oddrationale/mnist-in-csv)
- smallNORB (https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/)
- 20newsgroup (`fetch_20newsgroups` from sklearn library)

## Notebooks

The notebooks

- experiments-MNIST.ipynb
- experiments-NORBand20newsgroup.ipynb

contain the experiments and comparisons described in the Section 3 of the paper. `plotnine`, `umap-learn` and `matplotlib` are required for the plots.

We run the t-EM algorithm implemented by the function `EmSkew` from the R library `EMMMIXskew`. 

## Copyright

Authors

- Violeta Roizman (violeta.roizman@centralesupelec.fr)
- Matthieu Jonckheere
- Frédéric Pascal

Copyright (c) 2019 CentraleSupelec and UBA.

The python wrapper used to read smallNORB data is available [here](https://github.com/ndrplz/small_norb) (Copyright (c) 2017 Andrea Palazzi).
