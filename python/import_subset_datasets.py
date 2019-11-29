import numpy as np
from sklearn import decomposition #PCA
import random
import pandas as pd

def import_subset(dataset, labels, selected_cat, n_pca, n_sample):
    
    i = 1
    
    # n_sample from each category
    subset_labels = np.zeros((n_sample*len(selected_cat), ))  
    

    data_label = dataset.loc[selected_cat[0]==labels]
    sample = random.sample(range(data_label.shape[0]), n_sample)
    subset_data = data_label.iloc[sample, :]
    subset_labels[(n_sample*(i-1)):(n_sample*i)] = selected_cat[0]    

    for cat in selected_cat[1:]:
        i += 1
        data_label = dataset.loc[labels==cat]
        sample = random.sample(range(data_label.shape[0]), n_sample)
        subset_data = pd.concat([subset_data, data_label.iloc[sample, :]])
        subset_labels[(n_sample*(i-1)):(n_sample*i)] = cat  

    pca = decomposition.PCA(n_components = n_pca)
    pca.fit(subset_data)

    # returns true labels, even if it's noise
    return pca.transform(subset_data), np.array(subset_labels).astype(int), subset_data, pca.explained_variance_ratio_
