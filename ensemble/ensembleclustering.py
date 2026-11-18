import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components

import sklearn.preprocessing as preprocessing
import scipy.cluster.hierarchy as sch
from sklearn.metrics.cluster import rand_score, adjusted_rand_score

def cooccurence_matrix(y):
    # y is index of cluster for each sample
    y_reshaped = np.expand_dims(y, axis=-1)
    # co-occurence matrix, one if data points are in same cluster
    # cdist to calculate distance between cluster indice,s integers
    return (cdist(y_reshaped, y_reshaped, 'cityblock')==0).astype(int)

        
class EnsembleClustering():
    def __init__(self, base_estimators):
        self.base_estimators = base_estimators

    def fit(self, X):
        sim_matrix = None
        # pooling similarity matrices from estimators
        for model in self.base_estimators:
            if sim_matrix is None:
                print("setting sim_matrix")
                sim_matrix = cooccurence_matrix(model.fit_predict(X))
            else:
                sim_matrix += cooccurence_matrix(model.fit_predict(X))
        print(sim_matrix)
        self.sim_matrix = sim_matrix/sim_matrix.diagonal()
        # perform hierarchical clustering using the similarity matrix
        self.linked = sch.linkage(self.sim_matrix, method='ward')
        self.labels_ = sch.fcluster(self.linked, t=4, criterion='maxclust')

    def dendrogram(self):
        # Plot the dendrogram
        plt.figure(figsize=(20, 14))
        sch.dendrogram(self.linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample index')
        plt.ylabel('Distance')
        plt.savefig(f'ensemble_dendrogram.png')

    def fit_predict(self, X):
        self.fit(X)
        return labels_
    
    def metrics(self, true_labels):
        cluster_labels = self.labels_
        ri = rand_score(true_labels, cluster_labels)
        ari = adjusted_rand_score(true_labels, cluster_labels)
        print('Hierarchical Ensemble Clustering...')
        print(f'Rand score is {ri} and adjusted Rand score is {ari}')