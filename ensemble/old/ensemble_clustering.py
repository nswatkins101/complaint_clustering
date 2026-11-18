import random
import _pickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components

import sklearn.preprocessing as preprocessing
from sklearn.cluster import KMeans, MiniBatchKMeans
import scipy.cluster.hierarchy as sch
from sklearn.metrics.cluster import rand_score, adjusted_rand_score

from sentence_transformers import SentenceTransformer


class ClusterSimilarityMatrix():
    
    def __init__(self) -> None:
        self._is_fitted = False

    def fit(self, y_clusters):
        if not self._is_fitted:
            # for first time fit is called
            self._is_fitted = True
            self.similarity = self.to_binary_matrix(y_clusters)
            return self
        # for further calls, add binary matrix to similarity matrix
        self.similarity += self.to_binary_matrix(y_clusters)

    def to_binary_matrix(self, y_clusters):
        # y_clusters index of cluster for each sample
        y_reshaped = np.expand_dims(y_clusters, axis=-1)
        # co-occurence matrix, one if data points are in same cluster
        # cdist to calculate distance between cluster indice,s integers
        return (cdist(y_reshaped, y_reshaped, 'cityblock')==0).astype(int)


class EnsembleCustering():
    def __init__(self, base_estimators, aggregator, distances=False):
        self.base_estimators = base_estimators
        self.aggregator = aggregator
        self.distances = distances

    def fit(self, X):
        X_ = X.copy()

        clt_sim_matrix = ClusterSimilarityMatrix()
        for model in self.base_estimators:
            clt_sim_matrix.fit(model.fit_predict(X=X_))
        
        sim_matrix = clt_sim_matrix.similarity
        self.cluster_matrix = sim_matrix/sim_matrix.diagonal()

        if self.distances:
            self.cluster_matrix = np.abs(np.log(self.cluster_matrix + 1e-8)) # Avoid log(0)

    def fit_predict(self, X):
        self.fit(X)
        y = self.aggregator.fit_predict(self.cluster_matrix)
        return y
    

# load data
# unpickle complaints dictionary
with open('../data/many_complaints.bin', 'rb') as file:
    complaints_dict = pickle.load(file)
    
# Generate a list of 50 customer complaints
complaints = []
true_labels = []

for key, some_complaints in complaints_dict.items():
    for complaint in some_complaints:
            complaints.append(complaint)
            true_labels.append(key)

# Load a pre-trained Sentence Transformer model
model_names = ["all-MiniLM-L6-v2", "gte-base-en-v1.5", "gte-large-en-v1.5", "mxbai-embed-large-v1", "GIST-small-Embedding-v0"]
repos = ["all-MiniLM-L6-v2", "Alibaba-NLP/gte-base-en-v1.5", "Alibaba-NLP/gte-large-en-v1.5", "mixedbread-ai/mxbai-embed-large-v1", "avsolatorio/GIST-small-Embedding-v0"]
repos_dict = {name: repo for name,repo in zip(model_names,repos)}
print("loading sentence transformer...", end='', flush=True)
models = {name: SentenceTransformer(repos_dict[name], trust_remote_code=True) for name in model_names}
print("DONE")


# Encode sentences to get embeddings
print("encoding sentences...", end='', flush=True)
embeddings = {}
for name in model_names:
    vectors = models[name].encode(complaints)
    # normalize vectors so that distance is inner product, cosine similarity
    norm_vectors = preprocessing.normalize(vectors, norm="l2")
    embeddings[name] = vectors
print("DONE")
    

# pooling similarity matrices from k means
# Generating a "Cluster Forest"
NUM_KMEANS = 32
clustering_models = NUM_KMEANS*[
    # Note: Do not set a random_state, as the variability is crucial
    # This is a extreme simple K-Means
    MiniBatchKMeans(n_clusters=16, batch_size=64, n_init=1, max_iter=20)
]

clt_sim_matrix = ClusterSimilarityMatrix()
for name in model_names:
    for kmeans in clustering_models:
        clt_sim_matrix.fit( kmeans.fit_predict(X=embeddings[name]) ) 

sim_matrix = clt_sim_matrix.similarity
norm_sim_matrix = sim_matrix/sim_matrix.diagonal()    

# Print the similarity matrix
print(sim_matrix)
# Perform hierarchical clustering using the similarity matrix
linked = sch.linkage(norm_sim_matrix, method='ward')

# Plot the dendrogram
plt.figure(figsize=(20, 14))
sch.dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.savefig(f'ensemble_dendrogram.png')
cluster_labels = sch.fcluster(linked, t=4, criterion='maxclust')
print(cluster_labels)
ri = rand_score(true_labels, cluster_labels)
ari = adjusted_rand_score(true_labels, cluster_labels)
print('Ensemble Hierarchical Clustering...')
print(f'Rand score is {ri} and adjusted Rand score is {ari}')

# work absolutely perfectly, in fact finds mistakes Chat GPT made in the labelling
# and, it is scalable