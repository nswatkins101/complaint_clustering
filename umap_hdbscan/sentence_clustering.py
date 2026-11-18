from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import hdbscan
import umap.umap_ as umap
from sklearn.metrics.cluster import rand_score, adjusted_rand_score
import numpy as np
import pandas as pd
import random

# load data
from ../data/complaints.py import categories, complaints_dict

# Generate a list of 50 customer complaints
complaints = []
true_labels = []

for key, some_complaints in complaints_dict.items():
    for complaint in some_complaints:
            complaints.append(complaint)
            true_labels.append(key)


# Load a pre-trained Sentence Transformer model
model_names = ["all-MiniLM-L6-v2", "Alibaba-NLP/gte-base-en-v1.5", "avsolatorio/GIST-small-Embedding-v0"]
print("loading sentence transformer...", end='', flush=True)
models = {model_name: SentenceTransformer(model_name, trust_remote_code=True) for model_name in model_names}
print("DONE")


# Encode sentences to get embeddings
print("encoding sentences...", end='', flush=True)
embeddings = {name: models[name].encode(complaints) for name in model_names}
print("DONE")


# Define the number of clusters
num_clusters = 4

# Perform K-Means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
print("clustering with k means...", end='', flush=True)
# Get cluster assignments for training data
cluster_assignments = {name: kmeans.fit(embeddings[name]).labels_ for name in model_names}
print("DONE")


# Print the sentenceswith their ground truth labels and cluster assignments
#for complaint, label, cluster in zip(complaints, true_labels, cluster_assignments):
    #print(f"Complaint: '{complaint}'")
#    print(f"Predicted label: {cluster}\t True label: '{label}'")

for name in model_names:
    print(f"For model {name}")
    # rand index to calculate accuracy of clustering
    print("\tRand score: {0}".format(rand_score(true_labels, cluster_assignments[name])))
    print("\tAdjusted Rand score: {0}".format(adjusted_rand_score(true_labels, cluster_assignments[name])))


def generate_clusters(message_embeddings,
                      n_neighbors,
                      n_components, 
                      min_cluster_size,
                      random_state = None):
    """
    Generate HDBSCAN cluster object after reducing embedding dimensionality with UMAP
    """
    
    umap_embeddings = (umap.UMAP(n_neighbors=n_neighbors, 
                                n_components=n_components, 
                                metric='cosine', 
                                n_jobs=1, 
                                random_state=random_state)
                            .fit_transform(message_embeddings))

    clusters = hdbscan.HDBSCAN(min_cluster_size = min_cluster_size,
                               metric='euclidean', 
                               cluster_selection_method='eom').fit(umap_embeddings)

    return clusters
    
    
def score_clusters(clusters, prob_threshold = 0.50):
    """
    Returns the label count and cost of a given cluster supplied from running hdbscan
    cost is the percent of points with probability to be in their assigned cluster of less than the threshold 
    """
    
    cluster_labels = clusters.labels_
    label_count = len(np.unique(cluster_labels))
    total_num = len(clusters.labels_)
    cost = np.count_nonzero(clusters.probabilities_ < prob_threshold)/total_num
    ri = rand_score(true_labels, cluster_labels)
    ari = adjusted_rand_score(true_labels, cluster_labels)
    
    return label_count, cost, ri, ari
    
def random_search(embeddings, space, num_evals):
    """
    Randomly search hyperparameter space and limited number of times 
    and return a summary of the results
    """
    
    results = []
    
    for i in range(num_evals):
        n_neighbors = random.choice(space['n_neighbors'])
        n_components = random.choice(space['n_components'])
        min_cluster_size = random.choice(space['min_cluster_size'])
        
        clusters = generate_clusters(embeddings, 
                                     n_neighbors = n_neighbors, 
                                     n_components = n_components, 
                                     min_cluster_size = min_cluster_size, 
                                     random_state = 42)
    
        label_count, cost, ri, ari = score_clusters(clusters, prob_threshold = 0.05)
                
        results.append([i, n_neighbors, n_components, min_cluster_size, 
                        label_count, ri, ari])
    
    result_df = pd.DataFrame(results, columns=['run_id', 'n_neighbors', 'n_components', 
                                               'min_cluster_size', 'label_count', 'ri', 'ari'])
    
    return result_df.sort_values(by='ari')
    
space = {
    "n_neighbors": range(2, 10),
    "n_components": range(2, 10),
    "min_cluster_size": range(2,10),
    "random_state": 42
}
    
for name in model_names:
    print(f"For model {name}")
    random_use = random_search(embeddings[name], space, 500)
    print(random_use.tail(5))