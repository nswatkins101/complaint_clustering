import random

import numpy as np
import pandas as pd
import _pickle as pickle

from sklearn.cluster import KMeans
import umap.umap_ as umap
from sklearn.metrics.cluster import rand_score, adjusted_rand_score

from sentence_transformers import SentenceTransformer


from ..common.embed import *
from ..common.visualize import *


def generate_clusters(message_embeddings,
                      n_neighbors,
                      n_components, 
                      n_clusters,
                      random_state = None):
    """
    Generate kmeans cluster object after reducing dimensionality with UMAP
    """
    
    umap_embeddings = (umap.UMAP(n_neighbors=n_neighbors, 
                                n_components=n_components, 
                                metric='cosine', 
                                n_jobs=1, 
                                random_state=random_state)
                            .fit_transform(message_embeddings))

    clusters = KMeans(n_clusters=n_clusters, random_state=random_state).fit(umap_embeddings)

    return clusters
    
    
def score_clusters(clusters, true_labels):
    """
    Returns the label count and rand scores
    """
    
    cluster_labels = clusters.labels_
    label_count = len(np.unique(cluster_labels))
    total_num = len(clusters.labels_)
    ri = rand_score(true_labels, cluster_labels)
    ari = adjusted_rand_score(true_labels, cluster_labels)
    
    return label_count, ri, ari
    
def random_search(embeddings, space, num_evals, true_labels):
    """
    Randomly search hyperparameter space and limited number of times 
    and return a summary of the results
    """
    
    results = []
    
    for i in range(num_evals):
        n_neighbors = random.choice(space['n_neighbors'])
        n_components = random.choice(space['n_components'])
        n_clusters = random.choice(space['n_clusters'])
        
        clusters = generate_clusters(embeddings, 
                                     n_neighbors = n_neighbors, 
                                     n_components = n_components, 
                                     n_clusters=n_clusters, 
                                     random_state = 42)
    
        label_count, ri, ari = score_clusters(clusters, true_labels)
                
        results.append([i, n_neighbors, n_components, n_clusters, 
                        label_count, ri, ari])
    
    result_df = pd.DataFrame(results, columns=['run_id', 'n_neighbors', 'n_components', 
                                               'n_clusters', 'label_count', 'ri', 'ari'])
    
    return result_df.sort_values(by='ari')



def main():
    print('starting sentence_umap_kmeans...')
    
    # unpickle complaints dictionary
    with open('clustering/data/many_complaints.bin', 'rb') as file:
        complaints_dict = pickle.load(file)
        
    # Generate a list of 50 customer complaints
    complaints = []
    true_labels = []

    for key, some_complaints in complaints_dict.items():
        for complaint in some_complaints:
                complaints.append(complaint)
                true_labels.append(key)
            
    # define transformer models to be used
    print('making transformers...')
    model_names = ["all-MiniLM-L6-v2", "gte-base-en-v15", "gte-large-en-v15", "mxbai-embed-large-v1", "GIST-small-Embedding-v0"]
    repos = ["all-MiniLM-L6-v2", "Alibaba-NLP/gte-base-en-v1.5", "Alibaba-NLP/gte-large-en-v1.5", "mixedbread-ai/mxbai-embed-large-v1", "avsolatorio/GIST-small-Embedding-v0"]
    repos_dict = {name: repo for name,repo in zip(model_names,repos)}
    # embed sentences
    transformers = {name: embed(repos_dict[name]) for name in model_names}
    #embeddings = {name: embed(complaints, repos_dict[name]) for name in model_names}
    print('embedding sentences...')
    embeddings = {name: transformers[name](complaints) for name in model_names}
    
    space = {
        "n_neighbors": range(2, 50),
        "n_components": range(2, 10),
        "n_clusters": range(2,10),
        "random_state": 42
    }
    
    print('starting random search...')
    for name in model_names:
        print(f"For model {name}")
        random_use = random_search(embeddings[name], space, 500, true_labels)
        print(random_use.tail(5))
    
 
    
if __name__ == '__main__':
    main()