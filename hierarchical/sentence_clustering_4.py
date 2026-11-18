from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import hdbscan
import umap.umap_ as umap
from sklearn.metrics.cluster import rand_score, adjusted_rand_score
import sklearn.preprocessing as preprocessing
import scipy.cluster.hierarchy as sch
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random


def generate_clusters(message_embeddings,
                      n_neighbors,
                      n_components, 
                      n_clusters,
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

    clusters = KMeans(n_clusters=n_clusters, random_state=random_state).fit(umap_embeddings)

    return clusters
    
    
def score_clusters(clusters):
    """
    Returns the label count and rand scores
    """
    
    cluster_labels = clusters.labels_
    label_count = len(np.unique(cluster_labels))
    total_num = len(clusters.labels_)
    ri = rand_score(true_labels, cluster_labels)
    ari = adjusted_rand_score(true_labels, cluster_labels)
    
    return label_count, ri, ari
    
def random_search(embeddings, space, num_evals):
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
    
        label_count, ri, ari = score_clusters(clusters)
                
        results.append([i, n_neighbors, n_components, n_clusters, 
                        label_count, ri, ari])
    
    result_df = pd.DataFrame(results, columns=['run_id', 'n_neighbors', 'n_components', 
                                               'n_clusters', 'label_count', 'ri', 'ari'])
    
    return result_df.sort_values(by='ari')
    
# Categories of complaints
categories = [
    "Misunderstanding of Product Details",
    "Poor Web Experience",
    "Product Quality",
    "Customer Service Quality"
]

# Example complaints for each category
complaints_dict = {
    "Customer Service Issues": [
        "The representative ended the call abruptly.",
        "The chat support is slow and unresponsive.",
        "The customer service representative was very rude and unhelpful.",
        "The support team did not understand my issue and provided no solution.",
        "I never received a response to my email inquiry.",
        "I was on hold for over an hour before I could speak to someone.",
        "The live chat feature was unavailable when I needed assistance.",
        "The service rep was condescending and unprofessional.",
        "I was transferred multiple times and still didn't get my issue resolved.",
        "I was promised a callback that I never received.",
        "My issue was marked as resolved without any actual resolution.",
        "The customer service team gave me incorrect information.",
        "The representative ended the call abruptly.",
        "The chat support is slow and unresponsive.",
        "The customer service representative was very rude and unhelpful.",
        "The support team did not understand my issue and provided no solution.",
        "I never received a response to my email inquiry.",
        "I was on hold for over an hour before I could speak to someone.",
        "The live chat feature was unavailable when I needed assistance.",
        "The service rep was condescending and unprofessional.",
        "I was transferred multiple times and still didn't get my issue resolved.",
        "I was promised a callback that I never received.",
        "My issue was marked as resolved without any actual resolution.",
        "The customer service team gave me incorrect information.",
        "The representative ended the call abruptly.",
        "The chat support is slow and unresponsive.",
        "The customer service representative was very rude and unhelpful.",
        "The support team did not understand my issue and provided no solution.",
        "I never received a response to my email inquiry.",
        "I was on hold for over an hour before I could speak to someone.",
        "The live chat feature was unavailable when I needed assistance.",
        "The service rep was condescending and unprofessional.",
        "I was transferred multiple times and still didn't get my issue resolved.",
        "I was promised a callback that I never received.",
        "My issue was marked as resolved without any actual resolution.",
        "The customer service team gave me incorrect information.",
        "The representative ended the call abruptly.",
        "The chat support is slow and unresponsive.",
        "The customer service representative was very rude and unhelpful.",
        "The support team did not understand my issue and provided no solution.",
        "I never received a response to my email inquiry.",
        "I was on hold for over an hour before I could speak to someone.",
        "The live chat feature was unavailable when I needed assistance.",
        "The service rep was condescending and unprofessional.",
        "I was transferred multiple times and still didn't get my issue resolved.",
        "I was promised a callback that I never received.",
        "My issue was marked as resolved without any actual resolution.",
        "The customer service team gave me incorrect information."
    ],
    "Product Description and Expectation Issues": [
        "The dimensions listed on the website were incorrect.",
        "The functionality described in the details is not available in the product.",
        "The product description was unclear, and I received something different than expected.",
        "The features mentioned are not present in the product I received.",
        "The product was described as waterproof, but it is not.",
        "The product title was misleading and did not match the actual product.",
        "I thought the product included accessories as shown in the pictures, but it did not.",
        "The color of the item is not as vibrant as shown in the photos.",
        "The product looks very different from the advertised pictures.",
        "I believed the product was made from organic materials, but it wasn't.",
        "The product details mentioned a warranty, but there was none in the package.",
        "I was expecting a full set, but only one piece arrived.",
        "The dimensions listed on the website were incorrect.",
        "The functionality described in the details is not available in the product.",
        "The product description was unclear, and I received something different than expected.",
        "The features mentioned are not present in the product I received.",
        "The product was described as waterproof, but it is not.",
        "The product title was misleading and did not match the actual product.",
        "I thought the product included accessories as shown in the pictures, but it did not.",
        "The color of the item is not as vibrant as shown in the photos.",
        "The product looks very different from the advertised pictures.",
        "I believed the product was made from organic materials, but it wasn't.",
        "The product details mentioned a warranty, but there was none in the package.",
        "I was expecting a full set, but only one piece arrived.",
        "The dimensions listed on the website were incorrect.",
        "The functionality described in the details is not available in the product.",
        "The product description was unclear, and I received something different than expected.",
        "The features mentioned are not present in the product I received.",
        "The product was described as waterproof, but it is not.",
        "The product title was misleading and did not match the actual product.",
        "I thought the product included accessories as shown in the pictures, but it did not.",
        "The color of the item is not as vibrant as shown in the photos.",
        "The product looks very different from the advertised pictures.",
        "I believed the product was made from organic materials, but it wasn't.",
        "The product details mentioned a warranty, but there was none in the package.",
        "I was expecting a full set, but only one piece arrived.",
        "The dimensions listed on the website were incorrect.",
        "The functionality described in the details is not available in the product.",
        "The product description was unclear, and I received something different than expected.",
        "The features mentioned are not present in the product I received.",
        "The product was described as waterproof, but it is not.",
        "The product title was misleading and did not match the actual product.",
        "I thought the product included accessories as shown in the pictures, but it did not.",
        "The color of the item is not as vibrant as shown in the photos.",
        "The product looks very different from the advertised pictures.",
        "I believed the product was made from organic materials, but it wasn't.",
        "The product details mentioned a warranty, but there was none in the package."
    ],
    "Website and Navigation Issues": [
        "The site logged me out unexpectedly several times.",
        "The support website is difficult to navigate and find help.",
        "The product pages take too long to load.",
        "The website's layout is confusing and not user-friendly.",
        "The filter options are very limited and not useful.",
        "The site is not mobile-friendly and difficult to navigate on my phone.",
        "The website kept crashing while I was trying to place an order.",
        "The discount code field was not working.",
        "There is no option to save items to a wishlist.",
        "The site logged me out unexpectedly several times.",
        "The support website is difficult to navigate and find help.",
        "The product pages take too long to load.",
        "The website's layout is confusing and not user-friendly.",
        "The filter options are very limited and not useful.",
        "The site is not mobile-friendly and difficult to navigate on my phone.",
        "The website kept crashing while I was trying to place an order.",
        "The discount code field was not working.",
        "There is no option to save items to a wishlist.",
        "The site logged me out unexpectedly several times.",
        "The support website is difficult to navigate and find help.",
        "The product pages take too long to load.",
        "The website's layout is confusing and not user-friendly.",
        "The filter options are very limited and not useful.",
        "The site is not mobile-friendly and difficult to navigate on my phone.",
        "The website kept crashing while I was trying to place an order.",
        "The discount code field was not working.",
        "There is no option to save items to a wishlist.",
        "The site logged me out unexpectedly several times.",
        "The support website is difficult to navigate and find help.",
        "The product pages take too long to load.",
        "The website's layout is confusing and not user-friendly.",
        "The filter options are very limited and not useful.",
        "The site is not mobile-friendly and difficult to navigate on my phone.",
        "The website kept crashing while I was trying to place an order.",
        "The discount code field was not working.",
        "There is no option to save items to a wishlist.",
        "The site logged me out unexpectedly several times.",
        "The support website is difficult to navigate and find help.",
        "The product pages take too long to load.",
        "The website's layout is confusing and not user-friendly.",
        "The filter options are very limited and not useful.",
        "The site is not mobile-friendly and difficult to navigate on my phone.",
        "The website kept crashing while I was trying to place an order.",
        "The discount code field was not working.",
        "There is no option to save items to a wishlist.",
        "The site logged me out unexpectedly several times.",
        "The support website is difficult to navigate and find help.",
        "The product pages take too long to load.",
        "The website's layout is confusing and not user-friendly."
    ],
    "Product Quality and Condition Issues": [
        "The product did not work as advertised and is a waste of money.",
        "The stitching on the clothing item is already coming apart.",
        "The product has a very strong and unpleasant smell.",
        "The product stopped working after just one week of use.",
        "The paint on the product started peeling off after a few days.",
        "The product parts do not fit together properly.",
        "The item I received was damaged and of very poor quality.",
        "The product is very noisy and disturbs my household.",
        "The electronic device overheats quickly and is unsafe.",
        "The stitching on the clothing item is already coming apart.",
        "There were scratches and dents on the product.",
        "The battery life of the product is extremely poor.",
        "The packaging was inadequate and the product arrived broken.",
        "The product parts do not fit together properly.",
        "The product looks very different from the advertised pictures.",
        "The product did not work as advertised and is a waste of money.",
        "The stitching on the clothing item is already coming apart.",
        "The product has a very strong and unpleasant smell.",
        "The product stopped working after just one week of use.",
        "The paint on the product started peeling off after a few days.",
        "The product parts do not fit together properly.",
        "The item I received was damaged and of very poor quality.",
        "The product is very noisy and disturbs my household.",
        "The electronic device overheats quickly and is unsafe.",
        "The stitching on the clothing item is already coming apart.",
        "There were scratches and dents on the product.",
        "The battery life of the product is extremely poor.",
        "The packaging was inadequate and the product arrived broken.",
        "The product parts do not fit together properly.",
        "The product looks very different from the advertised pictures.",
        "The product did not work as advertised and is a waste of money.",
        "The stitching on the clothing item is already coming apart.",
        "The product has a very strong and unpleasant smell.",
        "The product stopped working after just one week of use.",
        "The paint on the product started peeling off after a few days.",
        "The product parts do not fit together properly.",
        "The item I received was damaged and of very poor quality.",
        "The product is very noisy and disturbs my household.",
        "The electronic device overheats quickly and is unsafe.",
        "The stitching on the clothing item is already coming apart.",
        "There were scratches and dents on the product.",
        "The battery life of the product is extremely poor.",
        "The packaging was inadequate and the product arrived broken.",
        "The product parts do not fit together properly.",
        "The product looks very different from the advertised pictures.",
        "The product did not work as advertised and is a waste of money.",
        "The stitching on the clothing item is already coming apart.",
        "The product has a very strong and unpleasant smell.",
        "The product stopped working after just one week of use."
    ]
}

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
    
    
space = {
    "n_neighbors": range(2, 50),
    "n_components": range(2, 10),
    "n_clusters": range(2,10),
    "random_state": 42
}
    
for name in model_names:
    print(f"For model {name}")
    cluster_labels = KMeans(n_clusters=4, random_state=42).fit(embeddings[name]).labels_
    ri = rand_score(true_labels, cluster_labels)
    ari = adjusted_rand_score(true_labels, cluster_labels)
    print(f'Rand score is {ri} and adjusted Rand score is {ari}')
    
    # Compute pairwise cosine similarity
    cosine_sim_matrix = cosine_similarity(embeddings[name], embeddings[name])
    # Create a distance matrix by converting the cosine similarity matrix
    distance_matrix = 1 - cosine_sim_matrix
    # Print the cosine similarity matrix
    print(cosine_sim_matrix)
    # Perform hierarchical clustering using the distance matrix
    linked = sch.linkage(distance_matrix, method='ward')

    # Plot the dendrogram
    plt.figure(figsize=(20, 14))
    sch.dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    plt.savefig(f'sentence_dendrogram_{name}.png')
    cluster_labels = sch.fcluster(linked, t=4, criterion='maxclust')
    print(cluster_labels)
    ri = rand_score(true_labels, cluster_labels)
    ari = adjusted_rand_score(true_labels, cluster_labels)
    print('Hierarchical Clustering...')
    print(f'Rand score is {ri} and adjusted Rand score is {ari}')
#    
#for name in model_names:
#    print(f"For model {name}")
#    cluster_labels = generate_clusters(embeddings[name], n_neighbors=30, n_components=3, n_clusters=4, random_state=42).labels_
#    ri = rand_score(true_labels, cluster_labels)
#    ari = adjusted_rand_score(true_labels, cluster_labels)
#    print(f'Rand index is {ri} and Adjusted Rand Index is {ari}')
#    for true_label, label in zip(true_labels, cluster_labels):
#        print(f'predicted label: {label}\t true label: {true_label}')