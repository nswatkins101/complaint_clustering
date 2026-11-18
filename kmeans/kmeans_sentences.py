from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import hdbscan
import umap.umap_ as umap
from sklearn.metrics.cluster import rand_score, adjusted_rand_score
import numpy as np
import pandas as pd
import random

# Categories of complaints
categories = [
    "Misunderstanding of Product Details",
    "Poor Web Experience",
    "Product Quality",
    "Customer Service Quality"
]

# Example complaints for each category
complaints_dict = {
    "Misunderstanding of Product Details": [
        "The product description was unclear, and I received something different than expected.",
        "I thought the product included accessories as shown in the pictures, but it did not.",
        "The sizing chart was confusing, and the item doesn't fit as expected.",
        "The product was described as waterproof, but it is not.",
        "The color of the item is not as vibrant as shown in the photos.",
        "The product title was misleading and did not match the actual product.",
        "I was expecting a full set, but only one piece arrived.",
        "The dimensions listed on the website were incorrect.",
        "I believed the product was made from organic materials, but it wasn't.",
        "The product details mentioned a warranty, but there was none in the package.",
        "The functionality described in the details is not available in the product.",
        "The product looks very different from the advertised pictures.",
        "The features mentioned are not present in the product I received."
    ],
    "Poor Web Experience": [
        "The website kept crashing while I was trying to place an order.",
        "The checkout process is overly complicated and frustrating.",
        "I couldn't find the search bar on the website.",
        "The site is not mobile-friendly and difficult to navigate on my phone.",
        "I experienced multiple errors during the payment process.",
        "The product pages take too long to load.",
        "The filter options are very limited and not useful.",
        "I had to enter my shipping details multiple times due to errors.",
        "The product recommendations are irrelevant to my interests.",
        "The site logged me out unexpectedly several times.",
        "The discount code field was not working.",
        "The website's layout is confusing and not user-friendly.",
        "There is no option to save items to a wishlist."
    ],
    "Product Quality": [
        "The item I received was damaged and of very poor quality.",
        "The product stopped working after just one week of use.",
        "The material of the product feels cheap and flimsy.",
        "The stitching on the clothing item is already coming apart.",
        "The product did not work as advertised and is a waste of money.",
        "The battery life of the product is extremely poor.",
        "There were scratches and dents on the product.",
        "The product has a very strong and unpleasant smell.",
        "The packaging was inadequate and the product arrived broken.",
        "The electronic device overheats quickly and is unsafe.",
        "The paint on the product started peeling off after a few days.",
        "The product is very noisy and disturbs my household.",
        "The product parts do not fit together properly."
    ],
    "Customer Service Quality": [
        "The customer service representative was very rude and unhelpful.",
        "I was on hold for over an hour before I could speak to someone.",
        "The support team did not understand my issue and provided no solution.",
        "I never received a response to my email inquiry.",
        "The chat support is slow and unresponsive.",
        "The customer service team gave me incorrect information.",
        "I was transferred multiple times and still didn't get my issue resolved.",
        "The representative ended the call abruptly.",
        "I was promised a callback that I never received.",
        "The support website is difficult to navigate and find help.",
        "The live chat feature was unavailable when I needed assistance.",
        "The service rep was condescending and unprofessional.",
        "My issue was marked as resolved without any actual resolution."
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
    
space = {
    "n_neighbors": range(2, 10),
    "n_components": range(2, 10),
    "n_clusters": range(2,6),
    "random_state": 42
}
    
for name in model_names:
    print(f"For model {name}")
    random_use = random_search(embeddings[name], space, 100)
    print(random_use.tail(10))