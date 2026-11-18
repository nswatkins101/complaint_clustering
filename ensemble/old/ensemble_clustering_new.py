from tempfile import mkdtemp
from shutil import rmtree

import _pickle as pickle

from sklearn.cluster import MiniBatchKMeans

from ensemble import *

# Unpickle complaints dictionary
with open('../data/many_complaints.bin', 'rb') as file:
    complaints_dict = pickle.load(file)

# Create list of unlabelled complaints and a list of their labels
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
transformers = {name: EmbeddingTransformer(repo_dict[name]) for name in model_names}

# Generating a "Cluster Forest"
cachedir = mkdtemp()
NUM_KMEANS = 32
estimators = NUM_KMEANS*[
    # Note: Do not set a random_state, as the variability is crucial
    # This is a extreme simple K-Means
    # use pipelines, each estimator is a transformer and then kmeans
    # memory=True so pipeline uses caching, 
    make_pipeline(transformers[name], MiniBatchKMeans(n_clusters=16, batch_size=64, n_init=1, max_iter=20), memory=cachedir) for name in model_names
]

print(estimators)

ensemble = EnsembleClustering(estimators)
ensemble.fit(complaints)
ensemble.dendrogram()
ensemble.metrics(true_labels)

# done with cached directory
rmtree(cachedir)