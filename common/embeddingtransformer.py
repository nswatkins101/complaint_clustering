from functools import cache
from joblib import Memory

import sklearn.preprocessing as preprocessing
from sklearn.base import BaseEstimator, TransformerMixin

from sentence_transformers import SentenceTransformer

def _encode(model, point):
    return model.encode(point)
    
class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, name, repo_path):
        self.name = name
        self.repo_path = repo_path
        self.model = SentenceTransformer(repo_path, trust_remote_code=True, cache_folder='clustering/sentencetransformers')
        self.cachedir = 'clustering/embeddings/name'
        self.memory = Memory(location=self.cachedir, verbose=10)
        # use memoization with functools.cache to speed up pipeline
        # cache works strangely with objects, because of hashing self, still hashing model, but as last not all of EmbeddingTransformer
        self.encode = self.memory.cache(_encode)
        
    def encode(self):
        pass
    
    def fit(self, X, y=None):
        return self  # The fit method typically does nothing for transformers
    
    # Encode sentences to get embeddings and normalize    
    def transform(self, X, y=None):
        # transform with embedding
        print("encoding sentences...", end='', flush=True)
        Xt = X.copy()  # Copy the input DataFrame to avoid modifying the original
        vectors = [self.encode(self.model, point) for point in Xt]
        # normalize vectors so that distance is inner product, cosine similarity
        embeddings = preprocessing.normalize(vectors, norm="l2")
        print("DONE")
        return embeddings
    
    def fit_transform(self, X, y=None):
        return self.transform(X)