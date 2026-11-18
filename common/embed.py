import sklearn.preprocessing as preprocessing
from sklearn.base import BaseEstimator, TransformerMixin

from sentence_transformers import SentenceTransformer


def embed(repo_path):
    """
    Loads transformers and embeds text
    """
    # Load a pre-trained Sentence Transformer model
    print(f"loading sentence transformer from {repo_path}...", end='', flush=True)
    model = SentenceTransformer(repo_path, trust_remote_code=True)
    print("DONE")

    # Encode sentences to get embeddings and normalize
    def encode(data):
        print("encoding sentences...", end='', flush=True)
        vectors = model.encode(data)
        # normalize vectors so that distance is inner product, cosine similarity
        embeddings = preprocessing.normalize(vectors, norm="l2")
        print("DONE")
        # it returns the embeddings with the model at repo_path
        return embeddings
    # return that function
    return encode