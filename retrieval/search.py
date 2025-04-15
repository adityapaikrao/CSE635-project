import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load FAISS index (CPU version)
index = faiss.read_index("retrieval/faiss_flat_index.idx")

# Load passages
with open("retrieval/passage_map.pkl", "rb") as f:
    passages = pickle.load(f)

# Load embedding model (CPU for now)
model = SentenceTransformer("intfloat/e5-small-v2", device="cpu")

def retrieve_passages(query, top_k=3):
    # Embed query
    query_embedding = model.encode(
        [f"query: {query}"],
        convert_to_numpy=True
    ).astype("float32")
    
    # Search index
    distances, indices = index.search(query_embedding, top_k)
    
    # Return results
    return [(passages[i], distances[0][rank]) for rank, i in enumerate(indices[0])]
