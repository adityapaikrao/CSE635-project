import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load corpus
print("Loading corpus...")
with open("data/diabetes_guidelines.txt", "r") as f:
    passages = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(passages)} passages")

# Load model - using CPU since dataset is tiny
model = SentenceTransformer("intfloat/e5-small-v2", device="cpu")

# Encode all passages (small enough to do in one batch)
print("Encoding passages...")
embeddings = model.encode(
    [f"passage: {p}" for p in passages],
    show_progress_bar=True,
    convert_to_numpy=True
).astype("float32")

print(f"Embeddings shape: {embeddings.shape}")

# Get dimensionality
dim = embeddings.shape[1]

# Create a simple flat index - perfect for small datasets
print("Creating FAISS index...")
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# Save the index
print("Saving index...")
faiss.write_index(index, "retrieval/faiss_flat_index.idx")  # Updated path

# Save passage-to-id map
with open("retrieval/passage_map.pkl", "wb") as f:  # Updated path
    pickle.dump(passages, f)

print(f"✅ Built FAISS index with {len(passages)} passages")

# Test search
print("\nTesting search...")
sample_query = "How to manage diabetes with diet?"
query_embedding = model.encode([f"query: {sample_query}"], convert_to_numpy=True).astype("float32")

# Search
k = min(3, len(passages))  # Return top 3 results or fewer if we have less passages
distances, indices = index.search(query_embedding, k)

# Display results
print(f"\nTop {k} results for query: '{sample_query}'")
for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
    print(f"{i+1}. Distance: {dist:.4f}")
    print(f"   Passage: {passages[idx]}")