import os
import pickle
import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

# Settings
faiss_index_dir = "faiss_index_final"  # Where the index was saved
index_file = "index"  # or "index_simple" if that was created
texts_file = "texts.pkl"
embedding_model_name = "neuml/pubmedbert-base-embeddings"

# Load the FAISS index
index_path = os.path.join(faiss_index_dir, index_file)
texts_path = os.path.join(faiss_index_dir, texts_file)

try:
    print(f"Loading FAISS index from {index_path}...")
    index = faiss.read_index(index_path)
    print(f"Loaded FAISS index with {index.ntotal} vectors")
    
    print(f"Loading texts from {texts_path}...")
    with open(texts_path, "rb") as f:
        texts = pickle.load(f)
    print(f"Loaded {len(texts)} texts")
    
    # Initialize the embedding function
    print(f"Loading embedding model {embedding_model_name}...")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    # Test a search
    print("\n--- Testing search functionality ---")
    
    # Use a part of an existing text as a query
    if len(texts) > 0:
        sample_text = texts[0]
        query = ' '.join(sample_text.split()[:10])  # First 10 words
        print(f"Query: '{query}'")
        
        # Get query embedding
        query_vector = embeddings.embed_query(query)
        query_vector_np = np.array(query_vector).astype('float32').reshape(1, -1)
        
        # Search
        print("Searching FAISS index...")
        k = 5  # Number of results to return
        distances, indices = index.search(query_vector_np, k)
        
        # Print results
        print(f"\nTop {k} results:")
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            if idx < len(texts) and idx >= 0:
                print(f"{i+1}. [Distance: {distance:.4f}] {texts[idx][:100]}...")
            else:
                print(f"{i+1}. [Invalid index: {idx}]")
        
        # Try a different query
        print("\n--- Testing with a different query ---")
        query2 = "medical research"
        print(f"Query: '{query2}'")
        
        # Get query embedding
        query_vector2 = embeddings.embed_query(query2)
        query_vector_np2 = np.array(query_vector2).astype('float32').reshape(1, -1)
        
        # Search
        distances2, indices2 = index.search(query_vector_np2, k)
        
        # Print results
        print(f"\nTop {k} results:")
        for i, (idx, distance) in enumerate(zip(indices2[0], distances2[0])):
            if idx < len(texts) and idx >= 0:
                print(f"{i+1}. [Distance: {distance:.4f}] {texts[idx][:100]}...")
            else:
                print(f"{i+1}. [Invalid index: {idx}]")
        
        print("\n✅ FAISS index tested successfully!")
    else:
        print("No texts available to test with!")
        
except Exception as e:
    print(f"Error testing FAISS index: {e}")
    print("Check that your index files exist and are valid.")