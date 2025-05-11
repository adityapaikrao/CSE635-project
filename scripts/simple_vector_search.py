import os
import pickle
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

class SimpleVectorDB:
    """
    A simple vector database implementation using NumPy for when FAISS is problematic.
    """
    
    def __init__(self, embeddings, texts=None):
        """Initialize the vector database with optional embeddings and texts."""
        self.embedding_function = embeddings
        self.embeddings = None
        self.texts = []
        
        if texts is not None:
            self.add_texts(texts)
    
    def add_texts(self, texts, embeddings=None):
        """Add texts to the database with optional pre-computed embeddings."""
        if embeddings is None:
            # Embed texts
            print(f"Embedding {len(texts)} texts...")
            embeddings = []
            for text in tqdm(texts):
                embedding = self.embedding_function.embed_query(text)
                embeddings.append(embedding)
            embeddings = np.array(embeddings, dtype=np.float32)
        
        # Store
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        
        self.texts.extend(texts)
        
        print(f"Database now contains {len(self.texts)} texts with {self.embeddings.shape[0]} embeddings")
        return len(texts)
    
    def similarity_search(self, query, k=5):
        """Search for similar texts to the query."""
        # Embed query
        query_embedding = np.array(self.embedding_function.embed_query(query), dtype=np.float32)
        
        # Calculate cosine similarity
        dot_product = np.dot(self.embeddings, query_embedding)
        query_norm = np.linalg.norm(query_embedding)
        corpus_norm = np.linalg.norm(self.embeddings, axis=1)
        cosine_similarities = dot_product / (query_norm * corpus_norm)
        
        # Get top k
        top_k_indices = np.argsort(cosine_similarities)[-k:][::-1]
        top_k_scores = cosine_similarities[top_k_indices]
        
        # Return results
        results = []
        for i, (idx, score) in enumerate(zip(top_k_indices, top_k_scores)):
            results.append({
                "index": int(idx),
                "score": float(score),
                "text": self.texts[idx]
            })
        
        return results
    
    def save(self, directory):
        """Save the database to a directory."""
        os.makedirs(directory, exist_ok=True)
        
        # Save embeddings
        embeddings_path = os.path.join(directory, "embeddings.npy")
        np.save(embeddings_path, self.embeddings)
        
        # Save texts
        texts_path = os.path.join(directory, "texts.pkl")
        with open(texts_path, "wb") as f:
            pickle.dump(self.texts, f)
        
        print(f"Vector database saved to {directory}")
        return directory
    
    @classmethod
    def load(cls, directory, embedding_function):
        """Load a database from a directory."""
        # Load embeddings
        embeddings_path = os.path.join(directory, "./embeddings.npy")
        embeddings = np.load(embeddings_path)
        
        # Load texts
        texts_path = os.path.join(directory, "texts.pkl")
        with open(texts_path, "rb") as f:
            texts = pickle.load(f)
        
        # Create instance
        db = cls(embedding_function)
        db.embeddings = embeddings
        db.texts = texts
        
        print(f"Loaded vector database with {len(db.texts)} texts")
        return db

# Usage example
if __name__ == "__main__":
    # Load saved embeddings and texts
    saved_embeddings = "faiss_index/embeddings.npy"
    saved_texts = "faiss_index/texts.pkl"
    output_dir = "numpy_vectordb"
    
    print("Loading saved embeddings and texts...")
    embeddings_array = np.load(saved_embeddings)
    with open(saved_texts, "rb") as f:
        texts = pickle.load(f)
    
    print(f"Loaded {len(texts)} texts and {embeddings_array.shape[0]} embeddings")
    
    # Initialize the embedding function
    embedding_model_name = "neuml/pubmedbert-base-embeddings"
    print(f"Loading embedding model {embedding_model_name}...")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    # Create and save vector database
    print("Creating vector database...")
    db = SimpleVectorDB(embeddings)
    db.add_texts(texts, embeddings=embeddings_array)
    db.save(output_dir)
    
    # Test search
    print("\n--- Testing search functionality ---")
    
    # Use a part of an existing text as a query
    if len(texts) > 0:
        sample_text = texts[0]
        query = ' '.join(sample_text.split()[:10])  # First 10 words
        print(f"Query: '{query}'")
        
        # Search
        results = db.similarity_search(query, k=5)
        
        # Print results
        print(f"\nTop 5 results:")
        for i, result in enumerate(results):
            text_preview = result["text"][:100] + "..." if len(result["text"]) > 100 else result["text"]
            print(f"{i+1}. [Score: {result['score']:.4f}] {text_preview}")
        
        # Try a different query
        print("\n--- Testing with a different query ---")
        query2 = "medical research"
        print(f"Query: '{query2}'")
        
        # Search
        results2 = db.similarity_search(query2, k=5)
        
        # Print results
        print(f"\nTop 5 results:")
        for i, result in enumerate(results2):
            text_preview = result["text"][:100] + "..." if len(result["text"]) > 100 else result["text"]
            print(f"{i+1}. [Score: {result['score']:.4f}] {text_preview}")
    
    print("\n✅ Vector database created and tested successfully!")
    print(f"You can load it with: db = SimpleVectorDB.load('{output_dir}', embedding_function)")