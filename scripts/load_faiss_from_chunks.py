import os
import json
import numpy as np
from glob import glob
import faiss
import pickle
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# === Settings ===
embedding_model_name = "neuml/pubmedbert-base-embeddings"
chunk_files_dir = "embeddings"
faiss_save_path = "faiss_index"

# === Step 1: Load chunks and precomputed embeddings ===
texts = []
vectors = []
skipped_count = 0

for file in sorted(glob(os.path.join(chunk_files_dir, "chunked_embeddings_*.json"))):
    print(f"Processing file: {file}")
    with open(file, "r", encoding="utf-8") as f:
        docs = json.load(f)
        for doc_idx, doc in enumerate(docs):
            if "chunked_sentences" in doc and "chunked_embeddings" in doc:
                if len(doc["chunked_sentences"]) != len(doc["chunked_embeddings"]):
                    print(f"⚠️ Warning: Mismatch in lengths for doc {doc_idx} in {file}")
                    continue
                    
                for i, (chunk, emb) in enumerate(zip(doc["chunked_sentences"], doc["chunked_embeddings"])):
                    if not chunk or not emb:
                        skipped_count += 1
                        continue
                        
                    try:
                        if isinstance(emb, list):
                            # Convert to numpy array and ensure float32
                            embedding_vector = np.array(emb, dtype=np.float32)
                            
                            if np.isnan(embedding_vector).any() or np.isinf(embedding_vector).any():
                                skipped_count += 1
                                continue
                                
                            texts.append(chunk)
                            vectors.append(embedding_vector)
                        else:
                            skipped_count += 1
                            print(f"⚠️ Skipping non-list embedding in doc {doc_idx}, chunk {i} in {file}")
                    except (ValueError, TypeError) as e:
                        skipped_count += 1
                        print(f"⚠️ Invalid embedding in doc {doc_idx}, chunk {i} in {file}: {str(e)[:50]}")

print(f"Loaded {len(texts)} chunks for FAISS indexing (skipped {skipped_count} invalid chunks).")

if len(texts) == 0:
    print("❌ No valid chunks found. Please check your embedding files.")
    exit(1)

# === Step 2: Create FAISS index manually ===

# Print debug info about the first few vectors
print("\nDebug - First vector:", vectors[0][:5], "...")
print(f"Vector type: {type(vectors[0])}, shape: {vectors[0].shape}, dtype: {vectors[0].dtype}")

# Ensure all vectors have the same dimension
dim = vectors[0].shape[0]
print(f"Vector dimension: {dim}")

# Create a fresh numpy array with the right shape and dtype
xb = np.empty((len(vectors), dim), dtype=np.float32)

# Copy each vector into the array
for i, vec in enumerate(vectors):
    if vec.shape[0] != dim:
        print(f"⚠️ Warning: Vector {i} has wrong dimension {vec.shape[0]} (expected {dim})")
        # Use zeros as placeholder for wrong dimension vectors
        xb[i] = np.zeros(dim, dtype=np.float32)
    else:
        xb[i] = vec

# Double-check the array
print(f"FAISS input shape: {xb.shape}, dtype: {xb.dtype}, contiguous: {xb.flags['C_CONTIGUOUS']}")

# Create a direct FAISS index
try:
    # Create a new FAISS index (try a basic one first)
    index = faiss.IndexFlatL2(dim)
    
    # Make sure the array is contiguous in memory
    xb = np.ascontiguousarray(xb)
    
    # Use low-level FAISS functions to add vectors
    index.add(xb)

    
    print(f"✅ Successfully added {xb.shape[0]} vectors to FAISS index using direct method.")
    print(f"Index now contains {index.ntotal} vectors.")
except Exception as e:
    print(f"Error with low-level FAISS operation: {e}")
    print("Trying to create and save index without using FAISS wrapper...")
    
    # Create directory if it doesn't exist
    os.makedirs(faiss_save_path, exist_ok=True)
    
    # Save the index directly
    try:
        # Try a simpler index type
        index = faiss.IndexFlatIP(dim)  # Inner product distance
        index.add(xb)  # One more attempt with standard add
        
        faiss.write_index(index, os.path.join(faiss_save_path, "index"))
        
        # Save the texts separately
        with open(os.path.join(faiss_save_path, "texts.pkl"), "wb") as f:
            pickle.dump(texts, f)
            
        print(f"✅ Saved index directly to {faiss_save_path}/")
        print(f"Index contains {index.ntotal} vectors and {len(texts)} texts.")
        
        # Print sample of indexed texts
        print("\nSample of indexed texts:")
        for i in range(min(3, len(texts))):
            print(f"{i+1}. {texts[i][:100]}...")
            
        exit(0)
    except Exception as e2:
        print(f"Direct save also failed: {e2}")
        print("Trying manual numpy save as fallback...")
        
        # Save embeddings as numpy array for future use
        np.save(os.path.join(faiss_save_path, "embeddings.npy"), xb)
        
        # Save texts as pickle
        with open(os.path.join(faiss_save_path, "texts.pkl"), "wb") as f:
            pickle.dump(texts, f)
            
        print(f"✅ Saved raw embeddings to {faiss_save_path}/embeddings.npy")
        print(f"✅ Saved texts to {faiss_save_path}/texts.pkl")
        print("You can load these files later and create a FAISS index.")
        exit(1)

# If we got here, the index was created successfully
try:
    # Create embeddings object
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    # Create LangChain FAISS wrapper
    db = FAISS(
        embedding_function=embeddings.embed_query,
        index=index,
        docstore=None,
        index_to_docstore_id=None
    )

    # Add texts manually
    db._texts = texts
    
    # === Step 3: Save index ===
    os.makedirs(faiss_save_path, exist_ok=True)
    
    # Save the FAISS index
    faiss.write_index(index, os.path.join(faiss_save_path, "index"))
    
    # Save the texts
    with open(os.path.join(faiss_save_path, "texts.pkl"), "wb") as f:
        pickle.dump(texts, f)
        
    print(f"✅ FAISS index saved to '{faiss_save_path}/index'")
    print(f"✅ Texts saved to '{faiss_save_path}/texts.pkl'")
    
    # Print a sample
    print("\nSample of indexed texts:")
    for i in range(min(3, len(texts))):
        print(f"{i+1}. {texts[i][:100]}...")
except Exception as e:
    print(f"Error saving index: {e}")
    exit(1)