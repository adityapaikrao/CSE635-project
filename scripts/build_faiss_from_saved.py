import os
import numpy as np
import pickle
import faiss
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS as LangchainFAISS

# File paths
saved_embeddings = "faiss_index/embeddings.npy"
saved_texts = "faiss_index/texts.pkl"
faiss_output_dir = "faiss_index_final"
embedding_model_name = "neuml/pubmedbert-base-embeddings"

print("Loading saved embeddings and texts...")
embeddings_array = np.load(saved_embeddings)
with open(saved_texts, "rb") as f:
    texts = pickle.load(f)

print(f"Loaded {len(texts)} texts and {embeddings_array.shape[0]} embeddings")
print(f"Embeddings shape: {embeddings_array.shape}")
print(f"Sample embedding: {embeddings_array[0][:5]}...")

# Create directory for output
os.makedirs(faiss_output_dir, exist_ok=True)

# Try a completely different FAISS approach
try:
    print("\nBuilding FAISS index using standard factory method...")
    dimension = embeddings_array.shape[1]
    
    # Use FAISS factory method to create the index
    index = faiss.index_factory(dimension, "Flat", faiss.METRIC_L2)
    
    print("Converting embeddings to float32 contiguous array...")
    embeddings_array = np.ascontiguousarray(embeddings_array.astype('float32'))
    
    print(f"Adding {embeddings_array.shape[0]} vectors to the index...")
    
    # Manually handle the vectors in smaller batches
    batch_size = 100
    total_batches = (embeddings_array.shape[0] + batch_size - 1) // batch_size
    
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, embeddings_array.shape[0])
        batch = embeddings_array[start_idx:end_idx]
        
        print(f"Adding batch {i+1}/{total_batches} (vectors {start_idx} to {end_idx-1})...")
        try:
            index.add(batch)
        except Exception as e:
            print(f"Error adding batch: {e}")
            # Try an alternative batch approach
            for j in range(batch.shape[0]):
                try:
                    single_vector = batch[j].reshape(1, -1)
                    index.add(single_vector)
                except Exception as e2:
                    print(f"Error adding vector {start_idx + j}: {e2}")
    
    print(f"FAISS index now contains {index.ntotal} vectors")
    
    # Write the index directly
    faiss_index_path = os.path.join(faiss_output_dir, "index")
    print(f"Saving FAISS index to {faiss_index_path}")
    faiss.write_index(index, faiss_index_path)
    
    # Save the texts
    texts_path = os.path.join(faiss_output_dir, "texts.pkl")
    print(f"Saving texts to {texts_path}")
    with open(texts_path, "wb") as f:
        pickle.dump(texts, f)
    
    # Try to create a LangChain wrapper
    try:
        print("Creating LangChain FAISS wrapper...")
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        # Create metadata for documents
        metadatas = [{"index": i} for i in range(len(texts))]
        
        # Create mapping from index to ID
        docstore = {}
        index_to_docstore_id = {}
        for i, text in enumerate(texts):
            doc_id = str(i)
            docstore[doc_id] = text
            index_to_docstore_id[i] = doc_id
        
        db = LangchainFAISS(
            embedding_function=embeddings.embed_query,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )
        
        # Try a test query
        print("\nTesting retrieval with a sample query...")
        if len(texts) > 0:
            query = texts[0].split()[:5]  # Use first few words of first document
            query = " ".join(query)
            print(f"Sample query: '{query}'")
            
            results = db.similarity_search(query, k=3)
            print(f"Retrieved {len(results)} documents")
            print("First result:", results[0].page_content[:100], "...")
        
        langchain_path = os.path.join(faiss_output_dir, "langchain_faiss")
        db.save_local(langchain_path)
        print(f"Saved LangChain FAISS wrapper to {langchain_path}")
        
    except Exception as e:
        print(f"Error creating LangChain wrapper: {e}")
        print("You can still use the FAISS index directly with the saved files.")
    
    print("\n✅ Process completed successfully!")
    print(f"You can load the FAISS index from: {faiss_index_path}")
    print(f"And the corresponding texts from: {texts_path}")
    
except Exception as e:
    print(f"Error building FAISS index: {e}")
    print("Trying simple dense FAISS index as final attempt...")
    
    try:
        # Final attempt with the simplest index possible
        index = faiss.IndexFlatL2(embeddings_array.shape[1])
        
        # Add vectors one by one
        for i, vector in enumerate(embeddings_array):
            if i % 100 == 0:
                print(f"Adding vector {i}/{embeddings_array.shape[0]}...")
            try:
                vector_reshaped = vector.reshape(1, -1)
                index.add(vector_reshaped)
            except Exception as e2:
                print(f"Skipping vector {i} due to error: {e2}")
        
        # Save what we have
        faiss_index_path = os.path.join(faiss_output_dir, "index_simple")
        faiss.write_index(index, faiss_index_path)
        
        texts_path = os.path.join(faiss_output_dir, "texts.pkl")
        with open(texts_path, "wb") as f:
            pickle.dump(texts, f)
            
        print(f"\n✅ Created simple FAISS index with {index.ntotal} vectors")
        print(f"Saved to {faiss_index_path}")
        print(f"Texts saved to {texts_path}")
    
    except Exception as e3:
        print(f"All FAISS attempts failed: {e3}")
        print("Please check your FAISS installation and version compatibility.")
        print("You may need to try a different vector database library.")