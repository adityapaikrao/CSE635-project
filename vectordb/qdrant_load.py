import json
import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import argparse

def connect_to_qdrant():
    return QdrantClient(host="localhost", port=6333)

def recreate_collection(client, collection_name="document_embeddings", vector_size=768):
    if client.collection_exists(collection_name):
        print(f"Collection '{collection_name}' already exists. Deleting it...")
        client.delete_collection(collection_name=collection_name)
        print(f"Collection '{collection_name}' deleted.")

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )
    print(f"Collection '{collection_name}' recreated.")

def upload_embeddings(client, collection_name, documents):
    points = []
    point_id = 0

    for doc in documents:
        for i, chunk in enumerate(doc.get('chunked_sentences', [])):
            embedding = doc.get('chunked_embeddings', [])[i]
            payload = {
                "id": doc.get('id'),
                "site": doc.get('site'),
                "url": doc.get('url'),
                "sentence": chunk
            }
            points.append(PointStruct(id=point_id, vector=embedding, payload=payload))
            point_id += 1

    if not points:
        print(" No points to upload. Skipping...")
        return

    client.upsert(
        collection_name=collection_name,
        points=points
    )
    print(f"Uploaded {len(points)} embeddings to '{collection_name}'.")

def load_and_process_batch(batch_file_path):
    with open(batch_file_path, 'r') as f:
        return json.load(f)

def process_batch_files(client, collection_name, batch_files):
    for batch_file in batch_files:
        print(f"\nProcessing file: {batch_file}")
        documents = load_and_process_batch(batch_file)
        upload_embeddings(client, collection_name, documents)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rewrite", type=str, required=True, help="yes/no: resets db if yes")
    args = parser.parse_args()

    client = connect_to_qdrant()
    collection_name = "document_embeddings"

    # Set to True ONLY if you want to delete and recreate collection
    recreate = args.rewrite.strip() == "yes"

    if recreate:
        recreate_collection(client, collection_name)
    else:
        if not client.collection_exists(collection_name):
            recreate_collection(client, collection_name)
        else:
            print(f"Using existing collection '{collection_name}'")

    # Find batch files in current directory
    # Look in the parent directory
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    batch_files = [os.path.join(parent_dir, f) for f in os.listdir(parent_dir) if f.startswith('chunked_embeddings_') and f.endswith('.json')
]


    print(f"\nFound {len(batch_files)} batch files: {batch_files}")
    if not batch_files:
        print("No batch files found. Exiting.")
        exit()

    # Upload all batch files
    process_batch_files(client, collection_name, batch_files)

    print("\nAll batches processed and uploaded!")
