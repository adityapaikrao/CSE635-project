import argparse
from simple_vector_search import SimpleVectorDB
from langchain_huggingface import HuggingFaceEmbeddings

def main():
    parser = argparse.ArgumentParser(description='Query the vector database')
    parser.add_argument('--db_path', default='numpy_vectordb', help='Path to the vector database')
    parser.add_argument('--query', required=True, help='Query string')
    parser.add_argument('--k', type=int, default=5, help='Number of results to return')
    parser.add_argument('--model', default='neuml/pubmedbert-base-embeddings', help='Embedding model name')
    args = parser.parse_args()
    
    # Load the embedding model
    print(f"Loading embedding model {args.model}...")
    embeddings = HuggingFaceEmbeddings(model_name=args.model)
    
    # Load the vector database
    print(f"Loading vector database from {args.db_path}...")
    db = SimpleVectorDB.load(args.db_path, embeddings)
    
    # Search
    print(f"Searching for: '{args.query}'")
    results = db.similarity_search(args.query, k=args.k)
    
    # Print results
    print(f"\nTop {args.k} results:")
    for i, result in enumerate(results):
        print(f"\n{i+1}. [Score: {result['score']:.4f}]")
        print("-" * 40)
        print(result["text"])
        print("-" * 40)
    
    return results

if __name__ == "__main__":
    main()