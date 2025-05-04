import loader
from tqdm import tqdm
import json
import os
from itertools import islice
import gc
import argparse

def save_checkpoint(documents, step, out_dir="../embeddings"):
    # Ensure the output directory exists
    os.makedirs(os.path.join(os.path.dirname(__file__), out_dir), exist_ok=True)
    
    output_path = os.path.join(os.path.dirname(__file__), out_dir, f"chunked_embeddings_{step}.json")
    with open(output_path, "w") as f:
        json.dump(documents, f, indent=2)
    print(f"Saved checkpoint: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, required=True, help="Batch index to process")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of docs per batch")
    parser.add_argument("--output_dir", type=str, default="../embeddings", help="Directory to save output files")
    args = parser.parse_args()

    # Path to the JSON file in the data directory
    json_file_path = os.path.join(os.path.dirname(__file__), "..", "data", "scraped_text.json")

    with open(json_file_path, 'r') as f:
        data = json.load(f)

    doc_generator = loader.create_documents(data)

    # Calculate start and end index for this batch
    start = args.step * args.batch_size
    end = start + args.batch_size

    documents_list = list(islice(doc_generator, start, end))

    if not documents_list:
        print("No more documents to process in this batch.")
        exit(1)  # Exit with non-zero code to indicate no more documents

    processed_docs = []
    for idx, doc in enumerate(tqdm(documents_list, desc=f"Processing batch {args.step}")):
        doc['chunked_sentences'] = loader.chunk_sentences(doc['sentences'])
        doc['chunked_embeddings'] = loader.get_embeddings(doc['chunked_sentences'])
        processed_docs.append(doc)

    save_checkpoint(processed_docs, step=args.step, out_dir=args.output_dir)

    # Optional cleanup
    del processed_docs
    del documents_list
    gc.collect()
    
    exit(0)  # Exit with code 0 to indicate success