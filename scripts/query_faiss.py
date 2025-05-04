import sys
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_function = HuggingFaceEmbeddings(model_name="neuml/pubmedbert-base-embeddings")

db = FAISS.load_local("faiss_index", embedding_function, allow_dangerous_deserialization=True)

query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("🔍 Enter your query: ")
results = db.similarity_search(query, k=5)

print(f"\nTop results for: '{query}'\n")
for i, doc in enumerate(results, 1):
    print(f"{i}. {doc.page_content}\n{'-'*80}")
