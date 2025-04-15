import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_pipeline import rag_pipeline

import json
from metrics_eval import evaluate_predictions


# Load your MedDialog QA dataset
with open("data/diabetes_qa.json", "r") as f:


    data = json.load(f)

predictions = []
references = []

for pair in data[:20]:  # limit to 20 for quick test
    question = pair["question"]
    reference = pair["reference"]

    print(f"\n🧠 Question: {question}")
    answer = rag_pipeline(question)
    print(f"🤖 Answer: {answer}")
    print(f"✅ Reference: {reference}")

    predictions.append(answer)
    references.append(reference)

# Evaluate the RAG output vs ground truth
evaluate_predictions(predictions, references)
