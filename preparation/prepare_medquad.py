from datasets import load_dataset
import json

dataset = load_dataset("abachaa/medquad", split="train")

# Keywords to filter
keywords = [
    "diabetes", "insulin", "blood sugar", "glucose", "hba1c", "hypoglycemia", "hyperglycemia",
    "type 1", "type 2", "gestational diabetes", "diabetic", "blood glucose"
]

def is_diabetes_related(text):
    text = text.lower()
    return any(k in text for k in keywords)

qa_pairs = []
for entry in dataset:
    q = entry["question"]
    a = entry["answer"]
    if is_diabetes_related(q) or is_diabetes_related(a):
        qa_pairs.append({"question": q, "reference": a})

with open("data/medquad_diabetes.json", "w") as f:
    json.dump(qa_pairs, f, indent=2)

print(f"✅ Saved {len(qa_pairs)} diabetes QA pairs from MedQuAD.")
