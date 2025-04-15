from datasets import load_dataset
import json

# Load the full dataset from Hugging Face
dataset = load_dataset("bigbio/meddialog", name="meddialog_en_bigbio_text", split="train", trust_remote_code=True)

keywords = [
    "diabetes", "blood sugar", "glucose", "insulin", "type 1", "type 2",
    "hypoglycemia", "hyperglycemia", "hba1c", "metformin"
]

def is_diabetes_related(text):
    text = text.lower()
    return any(k in text for k in keywords)

# Filter QA pairs
diabetes_qa = []
for entry in dataset:
    text = entry.get("text", "").strip()
    if is_diabetes_related(text):
        diabetes_qa.append({
            "question": text,
            "reference": entry.get("labels", [""])[0]  # Adjust based on dataset structure
        })

# Save to file
with open("data/healthsearchqa_diabetes.json", "w") as f:
    json.dump(diabetes_qa, f, indent=2)

print(f"✅ Saved {len(diabetes_qa)} diabetes-related QA pairs from HealthSearchQA.")
