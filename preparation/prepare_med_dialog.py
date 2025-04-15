from datasets import load_dataset
import json

# Use a valid configuration for the dataset
dataset = load_dataset("bigbio/meddialog", name="meddialog_en_bigbio_text", trust_remote_code=True)

# Define diabetes-related keywords
diabetes_keywords = [
    "diabetes", "type 1", "type 2", "prediabetes", "pre-diabetic",
    "hyperglycemia", "hypoglycemia", "glucose", "insulin", "hba1c",
    "blood sugar", "sugar level", "sugar levels", "sugar reading", 
    "high sugar", "low sugar", "sugar spikes", "spike in sugar",
    "insulin shot", "insulin pump", "glucose monitor", "continuous glucose",
    "metformin", "blood test", "a1c test", "glucose tolerance test",
    "frequent urination", "excessive thirst", "blurred vision",
    "diabetic foot", "diabetic neuropathy", "diabetic ketoacidosis", "dka",
    "diabetic diet", "low carb", "carbohydrate counting", "glycemic index",
    "diabetic meal plan", "sugar-free", "blood sugar control", 
    "controlling diabetes", "diet for diabetes", "exercise and diabetes",
    "gestational diabetes", "pcos and diabetes", "weight loss and sugar", 
    "obesity and diabetes", "diabetes medication", "diabetic medication"
]

# Function to check if text matches diabetes-related keywords
def is_diabetes_related(text):
    text = text.lower()
    return any(keyword in text for keyword in diabetes_keywords)

qa_pairs = []

def extract_qa_pairs(split):
    pairs = []
    dialog = []
    last_speaker = None
    for entry in split:
        text = entry["text"].strip()
        label = entry["labels"][0]  # "patient" or "doctor"

        # Filter dialogs based on diabetes-related keywords
        if not is_diabetes_related(text):
            continue

        if last_speaker == "patient" and label == "doctor":
            pairs.append({
                "question": dialog[-1],
                "reference": text
            })
        dialog.append(text)
        last_speaker = label
    return pairs

# Process all splits
qa_pairs += extract_qa_pairs(dataset["train"])
qa_pairs += extract_qa_pairs(dataset["validation"])
qa_pairs += extract_qa_pairs(dataset["test"])

# Save filtered QA pairs
with open("data/diabetes_qa.json", "w") as f:
    json.dump(qa_pairs, f, indent=2)

print(f"✅ Saved {len(qa_pairs)} diabetes-related QA pairs to data/diabetes_qa.json")
