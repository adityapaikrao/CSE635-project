import json

with open("data/meddialog_qa.json", "r") as f:
    all_qa = json.load(f)

# Expand keyword list to catch more variations
keywords = [
    # Core medical terms
    "diabetes", "type 1", "type 2", "prediabetes", "pre-diabetic",
    "hyperglycemia", "hypoglycemia", "glucose", "insulin", "hba1c",
    
    # Common layperson language
    "blood sugar", "sugar level", "sugar levels", "sugar reading", 
    "my sugar", "high sugar", "low sugar", "sugar spikes", "spike in sugar",
    
    # Treatment-related
    "insulin shot", "insulin pump", "glucose monitor", "continuous glucose",
    "metformin", "blood test", "a1c test", "glucose tolerance test",
    
    # Symptoms and complications
    "frequent urination", "excessive thirst", "blurred vision",
    "diabetic foot", "diabetic neuropathy", "diabetic ketoacidosis", "dka",
    
    # Lifestyle management
    "diabetic diet", "low carb", "carbohydrate counting", "glycemic index",
    "diabetic meal plan", "sugar-free", "blood sugar control", 
    "controlling diabetes", "diet for diabetes", "exercise and diabetes",
    
    # Related conditions
    "gestational diabetes", "pcos and diabetes", "weight loss and sugar", 
    "obesity and diabetes", "diabetes medication", "diabetic medication"
]


# Lowercase everything
def matches_diabetes_keywords(text):
    text = text.lower()
    return any(k in text for k in keywords)

# Filter using question OR reference
diabetes_qa = [
    qa for qa in all_qa
    if matches_diabetes_keywords(qa["question"]) or matches_diabetes_keywords(qa["reference"])
]

# Save everything matched
with open("data/diabetes_qa.json", "w") as f:
    json.dump(diabetes_qa, f, indent=2)

print(f"✅ Found {len(diabetes_qa)} diabetes-related QA pairs.")
