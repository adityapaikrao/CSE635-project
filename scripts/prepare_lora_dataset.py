import json
from pathlib import Path
import glob

Path("data/final").mkdir(parents=True, exist_ok=True)

# Load all datasets
files = glob.glob("data/prepared/*.json")
combined = []

for file in files:
    with open(file, encoding="utf-8") as f:
        data = json.load(f)

    print(f"🔄 Processing {file} ({len(data)} entries)")

    if "meddialog" in file or "remedi" in file:
        # Dialogues – extract patient-doctor turns
        for entry in data:
            dialogue = entry.get("dialogue", entry.get("information", []))
            if isinstance(dialogue, list):
                for i in range(len(dialogue) - 1):
                    q = dialogue[i]
                    a = dialogue[i + 1]
                    if isinstance(q, dict):
                        q = q.get("sentence", "")
                        a = dialogue[i + 1].get("sentence", "")
                    if q and a:
                        combined.append({
                            "instruction": f"Patient: {q}",
                            "input": "",
                            "output": a
                        })
    elif "llm_cgm" in file:
        for q in data:
            if isinstance(q, dict) and "question" in q:
                combined.append({
                    "instruction": f"Patient: {q['question']}",
                    "input": "",
                    "output": q.get("answer", "This depends on the patient's condition.")
                })
    else:
        # MedQuAD / HealthSearchQA / diabetes_qa style
        for qa in data:
            q = qa.get("question", "")
            a = qa.get("reference", qa.get("answer", ""))
            if q and a:
                combined.append({
                    "instruction": f"Patient: {q.strip()}",
                    "input": "",
                    "output": a.strip()
                })

print(f"\n✅ Total formatted QA pairs: {len(combined)}")

# Save as final JSON
with open("data/final/diabetes_lora.json", "w", encoding="utf-8") as f:
    json.dump(combined, f, indent=2, ensure_ascii=False)

print("📦 Saved to data/final/diabetes_lora.json")
