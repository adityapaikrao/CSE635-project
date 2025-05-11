import json
import os
import pandas as pd
from tqdm import tqdm
from bert_score import score as bert_score

ALIGNMENT_DATA_PATH = "data/alignment_dataset.json"
GENERATED_LOG_PATH = "logs/generated_responses.csv"  # Export from Streamlit history

# Load alignment dataset
with open(ALIGNMENT_DATA_PATH, "r", encoding="utf-8") as f:
    gold_examples = json.load(f)

# Load generated chatbot responses
gen_df = pd.read_csv(GENERATED_LOG_PATH, encoding='utf-8', on_bad_lines='skip')
gen_df.columns = [col.strip() for col in gen_df.columns]
gen_df["Question"] = gen_df["Question"].apply(lambda x: x.encode('latin1').decode('utf-8') if isinstance(x, str) else x)

results = []

# Match by patient + question
for item in tqdm(gold_examples):
    patient_id = item["patient_id"].lower()
    question = item["question"].strip().lower()
    gold_answer = item["ideal_response"].strip()

    row = gen_df[(gen_df["User ID"] == patient_id) & (gen_df["Question"].str.lower().str.strip() == question)]
    if row.empty:
        results.append({"patient_id": patient_id, "question": question, "match": False})
        continue

    pred = row.iloc[0]["Answer"].strip()
    P, R, F1 = bert_score([pred], [gold_answer], lang="es" if "¿" in question or "á" in question else "en")
    results.append({
        "patient_id": patient_id,
        "question": question,
        "generated_answer": pred,
        "ideal_answer": gold_answer,
        "BERTScore_F1": float(F1[0]),
        "match": True
    })

# Save results
out_df = pd.DataFrame(results)
out_df.to_csv("logs/alignment_eval_report.csv", index=False)
print("Saved alignment evaluation to logs/alignment_eval_report.csv")
