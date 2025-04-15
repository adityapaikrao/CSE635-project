import os
import json
import subprocess
from pathlib import Path
import glob

# Ensure data directory exists
Path("data/prepared").mkdir(parents=True, exist_ok=True)

def clone_repo(url, target_dir):
    if not os.path.exists(target_dir):
        print(f"Cloning {url} into {target_dir}...")
        subprocess.run(["git", "clone", url, target_dir])
    else:
        print(f"{target_dir} already exists, skipping clone.")

# ----------------------------
# 1. MedQuAD (manual XML parse)
# ----------------------------
def prepare_medquad():
    import xml.etree.ElementTree as ET

    xml_dirs = glob.glob("data/MedQuAD/*_QA")
    keywords = ["diabetes", "insulin", "blood sugar", "glucose", "hba1c", "hypoglycemia",
                "hyperglycemia", "type 1", "type 2", "gestational diabetes", "diabetic"]

    def is_diabetes(text):
        return any(k in text.lower() for k in keywords)

    qa_pairs = []
    for folder in xml_dirs:
        for file in os.listdir(folder):
            if file.endswith(".xml"):
                tree = ET.parse(os.path.join(folder, file))
                root = tree.getroot()
                for qa in root.findall(".//QAPair"):
                    q = qa.findtext("Question") or ""
                    a = qa.findtext("Answer") or ""
                    if is_diabetes(q) or is_diabetes(a):
                        qa_pairs.append({"question": q.strip(), "reference": a.strip()})


    with open("data/prepared/medquad_diabetes.json", "w") as f:
        json.dump(qa_pairs, f, indent=2)
    print(f"✅ Saved {len(qa_pairs)} diabetes QAs from MedQuAD")

# ----------------------------
# 2. MedDialog-EN (filter)
# ----------------------------
def prepare_meddialog():
    folder = "data/MedDialog"
    if not os.path.exists(folder):
        print("❌ MedDialog folder not found. Please manually download it from:")
        print("👉 https://drive.google.com/drive/folders/1g29ssimdZ6JzTST6Y8g6h-ogUNReBtJD")
        return

    txt_files = [f for f in os.listdir(folder) if f.endswith(".txt")]
    if not txt_files:
        print("❌ No .txt MedDialog files found in the folder.")
        return

    keywords = ["diabetes", "insulin", "blood sugar", "glucose", "hba1c", "hypoglycemia",
                "hyperglycemia", "type 1", "type 2", "gestational diabetes", "diabetic"]

    def is_diabetes(text):
        return any(k in text.lower() for k in keywords)

    diabetes_dialogues = []

    for filename in txt_files:
        with open(os.path.join(folder, filename), encoding="utf-8") as f:
            lines = f.read().split("\n")

        dialogue = []
        for line in lines:
            if line.strip() == "":
                if any(is_diabetes(turn) for turn in dialogue):
                    diabetes_dialogues.append({"dialogue": dialogue})
                dialogue = []
            else:
                dialogue.append(line.strip())

        if dialogue and any(is_diabetes(turn) for turn in dialogue):
            diabetes_dialogues.append({"dialogue": dialogue})

    with open("data/prepared/meddialog_diabetes.json", "w") as f:
        json.dump(diabetes_dialogues, f, indent=2)
    print(f"✅ Saved {len(diabetes_dialogues)} diabetes dialogues from MedDialog")

# ----------------------------
# 3. ReMeDi Dataset
# ----------------------------
def prepare_remedi():
    file_path = "data/ReMeDi/ReMeDi-base.json"
    if not os.path.exists(file_path):
        print("❌ ReMeDi-base.json not found. Please manually place it in data/ReMeDi/")
        return

    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    keywords = [
        "糖尿病", "血糖", "高血糖", "低血糖", "胰岛素", "葡萄糖",
        "HbA1c", "1型糖尿病", "2型糖尿病", "妊娠糖尿病"
    ]

    def is_diabetes(text):
        return any(k in text for k in keywords)

    diabetes_dialogs = []
    for dialog in data:
        for turn in dialog.get("information", []):
            if is_diabetes(turn.get("sentence", "")):
                diabetes_dialogs.append(dialog)
                break

    with open("data/prepared/remedi_diabetes.json", "w", encoding="utf-8") as f:
        json.dump(diabetes_dialogs, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(diabetes_dialogs)} diabetes dialogues from ReMeDi")

# ----------------------------
# 4. LLM-CGM benchmark (copy only)
# ----------------------------
def prepare_llm_cgm():
    import sys
    import importlib.util

    file_path = "data/LLM-CGM/src/benchmarkQandA.py"
    if not os.path.exists(file_path):
        print("❌ benchmarkQandA.py not found.")
        return

    # Dynamically import the get_questions function from the .py file
    spec = importlib.util.spec_from_file_location("benchmarkQandA", file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["benchmarkQandA"] = module
    spec.loader.exec_module(module)

    if hasattr(module, "get_questions"):
        questions = module.get_questions()
        output_path = "data/prepared/llm_cgm_questions.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved {len(questions)} CGM questions from benchmarkQandA.py")
    else:
        print("❌ get_questions() not found in benchmarkQandA.py")



# Run all
if __name__ == "__main__":
    print("🚀 Starting dataset preparation...")

    clone_repo("https://github.com/abachaa/MedQuAD.git", "data/MedQuAD")
    clone_repo("https://github.com/lizhealey/LLM-CGM.git", "data/LLM-CGM")

    # Skip broken ones
    print("\n⚠️ Skipping MedDialog and ReMeDi cloning (manual download required)")
    print("📥 MedDialog ➜ https://drive.google.com/drive/folders/1g29ssimdZ6JzTST6Y8g6h-ogUNReBtJD")
    print("📥 ReMeDi ➜ Obtain ReMeDi-base.json manually and place in data/ReMeDi/\n")

    prepare_medquad()
    prepare_meddialog()
    prepare_remedi()
    prepare_llm_cgm()

    print("\n✅ All done! Prepared data is in `data/prepared/`.")
