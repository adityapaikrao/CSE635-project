import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel, PeftConfig
import evaluate
from tqdm import tqdm

# === Settings ===
model_name = "bigscience/bloomz-560m"
adapter_path = "./lora-bloomz-diabetes"  # Keep the original path
dataset_path = "data/prepared/medquad_diabetes.json"  # change as needed
max_examples = 50  # evaluate on first N examples

# === Load model + tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16
).to("cuda:0")


# Find the adapter configuration file
for root, dirs, files in os.walk(adapter_path):
    if "adapter_config.json" in files:
        config_path = os.path.join(root, "adapter_config.json")
        print(f"Found adapter config at: {config_path}")
        adapter_path = root
        break
else:
    print(f"❌ No adapter_config.json found in {adapter_path} or its subdirectories")
    exit(1)

print(f"Loading adapter from: {adapter_path}")
model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
    is_trainable=False,
    local_files_only=True
)

# Rest of your code remains the same

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

# === Load dataset ===
with open(dataset_path) as f:
    data = json.load(f)

questions, references, predictions = [], [], []

# === Evaluate ===
print(f"🚀 Evaluating {max_examples} examples from {dataset_path}")
for item in tqdm(data[:max_examples]):
    q = item.get("question") or item.get("instruction")
    a = item.get("reference") or item.get("output")

    prompt = f"Patient: {q.strip()}"
    gen = generator(prompt, max_new_tokens=150, do_sample=False)[0]["generated_text"]
    generated = gen.split("Patient:")[-1].strip().split("\n")[-1].strip()

    questions.append(q.strip())
    references.append(a.strip())
    predictions.append(generated)

# === Scoring ===
bleu = evaluate.load("bleu").compute(predictions=predictions, references=[[r] for r in references])
rouge = evaluate.load("rouge").compute(predictions=predictions, references=references)
bertscore = evaluate.load("bertscore").compute(predictions=predictions, references=references, lang="en")

print("\n===== EVALUATION METRICS =====")
print(f"BLEU: {bleu['bleu']:.4f}")
print(f"ROUGE-1: {rouge['rouge1']:.4f}")
print(f"ROUGE-2: {rouge['rouge2']:.4f}")
print(f"ROUGE-L: {rouge['rougeL']:.4f}")
print(f"BERTScore-F1: {sum(bertscore['f1'])/len(bertscore['f1']):.4f}")
print("================================")
