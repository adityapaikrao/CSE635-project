from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

app = Flask(__name__)

MODEL_NAME = "bigscience/bloomz-1b7"
LORA_PATH = "./lora-bloomz-diabetes"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=bnb_config
)

model = PeftModel.from_pretrained(model, LORA_PATH, is_trainable=False)
model.eval()

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.json
        question = data.get("question", "")
        context = data.get("context", "")
        conditions = data.get("conditions", "")
        language = data.get("language", "EN")

        prompt = f"""You are a medically‑accurate diabetes assistant.
Respond in the same language as the question ({'Spanish' if language == 'ES' else 'English'}).

PATIENT CONDITIONS: {conditions}
RETRIEVED INFO:
{context}

QUESTION: {question}

ANSWER:"""

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True
            )

        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = response_text[len(prompt):].strip()
        return jsonify({"response": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
