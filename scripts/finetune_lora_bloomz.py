import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from pathlib import Path

# --- Hardware check ---
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"Number of CUDA devices: {device_count}")
    print(f"Using device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available. Using CPU.")

# --- Load extended JSON dataset ---
with open("data/final/diabetes_lora.json", "r") as f:
    qa_data = json.load(f)

dataset = Dataset.from_list(qa_data)

# --- Model and Tokenizer ---
model_name = "bigscience/bloomz-1b7"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# --- Quantized 4-bit model loading ---
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    use_safetensors=True,            # ✅ Use .safetensors if available (much better than .bin for memory)
    trust_remote_code=True,
    low_cpu_mem_usage=True           # ✅ This prevents mmap!
)

# --- LoRA setup ---
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    task_type=TaskType.CAUSAL_LM,
    lora_dropout=0.05,
    bias="none",
    target_modules=["query_key_value"]
)
model = get_peft_model(model, peft_config)

print(f"LoRA model loaded on device: {next(model.parameters()).device}")

# --- Tokenization ---
def preprocess(example):
    prompt = f"{example['instruction']}\n\n{example['input']}" if example["input"] else example["instruction"]
    full_text = f"{prompt}\n\n{example['output']}"

    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=1024,
        padding="max_length",
        return_tensors=None,
    )

    tokenized["labels"] = tokenized["input_ids"].copy()

    prompt_tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=1024,
        padding="max_length",
        return_tensors=None,
    )
    prompt_len = len([t for t in prompt_tokens["input_ids"] if t != tokenizer.pad_token_id])
    tokenized["labels"][:prompt_len] = [-100] * prompt_len

    for i in range(len(tokenized["labels"])):
        if tokenized["input_ids"][i] == tokenizer.pad_token_id:
            tokenized["labels"][i] = -100

    return tokenized

tokenized_dataset = dataset.map(preprocess)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

# --- Training Arguments ---
args = TrainingArguments(
    output_dir="lora-bloomz1b7-diabetes",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=100,
    save_total_limit=1,
    report_to="none",
    fp16=True,
    dataloader_pin_memory=True,
)

# --- Trainer ---
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# --- Train and Save ---
trainer.train()

model.save_pretrained("lora-bloomz-diabetes")
tokenizer.save_pretrained("lora-bloomz-diabetes")
print("✅ Training complete. Adapter saved to lora-bloomz-diabetes/")
