import json
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import get_peft_model, LoraConfig, TaskType
from pathlib import Path

# Check CUDA availability and device count first
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"Number of CUDA devices: {device_count}")
    
    # Use device 0 (first available GPU)
    device = torch.device("cuda:0")
    print(f"Using device: {device}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available. Using CPU instead.")
    device = torch.device("cpu")

# ✅ Load JSON dataset
with open("data/final/diabetes_lora.json", "r") as f:
    qa_data = json.load(f)

dataset = Dataset.from_list(qa_data)

# ✅ Model + Tokenizer
model_name = "bigscience/bloomz-560m"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set

# Load model to the available GPU or CPU
if torch.cuda.is_available():
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        device_map="auto"  # Let it automatically use available GPU
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": "cpu"}  # Use CPU
    )

# ✅ Apply LoRA using PEFT
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    task_type=TaskType.CAUSAL_LM,
    lora_dropout=0.05,
    bias="none",
    target_modules=["query_key_value"] if "bloom" in model.config.architectures[0].lower() else ["q_proj", "v_proj"]
)

model = get_peft_model(model, peft_config)

# Verify model device
print(f"Model device: {next(model.parameters()).device}")

# ✅ Tokenize - Fixed function
def preprocess(example):
    # Format the prompt with input
    prompt = f"{example['instruction']}\n\n{example['input']}" if example["input"] else example["instruction"]
    
    # Combine prompt and target for language modeling
    full_text = f"{prompt}\n\n{example['output']}"
    
    # Tokenize the full text
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors=None,
    )
    
    # Create labels by copying input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    # Mask prompt tokens in labels (set to -100 so they're ignored in loss calculation)
    prompt_tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors=None,
    )
    prompt_len = len([t for t in prompt_tokens["input_ids"] if t != tokenizer.pad_token_id])
    
    # Set prompt part of labels to -100 (ignore in loss)
    tokenized["labels"][:prompt_len] = [-100] * prompt_len
    
    # Check for potential padding at the end and mask it in labels
    for i in range(len(tokenized["labels"])):
        if tokenized["input_ids"][i] == tokenizer.pad_token_id:
            tokenized["labels"][i] = -100
    
    return tokenized

# Process the dataset
tokenized_dataset = dataset.map(preprocess)

# ✅ Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

# ✅ Training args
args = TrainingArguments(
    output_dir="lora-bloomz-diabetes",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=100,
    save_total_limit=1,
    report_to="none",
    bf16=False,
    fp16=torch.cuda.is_available(),  # Only use fp16 if CUDA is available
    gradient_accumulation_steps=2,
    dataloader_pin_memory=torch.cuda.is_available(),
)

# ✅ Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# ✅ Start training
trainer.train()

# ✅ Save final LoRA adapter
model.save_pretrained("lora-bloomz-diabetes")
tokenizer.save_pretrained("lora-bloomz-diabetes")

print("✅ Training complete. Adapter saved to lora-bloomz-diabetes/")