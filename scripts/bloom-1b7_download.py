from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-1b7", force_download=True)
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-1b7")
print("Model loaded successfully!")