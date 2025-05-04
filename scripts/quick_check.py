import json

with open("data/final/diabetes_lora.json") as f:
    new_data = json.load(f)

print(f"Loaded {len(new_data)} examples.")
print("Sample:")
print(new_data[0])
