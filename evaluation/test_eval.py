# evaluation/test_eval.py
from evaluation.metrics_eval import evaluate_predictions

# Sample test
preds = [
    "You should check sugar after meals and avoid high carbs."
]
refs = [
    "It's recommended to monitor sugar after eating and limit carbohydrates."
]

evaluate_predictions(preds, refs)