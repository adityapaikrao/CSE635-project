import pandas as pd
import evaluate
from rouge_score import rouge_scorer
import numpy as np

# Load data
df = pd.read_csv("chatbot_eval_output.csv")

# Verify columns
print("Available columns:", df.columns.tolist())
print("\nSample data:")
print(df.head(3))

# Separate by language
df_en = df[df["language"] == "en"]
df_es = df[df["language"] == "es"]

# Initialize metrics
bleu = evaluate.load("sacrebleu")
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
bertscore = evaluate.load("bertscore")

def evaluate_metrics(df_subset, lang_code):
    refs = df_subset["reference_answer"].tolist()
    preds = df_subset["generated"].tolist()  # Note: Typo in your original column name?
    
    # Verify we have data
    if len(preds) == 0:
        print(f"\n⚠️ No {lang_code.upper()} examples to evaluate")
        return

    print(f"\n🔍 Evaluating {lang_code.upper()} - {len(preds)} examples")

    # BLEU
    bleu_result = bleu.compute(predictions=preds, references=[[r] for r in refs])
    print(f"BLEU: {bleu_result['score']:.2f}")

    # ROUGE
    rouge_scores = [rouge.score(ref, pred)['rougeL'].fmeasure for ref, pred in zip(refs, preds)]
    print(f"ROUGE-L: {np.mean(rouge_scores):.2f} ± {np.std(rouge_scores):.2f}")

    # BERTScore
    bert_result = bertscore.compute(predictions=preds, references=refs, lang=lang_code)
    print(f"BERTScore F1: {np.mean(bert_result['f1']):.2f} ± {np.std(bert_result['f1']):.2f}")

# Run evaluations
evaluate_metrics(df_en, "en")
evaluate_metrics(df_es, "es")