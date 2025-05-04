import os
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sklearn.metrics import accuracy_score
from datetime import datetime
from difflib import get_close_matches
from sentence_transformers import SentenceTransformer, util

def compute_bleu(preds, refs):
    refs_tokenized = [[r.split()] for r in refs]
    preds_tokenized = [p.split() for p in preds]
    smoothie = SmoothingFunction().method4  # 🔹 Added smoothing
    return corpus_bleu(refs_tokenized, preds_tokenized, smoothing_function=smoothie)

def compute_rouge(preds, refs):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = [scorer.score(ref, pred)['rougeL'].fmeasure for pred, ref in zip(preds, refs)]
    return sum(scores) / len(scores)

def compute_bert_score(preds, refs):
    P, R, F1 = bert_score(preds, refs, lang="en", verbose=False)
    print(f"\n🔬 BERTScore Breakdown:")
    print(f"Precision: {P.mean().item():.4f}, Recall: {R.mean().item():.4f}, F1: {F1.mean().item():.4f}")
    return F1.mean().item()

def compute_exact_match(preds, refs):
    def normalize(text):
        return text.strip().lower().replace(".", "").replace(",", "")
    preds_norm = [normalize(p) for p in preds]
    refs_norm = [normalize(r) for r in refs]
    return accuracy_score(refs_norm, preds_norm)

def compute_sbert_cosine(preds, refs):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    emb_preds = model.encode(preds, convert_to_tensor=True)
    emb_refs = model.encode(refs, convert_to_tensor=True)
    cosine_scores = util.cos_sim(emb_preds, emb_refs).diagonal()
    return cosine_scores.cpu().numpy().tolist(), cosine_scores.mean().item()

def get_column_by_guess(df, desired_col):
    cols = df.columns.tolist()
    close = get_close_matches(desired_col, cols, n=1, cutoff=0.6)
    if close:
        print(f"[INFO] Matched '{desired_col}' to '{close[0]}' in CSV.")
        return close[0]
    else:
        raise KeyError(f"Column '{desired_col}' not found and no close match in: {cols}")

def log_mismatches(df, predictions, references):
    print("\n🔍 Sample mismatches:\n")
    mismatch_count = 0
    for i in range(len(predictions)):
        pred_norm = predictions[i].strip().lower()
        ref_norm = references[i].strip().lower()
        if pred_norm != ref_norm:
            mismatch_count += 1
            print(f"Q{i+1}: {df.iloc[i]['question']}")
            print(f"Ref : {references[i]}")
            print(f"Gen : {predictions[i]}")
            print("-" * 50)
        if mismatch_count >= 5:
            break

def main():
    eval_path = "mock_eval_100.csv"
    log_file = "evaluation_log.txt"

    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"{eval_path} not found.")

    df = pd.read_csv(eval_path, sep=",")
    df.columns = df.columns.str.strip().str.lower()
    print(f"[INFO] Parsed columns: {df.columns.tolist()}")

    pred_col = get_column_by_guess(df, "generated")
    ref_col = get_column_by_guess(df, "reference_answer")

    predictions = df[pred_col].astype(str).tolist()
    references = df[ref_col].astype(str).tolist()

    bleu = compute_bleu(predictions, references)
    rouge = compute_rouge(predictions, references)
    bert = compute_bert_score(predictions, references)
    exact = compute_exact_match(predictions, references)
    cosine_scores, cosine_avg = compute_sbert_cosine(predictions, references)

    df['sbert_cosine'] = cosine_scores
    enriched_path = "evaluation_with_scores.csv"
    df.to_csv(enriched_path, index=False)
    print(f"\n📁 Exported evaluation with SBERT scores to '{enriched_path}'")

    output = (
        f"\n📊 Evaluation Metrics ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}):\n"
        f"BLEU: {bleu:.4f}\n"
        f"ROUGE-L: {rouge:.4f}\n"
        f"BERTScore (F1): {bert:.4f}\n"
        f"Exact Match Accuracy: {exact:.4f}\n"
        f"SBERT Cosine Similarity: {cosine_avg:.4f}\n"
    )

    print(output)
    log_mismatches(df, predictions, references)

    with open(log_file, "a") as f:
        f.write(output)


if __name__ == "__main__":
    main()
