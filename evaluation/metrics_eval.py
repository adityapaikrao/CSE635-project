# evaluation/metrics_eval.py
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from bert_score import score as bert_score

def calculate_bleu(prediction, reference):
    """Calculate BLEU score between prediction and reference."""
    if isinstance(prediction, str):
        prediction = prediction.split()
    if isinstance(reference, str):
        reference = [reference.split()]
    elif isinstance(reference, list) and isinstance(reference[0], str):
        reference = [r.split() for r in reference]
        
    smoothie = SmoothingFunction().method1
    try:
        return sentence_bleu(reference, prediction, smoothing_function=smoothie)
    except Exception as e:
        print(f"BLEU calculation error: {e}")
        return 0.0

def calculate_rouge(prediction, reference):
    """Calculate ROUGE scores between prediction and reference."""
    if not prediction or not reference:
        return {'rouge-1': {'f': 0.0}, 'rouge-2': {'f': 0.0}, 'rouge-l': {'f': 0.0}}
    
    rouge = Rouge()
    try:
        scores = rouge.get_scores(prediction, reference)[0]
        return scores
    except Exception as e:
        print(f"ROUGE calculation error: {e}")
        return {'rouge-1': {'f': 0.0}, 'rouge-2': {'f': 0.0}, 'rouge-l': {'f': 0.0}}

def calculate_bertscore(predictions, references, lang="en"):
    """Calculate BERTScore between predictions and references."""
    try:
        P, R, F1 = bert_score(predictions, references, lang=lang)
        return {
            'precision': P.mean().item(),
            'recall': R.mean().item(),
            'f1': F1.mean().item()
        }
    except Exception as e:
        print(f"BERTScore calculation error: {e}")
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

def evaluate_predictions(predictions, references, verbose=True):
    """
    Evaluate a list of predictions against references using multiple metrics.
    
    Args:
        predictions: List of prediction strings
        references: List of reference strings
        verbose: Whether to print results
        
    Returns:
        Dictionary with evaluation results
    """
    if len(predictions) != len(references):
        raise ValueError(f"Number of predictions ({len(predictions)}) doesn't match references ({len(references)})")
    
    results = {
        'bleu': [],
        'rouge-1': [],
        'rouge-2': [],
        'rouge-l': [],
        'bertscore': {'precision': [], 'recall': [], 'f1': []}
    }
    
    # Individual scores
    for pred, ref in zip(predictions, references):
        # BLEU
        bleu_score = calculate_bleu(pred, ref)
        results['bleu'].append(bleu_score)
        
        # ROUGE
        rouge_scores = calculate_rouge(pred, ref)
        results['rouge-1'].append(rouge_scores['rouge-1']['f'])
        results['rouge-2'].append(rouge_scores['rouge-2']['f'])
        results['rouge-l'].append(rouge_scores['rouge-l']['f'])
    
    # BERTScore (batch calculation)
    bert_scores = calculate_bertscore(predictions, references)
    results['bertscore'] = bert_scores
    
    # Calculate averages
    avg_results = {
        'bleu': np.mean(results['bleu']),
        'rouge-1': np.mean(results['rouge-1']),
        'rouge-2': np.mean(results['rouge-2']),
        'rouge-l': np.mean(results['rouge-l']),
        'bertscore-precision': bert_scores['precision'],
        'bertscore-recall': bert_scores['recall'],
        'bertscore-f1': bert_scores['f1']
    }
    
    if verbose:
        print("\n===== Evaluation Results =====")
        print(f"BLEU: {avg_results['bleu']:.4f}")
        print(f"ROUGE-1: {avg_results['rouge-1']:.4f}")
        print(f"ROUGE-2: {avg_results['rouge-2']:.4f}")
        print(f"ROUGE-L: {avg_results['rouge-l']:.4f}")
        print(f"BERTScore-P: {avg_results['bertscore-precision']:.4f}")
        print(f"BERTScore-R: {avg_results['bertscore-recall']:.4f}")
        print(f"BERTScore-F1: {avg_results['bertscore-f1']:.4f}")
        print("=============================\n")
    
    return avg_results