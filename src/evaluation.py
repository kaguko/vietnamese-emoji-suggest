"""
Evaluation utilities for Vietnamese Emoji Suggestion System.
"""

from typing import List, Dict, Optional, Any
import numpy as np
from collections import Counter
import pandas as pd


def precision_at_k(true_labels: List[str], predictions: List[str], k: int = 3) -> float:
    """
    Calculate precision@k - what fraction of top-k predictions are correct.
    
    Args:
        true_labels: List of correct emoji labels
        predictions: List of predicted emojis
        k: Number of predictions to consider
        
    Returns:
        Precision score (0-1)
    """
    predictions = predictions[:k]
    true_set = set(true_labels)
    pred_set = set(predictions)
    
    if not predictions:
        return 0.0
    
    correct = len(true_set & pred_set)
    return correct / len(predictions)


def recall_at_k(true_labels: List[str], predictions: List[str], k: int = 3) -> float:
    """
    Calculate recall@k - what fraction of true labels are in top-k predictions.
    
    Args:
        true_labels: List of correct emoji labels
        predictions: List of predicted emojis
        k: Number of predictions to consider
        
    Returns:
        Recall score (0-1)
    """
    predictions = predictions[:k]
    true_set = set(true_labels)
    pred_set = set(predictions)
    
    if not true_set:
        return 0.0
    
    correct = len(true_set & pred_set)
    return correct / len(true_set)


def hit_rate_at_k(true_labels: List[str], predictions: List[str], k: int = 3) -> float:
    """
    Calculate hit rate@k - is there at least one correct prediction in top-k.
    
    Args:
        true_labels: List of correct emoji labels
        predictions: List of predicted emojis
        k: Number of predictions to consider
        
    Returns:
        1.0 if at least one hit, 0.0 otherwise
    """
    predictions = predictions[:k]
    true_set = set(true_labels)
    pred_set = set(predictions)
    
    return 1.0 if len(true_set & pred_set) > 0 else 0.0


def mrr(true_labels: List[str], predictions: List[str]) -> float:
    """
    Calculate Mean Reciprocal Rank for a single prediction.
    
    Args:
        true_labels: List of correct emoji labels
        predictions: List of predicted emojis (ranked)
        
    Returns:
        Reciprocal rank (0-1)
    """
    true_set = set(true_labels)
    
    for i, pred in enumerate(predictions):
        if pred in true_set:
            return 1.0 / (i + 1)
    
    return 0.0


def ndcg_at_k(true_labels: List[str], predictions: List[str], k: int = 3) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at k.
    
    Args:
        true_labels: List of correct emoji labels
        predictions: List of predicted emojis
        k: Number of predictions to consider
        
    Returns:
        NDCG score (0-1)
    """
    predictions = predictions[:k]
    true_set = set(true_labels)
    
    # DCG
    dcg = 0.0
    for i, pred in enumerate(predictions):
        if pred in true_set:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because positions are 1-indexed
    
    # Ideal DCG (all correct predictions at top)
    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(true_labels), k)))
    
    if ideal_dcg == 0:
        return 0.0
    
    return dcg / ideal_dcg


def evaluate_model(
    model: Any,
    test_data: List[Dict],
    k: int = 3,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Model with .suggest() method
        test_data: List of dicts with 'text' and 'emoji_1', 'emoji_2', 'emoji_3'
        k: Number of predictions for metrics
        verbose: Print detailed results
        
    Returns:
        Dict with evaluation metrics
    """
    precisions = []
    recalls = []
    hit_rates = []
    mrrs = []
    ndcgs = []
    
    detailed_results = []
    
    for sample in test_data:
        text = sample['text']
        true_emojis = [
            sample.get('emoji_1'),
            sample.get('emoji_2'),
            sample.get('emoji_3')
        ]
        true_emojis = [e for e in true_emojis if e]  # Remove None/empty
        
        predictions = model.suggest(text)
        
        # Calculate metrics
        prec = precision_at_k(true_emojis, predictions, k)
        rec = recall_at_k(true_emojis, predictions, k)
        hr = hit_rate_at_k(true_emojis, predictions, k)
        rr = mrr(true_emojis, predictions)
        ndcg_score = ndcg_at_k(true_emojis, predictions, k)
        
        precisions.append(prec)
        recalls.append(rec)
        hit_rates.append(hr)
        mrrs.append(rr)
        ndcgs.append(ndcg_score)
        
        if verbose:
            detailed_results.append({
                'text': text,
                'true': true_emojis,
                'pred': predictions,
                'precision': prec,
                'hit': hr > 0
            })
    
    results = {
        f'precision@{k}': np.mean(precisions),
        f'recall@{k}': np.mean(recalls),
        f'hit_rate@{k}': np.mean(hit_rates),
        'mrr': np.mean(mrrs),
        f'ndcg@{k}': np.mean(ndcgs),
        'num_samples': len(test_data)
    }
    
    if verbose:
        results['detailed'] = detailed_results
    
    return results


def compare_models(
    models: Dict[str, Any],
    test_data: List[Dict],
    k: int = 3
) -> pd.DataFrame:
    """
    Compare multiple models on the same test data.
    
    Args:
        models: Dict of {model_name: model}
        test_data: List of test samples
        k: Number of predictions for metrics
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for name, model in models.items():
        metrics = evaluate_model(model, test_data, k)
        metrics['model'] = name
        results.append(metrics)
    
    df = pd.DataFrame(results)
    df = df.set_index('model')
    
    return df


def error_analysis(
    model: Any,
    test_data: List[Dict],
    k: int = 3
) -> Dict[str, Any]:
    """
    Perform error analysis on model predictions.
    
    Args:
        model: Model with .suggest() method
        test_data: List of test samples
        k: Number of predictions
        
    Returns:
        Dict with error analysis results
    """
    errors = []
    correct = []
    
    for sample in test_data:
        text = sample['text']
        true_emojis = [
            sample.get('emoji_1'),
            sample.get('emoji_2'),
            sample.get('emoji_3')
        ]
        true_emojis = [e for e in true_emojis if e]
        
        predictions = model.suggest(text)
        
        if hit_rate_at_k(true_emojis, predictions, k) == 0:
            errors.append({
                'text': text,
                'emotion': sample.get('primary_emotion', 'unknown'),
                'intensity': sample.get('intensity', 0),
                'true': true_emojis,
                'pred': predictions,
                'text_length': len(text.split())
            })
        else:
            correct.append({
                'text': text,
                'emotion': sample.get('primary_emotion', 'unknown'),
                'true': true_emojis,
                'pred': predictions
            })
    
    # Analyze errors
    error_emotions = Counter([e['emotion'] for e in errors])
    correct_emotions = Counter([c['emotion'] for c in correct])
    
    # Error rate by emotion
    emotion_error_rates = {}
    for emotion in set(list(error_emotions.keys()) + list(correct_emotions.keys())):
        err_count = error_emotions.get(emotion, 0)
        cor_count = correct_emotions.get(emotion, 0)
        total = err_count + cor_count
        if total > 0:
            emotion_error_rates[emotion] = err_count / total
    
    # Text length analysis
    error_lengths = [e['text_length'] for e in errors]
    avg_error_length = np.mean(error_lengths) if error_lengths else 0
    
    return {
        'total_errors': len(errors),
        'total_correct': len(correct),
        'error_rate': len(errors) / len(test_data) if test_data else 0,
        'errors_by_emotion': dict(error_emotions),
        'emotion_error_rates': emotion_error_rates,
        'avg_error_text_length': avg_error_length,
        'error_samples': errors[:10],  # First 10 errors for inspection
    }


def inter_rater_agreement(
    rater1_labels: List[str],
    rater2_labels: List[str]
) -> float:
    """
    Calculate Cohen's Kappa for inter-rater agreement.
    
    Args:
        rater1_labels: Labels from first rater
        rater2_labels: Labels from second rater
        
    Returns:
        Kappa score (-1 to 1)
    """
    if len(rater1_labels) != len(rater2_labels):
        raise ValueError("Raters must have same number of labels")
    
    n = len(rater1_labels)
    if n == 0:
        return 0.0
    
    # All unique labels
    all_labels = list(set(rater1_labels + rater2_labels))
    
    # Observed agreement
    observed = sum(1 for a, b in zip(rater1_labels, rater2_labels) if a == b) / n
    
    # Expected agreement by chance
    expected = 0.0
    for label in all_labels:
        p1 = sum(1 for l in rater1_labels if l == label) / n
        p2 = sum(1 for l in rater2_labels if l == label) / n
        expected += p1 * p2
    
    # Kappa
    if expected == 1:
        return 1.0
    
    kappa = (observed - expected) / (1 - expected)
    return kappa


if __name__ == "__main__":
    # Example usage
    print("Evaluation module loaded successfully!")
    
    # Test precision@k
    true = ["😊", "🎉", "🥳"]
    pred = ["😊", "😄", "🎉"]
    print(f"Precision@3: {precision_at_k(true, pred, 3):.2f}")
    print(f"Recall@3: {recall_at_k(true, pred, 3):.2f}")
    print(f"Hit Rate@3: {hit_rate_at_k(true, pred, 3):.2f}")
    print(f"MRR: {mrr(true, pred):.2f}")
    print(f"NDCG@3: {ndcg_at_k(true, pred, 3):.2f}")
