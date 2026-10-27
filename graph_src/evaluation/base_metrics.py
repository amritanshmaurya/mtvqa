import re
import string
from typing import List, Dict, Optional
from collections import Counter


def normalize_answer(s: str) -> str:

    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', '', s)
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = ' '.join(s.split())
    return s.strip()


def exact_match(prediction: str, reference: str) -> float:

    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:

    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return f1


def compute_bertscore(
    predictions: List[str],
    references: List[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
    device: str = "cuda:0"
) -> Dict[str, float]:

    try:
        from bert_score import score as bert_score_fn

        P, R, F1 = bert_score_fn(
            predictions, references,
            model_type=model_type,
            device=device,
            verbose=False
        )

        return {
            'bertscore_precision': P.mean().item(),
            'bertscore_recall': R.mean().item(),
            'bertscore_f1': F1.mean().item()
        }
    except ImportError:
        print("Warning: bert_score not installed. Skipping BERTScore.")
        return {
            'bertscore_precision': 0.0,
            'bertscore_recall': 0.0,
            'bertscore_f1': 0.0
        }
    except Exception as e:
        print(f"Warning: BERTScore computation failed: {e}")
        return {
            'bertscore_precision': 0.0,
            'bertscore_recall': 0.0,
            'bertscore_f1': 0.0
        }


def compute_all_metrics(
    predictions: List[str],
    references: List[str],
    compute_bert: bool = True,
    device: str = "cuda:0"
) -> Dict[str, float]:

    assert len(predictions) == len(references), \
        f"Length mismatch: {len(predictions)} predictions vs {len(references)} references"

    n = len(predictions)
    em_scores = []
    f1_scores = []

    for pred, ref in zip(predictions, references):
        em_scores.append(exact_match(pred, ref))
        f1_scores.append(token_f1(pred, ref))

    metrics = {
        'exact_match': sum(em_scores) / n,
        'token_f1': sum(f1_scores) / n,
        'num_samples': n
    }

    if compute_bert:
        bert_results = compute_bertscore(predictions, references, device=device)
        metrics.update(bert_results)

    return metrics


def compute_retrieval_recall(
    retrieved_texts: List[List[str]],
    answers: List[str],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:

    results = {}

    for k in k_values:
        hits = 0
        for retrieved, answer in zip(retrieved_texts, answers):
            top_k_text = ' '.join(retrieved[:k]).lower()
            if normalize_answer(answer) in normalize_answer(top_k_text):
                hits += 1
        results[f'recall@{k}'] = hits / len(answers) if answers else 0.0

    return results
