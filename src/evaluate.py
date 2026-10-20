"""Evaluation, post-processing, and metric logic."""

from __future__ import annotations

import collections
import re
import string
from typing import Any

import numpy as np

from .calibrate import apply_calibration
from .calibration_metrics import compute_calibration_bundle


def normalize_answer(text: str) -> str:
    """Lowercase, remove punctuation/articles, and normalize spaces."""
    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        return "".join(ch for ch in s if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    if len(pred_tokens) == 0 and len(truth_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 0.0

    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def metric_max_over_ground_truths(metric_fn, prediction: str, gold_answers: list[str]) -> float:
    if not gold_answers:
        gold_answers = [""]
    return max(metric_fn(prediction, g) for g in gold_answers)


def _best_valid_span(start_logits: np.ndarray, end_logits: np.ndarray, offsets: list[Any], context: str, n_best_size: int, max_answer_len: int) -> tuple[str, float]:
    start_indexes = np.argsort(start_logits)[-n_best_size:][::-1]
    end_indexes = np.argsort(end_logits)[-n_best_size:][::-1]
    best_text = ""
    best_score = -1e9
    for s_idx in start_indexes:
        for e_idx in end_indexes:
            if s_idx >= len(offsets) or e_idx >= len(offsets):
                continue
            if offsets[s_idx] is None or offsets[e_idx] is None:
                continue
            if e_idx < s_idx:
                continue
            if (e_idx - s_idx + 1) > max_answer_len:
                continue
            start_char, _ = offsets[s_idx]
            _, end_char = offsets[e_idx]
            text = context[start_char:end_char]
            score = float(start_logits[s_idx] + end_logits[e_idx])
            if score > best_score:
                best_score = score
                best_text = text
    return best_text, best_score


def postprocess_predictions(
    examples: list[dict[str, Any]],
    features: list[dict[str, Any]],
    all_start_logits: np.ndarray,
    all_end_logits: np.ndarray,
    all_answerability_logits: np.ndarray,
    threshold: float = 0.0,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    use_calibration: bool = True,
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """Convert feature-level logits into example-level predictions."""
    example_id_to_index = {ex["id"]: i for i, ex in enumerate(examples)}
    features_per_example: dict[int, list[int]] = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    predictions: dict[str, str] = {}
    metadata: dict[str, dict[str, Any]] = {}
    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))

    for ex_idx, ex in enumerate(examples):
        feature_indices = features_per_example.get(ex_idx, [])
        best_text = ""
        best_span_score = -1e9
        best_null_score = -1e9
        best_ans_prob = 0.0

        for f_idx in feature_indices:
            start = all_start_logits[f_idx]
            end = all_end_logits[f_idx]
            offsets = features[f_idx]["offset_mapping"]
            ans_prob = float(sigmoid(all_answerability_logits[f_idx]))
            # In BERT-style inputs with pair formatting, CLS is index 0.
            cls_idx = 0
            null_score = float(start[cls_idx] + end[cls_idx])
            text, span_score = _best_valid_span(start, end, offsets, ex["context"], n_best_size, max_answer_length)

            if span_score > best_span_score:
                best_span_score = span_score
                best_text = text
                best_ans_prob = ans_prob
            if null_score > best_null_score:
                best_null_score = null_score

        pred_text = best_text
        if use_calibration:
            decision = apply_calibration(best_span_score, best_null_score, best_ans_prob, threshold)
            if decision.predict_no_answer:
                pred_text = ""
        elif (best_null_score - best_span_score) > threshold:
            pred_text = ""

        predictions[ex["id"]] = pred_text
        metadata[ex["id"]] = {
            "predicted_text": pred_text,
            "best_span_score": best_span_score,
            "null_score": best_null_score,
            "answerability_prob": best_ans_prob,
        }
    return predictions, metadata


def compute_per_example_scores(
    examples: list[dict[str, Any]],
    predictions: dict[str, str],
    references: dict[str, list[str]],
    reference_answerable: dict[str, int],
    metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Per-example EM, F1, answerability match, and probs for bootstrap / calibration.

    Order follows ``examples`` iteration order.
    """
    em_list: list[float] = []
    f1_list: list[float] = []
    ans_correct: list[float] = []
    probs: list[float] = []
    gold_ans: list[int] = []

    for ex in examples:
        qid = ex["id"]
        pred = predictions[qid]
        gold_answers = references.get(qid, [""])
        em_list.append(metric_max_over_ground_truths(exact_match_score, pred, gold_answers))
        f1_list.append(metric_max_over_ground_truths(f1_score, pred, gold_answers))
        pred_answerable = 0 if pred.strip() == "" else 1
        ans_correct.append(float(pred_answerable == reference_answerable.get(qid, 0)))
        probs.append(float(metadata[qid]["answerability_prob"]))
        gold_ans.append(int(reference_answerable.get(qid, 0)))

    em_arr = np.array(em_list, dtype=np.float64)
    f1_arr = np.array(f1_list, dtype=np.float64)
    ans_arr = np.array(ans_correct, dtype=np.float64)
    p_arr = np.array(probs, dtype=np.float64)
    g_arr = np.array(gold_ans, dtype=np.float64)

    cal = compute_calibration_bundle(p_arr, g_arr, em_arr)
    return {
        "per_example_em": em_list,
        "per_example_f1": f1_list,
        "per_example_answerability_correct": ans_correct,
        "answerability_prob": probs,
        "gold_answerable": gold_ans,
        "calibration_metrics": cal,
    }


def compute_qa_metrics(predictions: dict[str, str], references: dict[str, list[str]], reference_answerable: dict[str, int]) -> dict[str, float]:
    """Compute EM/F1 and answerability accuracy with multiple gold answers."""
    em_scores = []
    f1_scores = []
    ans_scores = []
    for qid, pred in predictions.items():
        gold_answers = references.get(qid, [""])
        em_scores.append(metric_max_over_ground_truths(exact_match_score, pred, gold_answers))
        f1_scores.append(metric_max_over_ground_truths(f1_score, pred, gold_answers))
        pred_answerable = 0 if pred.strip() == "" else 1
        ans_scores.append(float(pred_answerable == reference_answerable.get(qid, 0)))
    return {
        "exact_match": float(np.mean(em_scores) * 100),
        "f1": float(np.mean(f1_scores) * 100),
        "answerability_accuracy": float(np.mean(ans_scores) * 100),
    }

