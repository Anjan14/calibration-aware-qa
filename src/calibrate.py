"""Calibration and threshold sweep utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class CalibratedDecision:
    """Stores calibrated scoring pieces for one example."""

    calibrated_span_score: float
    calibrated_null_score: float
    predict_no_answer: bool


def apply_calibration(best_span_score: float, null_score: float, answerability_prob: float, threshold: float) -> CalibratedDecision:
    """
    Apply interpretable calibration rule.

    If calibrated null score exceeds calibrated span score by threshold margin,
    we abstain and output no-answer.
    """
    calibrated_span = best_span_score * answerability_prob
    calibrated_null = null_score * (1.0 - answerability_prob)
    predict_no_answer = (calibrated_null - calibrated_span) > threshold
    return CalibratedDecision(calibrated_span, calibrated_null, predict_no_answer)


def sweep_thresholds(
    thresholds: Iterable[float],
    best_span_scores: list[float],
    null_scores: list[float],
    answerability_probs: list[float],
    gold_answerable: list[int],
) -> tuple[float, list[dict[str, float]]]:
    """Search thresholds and pick best by answerability accuracy."""
    results: list[dict[str, float]] = []
    best_threshold = 0.0
    best_acc = -1.0
    for t in thresholds:
        preds = []
        for b, n, p in zip(best_span_scores, null_scores, answerability_probs):
            decision = apply_calibration(b, n, p, t)
            preds.append(0 if decision.predict_no_answer else 1)
        acc = float(np.mean(np.array(preds) == np.array(gold_answerable)))
        results.append({"threshold": float(t), "answerability_accuracy": acc})
        if acc > best_acc:
            best_acc = acc
            best_threshold = float(t)
    return best_threshold, results

