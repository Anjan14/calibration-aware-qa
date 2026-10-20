"""Calibration and selective-prediction metrics (ECE, Brier, risk–coverage AURC)."""

from __future__ import annotations

import numpy as np


def _clip_probs(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    return np.clip(p, eps, 1.0 - eps)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error (ECE) for binary labels in [0, 1].

    Bins predictions by confidence; compares bin accuracy to mean confidence.
    """
    probs = _clip_probs(probs)
    y = np.asarray(labels, dtype=np.float64).ravel()
    if len(probs) != len(y):
        raise ValueError("probs and labels must have the same length.")
    if len(y) == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        acc = float(y[mask].mean())
        conf = float(probs[mask].mean())
        ece += (n / len(y)) * abs(acc - conf)
    return float(ece)


def brier_score_binary(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared error between predicted probability and binary label."""
    probs = _clip_probs(probs)
    y = np.asarray(labels, dtype=np.float64).ravel()
    if len(probs) != len(y):
        raise ValueError("probs and labels must have the same length.")
    if len(y) == 0:
        return 0.0
    return float(np.mean((probs - y) ** 2))


def area_under_risk_coverage_curve(confidence: np.ndarray, correct: np.ndarray) -> float:
    """
    Area under the risk–coverage curve (selective prediction).

    Sorts examples by descending confidence; at each coverage k/n, risk = 1 - accuracy
    on the k most confident examples. Integrates risk vs coverage in [0, 1].

    Lower is better (less average risk when sweeping coverage).
    """
    conf = np.asarray(confidence, dtype=np.float64).ravel()
    corr = np.asarray(correct, dtype=np.float64).ravel()
    if len(conf) != len(corr):
        raise ValueError("confidence and correct must have the same length.")
    n = len(conf)
    if n == 0:
        return 0.0
    order = np.argsort(-conf)
    c_sorted = corr[order]
    cum_acc = np.cumsum(c_sorted) / np.arange(1, n + 1, dtype=np.float64)
    risk = 1.0 - cum_acc
    coverage = np.arange(1, n + 1, dtype=np.float64) / float(n)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(risk, coverage))
    return float(np.trapz(risk, coverage))


def compute_calibration_bundle(
    answerability_probs: np.ndarray,
    gold_answerable: np.ndarray,
    em_correct: np.ndarray,
) -> dict[str, float]:
    """
    Aggregate metrics for publication: answerability ECE/Brier and selective AURC on EM.

    - ECE/Brier use model probability of answerable vs gold 0/1 label.
    - AURC uses same confidence ordering vs per-example EM correctness (selective QA).
    """
    probs = np.asarray(answerability_probs, dtype=np.float64).ravel()
    gold = np.asarray(gold_answerable, dtype=np.float64).ravel()
    em = np.asarray(em_correct, dtype=np.float64).ravel()
    if not (len(probs) == len(gold) == len(em)):
        raise ValueError("answerability_probs, gold_answerable, em_correct length mismatch.")

    return {
        "ece_answerability": expected_calibration_error(probs, gold),
        "brier_answerability": brier_score_binary(probs, gold),
        "aurc_risk_coverage_em": area_under_risk_coverage_curve(probs, em),
    }
