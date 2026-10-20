"""Helpers for downstream error/confidence analysis."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def confidence_buckets(prediction_metadata: dict[str, dict[str, Any]], bucket_size: float = 0.1) -> dict[str, int]:
    """Group examples by answerability probability buckets."""
    buckets: dict[str, int] = defaultdict(int)
    for item in prediction_metadata.values():
        p = float(item.get("answerability_prob", 0.0))
        start = max(0.0, min(0.9, (p // bucket_size) * bucket_size))
        end = min(1.0, start + bucket_size)
        key = f"{start:.1f}-{end:.1f}"
        buckets[key] += 1
    return dict(sorted(buckets.items()))


def group_by_correctness(predictions: dict[str, str], gold_answers: dict[str, list[str]]) -> dict[str, list[str]]:
    """Separate IDs where prediction exactly matches any gold normalized answer."""
    from .evaluate import normalize_answer

    out = {"correct": [], "incorrect": []}
    for qid, pred in predictions.items():
        norm_pred = normalize_answer(pred)
        norms = {normalize_answer(g) for g in gold_answers.get(qid, [""])}
        if norm_pred in norms:
            out["correct"].append(qid)
        else:
            out["incorrect"].append(qid)
    return out

