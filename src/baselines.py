"""Simple baseline systems for comparison."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .evaluate import compute_qa_metrics


def first_three_words_baseline(context: str) -> str:
    """Return first 3 words as a trivial deterministic baseline."""
    words = context.split()
    return " ".join(words[:3])


def lexical_overlap_sentence_baseline(question: str, context: str) -> str:
    """Pick sentence with highest lexical overlap with question."""
    q_tokens = [t for t in re.findall(r"\w+", question.lower()) if t]
    q_counts = Counter(q_tokens)
    sentences = re.split(r"(?<=[.!?])\s+", context.strip())
    best_sentence = sentences[0] if sentences else ""
    best_score = -1
    for sent in sentences:
        s_tokens = [t for t in re.findall(r"\w+", sent.lower()) if t]
        score = sum((Counter(s_tokens) & q_counts).values())
        if score > best_score:
            best_score = score
            best_sentence = sent
    return best_sentence


def evaluate_baselines(examples: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Evaluate both baselines on QA metrics."""
    references = {}
    answerable = {}
    pred_a = {}
    pred_b = {}
    for ex in examples:
        qid = ex["id"]
        ans = ex["answers"]["text"]
        references[qid] = ans if ans else [""]
        answerable[qid] = 1 if len(ans) > 0 else 0
        pred_a[qid] = first_three_words_baseline(ex["context"]) if answerable[qid] else ""
        pred_b[qid] = lexical_overlap_sentence_baseline(ex["question"], ex["context"]) if answerable[qid] else ""
    return {
        "first_three_words": compute_qa_metrics(pred_a, references, answerable),
        "lexical_overlap_sentence": compute_qa_metrics(pred_b, references, answerable),
    }

