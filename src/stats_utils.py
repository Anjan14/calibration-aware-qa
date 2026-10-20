"""Bootstrap confidence intervals and paired comparisons for publication."""

from __future__ import annotations

from typing import Any

import numpy as np


def bootstrap_mean_ci(
    values: np.ndarray,
    n_bootstrap: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Nonparametric bootstrap CI for the mean of a 1D array.

    Returns mean point estimate and percentile CI.
    """
    x = np.asarray(values, dtype=np.float64).ravel()
    n = len(x)
    if n == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0, "n_bootstrap": n_bootstrap}

    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        means[i] = float(np.mean(x[idx]))

    return {
        "mean": float(np.mean(x)),
        "ci_low": float(np.percentile(means, 100 * alpha / 2)),
        "ci_high": float(np.percentile(means, 100 * (1 - alpha / 2))),
        "n": n,
        "n_bootstrap": n_bootstrap,
        "alpha": alpha,
    }


def bootstrap_paired_diff_ci(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Paired bootstrap for mean(A - B); useful for comparing two systems on the same examples.
    """
    a = np.asarray(scores_a, dtype=np.float64).ravel()
    b = np.asarray(scores_b, dtype=np.float64).ravel()
    if len(a) != len(b):
        raise ValueError("scores_a and scores_b must have the same length.")
    diff = a - b
    n = len(diff)
    if n == 0:
        return {
            "mean_diff": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "n": 0,
            "n_bootstrap": n_bootstrap,
        }

    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        means[i] = float(np.mean(diff[idx]))

    return {
        "mean_diff": float(np.mean(diff)),
        "ci_low": float(np.percentile(means, 100 * alpha / 2)),
        "ci_high": float(np.percentile(means, 100 * (1 - alpha / 2))),
        "n": n,
        "n_bootstrap": n_bootstrap,
        "alpha": alpha,
    }
