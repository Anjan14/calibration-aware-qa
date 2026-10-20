"""Assemble publication-ready metric bundles (bootstrap, tables)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .publication_tables import write_metrics_csv, write_metrics_latex
from .stats_utils import bootstrap_mean_ci, bootstrap_paired_diff_ci


def bootstrap_metrics_from_per_example(
    per_example: dict[str, Any],
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap CIs for mean EM, mean F1, and answerability accuracy (0–100 scale)."""
    em = np.asarray(per_example["per_example_em"], dtype=np.float64) * 100.0
    f1 = np.asarray(per_example["per_example_f1"], dtype=np.float64) * 100.0
    ans = np.asarray(per_example["per_example_answerability_correct"], dtype=np.float64) * 100.0

    return {
        "exact_match_pct": bootstrap_mean_ci(em, n_bootstrap=n_bootstrap, seed=seed),
        "f1_pct": bootstrap_mean_ci(f1, n_bootstrap=n_bootstrap, seed=seed),
        "answerability_accuracy_pct": bootstrap_mean_ci(ans, n_bootstrap=n_bootstrap, seed=seed + 1),
    }


def paired_report(
    per_a: dict[str, Any],
    per_b: dict[str, Any],
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired bootstrap on per-example differences (A minus B), same qid order required."""
    for key in ("per_example_em", "per_example_f1", "per_example_answerability_correct"):
        if len(per_a[key]) != len(per_b[key]):
            raise ValueError(f"Length mismatch for {key}; align examples before comparing.")

    em_a = np.asarray(per_a["per_example_em"], dtype=np.float64) * 100.0
    em_b = np.asarray(per_b["per_example_em"], dtype=np.float64) * 100.0
    f1_a = np.asarray(per_a["per_example_f1"], dtype=np.float64) * 100.0
    f1_b = np.asarray(per_b["per_example_f1"], dtype=np.float64) * 100.0
    ans_a = np.asarray(per_a["per_example_answerability_correct"], dtype=np.float64) * 100.0
    ans_b = np.asarray(per_b["per_example_answerability_correct"], dtype=np.float64) * 100.0

    return {
        "paired_diff_em_pct": bootstrap_paired_diff_ci(em_a, em_b, n_bootstrap=n_bootstrap, seed=seed),
        "paired_diff_f1_pct": bootstrap_paired_diff_ci(f1_a, f1_b, n_bootstrap=n_bootstrap, seed=seed + 1),
        "paired_diff_answerability_pct": bootstrap_paired_diff_ci(ans_a, ans_b, n_bootstrap=n_bootstrap, seed=seed + 2),
    }


def mean_qa_metrics_from_per_example(per_example: dict[str, Any]) -> dict[str, float]:
    """Aggregate EM/F1/answerability accuracy (percent) from per-example lists."""
    em = np.asarray(per_example["per_example_em"], dtype=np.float64)
    f1 = np.asarray(per_example["per_example_f1"], dtype=np.float64)
    ans = np.asarray(per_example["per_example_answerability_correct"], dtype=np.float64)
    return {
        "exact_match": float(np.mean(em) * 100.0),
        "f1": float(np.mean(f1) * 100.0),
        "answerability_accuracy": float(np.mean(ans) * 100.0),
    }


def make_table_row(label: str, metrics: dict[str, float], calibration: dict[str, float]) -> dict[str, Any]:
    """Single row combining QA metrics and calibration metrics for CSV/LaTeX."""
    return {"method": label, **metrics, **calibration}


def export_table(
    rows: list[dict[str, Any]],
    out_dir: str | Path,
    base_name: str = "paper_table",
) -> tuple[Path, Path]:
    """Write CSV and LaTeX side by side."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{base_name}.csv"
    tex_path = out_dir / f"{base_name}.tex"
    write_metrics_csv(rows, csv_path)
    preferred = [
        "method",
        "exact_match",
        "f1",
        "answerability_accuracy",
        "ece_answerability",
        "brier_answerability",
        "aurc_risk_coverage_em",
    ]
    cols = [c for c in preferred if any(c in r for r in rows)]
    if not cols:
        cols = sorted({k for r in rows for k in r.keys()})
    write_metrics_latex(rows, tex_path, columns=cols)
    return csv_path, tex_path
