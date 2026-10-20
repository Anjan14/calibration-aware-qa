"""Generate slide-ready plots from outputs_collab artifacts.

Usage:
    python generate_presentation_plots.py
    python generate_presentation_plots.py --outputs-dir outputs_collab --save-dir outputs_collab/presentation_plots
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate presentation graphs from training/eval artifacts."
    )
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default="outputs_collab",
        help="Directory containing training_summary.json, final_eval.json, and logs/train.log",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Where to save generated plots (default: <outputs-dir>/presentation_plots)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Image DPI for exported plots.",
    )
    return parser.parse_args()


def _resolve_outputs_dir(raw_outputs_dir: str) -> Path:
    """Resolve outputs directory robustly across common launch directories."""
    candidate = Path(raw_outputs_dir).expanduser()
    if candidate.exists():
        return candidate

    script_dir = Path(__file__).resolve().parent
    fallback_candidates = [
        script_dir / raw_outputs_dir,
        script_dir.parent / raw_outputs_dir,
        script_dir.parent.parent / raw_outputs_dir,
    ]
    for path in fallback_candidates:
        if path.exists():
            return path

    return candidate


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _plot_epoch_metrics(summary: dict[str, Any], save_dir: Path, dpi: int) -> None:
    epochs = summary.get("epochs", [])
    if not epochs:
        raise ValueError("No epoch records found in training_summary.json")

    x = [int(e["epoch"]) for e in epochs]
    raw_em = [float(e["raw_metrics"]["exact_match"]) for e in epochs]
    cal_em = [float(e["calibrated_metrics"]["exact_match"]) for e in epochs]
    raw_f1 = [float(e["raw_metrics"]["f1"]) for e in epochs]
    cal_f1 = [float(e["calibrated_metrics"]["f1"]) for e in epochs]
    raw_ans = [float(e["raw_metrics"]["answerability_accuracy"]) for e in epochs]
    cal_ans = [float(e["calibrated_metrics"]["answerability_accuracy"]) for e in epochs]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    configs = [
        ("Exact Match (%)", raw_em, cal_em),
        ("F1 (%)", raw_f1, cal_f1),
        ("Answerability Accuracy (%)", raw_ans, cal_ans),
    ]
    for ax, (title, raw_values, cal_values) in zip(axes, configs):
        ax.plot(x, raw_values, marker="o", linewidth=2, label="Raw")
        ax.plot(x, cal_values, marker="o", linewidth=2, label="Calibrated")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.grid(alpha=0.25)
        ax.legend()

    fig.suptitle("Raw vs Calibrated Metrics Across Epochs", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_dir / "01_epoch_raw_vs_calibrated_metrics.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_calibration_metrics(summary: dict[str, Any], save_dir: Path, dpi: int) -> None:
    epochs = summary.get("epochs", [])
    x = [int(e["epoch"]) for e in epochs]
    ece = [float(e["calibration_metrics"]["ece_answerability"]) for e in epochs]
    brier = [float(e["calibration_metrics"]["brier_answerability"]) for e in epochs]
    aurc = [float(e["calibration_metrics"]["aurc_risk_coverage_em"]) for e in epochs]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, y, label in zip(
        axes,
        [ece, brier, aurc],
        ["ECE (lower is better)", "Brier Score (lower is better)", "AURC (lower is better)"],
    ):
        ax.plot(x, y, marker="o", linewidth=2, color="#8e44ad")
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)

    fig.suptitle("Calibration Quality Trends", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_dir / "02_calibration_metrics_by_epoch.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_threshold_sweep(final_eval: dict[str, Any], save_dir: Path, dpi: int) -> None:
    sweep = final_eval.get("threshold_sweep", [])
    if not sweep:
        return

    thresholds = [float(x["threshold"]) for x in sweep]
    ans_acc_pct = [100.0 * float(x["answerability_accuracy"]) for x in sweep]
    best_t = float(final_eval.get("best_threshold", 0.0))

    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.plot(thresholds, ans_acc_pct, linewidth=2.2, color="#1f77b4")
    ax.axvline(best_t, linestyle="--", color="#d62728", label=f"Best threshold = {best_t:.2f}")
    ax.set_title("Threshold Sweep on Answerability Accuracy")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Answerability Accuracy (%)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_dir / "03_threshold_sweep_answerability.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_final_comparison_with_ci(final_eval: dict[str, Any], save_dir: Path, dpi: int) -> None:
    raw = final_eval.get("raw_metrics", {})
    cal = final_eval.get("calibrated_metrics", {})
    boot = final_eval.get("bootstrap_calibrated", {})

    labels = ["EM", "F1", "Ans Acc"]
    raw_vals = [
        float(raw.get("exact_match", 0.0)),
        float(raw.get("f1", 0.0)),
        float(raw.get("answerability_accuracy", 0.0)),
    ]
    cal_vals = [
        float(cal.get("exact_match", 0.0)),
        float(cal.get("f1", 0.0)),
        float(cal.get("answerability_accuracy", 0.0)),
    ]

    ci_map = {
        "EM": boot.get("exact_match_pct", {}),
        "F1": boot.get("f1_pct", {}),
        "Ans Acc": boot.get("answerability_accuracy_pct", {}),
    }
    lower = []
    upper = []
    for lab, val in zip(labels, cal_vals):
        ci = ci_map.get(lab, {})
        lo = float(ci.get("ci_low", val))
        hi = float(ci.get("ci_high", val))
        lower.append(max(0.0, val - lo))
        upper.append(max(0.0, hi - val))

    x = range(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    raw_bars = ax.bar([i - w / 2 for i in x], raw_vals, width=w, label="Raw", color="#7f8c8d")
    cal_bars = ax.bar([i + w / 2 for i in x], cal_vals, width=w, label="Calibrated", color="#27ae60")
    ax.errorbar(
        [i + w / 2 for i in x],
        cal_vals,
        yerr=[lower, upper],
        fmt="none",
        ecolor="black",
        capsize=4,
        linewidth=1.2,
        label="Calibrated 95% CI",
    )

    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Score (%)")
    ax.set_title("Final Evaluation: Raw vs Calibrated (with Bootstrap CI)")
    ax.set_ylim(min(raw_vals + cal_vals) - 3, max(raw_vals + cal_vals) + 3)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    for b in list(raw_bars) + list(cal_bars):
        h = b.get_height()
        ax.annotate(f"{h:.2f}", (b.get_x() + b.get_width() / 2, h), ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_dir / "04_final_raw_vs_calibrated_with_ci.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _parse_log_time(line: str) -> dt.datetime | None:
    # Example: 2026-04-20 18:06:37,955 | INFO | ...
    stamp = line.split(" | ")[0].strip()
    try:
        return dt.datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S,%f")
    except ValueError:
        return None


def _plot_training_timeline(log_path: Path, save_dir: Path, dpi: int) -> None:
    if not log_path.exists():
        return

    lines = [ln for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return

    start_ts = _parse_log_time(lines[0])
    if start_ts is None:
        return

    epoch_finish_minutes: list[float] = []
    epoch_labels: list[str] = []
    for ln in lines:
        if "finished. Avg train loss" not in ln:
            continue
        ts = _parse_log_time(ln)
        if ts is None:
            continue
        minute = (ts - start_ts).total_seconds() / 60.0
        epoch_finish_minutes.append(minute)
        epoch_labels.append(f"Epoch {len(epoch_labels) + 1}")

    end_ts = _parse_log_time(lines[-1])
    total_minutes = (end_ts - start_ts).total_seconds() / 60.0 if end_ts else None
    if not epoch_finish_minutes:
        return

    fig, ax = plt.subplots(figsize=(9.5, 5))
    bars = ax.bar(epoch_labels, epoch_finish_minutes, color="#f39c12")
    ax.set_title("Cumulative Training Time by Epoch")
    ax.set_ylabel("Minutes since start")
    ax.grid(axis="y", alpha=0.25)

    for b, val in zip(bars, epoch_finish_minutes):
        ax.annotate(f"{val:.1f}m", (b.get_x() + b.get_width() / 2, val), ha="center", va="bottom", fontsize=9)

    if total_minutes is not None:
        ax.text(
            0.98,
            0.02,
            f"Total runtime: {total_minutes:.1f} min",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )

    fig.tight_layout()
    fig.savefig(save_dir / "05_training_timeline_minutes.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outputs_dir = _resolve_outputs_dir(args.outputs_dir)
    save_dir = Path(args.save_dir) if args.save_dir else outputs_dir / "presentation_plots"
    save_dir.mkdir(parents=True, exist_ok=True)

    training_summary = _read_json(outputs_dir / "training_summary.json")
    final_eval = _read_json(outputs_dir / "final_eval.json")

    _plot_epoch_metrics(training_summary, save_dir, args.dpi)
    _plot_calibration_metrics(training_summary, save_dir, args.dpi)
    _plot_threshold_sweep(final_eval, save_dir, args.dpi)
    _plot_final_comparison_with_ci(final_eval, save_dir, args.dpi)
    _plot_training_timeline(outputs_dir / "logs" / "train.log", save_dir, args.dpi)

    print(f"Generated presentation plots in: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
