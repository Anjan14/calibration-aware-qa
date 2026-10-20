"""CLI: multi-seed training runs and aggregated summary for publication."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.config import DatasetConfig, ModelConfig, TrainingConfig
from src.train import run_training
from src.utils import load_json, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run training for multiple seeds; aggregate last-epoch metrics.")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Random seeds (e.g. 42 43 44).")
    p.add_argument("--output-dir", type=str, default="outputs/multi_seed", help="Base directory; each run uses output-dir/seed_<s>.")
    p.add_argument("--train-file", type=str, default=None)
    p.add_argument("--validation-file", type=str, default=None)
    p.add_argument("--model-name", type=str, default="bert-base-uncased")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--train-batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--cache-dir", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(args.output_dir)
    base.mkdir(parents=True, exist_ok=True)
    use_hf = not (args.train_file and args.validation_file)

    runs: list[dict] = []
    for seed in args.seeds:
        run_dir = base / f"seed_{seed}"
        dataset_cfg = DatasetConfig(
            use_hf_dataset=use_hf,
            train_file=args.train_file,
            validation_file=args.validation_file,
            cache_dir=args.cache_dir,
        )
        model_cfg = ModelConfig(model_name=args.model_name)
        train_cfg = TrainingConfig(
            output_dir=str(run_dir),
            seed=seed,
            epochs=args.epochs,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
        )
        run_training(dataset_cfg, model_cfg, train_cfg)
        summary_path = run_dir / "training_summary.json"
        summary = load_json(summary_path)
        epochs = summary.get("epochs", [])
        last = epochs[-1] if epochs else {}
        runs.append(
            {
                "seed": seed,
                "output_dir": str(run_dir),
                "last_epoch": last.get("epoch"),
                "calibrated_metrics": last.get("calibrated_metrics"),
                "raw_metrics": last.get("raw_metrics"),
                "calibration_metrics": last.get("calibration_metrics"),
                "best_threshold": last.get("best_threshold"),
            }
        )

    # Aggregate mean/std across seeds for key scalars
    keys = ["exact_match", "f1", "answerability_accuracy"]
    agg: dict = {"runs": runs}
    for k in keys:
        vals = [r["calibrated_metrics"][k] for r in runs if r.get("calibrated_metrics")]
        if vals:
            arr = np.array(vals, dtype=np.float64)
            agg[f"calibrated_{k}_mean"] = float(arr.mean())
            agg[f"calibrated_{k}_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

    cal_keys = ["ece_answerability", "brier_answerability", "aurc_risk_coverage_em"]
    for ck in cal_keys:
        vals = [r["calibration_metrics"][ck] for r in runs if r.get("calibration_metrics") and ck in r["calibration_metrics"]]
        if vals:
            arr = np.array(vals, dtype=np.float64)
            agg[f"{ck}_mean"] = float(arr.mean())
            agg[f"{ck}_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

    save_json(base / "multi_seed_summary.json", agg)
    print(f"Wrote {base / 'multi_seed_summary.json'}")


if __name__ == "__main__":
    main()
