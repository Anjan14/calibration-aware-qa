"""CLI entry point for evaluating a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import default_data_collator

from src.calibrate import sweep_thresholds
from src.config import DatasetConfig
from src.data import load_qa_datasets
from src.evaluate import compute_per_example_scores, compute_qa_metrics, postprocess_predictions
from src.model import CalibrationAwareQAModel
from src.paper_report import bootstrap_metrics_from_per_example
from src.preprocess import build_tokenizer, prepare_validation_features
from src.utils import save_json


def _load_state_dict(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint for calibration-aware extractive QA.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint directory or model.pt path.")
    parser.add_argument("--train-file", type=str, default=None, help="Optional local train JSON.")
    parser.add_argument("--validation-file", type=str, default=None, help="Optional local validation JSON.")
    parser.add_argument("--model-name", type=str, default="bert-base-uncased", help="Base encoder name.")
    parser.add_argument("--output-json", type=str, default="outputs/eval_results.json", help="Evaluation output path.")
    parser.add_argument("--cache-dir", type=str, default=None, help="HF cache directory.")
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10_000,
        help="Bootstrap resamples for CIs on per-example EM/F1/answerability (0 to skip).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for bootstrap.")
    return parser.parse_args()


def _refs(examples):
    refs, ans = {}, {}
    for ex in examples:
        txt = ex["answers"]["text"]
        refs[ex["id"]] = txt if txt else [""]
        ans[ex["id"]] = 1 if len(txt) > 0 else 0
    return refs, ans


def main() -> None:
    """
    Example:
    python run_eval.py --checkpoint outputs/checkpoints/epoch_3 --output-json outputs/final_eval.json
    """
    args = parse_args()
    use_hf = not (args.train_file and args.validation_file)
    dcfg = DatasetConfig(
        use_hf_dataset=use_hf,
        train_file=args.train_file,
        validation_file=args.validation_file,
        cache_dir=args.cache_dir,
    )
    datasets = load_qa_datasets(dcfg)
    tokenizer = build_tokenizer(args.model_name, args.cache_dir)

    val_features = datasets["validation"].map(
        lambda x: prepare_validation_features(x, tokenizer, dcfg),
        batched=True,
        remove_columns=datasets["validation"].column_names,
    )
    model_features = val_features.remove_columns(["example_id", "offset_mapping"])
    model_features.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids"])
    loader = DataLoader(model_features, batch_size=16, shuffle=False, collate_fn=default_data_collator)

    model = CalibrationAwareQAModel(model_name=args.model_name)
    ckpt_path = Path(args.checkpoint)
    model_file = ckpt_path / "model.pt" if ckpt_path.is_dir() else ckpt_path
    state = _load_state_dict(model_file)
    model.load_state_dict(state)
    model.eval()

    all_start, all_end, all_ans = [], [], []
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
            )
            all_start.append(outputs.start_logits.numpy())
            all_end.append(outputs.end_logits.numpy())
            all_ans.append(outputs.answerability_logits.numpy())

    start_logits = np.concatenate(all_start, axis=0)
    end_logits = np.concatenate(all_end, axis=0)
    ans_logits = np.concatenate(all_ans, axis=0)

    examples = [dict(x) for x in datasets["validation"]]
    features = [dict(x) for x in val_features]
    references, answerable = _refs(examples)

    raw_preds, raw_meta = postprocess_predictions(
        examples, features, start_logits, end_logits, ans_logits, threshold=0.0, use_calibration=False
    )
    raw_metrics = compute_qa_metrics(raw_preds, references, answerable)

    scores, nulls, probs, gold = [], [], [], []
    for ex in examples:
        m = raw_meta[ex["id"]]
        scores.append(float(m["best_span_score"]))
        nulls.append(float(m["null_score"]))
        probs.append(float(m["answerability_prob"]))
        gold.append(answerable[ex["id"]])
    best_threshold, sweep = sweep_thresholds(np.linspace(-10.0, 10.0, 41), scores, nulls, probs, gold)
    cal_preds, cal_meta = postprocess_predictions(
        examples,
        features,
        start_logits,
        end_logits,
        ans_logits,
        threshold=best_threshold,
        use_calibration=True,
    )
    cal_metrics = compute_qa_metrics(cal_preds, references, answerable)
    per_ex = compute_per_example_scores(examples, cal_preds, references, answerable, cal_meta)
    calib_only = per_ex.get("calibration_metrics", {})

    bootstrap_block = {}
    if args.bootstrap_samples and args.bootstrap_samples > 0:
        bootstrap_block = bootstrap_metrics_from_per_example(
            per_ex, n_bootstrap=args.bootstrap_samples, seed=args.seed
        )

    out_path = Path(args.output_json)
    result = {
        "raw_metrics": raw_metrics,
        "calibrated_metrics": cal_metrics,
        "calibration_metrics": calib_only,
        "best_threshold": best_threshold,
        "threshold_sweep": sweep,
        "bootstrap_calibrated": bootstrap_block,
    }
    save_json(out_path, result)
    stem = out_path.stem
    save_json(out_path.parent / f"{stem}_per_example.json", per_ex)
    save_json(out_path.parent / f"{stem}_predictions.json", cal_preds)
    save_json(out_path.parent / f"{stem}_metadata.json", cal_meta)


if __name__ == "__main__":
    main()
