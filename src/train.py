"""Manual training pipeline for calibration-aware extractive QA."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup

from .baselines import evaluate_baselines
from .calibrate import sweep_thresholds
from .config import DatasetConfig, ModelConfig, TrainingConfig
from .data import load_qa_datasets
from .evaluate import compute_per_example_scores, compute_qa_metrics, postprocess_predictions
from .model import CalibrationAwareQAModel
from .preprocess import build_tokenizer, prepare_train_features, prepare_validation_features
from .utils import build_logger, ensure_dir, save_json, set_seed


def _build_references(examples: list[dict[str, Any]]) -> tuple[dict[str, list[str]], dict[str, int]]:
    refs: dict[str, list[str]] = {}
    answerable: dict[str, int] = {}
    for ex in examples:
        texts = ex["answers"]["text"]
        refs[ex["id"]] = texts if texts else [""]
        answerable[ex["id"]] = 1 if len(texts) > 0 else 0
    return refs, answerable


def _plot_series(x: list[int], y: list[float], title: str, ylabel: str, out_path: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(x, y, marker="o")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _collect_logits(model: CalibrationAwareQAModel, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_start, all_end, all_ans = [], [], []
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k in {"input_ids", "attention_mask", "token_type_ids"}}
            outputs = model(**inputs)
            all_start.append(outputs.start_logits.cpu().numpy())
            all_end.append(outputs.end_logits.cpu().numpy())
            all_ans.append(outputs.answerability_logits.cpu().numpy())
    return np.concatenate(all_start, axis=0), np.concatenate(all_end, axis=0), np.concatenate(all_ans, axis=0)


def run_training(dataset_cfg: DatasetConfig, model_cfg: ModelConfig, train_cfg: TrainingConfig) -> dict[str, Any]:
    """Train model and save checkpoints/metrics/predictions/plots."""
    out = ensure_dir(train_cfg.output_dir)
    logs_dir = ensure_dir(out / "logs")
    ckpt_dir = ensure_dir(out / "checkpoints")
    pred_dir = ensure_dir(out / "predictions")
    plots_dir = ensure_dir(out / "plots")

    logger = build_logger("qa_train", logs_dir / "train.log")
    set_seed(train_cfg.seed)
    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")

    logger.info("Loading tokenizer and datasets...")
    tokenizer = build_tokenizer(model_cfg.model_name, dataset_cfg.cache_dir)
    datasets = load_qa_datasets(dataset_cfg)

    train_features = datasets["train"].map(
        lambda x: prepare_train_features(x, tokenizer, dataset_cfg),
        batched=True,
        remove_columns=datasets["train"].column_names,
    )
    val_features = datasets["validation"].map(
        lambda x: prepare_validation_features(x, tokenizer, dataset_cfg),
        batched=True,
        remove_columns=datasets["validation"].column_names,
    )

    train_features.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "start_positions", "end_positions", "answerability_labels"])
    val_model_features = val_features.remove_columns(["example_id", "offset_mapping"])
    val_model_features.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids"])

    train_loader = DataLoader(train_features, batch_size=train_cfg.train_batch_size, shuffle=True, collate_fn=default_data_collator)
    val_loader = DataLoader(val_model_features, batch_size=train_cfg.eval_batch_size, shuffle=False, collate_fn=default_data_collator)

    model = CalibrationAwareQAModel(model_name=model_cfg.model_name, lambda_answerability=model_cfg.lambda_answerability).to(device)
    optimizer = AdamW(model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    total_steps = max(1, len(train_loader) * train_cfg.epochs)
    warmup_steps = int(total_steps * train_cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    val_examples = [dict(x) for x in datasets["validation"]]
    val_feature_list = [dict(x) for x in val_features]
    references, answerability = _build_references(val_examples)

    baseline_metrics = evaluate_baselines(val_examples)
    save_json(pred_dir / "baseline_metrics.json", baseline_metrics)
    logger.info("Baseline metrics saved.")

    loss_curve, em_curve, f1_curve = [], [], []
    summary: dict[str, Any] = {
        "dataset_config": asdict(dataset_cfg),
        "model_config": asdict(model_cfg),
        "training_config": asdict(train_cfg),
        "epochs": [],
    }

    thresholds = np.linspace(train_cfg.threshold_min, train_cfg.threshold_max, train_cfg.threshold_steps)
    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            if loss is None:
                raise RuntimeError("Loss should not be None during training.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running_loss += float(loss.item())
            if step % 100 == 0:
                logger.info("Epoch %d Step %d | Loss %.4f", epoch, step, running_loss / step)

        epoch_loss = running_loss / max(1, len(train_loader))
        loss_curve.append(epoch_loss)
        logger.info("Epoch %d finished. Avg train loss: %.4f", epoch, epoch_loss)

        start_logits, end_logits, ans_logits = _collect_logits(model, val_loader, device)
        raw_preds, raw_meta = postprocess_predictions(
            val_examples,
            val_feature_list,
            start_logits,
            end_logits,
            ans_logits,
            threshold=0.0,
            max_answer_length=dataset_cfg.max_answer_length,
            use_calibration=False,
        )
        raw_metrics = compute_qa_metrics(raw_preds, references, answerability)

        score_list, null_list, prob_list, gold_list = [], [], [], []
        for ex in val_examples:
            m = raw_meta[ex["id"]]
            score_list.append(float(m["best_span_score"]))
            null_list.append(float(m["null_score"]))
            prob_list.append(float(m["answerability_prob"]))
            gold_list.append(int(answerability[ex["id"]]))

        best_threshold, sweep_results = sweep_thresholds(thresholds, score_list, null_list, prob_list, gold_list)
        cal_preds, cal_meta = postprocess_predictions(
            val_examples,
            val_feature_list,
            start_logits,
            end_logits,
            ans_logits,
            threshold=best_threshold,
            max_answer_length=dataset_cfg.max_answer_length,
            use_calibration=True,
        )
        cal_metrics = compute_qa_metrics(cal_preds, references, answerability)
        per_ex = compute_per_example_scores(val_examples, cal_preds, references, answerability, cal_meta)
        calibration_snapshot = per_ex["calibration_metrics"]
        em_curve.append(cal_metrics["exact_match"])
        f1_curve.append(cal_metrics["f1"])

        save_json(pred_dir / f"epoch_{epoch}_predictions.json", cal_preds)
        save_json(pred_dir / f"epoch_{epoch}_metadata.json", cal_meta)
        save_json(pred_dir / f"epoch_{epoch}_threshold_sweep.json", sweep_results)
        save_json(pred_dir / f"epoch_{epoch}_per_example.json", per_ex)

        ckpt_path = ckpt_dir / f"epoch_{epoch}"
        ensure_dir(ckpt_path)
        model.encoder.config.save_pretrained(ckpt_path)
        torch.save(model.state_dict(), ckpt_path / "model.pt")
        tokenizer.save_pretrained(ckpt_path)

        epoch_entry = {
            "epoch": epoch,
            "train_loss": epoch_loss,
            "raw_metrics": raw_metrics,
            "calibrated_metrics": cal_metrics,
            "calibration_metrics": calibration_snapshot,
            "best_threshold": best_threshold,
        }
        summary["epochs"].append(epoch_entry)
        logger.info("Epoch %d metrics: %s", epoch, epoch_entry)

    _plot_series(list(range(1, train_cfg.epochs + 1)), loss_curve, "Training Loss by Epoch", "Loss", plots_dir / "loss_curve.png")
    _plot_series(list(range(1, train_cfg.epochs + 1)), em_curve, "Validation EM by Epoch", "EM", plots_dir / "val_em_curve.png")
    _plot_series(list(range(1, train_cfg.epochs + 1)), f1_curve, "Validation F1 by Epoch", "F1", plots_dir / "val_f1_curve.png")

    save_json(out / "training_summary.json", summary)
    logger.info("Training complete. Summary saved to %s", out / "training_summary.json")
    return summary

