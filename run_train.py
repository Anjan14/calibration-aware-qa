"""CLI entry point for training calibration-aware extractive QA."""

from __future__ import annotations

import argparse

from src.config import DatasetConfig, ModelConfig, TrainingConfig
from src.train import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train calibration-aware extractive QA model.")
    parser.add_argument("--train-file", type=str, default=None, help="Local SQuAD train JSON path.")
    parser.add_argument("--validation-file", type=str, default=None, help="Local SQuAD validation JSON path.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory for checkpoints/logs/plots.")
    parser.add_argument("--model-name", type=str, default="bert-base-uncased", help="Hugging Face encoder model name.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--train-batch-size", type=int, default=8, help="Train batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=16, help="Eval batch size.")
    parser.add_argument("--learning-rate", type=float, default=3e-5, help="Learning rate.")
    parser.add_argument("--cache-dir", type=str, default=None, help="Optional HF cache directory.")
    return parser.parse_args()


def main() -> None:
    """
    Usage examples:
    - Hosted SQuAD2.0:
      python run_train.py --output-dir outputs
    - Local files:
      python run_train.py --train-file /path/train-v2.0.json --validation-file /path/dev-v2.0.json
    """
    args = parse_args()
    use_hf = not (args.train_file and args.validation_file)
    dataset_cfg = DatasetConfig(
        use_hf_dataset=use_hf,
        train_file=args.train_file,
        validation_file=args.validation_file,
        cache_dir=args.cache_dir,
    )
    model_cfg = ModelConfig(model_name=args.model_name)
    train_cfg = TrainingConfig(
        output_dir=args.output_dir,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
    )
    run_training(dataset_cfg, model_cfg, train_cfg)


if __name__ == "__main__":
    main()

