"""Configuration objects for the QA project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetConfig:
    """Dataset loading and tokenization configuration."""

    dataset_name: str = "squad_v2"
    use_hf_dataset: bool = True
    train_file: str | None = None
    validation_file: str | None = None
    cache_dir: str | None = None

    max_length: int = 384
    doc_stride: int = 128
    max_answer_length: int = 30
    padding: str = "max_length"
    truncation: str = "only_second"


@dataclass
class ModelConfig:
    """Model architecture and optimization objective configuration."""

    model_name: str = "bert-base-uncased"
    lambda_answerability: float = 0.5


@dataclass
class TrainingConfig:
    """Training and checkpointing configuration."""

    output_dir: str = "outputs"
    seed: int = 42
    learning_rate: float = 3e-5
    train_batch_size: int = 8
    eval_batch_size: int = 16
    epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    num_workers: int = 0
    device: str = "cuda"

    threshold_min: float = -10.0
    threshold_max: float = 10.0
    threshold_steps: int = 41

    def output_path(self) -> Path:
        return Path(self.output_dir)

