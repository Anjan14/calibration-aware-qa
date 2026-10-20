"""Dataset loading utilities for hosted and local SQuAD-format data."""

from __future__ import annotations

from pathlib import Path

from datasets import DatasetDict, load_dataset

from .config import DatasetConfig


def load_qa_datasets(cfg: DatasetConfig) -> DatasetDict:
    """
    Load QA datasets from Hugging Face or local SQuAD-format JSON.

    Expected local JSON format is standard SQuAD:
    {
      "data": [
        {
          "title": "...",
          "paragraphs": [
            {
              "context": "...",
              "qas": [
                {
                  "id": "...",
                  "question": "...",
                  "answers": {"text": [...], "answer_start": [...]},
                  "is_impossible": false
                }
              ]
            }
          ]
        }
      ]
    }
    """
    if cfg.use_hf_dataset:
        return load_dataset(cfg.dataset_name, cache_dir=cfg.cache_dir)

    if not cfg.train_file or not cfg.validation_file:
        raise ValueError("For local files, both train_file and validation_file must be provided.")

    train_path = Path(cfg.train_file)
    valid_path = Path(cfg.validation_file)
    if not train_path.exists() or not valid_path.exists():
        raise FileNotFoundError("Local dataset paths do not exist. Please update train/validation file paths.")

    # The `squad_v2` loader can parse standard SQuAD JSON when local files are provided.
    return load_dataset(
        "squad_v2",
        data_files={"train": str(train_path), "validation": str(valid_path)},
        cache_dir=cfg.cache_dir,
    )

