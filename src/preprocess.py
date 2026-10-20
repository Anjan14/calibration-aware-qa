"""Preprocessing logic for extractive QA with overflow windows."""

from __future__ import annotations

from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from .config import DatasetConfig


def build_tokenizer(model_name: str, cache_dir: str | None = None) -> PreTrainedTokenizerFast:
    """Create a fast tokenizer required for offset mapping."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=cache_dir)
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise TypeError("A fast tokenizer is required for QA preprocessing.")
    return tokenizer


def prepare_train_features(examples: dict[str, list[Any]], tokenizer: PreTrainedTokenizerFast, cfg: DatasetConfig) -> dict[str, list[Any]]:
    """
    Convert raw examples into training features with start/end labels.

    Key behavior:
    - Long contexts are split into multiple features using doc stride.
    - Gold answer spans are mapped from character offsets to token positions.
    - Unanswerable examples and non-containing overflow chunks use CLS as null label.
    - Answerability label remains tied to the original sample.
    """
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation=cfg.truncation,
        max_length=cfg.max_length,
        stride=cfg.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=cfg.padding,
    )
    sample_map = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    start_positions: list[int] = []
    end_positions: list[int] = []
    answerability_labels: list[int] = []

    for feature_idx, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][feature_idx]
        cls_idx = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized.sequence_ids(feature_idx)
        sample_idx = sample_map[feature_idx]
        answers = examples["answers"][sample_idx]
        is_impossible = bool(examples.get("is_impossible", [False] * len(sample_map))[sample_idx])

        # is_impossible==True means no-answer in SQuAD2.0. If no gold text is present,
        # we use CLS as both start/end token labels.
        if is_impossible or len(answers["answer_start"]) == 0:
            start_positions.append(cls_idx)
            end_positions.append(cls_idx)
            answerability_labels.append(0)
            continue

        answerability_labels.append(1)
        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        # Locate context token range for this feature. sequence_id==1 corresponds to context.
        token_start = 0
        while token_start < len(sequence_ids) and sequence_ids[token_start] != 1:
            token_start += 1
        token_end = len(sequence_ids) - 1
        while token_end >= 0 and sequence_ids[token_end] != 1:
            token_end -= 1

        # If answer is not fully inside this overflow chunk, assign CLS labels.
        if token_start >= len(offsets) or token_end < 0 or offsets[token_start][0] > start_char or offsets[token_end][1] < end_char:
            start_positions.append(cls_idx)
            end_positions.append(cls_idx)
            continue

        # Move token_start/token_end to exact answer boundaries.
        while token_start <= token_end and offsets[token_start][0] <= start_char:
            token_start += 1
        start_positions.append(token_start - 1)

        while token_end >= 0 and offsets[token_end][1] >= end_char:
            token_end -= 1
        end_positions.append(token_end + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    tokenized["answerability_labels"] = answerability_labels
    return tokenized


def prepare_validation_features(examples: dict[str, list[Any]], tokenizer: PreTrainedTokenizerFast, cfg: DatasetConfig) -> dict[str, list[Any]]:
    """
    Convert validation examples into model-ready features.

    We preserve `example_id` and clean offset mapping so non-context tokens are None.
    This simplifies safe text extraction during post-processing.
    """
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation=cfg.truncation,
        max_length=cfg.max_length,
        stride=cfg.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=cfg.padding,
    )
    sample_map = tokenized.pop("overflow_to_sample_mapping")
    example_ids: list[str] = []

    for feature_idx in range(len(tokenized["input_ids"])):
        sample_idx = sample_map[feature_idx]
        example_ids.append(examples["id"][sample_idx])
        sequence_ids = tokenized.sequence_ids(feature_idx)
        offsets = tokenized["offset_mapping"][feature_idx]
        tokenized["offset_mapping"][feature_idx] = [
            offset if sequence_ids[token_idx] == 1 else None for token_idx, offset in enumerate(offsets)
        ]

    tokenized["example_id"] = example_ids
    return tokenized

