"""Model definition: shared encoder + span and answerability heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoModel


@dataclass
class QAOutput:
    """Structured output for training/evaluation pipelines."""

    loss: torch.Tensor | None
    start_logits: torch.Tensor
    end_logits: torch.Tensor
    answerability_logits: torch.Tensor


class CalibrationAwareQAModel(nn.Module):
    """Shared-encoder extractive QA model with answerability classification."""

    def __init__(self, model_name: str = "bert-base-uncased", lambda_answerability: float = 0.5) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.qa_outputs = nn.Linear(hidden_size, 2)
        self.answerability_head = nn.Linear(hidden_size, 1)
        self.lambda_answerability = lambda_answerability

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        start_positions: torch.Tensor | None = None,
        end_positions: torch.Tensor | None = None,
        answerability_labels: torch.Tensor | None = None,
    ) -> QAOutput:
        # Encoder returns token-level hidden states. We use:
        # - all tokens for span prediction
        # - CLS token for answerability prediction.
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        sequence_output = outputs.last_hidden_state
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        cls_output = sequence_output[:, 0, :]
        answerability_logits = self.answerability_head(cls_output).squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None and answerability_labels is not None:
            ce_loss = nn.CrossEntropyLoss()
            bce_loss = nn.BCEWithLogitsLoss()

            start_loss = ce_loss(start_logits, start_positions)
            end_loss = ce_loss(end_logits, end_positions)
            ans_loss = bce_loss(answerability_logits, answerability_labels.float())
            total_loss = start_loss + end_loss + self.lambda_answerability * ans_loss

        return QAOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            answerability_logits=answerability_logits,
        )

