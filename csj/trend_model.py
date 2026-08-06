from __future__ import annotations

import torch
from torch import nn


class KronosTrendWrapper(nn.Module):
    """Training-only wrapper that leaves the predictor's public API unchanged."""

    def __init__(self, predictor: nn.Module, endpoint_count: int = 3) -> None:
        super().__init__()
        if endpoint_count != 3:
            raise ValueError("The V2 trend head must predict day1/day2/day3")
        if not hasattr(predictor, "d_model"):
            raise ValueError("Predictor must expose d_model")
        self.predictor = predictor
        self.trend_head = nn.Linear(int(predictor.d_model), endpoint_count)

    def forward(self, *args: object, **kwargs: object) -> object:
        return self.predictor(*args, **kwargs)

    def direction_logits(
        self,
        context_s1_ids: torch.Tensor,
        context_s2_ids: torch.Tensor,
        context_stamp: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return endpoint logits from context tokens only.

        There is deliberately no target-token argument. Callers must tokenize and
        slice the context before invoking this method.
        """

        if (
            context_s1_ids.ndim != 2
            or context_s2_ids.shape != context_s1_ids.shape
        ):
            raise ValueError(
                "Context token tensors must have matching [batch, time] shapes"
            )
        if (
            context_stamp is not None
            and context_stamp.shape[:2] != context_s1_ids.shape
        ):
            raise ValueError("Context timestamps must align with context tokens")
        if padding_mask is not None and padding_mask.shape != context_s1_ids.shape:
            raise ValueError("padding_mask must align with context tokens")

        _, hidden = self.predictor.decode_s1(
            context_s1_ids,
            context_s2_ids,
            context_stamp,
            padding_mask,
        )
        if padding_mask is None:
            final_hidden = hidden[:, -1, :]
        else:
            valid = padding_mask == 0
            positions = torch.arange(
                valid.shape[1], device=valid.device, dtype=torch.long
            ).expand_as(valid)
            final_positions = positions.masked_fill(~valid, -1).max(dim=1).values
            if torch.any(final_positions < 0):
                raise ValueError(
                    "Every direction sample needs at least one context token"
                )
            batch_positions = torch.arange(hidden.shape[0], device=hidden.device)
            final_hidden = hidden[batch_positions, final_positions]
        return self.trend_head(final_hidden)
