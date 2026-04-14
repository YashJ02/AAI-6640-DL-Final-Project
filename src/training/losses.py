"""Custom loss functions used by training loops."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional class weighting and label smoothing."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

        # Register alpha as buffer so it moves with model.to(device).
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)

        # Build smoothed one-hot targets for soft-label cross entropy.
        target_dist = F.one_hot(targets, num_classes=num_classes).float()
        if self.label_smoothing > 0.0:
            smooth = self.label_smoothing / num_classes
            target_dist = (1.0 - self.label_smoothing) * target_dist + smooth

        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        # p_t is probability assigned to target distribution.
        p_t = (probs * target_dist).sum(dim=-1).clamp(min=1e-8, max=1.0)
        ce_loss = -(target_dist * log_probs).sum(dim=-1)
        focal_factor = (1.0 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_factor = (self.alpha.unsqueeze(0).to(logits.device) * target_dist).sum(dim=-1)
        else:
            alpha_factor = torch.ones_like(ce_loss)

        loss = alpha_factor * focal_factor * ce_loss

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()
