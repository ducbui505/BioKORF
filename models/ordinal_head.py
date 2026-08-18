"""Cumulative ordinal-logistic components for five ordered classes."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


CLASS_COUNT = 5
THRESHOLD_COUNT = CLASS_COUNT - 1


def ordered_class_targets(labels: Tensor, label_base: int = 1) -> Tensor:
    """Convert class labels to four cumulative targets ``I(y > k)``.

    BioKORF datasets use labels 1..5.  ``label_base=0`` is supported
    explicitly for callers whose labels have already been converted to 0..4;
    ambiguous mini-batches are never guessed from their observed values.
    """
    if label_base not in (0, 1):
        raise ValueError("label_base must be 0 or 1")
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if torch.is_floating_point(labels) and labels.numel():
        if not torch.all(labels == labels.round()):
            raise ValueError("labels must contain integer class values")
    class_values = labels.long() + (1 - label_base)
    if class_values.numel() and (
        torch.any(class_values < 1) or torch.any(class_values > CLASS_COUNT)
    ):
        expected = "1..5" if label_base == 1 else "0..4"
        raise ValueError(f"labels must use the configured {expected} convention")
    cut_points = torch.arange(1, CLASS_COUNT, device=labels.device)
    return (class_values.unsqueeze(1) > cut_points.unsqueeze(0)).to(torch.float32)


class OrdinalCumulativeHead(nn.Module):
    """Learn one severity score and four strictly ordered thresholds."""

    def __init__(self, input_dim: int = 64) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.input_dim = int(input_dim)
        self.severity = nn.Linear(self.input_dim, 1)
        self.first_threshold = nn.Parameter(torch.zeros(1))
        self.raw_threshold_increments = nn.Parameter(torch.zeros(THRESHOLD_COUNT - 1))

    def ordered_thresholds(self) -> Tensor:
        increments = F.softplus(self.raw_threshold_increments) + torch.finfo(
            self.raw_threshold_increments.dtype
        ).eps
        later = self.first_threshold + torch.cumsum(increments, dim=0)
        return torch.cat((self.first_threshold, later), dim=0)

    def forward(self, latent: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if latent.ndim != 2 or latent.shape[1] != self.input_dim:
            raise ValueError(f"latent must have shape [B, {self.input_dim}]")
        severity_score = self.severity(latent)
        thresholds = self.ordered_thresholds()
        ordinal_logits = severity_score - thresholds.unsqueeze(0)
        return ordinal_logits, severity_score, thresholds

    @staticmethod
    def cumulative_to_class_probabilities(cumulative_probabilities: Tensor) -> Tensor:
        if cumulative_probabilities.ndim != 2 or cumulative_probabilities.shape[1] != 4:
            raise ValueError("cumulative_probabilities must have shape [B, 4]")
        q1, q2, q3, q4 = cumulative_probabilities.unbind(dim=1)
        return torch.stack((1.0 - q1, q1 - q2, q2 - q3, q3 - q4, q4), dim=1)

    @staticmethod
    def validate_probabilities(
        cumulative_probabilities: Tensor,
        class_probabilities: Tensor,
        tolerance: float = 1e-6,
    ) -> bool:
        if not torch.isfinite(cumulative_probabilities).all() or not torch.isfinite(
            class_probabilities
        ).all():
            return False
        monotonic = torch.all(
            cumulative_probabilities[:, :-1] + tolerance
            >= cumulative_probabilities[:, 1:]
        )
        nonnegative = torch.all(class_probabilities >= -tolerance)
        normalized = torch.allclose(
            class_probabilities.sum(dim=1),
            torch.ones(class_probabilities.shape[0], device=class_probabilities.device),
            atol=tolerance,
            rtol=0.0,
        )
        return bool(monotonic and nonnegative and normalized)

    def probabilities(self, ordinal_logits: Tensor) -> tuple[Tensor, Tensor]:
        if ordinal_logits.ndim != 2 or ordinal_logits.shape[1] != THRESHOLD_COUNT:
            raise ValueError("ordinal_logits must have shape [B, 4]")
        cumulative = torch.sigmoid(ordinal_logits)
        class_probabilities = self.cumulative_to_class_probabilities(cumulative)
        return cumulative, class_probabilities


class OrdinalCumulativeLoss(nn.Module):
    """Unweighted cumulative binary cross-entropy for five ordered classes."""

    def __init__(self, label_base: int = 1) -> None:
        super().__init__()
        if label_base not in (0, 1):
            raise ValueError("label_base must be 0 or 1")
        self.label_base = label_base
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, ordinal_logits: Tensor, labels: Tensor) -> Tensor:
        if ordinal_logits.ndim != 2 or ordinal_logits.shape[1] != THRESHOLD_COUNT:
            raise ValueError("ordinal_logits must have shape [B, 4]")
        targets = ordered_class_targets(labels.to(ordinal_logits.device), self.label_base)
        if targets.shape[0] != ordinal_logits.shape[0]:
            raise ValueError("ordinal_logits and labels must have equal batch size")
        return self.loss(ordinal_logits, targets.to(dtype=ordinal_logits.dtype))

