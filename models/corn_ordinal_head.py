"""CORN-style conditional ordinal head and masked binary loss."""

from __future__ import annotations

import torch
from torch import Tensor, nn


CLASS_COUNT = 5
TASK_COUNT = CLASS_COUNT - 1


def corn_targets_and_masks(labels: Tensor) -> tuple[Tensor, Tensor]:
    """Build four conditional targets/masks from raw BioKORF labels 1..5."""
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if torch.is_floating_point(labels) and labels.numel() and not torch.all(
        labels == labels.round()
    ):
        raise ValueError("labels must contain integer values")
    labels = labels.long()
    if labels.numel() and (torch.any(labels < 1) or torch.any(labels > CLASS_COUNT)):
        raise ValueError("raw CORN labels must be in 1..5")
    tasks = torch.arange(TASK_COUNT, device=labels.device)
    masks = labels.unsqueeze(1) > tasks.unsqueeze(0)
    targets = labels.unsqueeze(1) > (tasks + 1).unsqueeze(0)
    return targets.to(torch.float32), masks


def training_pos_weights(labels: Tensor, maximum: float = 10.0) -> Tensor:
    """Compute task weights from training labels only."""
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    targets, masks = corn_targets_and_masks(labels)
    weights: list[Tensor] = []
    for task_index in range(TASK_COUNT):
        eligible_targets = targets[masks[:, task_index], task_index]
        positive_count = eligible_targets.sum()
        negative_count = eligible_targets.numel() - positive_count
        if positive_count <= 0 or negative_count <= 0:
            raise ValueError(
                f"CORN task {task_index + 1} needs positive and negative training samples"
            )
        weight = negative_count / positive_count
        weights.append(torch.clamp(weight, max=maximum))
    return torch.stack(weights).to(torch.float32)


class CORNOrdinalHead(nn.Module):
    """Shared nonlinear projection followed by four conditional classifiers."""

    def __init__(self, input_dim: int = 64, hidden_dim: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.input_dim = int(input_dim)
        self.shared = nn.Sequential(
            nn.Linear(self.input_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )
        self.heads = nn.ModuleList(nn.Linear(int(hidden_dim), 1) for _ in range(TASK_COUNT))

    def forward(self, latent: Tensor) -> Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.input_dim:
            raise ValueError(f"latent must have shape [B, {self.input_dim}]")
        shared = self.shared(latent)
        return torch.cat([head(shared) for head in self.heads], dim=1)

    @staticmethod
    def probabilities(conditional_logits: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if conditional_logits.ndim != 2 or conditional_logits.shape[1] != TASK_COUNT:
            raise ValueError("conditional_logits must have shape [B, 4]")
        conditional = torch.sigmoid(conditional_logits)
        cumulative = torch.cumprod(conditional, dim=1)
        q1, q2, q3, q4 = cumulative.unbind(dim=1)
        classes = torch.stack((1.0 - q1, q1 - q2, q2 - q3, q3 - q4, q4), dim=1)
        return conditional, cumulative, classes

    @staticmethod
    def primary_decoder(class_probabilities: Tensor) -> Tensor:
        return class_probabilities.argmax(dim=1)

    @staticmethod
    def threshold_count_decoder(cumulative_probabilities: Tensor) -> Tensor:
        """Return diagnostic semantic classes 1..5; not the primary decoder."""
        return 1 + (cumulative_probabilities >= 0.5).sum(dim=1)

    @staticmethod
    def validate_probabilities(
        cumulative_probabilities: Tensor,
        class_probabilities: Tensor,
        tolerance: float = 1e-6,
    ) -> bool:
        finite = torch.isfinite(cumulative_probabilities).all() and torch.isfinite(
            class_probabilities
        ).all()
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
        return bool(finite and monotonic and nonnegative and normalized)


class CORNOrdinalLoss(nn.Module):
    """Mean of valid task-wise weighted conditional BCE losses."""

    def __init__(self, pos_weights: Tensor | None = None) -> None:
        super().__init__()
        weights = torch.ones(TASK_COUNT) if pos_weights is None else pos_weights.detach().clone()
        if weights.shape != (TASK_COUNT,) or not torch.isfinite(weights).all():
            raise ValueError("pos_weights must be a finite tensor with shape [4]")
        if torch.any(weights <= 0):
            raise ValueError("all pos_weights must be positive")
        self.register_buffer("pos_weights", weights.to(torch.float32))

    def forward(self, conditional_logits: Tensor, labels: Tensor) -> Tensor:
        if conditional_logits.ndim != 2 or conditional_logits.shape[1] != TASK_COUNT:
            raise ValueError("conditional_logits must have shape [B, 4]")
        targets, masks = corn_targets_and_masks(labels.to(conditional_logits.device))
        losses: list[Tensor] = []
        for task_index in range(TASK_COUNT):
            eligible = masks[:, task_index]
            if eligible.any():
                loss = nn.functional.binary_cross_entropy_with_logits(
                    conditional_logits[eligible, task_index],
                    targets[eligible, task_index].to(conditional_logits.dtype),
                    pos_weight=self.pos_weights[task_index].to(conditional_logits.dtype),
                )
                losses.append(loss)
        if not losses:
            raise ValueError("batch has no valid CORN tasks")
        return torch.stack(losses).mean()

