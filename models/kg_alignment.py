"""Frozen Drug-KG targets and unique-drug cosine alignment."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


DRUG_COUNT = 757
KG_DIM = 128
DEFAULT_KG_ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "data_processed"
    / "kg_features"
    / "biokorf_kg_embeddings.pt"
)


class DrugKnowledgeAlignment(nn.Module):
    """Align a shared prediction representation to frozen Drug-KG targets."""

    def __init__(
        self,
        input_dim: int,
        artifact_path: str | Path = DEFAULT_KG_ARTIFACT,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        path = Path(artifact_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Pretrained KG embedding artifact not found: {path}")
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        required = {"drug_embeddings", "drug_available_mask", "drug_matrix_indices"}
        missing = required.difference(artifact)
        if missing:
            raise KeyError(f"KG artifact is missing drug fields: {sorted(missing)}")
        embeddings = artifact["drug_embeddings"].detach().to(torch.float32).contiguous()
        mask = artifact["drug_available_mask"].detach().to(torch.bool).contiguous()
        indices = artifact["drug_matrix_indices"].detach().to(torch.long)
        if tuple(embeddings.shape) != (DRUG_COUNT, KG_DIM):
            raise ValueError(f"Expected drug_embeddings [{DRUG_COUNT}, {KG_DIM}]")
        if tuple(mask.shape) != (DRUG_COUNT,):
            raise ValueError(f"Expected drug_available_mask [{DRUG_COUNT}]")
        if not torch.equal(indices, torch.arange(DRUG_COUNT)):
            raise ValueError("Drug KG ordering is not exactly 0..756")
        if not torch.isfinite(embeddings).all():
            raise ValueError("Drug KG embeddings contain non-finite values")
        self.artifact_path = path
        self.projection = nn.Sequential(nn.Linear(int(input_dim), KG_DIM), nn.LayerNorm(KG_DIM))
        self.register_buffer("drug_embeddings", embeddings, persistent=True)
        self.register_buffer("drug_available_mask", mask, persistent=True)

    @staticmethod
    def _validate_indices(drug_index: Tensor) -> None:
        if drug_index.dtype != torch.long or drug_index.ndim != 1:
            raise TypeError("drug_index must be a one-dimensional LongTensor")
        if drug_index.numel() and (
            torch.any(drug_index < 0) or torch.any(drug_index >= DRUG_COUNT)
        ):
            raise ValueError("drug_index contains an index outside 0..756")

    def project(self, h_drug: Tensor) -> Tensor:
        if h_drug.ndim != 2:
            raise ValueError("H_drug must be two-dimensional")
        return self.projection(h_drug)

    def forward(
        self, h_drug: Tensor, drug_index: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        self._validate_indices(drug_index)
        if h_drug.shape[0] != drug_index.shape[0]:
            raise ValueError("H_drug and drug_index must have equal batch size")
        device = self.drug_embeddings.device
        drug_index = drug_index.to(device)
        projected = self.project(h_drug.to(device))
        available = self.drug_available_mask.index_select(0, drug_index)
        available_indices = drug_index[available]
        if available_indices.numel() == 0:
            differentiable_zero = projected.sum() * 0.0
            return (
                projected,
                differentiable_zero,
                differentiable_zero.detach(),
                available,
                available_indices,
            )

        unique_indices, inverse = torch.unique(
            available_indices, sorted=True, return_inverse=True
        )
        available_projected = projected[available]
        grouped = torch.zeros(
            (unique_indices.numel(), projected.shape[1]),
            dtype=projected.dtype,
            device=device,
        )
        grouped.index_add_(0, inverse, available_projected)
        counts = torch.bincount(inverse, minlength=unique_indices.numel()).to(projected.dtype)
        grouped = grouped / counts.unsqueeze(1)
        targets = self.drug_embeddings.index_select(0, unique_indices)
        projected_norm = F.normalize(grouped, dim=-1)
        target_norm = F.normalize(targets, dim=-1)
        cosine = F.cosine_similarity(projected_norm, target_norm, dim=-1)
        alignment_loss = (1.0 - cosine).mean()
        return projected, alignment_loss, cosine.mean(), available, unique_indices

