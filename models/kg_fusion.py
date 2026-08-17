"""Frozen KG feature lookup and gated residual fusion for BioKORF."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn


DRUG_COUNT = 757
SIDE_EFFECT_COUNT = 994
KG_EMBEDDING_DIM = 128
KG_INPUT_DIM = 258
PAIR_DIM = 384


class FrozenKGFeatures(nn.Module):
    """Load ordered pretrained anchor features as non-trainable buffers."""

    def __init__(self, artifact_path: str | Path) -> None:
        super().__init__()
        self.artifact_path = Path(artifact_path).resolve()
        if not self.artifact_path.is_file():
            raise FileNotFoundError(f"Pretrained KG embedding artifact not found: {self.artifact_path}")
        artifact = torch.load(self.artifact_path, map_location="cpu", weights_only=False)
        required = {
            "drug_embeddings",
            "side_embeddings",
            "drug_available_mask",
            "side_available_mask",
            "drug_matrix_indices",
            "side_matrix_indices",
        }
        missing = required.difference(artifact)
        if missing:
            raise KeyError(f"KG artifact is missing keys: {sorted(missing)}")

        drug_embeddings = artifact["drug_embeddings"].detach().to(torch.float32).contiguous()
        side_embeddings = artifact["side_embeddings"].detach().to(torch.float32).contiguous()
        drug_mask = artifact["drug_available_mask"].detach().to(torch.bool).contiguous()
        side_mask = artifact["side_available_mask"].detach().to(torch.bool).contiguous()
        drug_indices = artifact["drug_matrix_indices"].detach().to(torch.long)
        side_indices = artifact["side_matrix_indices"].detach().to(torch.long)

        if tuple(drug_embeddings.shape) != (DRUG_COUNT, KG_EMBEDDING_DIM):
            raise ValueError(f"Unexpected drug KG shape: {tuple(drug_embeddings.shape)}")
        if tuple(side_embeddings.shape) != (SIDE_EFFECT_COUNT, KG_EMBEDDING_DIM):
            raise ValueError(f"Unexpected side-effect KG shape: {tuple(side_embeddings.shape)}")
        if tuple(drug_mask.shape) != (DRUG_COUNT,) or tuple(side_mask.shape) != (SIDE_EFFECT_COUNT,):
            raise ValueError("Unexpected KG availability-mask shape")
        if not torch.equal(drug_indices, torch.arange(DRUG_COUNT)):
            raise ValueError("Drug KG matrix ordering is not exactly 0..756")
        if not torch.equal(side_indices, torch.arange(SIDE_EFFECT_COUNT)):
            raise ValueError("Side-effect KG matrix ordering is not exactly 0..993")
        if not torch.isfinite(drug_embeddings).all() or not torch.isfinite(side_embeddings).all():
            raise ValueError("KG artifact contains non-finite embeddings")

        self.register_buffer("drug_embeddings", drug_embeddings, persistent=True)
        self.register_buffer("side_embeddings", side_embeddings, persistent=True)
        self.register_buffer("drug_available_mask", drug_mask, persistent=True)
        self.register_buffer("side_available_mask", side_mask, persistent=True)

    @staticmethod
    def _validate_indices(indices: Tensor, count: int, name: str) -> None:
        if indices.dtype != torch.long or indices.ndim != 1:
            raise TypeError(f"{name} must be a one-dimensional LongTensor")
        if indices.numel() and (torch.any(indices < 0) or torch.any(indices >= count)):
            raise ValueError(f"{name} contains an index outside 0..{count - 1}")

    def forward(
        self, drug_index: Tensor, side_effect_index: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        self._validate_indices(drug_index, DRUG_COUNT, "drug_index")
        self._validate_indices(side_effect_index, SIDE_EFFECT_COUNT, "side_effect_index")
        if drug_index.shape != side_effect_index.shape:
            raise ValueError("drug_index and side_effect_index must have matching shapes")
        device = self.drug_embeddings.device
        drug_index = drug_index.to(device)
        side_effect_index = side_effect_index.to(device)
        z_drug = self.drug_embeddings.index_select(0, drug_index)
        z_side = self.side_embeddings.index_select(0, side_effect_index)
        drug_mask = self.drug_available_mask.index_select(0, drug_index).unsqueeze(1)
        side_mask = self.side_available_mask.index_select(0, side_effect_index).unsqueeze(1)
        kg_input = torch.cat(
            (z_drug, z_side, drug_mask.to(z_drug.dtype), side_mask.to(z_drug.dtype)), dim=1
        )
        return z_drug, z_side, drug_mask, side_mask, kg_input


class GatedKGFusion(nn.Module):
    """Fuse a 258-d KG context into a 384-d MSSF pair representation."""

    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()
        self.kg_projection = nn.Sequential(
            nn.Linear(KG_INPUT_DIM, PAIR_DIM),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(PAIR_DIM * 2, PAIR_DIM),
            nn.Sigmoid(),
        )
        self.output_norm = nn.LayerNorm(PAIR_DIM)

    def forward(
        self, h_pair: Tensor, kg_input: Tensor, return_debug: bool = False
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        if h_pair.ndim != 2 or h_pair.shape[1] != PAIR_DIM:
            raise ValueError(f"H_pair must have shape [B, {PAIR_DIM}]")
        if kg_input.ndim != 2 or kg_input.shape != (h_pair.shape[0], KG_INPUT_DIM):
            raise ValueError(f"KG_input must have shape [B, {KG_INPUT_DIM}]")
        kg_projected = self.kg_projection(kg_input)
        kg_gate = self.gate(torch.cat((h_pair, kg_projected), dim=1))
        h_fused = self.output_norm(h_pair + kg_gate * kg_projected)
        if return_debug:
            return h_fused, kg_projected, kg_gate
        return h_fused
