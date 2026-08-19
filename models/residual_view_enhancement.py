"""Zero-initialized gated residual enhancement for the EN-add branch."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


FEATURE_DIM = 128


class ResidualViewEnhancement(nn.Module):
    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()
        self.residual_projection = nn.Sequential(
            nn.Linear(FEATURE_DIM * 2, FEATURE_DIM),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(FEATURE_DIM, FEATURE_DIM),
        )
        final_linear = self.residual_projection[-1]
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

        self.gate = nn.Linear(FEATURE_DIM * 2, FEATURE_DIM)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, math.log(0.1 / 0.9))

    def forward(
        self, h_en_add: Tensor, h_drug_view: Tensor, h_side_view: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        expected = h_en_add.shape
        if h_en_add.ndim != 2 or h_en_add.shape[1] != FEATURE_DIM:
            raise ValueError("H_en_add must have shape [B, 128]")
        if h_drug_view.shape != expected or h_side_view.shape != expected:
            raise ValueError("View-aware entity representations must match H_en_add")
        view_pair = torch.cat((h_drug_view, h_side_view), dim=1)
        v_residual = self.residual_projection(view_pair)
        residual_gate = torch.sigmoid(
            self.gate(torch.cat((h_en_add, v_residual), dim=1))
        )
        correction = residual_gate * v_residual
        enhanced = h_en_add + correction
        return enhanced, v_residual, residual_gate, correction

