"""Sample-local view-aware encoders for BioKORF similarity sources."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


EMBEDDING_DIM = 128
ATTENTION_HEADS = 4


class ResidualViewBlock(nn.Module):
    """One transformer-style block whose sequence axis is the view axis."""

    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            EMBEDDING_DIM,
            ATTENTION_HEADS,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(EMBEDDING_DIM)
        self.ffn = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, EMBEDDING_DIM),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(EMBEDDING_DIM)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        attended, weights = self.self_attention(
            tokens,
            tokens,
            tokens,
            need_weights=True,
            average_attn_weights=False,
        )
        after_attention = self.attention_norm(tokens + attended)
        output = self.ffn_norm(after_attention + self.ffn(after_attention))
        return output, weights


class ViewAttentionPool(nn.Module):
    """Pool view tokens with one learned query and sample-specific weights."""

    def __init__(self) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.empty(EMBEDDING_DIM))
        nn.init.normal_(self.query, mean=0.0, std=EMBEDDING_DIM**-0.5)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        scores = torch.einsum("bvd,d->bv", tokens, self.query) / math.sqrt(
            EMBEDDING_DIM
        )
        weights = torch.softmax(scores, dim=1)
        pooled = torch.einsum("bv,bvd->bd", weights, tokens)
        return pooled, weights


class ViewEncoder(nn.Module):
    """Independent view projections followed by sample-local self-attention."""

    def __init__(self, view_count: int, view_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.view_count = int(view_count)
        self.view_dim = int(view_dim)
        self.projections = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.view_dim, EMBEDDING_DIM),
                nn.LayerNorm(EMBEDDING_DIM),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in range(self.view_count)
        )
        self.view_embedding = nn.Parameter(
            torch.empty(self.view_count, EMBEDDING_DIM)
        )
        nn.init.normal_(self.view_embedding, mean=0.0, std=EMBEDDING_DIM**-0.5)
        self.block = ResidualViewBlock(dropout)
        self.pool = ViewAttentionPool()

    def forward(
        self, flattened_views: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        expected = self.view_count * self.view_dim
        if flattened_views.ndim != 2 or flattened_views.shape[1] != expected:
            raise ValueError(f"Expected flattened views with shape [B, {expected}]")
        views = flattened_views.reshape(-1, self.view_count, self.view_dim)
        projected = torch.stack(
            [
                projection(views[:, index, :])
                for index, projection in enumerate(self.projections)
            ],
            dim=1,
        )
        tokens_before_attention = projected + self.view_embedding.unsqueeze(0)
        tokens_after_attention, attention_weights = self.block(
            tokens_before_attention
        )
        pooled, pooling_weights = self.pool(tokens_after_attention)
        return (
            tokens_before_attention,
            tokens_after_attention,
            attention_weights,
            pooling_weights,
            pooled,
        )


class DrugViewEncoder(ViewEncoder):
    def __init__(self) -> None:
        super().__init__(view_count=11, view_dim=757, dropout=0.2)


class SideEffectViewEncoder(ViewEncoder):
    def __init__(self) -> None:
        super().__init__(view_count=4, view_dim=994, dropout=0.2)
