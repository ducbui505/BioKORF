"""Self-supervised link-reconstruction components for BioKORF KG pretraining."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class BioKORFDistMultDecoder(nn.Module):
    """Relation-aware DistMult decoder over original biological relations."""

    def __init__(self, num_relations: int, embedding_dim: int = 128) -> None:
        super().__init__()
        if num_relations <= 0 or embedding_dim <= 0:
            raise ValueError("num_relations and embedding_dim must be positive")
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.relation_embedding.weight)

    def forward(
        self,
        node_embeddings: Tensor,
        source_index: Tensor,
        relation_index: Tensor,
        target_index: Tensor,
    ) -> Tensor:
        if node_embeddings.ndim != 2 or node_embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"node_embeddings must have shape [num_nodes, {self.embedding_dim}]"
            )
        for name, tensor in (
            ("source_index", source_index),
            ("relation_index", relation_index),
            ("target_index", target_index),
        ):
            if tensor.dtype != torch.long or tensor.ndim != 1:
                raise TypeError(f"{name} must be a one-dimensional LongTensor")
        if not (source_index.numel() == relation_index.numel() == target_index.numel()):
            raise ValueError("Source, relation, and target index lengths must match")
        if source_index.numel() and (
            torch.any(source_index < 0)
            or torch.any(target_index < 0)
            or torch.any(source_index >= node_embeddings.shape[0])
            or torch.any(target_index >= node_embeddings.shape[0])
        ):
            raise ValueError("A triple contains a node index outside the graph")
        if relation_index.numel() and (
            torch.any(relation_index < 0)
            or torch.any(relation_index >= self.num_relations)
        ):
            raise ValueError("A triple uses a non-original or unknown relation index")
        source = node_embeddings.index_select(0, source_index)
        relation = self.relation_embedding(relation_index)
        target = node_embeddings.index_select(0, target_index)
        return (source * relation * target).sum(dim=-1)
