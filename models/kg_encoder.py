"""Leakage-safe relational GCN encoder for the BioKORF biomedical graph."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.nn import RGCNConv


class BioKORFKGEncoder(nn.Module):
    """Encode graph nodes from node-type features and typed graph structure."""

    def __init__(
        self,
        num_node_types: int,
        num_relations: int,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_bases: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_node_types <= 0 or num_relations <= 0:
            raise ValueError("num_node_types and num_relations must be positive")
        if hidden_dim <= 0 or output_dim <= 0 or num_bases <= 0:
            raise ValueError("hidden_dim, output_dim, and num_bases must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.num_node_types = num_node_types
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_bases = num_bases

        self.node_type_embedding = nn.Embedding(num_node_types, hidden_dim)
        self.rgcn1 = RGCNConv(
            hidden_dim,
            hidden_dim,
            num_relations=num_relations,
            num_bases=num_bases,
            aggr="mean",
            root_weight=True,
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.rgcn2 = RGCNConv(
            hidden_dim,
            output_dim,
            num_relations=num_relations,
            num_bases=num_bases,
            aggr="mean",
            root_weight=True,
        )
        self.output_norm = nn.LayerNorm(output_dim)
        self.missing_drug_kg_embedding = nn.Parameter(torch.empty(output_dim))
        self.missing_side_kg_embedding = nn.Parameter(torch.empty(output_dim))
        self.reset_fallback_parameters()

    def reset_fallback_parameters(self) -> None:
        nn.init.normal_(self.missing_drug_kg_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.missing_side_kg_embedding, mean=0.0, std=0.02)

    def forward(
        self,
        node_type_index: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
    ) -> Tensor:
        """Return one output embedding per graph node."""
        self._validate_graph_inputs(node_type_index, edge_index, edge_type)
        features = self.node_type_embedding(node_type_index)
        features = self.rgcn1(features, edge_index, edge_type)
        features = self.activation(features)
        features = self.dropout(features)
        features = self.rgcn2(features, edge_index, edge_type)
        return self.output_norm(features)

    def _validate_graph_inputs(
        self, node_type_index: Tensor, edge_index: Tensor, edge_type: Tensor
    ) -> None:
        if node_type_index.dtype != torch.long or node_type_index.ndim != 1:
            raise TypeError("node_type_index must be a LongTensor with shape [num_nodes]")
        if edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise TypeError("edge_index must be a LongTensor with shape [2, num_edges]")
        if edge_type.dtype != torch.long or edge_type.ndim != 1:
            raise TypeError("edge_type must be a LongTensor with shape [num_edges]")
        if edge_index.shape[1] != edge_type.numel():
            raise ValueError("edge_index and edge_type must contain the same number of edges")
        if node_type_index.numel() == 0:
            raise ValueError("The graph must contain at least one node")
        if torch.any(node_type_index < 0) or torch.any(node_type_index >= self.num_node_types):
            raise ValueError("node_type_index contains a value outside the configured range")
        if edge_type.numel() and (
            torch.any(edge_type < 0) or torch.any(edge_type >= self.num_relations)
        ):
            raise ValueError("edge_type contains a value outside the configured range")
        if edge_index.numel() and (
            torch.any(edge_index < 0) or torch.any(edge_index >= node_type_index.numel())
        ):
            raise ValueError("edge_index contains an endpoint outside the node range")

    def _extract_anchor_embeddings(
        self,
        node_embeddings: Tensor,
        graph_node_index: Tensor,
        kg_available_mask: Tensor,
        missing_embedding: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if node_embeddings.ndim != 2 or node_embeddings.shape[1] != self.output_dim:
            raise ValueError(
                f"node_embeddings must have shape [num_nodes, {self.output_dim}]"
            )
        if graph_node_index.dtype != torch.long or graph_node_index.ndim != 1:
            raise TypeError("graph_node_index must be a one-dimensional LongTensor")
        if kg_available_mask.dtype != torch.bool or kg_available_mask.ndim != 1:
            raise TypeError("kg_available_mask must be a one-dimensional BoolTensor")
        if graph_node_index.numel() != kg_available_mask.numel():
            raise ValueError("Anchor indices and availability mask lengths must match")
        if graph_node_index.numel() and (
            torch.any(graph_node_index < 0)
            or torch.any(graph_node_index >= node_embeddings.shape[0])
        ):
            raise ValueError("An anchor graph index is outside the node embedding range")
        anchor_embeddings = node_embeddings.index_select(0, graph_node_index)
        replacement = missing_embedding.unsqueeze(0).expand_as(anchor_embeddings)
        anchor_embeddings = torch.where(
            kg_available_mask.unsqueeze(1), anchor_embeddings, replacement
        )
        return anchor_embeddings, kg_available_mask

    def extract_drug_anchor_embeddings(
        self,
        node_embeddings: Tensor,
        graph_node_index: Tensor,
        kg_available_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return self._extract_anchor_embeddings(
            node_embeddings,
            graph_node_index,
            kg_available_mask,
            self.missing_drug_kg_embedding,
        )

    def extract_side_anchor_embeddings(
        self,
        node_embeddings: Tensor,
        graph_node_index: Tensor,
        kg_available_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return self._extract_anchor_embeddings(
            node_embeddings,
            graph_node_index,
            kg_available_mask,
            self.missing_side_kg_embedding,
        )

    def rgcn_parameter_count(self) -> int:
        """Return the number of trainable parameters in the two R-GCN layers."""
        return sum(
            parameter.numel()
            for module in (self.rgcn1, self.rgcn2)
            for parameter in module.parameters()
            if parameter.requires_grad
        )
