"""Batch-independent clean MSSF backbone for BioKORF experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import TensorDataset

from model import (
    Classifier,
    CrossProduction,
    EncoderAddition,
    EncoderConnection,
    GaussianParametrizer,
    Preprocess,
)


DRUG_COUNT = 757
SIDE_EFFECT_COUNT = 994
DRUG_VIEW_COUNT = 11
SIDE_VIEW_COUNT = 4
BRANCH_DIM = 128
PAIR_DIM = 384
DEFAULT_LATENT_DIM = 64
CLASS_COUNT = 5


@dataclass(frozen=True)
class MSSFCleanConfig:
    """Minimal configuration used by the clean MSSF backbone."""

    dropout: float = 0.5
    gp: int = DEFAULT_LATENT_DIM


def build_indexed_pair_dataset(
    drug_features: Tensor,
    side_features: Tensor,
    drug_index: Tensor,
    side_effect_index: Tensor,
    frequency_label: Tensor,
) -> TensorDataset:
    """Keep pair indices and labels attached to each feature-row sample."""
    tensors = (drug_features, side_features, drug_index, side_effect_index, frequency_label)
    lengths = [tensor.shape[0] for tensor in tensors]
    if len(set(lengths)) != 1:
        raise ValueError(f"All indexed dataset tensors must have equal length; found {lengths}")
    if drug_features.ndim != 2 or drug_features.shape[1] != DRUG_COUNT * DRUG_VIEW_COUNT:
        raise ValueError(
            f"drug_features must have shape [N, {DRUG_COUNT * DRUG_VIEW_COUNT}]"
        )
    if side_features.ndim != 2 or side_features.shape[1] != SIDE_EFFECT_COUNT * SIDE_VIEW_COUNT:
        raise ValueError(
            f"side_features must have shape [N, {SIDE_EFFECT_COUNT * SIDE_VIEW_COUNT}]"
        )
    for tensor, label, upper_bound in (
        (drug_index, "drug_index", DRUG_COUNT),
        (side_effect_index, "side_effect_index", SIDE_EFFECT_COUNT),
    ):
        if tensor.dtype != torch.long or tensor.ndim != 1:
            raise TypeError(f"{label} must be a one-dimensional LongTensor")
        if tensor.numel() and (torch.any(tensor < 0) or torch.any(tensor >= upper_bound)):
            raise ValueError(f"{label} contains values outside 0..{upper_bound - 1}")
    if frequency_label.ndim != 1:
        raise ValueError("frequency_label must be one-dimensional")
    return TensorDataset(*tensors)


class MSSFClean(nn.Module):
    """MSSF with direct three-branch concatenation and no batch-axis attention."""

    def __init__(self, config: MSSFCleanConfig | Any | None = None) -> None:
        super().__init__()
        config = config or MSSFCleanConfig()
        if not hasattr(config, "dropout") or not hasattr(config, "gp"):
            raise TypeError("config must provide dropout and gp attributes")
        if int(config.gp) <= 0:
            raise ValueError("Latent dimension gp must be positive")

        self.config = config
        self.feature_nums = DRUG_VIEW_COUNT * SIDE_VIEW_COUNT
        self.encoderConnection = EncoderConnection(
            drugs_inputdim=DRUG_COUNT * DRUG_VIEW_COUNT,
            sides_inputdim=SIDE_EFFECT_COUNT * SIDE_VIEW_COUNT,
            latent_dim=256,
            feature_dim=BRANCH_DIM,
            heads=4,
            args=config,
        )
        self.encoderAddition = EncoderAddition(
            drugs_inputdim=DRUG_COUNT,
            sides_inputdim=SIDE_EFFECT_COUNT,
            latent_dim=256,
            feature_dim=BRANCH_DIM,
            heads=4,
            args=config,
        )

        # Existing branch attention scores are [heads, B, B], so they are
        # removed from the clean copy to make every branch sample-independent.
        self.encoderConnection.attention = nn.Identity()
        self.encoderAddition.attention = nn.Identity()

        self.preprocess = Preprocess(
            drug_inputdim=DRUG_COUNT,
            side_inputdim=SIDE_EFFECT_COUNT,
            embeddim=BRANCH_DIM,
            args=config,
        )
        self.crossProduction = CrossProduction(
            cross_dim=BRANCH_DIM,
            feature_dim=BRANCH_DIM,
            input_channel=self.feature_nums,
        )
        self.gaussian_parametrizer = GaussianParametrizer(
            feature_dim=PAIR_DIM, latent_dim=int(config.gp)
        )
        self.classifier = Classifier(
            latent_dim=int(config.gp), classes=CLASS_COUNT, args=config
        )
        self.classification_loss = nn.CrossEntropyLoss()

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Preserve MSSF Bayesian sampling; evaluation deterministically uses mu."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            return torch.randn_like(std).mul(std).add_(mu)
        return mu

    def frequency_classification_loss(self, logits: Tensor, frequency_label: Tensor) -> Tensor:
        """Preserve the five-class CrossEntropyLoss label convention."""
        return self.classification_loss(logits, frequency_label.long().to(logits.device) - 1)

    def forward(
        self,
        drugs: Tensor,
        sides: Tensor,
        device: torch.device | str | None = None,
        return_debug: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]
    ]:
        target_device = torch.device(device) if device is not None else next(self.parameters()).device
        drugs = drugs.to(target_device)
        sides = sides.to(target_device)

        h_en_con, rec_con = self.encoderConnection(drugs, sides)
        h_en_add, rec_add = self.encoderAddition(drugs, sides)
        processed_drugs, processed_sides = self.preprocess(drugs, sides)
        h_cnn_im = self.crossProduction(processed_drugs, processed_sides)

        # Deliberately no top-level batch-axis attention or replacement attention.
        h_pair = torch.cat((h_en_con, h_en_add, h_cnn_im), dim=1)
        mu, logvar = self.gaussian_parametrizer(h_pair)
        latent = self.reparameterize(mu, logvar)
        logits = self.classifier(latent)

        outputs = (logits, rec_con, rec_add, mu, logvar)
        if not return_debug:
            return outputs
        debug = {
            "H_en_con": h_en_con,
            "H_en_add": h_en_add,
            "H_cnn_im": h_cnn_im,
            "H_pair": h_pair,
            "latent": latent,
            "logits": logits,
        }
        return (*outputs, debug)


# Compatibility alias for future clean-pipeline callers familiar with Mulmodel.
MulmodelClean = MSSFClean
