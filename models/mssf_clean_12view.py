"""MSSF-clean with one appended Drug view and unchanged Side-effect views."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from model import (
    Classifier,
    CrossProduction,
    EncoderAddition,
    EncoderConnection,
    GaussianParametrizer,
    Preprocess,
)
from models.mssf_clean import MSSFCleanConfig


DRUG_COUNT = 757
SIDE_EFFECT_COUNT = 994
DRUG_VIEW_COUNT = 12
SIDE_VIEW_COUNT = 4
BRANCH_DIM = 128
PAIR_DIM = 384
DEFAULT_LATENT_DIM = 64
CLASS_COUNT = 5


class EncoderAddition12(EncoderAddition):
    """The CLEAN addition encoder with twelve Drug chunks instead of eleven."""

    def forward(
        self, drug_features: Tensor, side_features: Tensor
    ) -> tuple[Tensor, Tensor]:
        drug_views = drug_features.chunk(DRUG_VIEW_COUNT, dim=1)
        side_views = side_features.chunk(SIDE_VIEW_COUNT, dim=1)
        if len(drug_views) != DRUG_VIEW_COUNT or any(
            view.shape[1] != DRUG_COUNT for view in drug_views
        ):
            raise ValueError("Drug input does not contain twelve 757-column views")
        if len(side_views) != SIDE_VIEW_COUNT or any(
            view.shape[1] != SIDE_EFFECT_COUNT for view in side_views
        ):
            raise ValueError("Side-effect input does not contain four 994-column views")
        drugs = drug_views[0]
        for view in drug_views[1:]:
            drugs = drugs + view
        sides = side_views[0]
        for view in side_views[1:]:
            sides = sides + view
        add_features = torch.cat((drugs, sides), dim=1)
        encoded = self.l2(self.attention(self.l1(add_features)))
        return encoded, self.l3(encoded)


class Preprocess12(Preprocess):
    """The CLEAN per-view preprocessor extended by one identical Drug branch."""

    def __init__(self, drug_inputdim: int, side_inputdim: int, embeddim: int, args: Any):
        super().__init__(drug_inputdim, side_inputdim, embeddim, args)
        self.drug12_pre = nn.Sequential(
            nn.Linear(self.drug_inputdim, self.embdeddim),
            nn.BatchNorm1d(self.embdeddim),
            self.reluDrop,
        )

    def forward(
        self, drug_features: Tensor, side_features: Tensor
    ) -> tuple[list[Tensor], list[Tensor]]:
        drug_views = drug_features.chunk(DRUG_VIEW_COUNT, dim=1)
        side_views = side_features.chunk(SIDE_VIEW_COUNT, dim=1)
        if len(drug_views) != DRUG_VIEW_COUNT or any(
            view.shape[1] != DRUG_COUNT for view in drug_views
        ):
            raise ValueError("Drug input does not contain twelve 757-column views")
        if len(side_views) != SIDE_VIEW_COUNT or any(
            view.shape[1] != SIDE_EFFECT_COUNT for view in side_views
        ):
            raise ValueError("Side-effect input does not contain four 994-column views")
        drug_layers = [getattr(self, f"drug{index}_pre") for index in range(1, 13)]
        side_layers = [getattr(self, f"side{index}_pre") for index in range(1, 5)]
        return (
            [layer(view) for layer, view in zip(drug_layers, drug_views, strict=True)],
            [layer(view) for layer, view in zip(side_layers, side_views, strict=True)],
        )


class MSSFClean12View(nn.Module):
    """Exact CLEAN computation with a twelfth appended Drug similarity view."""

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
        self.encoderAddition = EncoderAddition12(
            drugs_inputdim=DRUG_COUNT,
            sides_inputdim=SIDE_EFFECT_COUNT,
            latent_dim=256,
            feature_dim=BRANCH_DIM,
            heads=4,
            args=config,
        )
        self.encoderConnection.attention = nn.Identity()
        self.encoderAddition.attention = nn.Identity()
        self.preprocess = Preprocess12(
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
        if self.training:
            std = torch.exp(0.5 * logvar)
            return torch.randn_like(std).mul(std).add_(mu)
        return mu

    def frequency_classification_loss(
        self, logits: Tensor, frequency_label: Tensor
    ) -> Tensor:
        return self.classification_loss(
            logits, frequency_label.long().to(logits.device) - 1
        )

    def forward(
        self,
        drugs: Tensor,
        sides: Tensor,
        device: torch.device | str | None = None,
        return_debug: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]
    ]:
        target_device = (
            torch.device(device) if device is not None else next(self.parameters()).device
        )
        drugs = drugs.to(target_device)
        sides = sides.to(target_device)
        if drugs.ndim != 2 or drugs.shape[1] != DRUG_COUNT * DRUG_VIEW_COUNT:
            raise ValueError("Drug input must have shape [B, 9084]")
        if sides.ndim != 2 or sides.shape[1] != SIDE_EFFECT_COUNT * SIDE_VIEW_COUNT:
            raise ValueError("Side-effect input must have shape [B, 3976]")

        h_en_con, rec_con = self.encoderConnection(drugs, sides)
        h_en_add, rec_add = self.encoderAddition(drugs, sides)
        processed_drugs, processed_sides = self.preprocess(drugs, sides)
        h_cnn_im = self.crossProduction(processed_drugs, processed_sides)
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


MulmodelClean12View = MSSFClean12View
