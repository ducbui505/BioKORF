"""MSSF view-aware controlled replacement model."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from models.mssf_clean import MSSFClean, MSSFCleanConfig
from models.view_aware_encoder import DrugViewEncoder, SideEffectViewEncoder


class BioKORFViewAware(MSSFClean):
    """Replace CLEAN EN-con/EN-add with independent entity view encoders."""

    def __init__(self, config: MSSFCleanConfig | Any | None = None) -> None:
        super().__init__(config)
        # These branches are deliberately absent rather than computed and ignored.
        del self.encoderConnection
        del self.encoderAddition
        self.drug_view_encoder = DrugViewEncoder()
        self.side_view_encoder = SideEffectViewEncoder()

    def forward(
        self,
        drugs: Tensor,
        sides: Tensor,
        device: torch.device | str | None = None,
        return_debug: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor] | tuple[
        Tensor, Tensor, Tensor, dict[str, Tensor]
    ]:
        target_device = (
            torch.device(device) if device is not None else next(self.parameters()).device
        )
        drugs = drugs.to(target_device)
        sides = sides.to(target_device)
        (
            drug_before,
            drug_after,
            drug_attention,
            drug_pooling,
            h_drug_view,
        ) = self.drug_view_encoder(drugs)
        (
            side_before,
            side_after,
            side_attention,
            side_pooling,
            h_side_view,
        ) = self.side_view_encoder(sides)

        # Preserve the CLEAN CNN-im computation exactly.
        processed_drugs, processed_sides = self.preprocess(drugs, sides)
        h_cnn_im = self.crossProduction(processed_drugs, processed_sides)
        h_pair_view = torch.cat((h_drug_view, h_side_view, h_cnn_im), dim=1)
        mu, logvar = self.gaussian_parametrizer(h_pair_view)
        latent = self.reparameterize(mu, logvar)
        logits = self.classifier(latent)

        outputs = (logits, mu, logvar)
        if not return_debug:
            return outputs
        debug = {
            "drug_tokens_before_attention": drug_before,
            "drug_tokens_after_attention": drug_after,
            "drug_view_attention_weights": drug_attention,
            "drug_view_pooling_weights": drug_pooling,
            "H_drug_view": h_drug_view,
            "side_tokens_before_attention": side_before,
            "side_tokens_after_attention": side_after,
            "side_view_attention_weights": side_attention,
            "side_view_pooling_weights": side_pooling,
            "H_side_view": h_side_view,
            "H_cnn_im": h_cnn_im,
            "H_pair_view": h_pair_view,
            "latent": latent,
            "logits": logits,
        }
        return (*outputs, debug)


MSSFViewAware = BioKORFViewAware
