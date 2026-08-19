"""MSSF-clean with zero-initialized residual view enhancement of EN-add."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from models.mssf_clean import MSSFClean, MSSFCleanConfig
from models.residual_view_enhancement import ResidualViewEnhancement
from models.view_aware_encoder import DrugViewEncoder, SideEffectViewEncoder


class BioKORFResidualViewAware(MSSFClean):
    def __init__(self, config: MSSFCleanConfig | Any | None = None) -> None:
        super().__init__(config)
        self.drug_view_encoder = DrugViewEncoder()
        self.side_view_encoder = SideEffectViewEncoder()
        self.residual_view_enhancement = ResidualViewEnhancement(dropout=0.2)

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
        h_en_con, rec_con = self.encoderConnection(drugs, sides)
        h_en_add, rec_add = self.encoderAddition(drugs, sides)
        (
            _drug_before,
            _drug_after,
            drug_attention,
            drug_pooling,
            h_drug_view,
        ) = self.drug_view_encoder(drugs)
        (
            _side_before,
            _side_after,
            side_attention,
            side_pooling,
            h_side_view,
        ) = self.side_view_encoder(sides)
        (
            h_en_add_enhanced,
            v_residual,
            residual_gate,
            residual_correction,
        ) = self.residual_view_enhancement(h_en_add, h_drug_view, h_side_view)

        processed_drugs, processed_sides = self.preprocess(drugs, sides)
        h_cnn_im = self.crossProduction(processed_drugs, processed_sides)
        h_pair_residual = torch.cat(
            (h_en_con, h_en_add_enhanced, h_cnn_im), dim=1
        )
        mu, logvar = self.gaussian_parametrizer(h_pair_residual)
        latent = self.reparameterize(mu, logvar)
        logits = self.classifier(latent)

        outputs = (logits, rec_con, rec_add, mu, logvar)
        if not return_debug:
            return outputs
        debug = {
            "H_en_con": h_en_con,
            "H_en_add": h_en_add,
            "H_cnn_im": h_cnn_im,
            "H_drug_view": h_drug_view,
            "H_side_view": h_side_view,
            "drug_view_attention_weights": drug_attention,
            "side_view_attention_weights": side_attention,
            "drug_view_pooling_weights": drug_pooling,
            "side_view_pooling_weights": side_pooling,
            "V_residual": v_residual,
            "residual_gate": residual_gate,
            "residual_correction": residual_correction,
            "H_en_add_enhanced": h_en_add_enhanced,
            "H_pair_residual": h_pair_residual,
            "latent": latent,
            "logits": logits,
        }
        return (*outputs, debug)


MSSFResidualViewAware = BioKORFResidualViewAware
