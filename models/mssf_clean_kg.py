"""MSSF-clean with frozen pretrained KG gated residual fusion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from models.kg_fusion import FrozenKGFeatures, GatedKGFusion
from models.mssf_clean import MSSFClean, MSSFCleanConfig


DEFAULT_KG_ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "data_processed"
    / "kg_features"
    / "biokorf_kg_embeddings.pt"
)


class BioKORFCleanKG(MSSFClean):
    """Clean three-branch MSSF backbone followed by frozen KG fusion."""

    def __init__(
        self,
        config: MSSFCleanConfig | Any | None = None,
        kg_artifact_path: str | Path = DEFAULT_KG_ARTIFACT,
    ) -> None:
        super().__init__(config=config)
        self.kg_features = FrozenKGFeatures(kg_artifact_path)
        self.kg_fusion = GatedKGFusion(dropout=0.2)

    def forward(
        self,
        drugs: Tensor,
        sides: Tensor,
        drug_index: Tensor,
        side_effect_index: Tensor,
        device: torch.device | str | None = None,
        return_debug: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]
    ]:
        target_device = torch.device(device) if device is not None else next(self.parameters()).device
        drugs = drugs.to(target_device)
        sides = sides.to(target_device)
        drug_index = drug_index.to(target_device)
        side_effect_index = side_effect_index.to(target_device)

        h_en_con, rec_con = self.encoderConnection(drugs, sides)
        h_en_add, rec_add = self.encoderAddition(drugs, sides)
        processed_drugs, processed_sides = self.preprocess(drugs, sides)
        h_cnn_im = self.crossProduction(processed_drugs, processed_sides)
        h_pair = torch.cat((h_en_con, h_en_add, h_cnn_im), dim=1)

        z_drug, z_side, drug_mask, side_mask, kg_input = self.kg_features(
            drug_index, side_effect_index
        )
        h_fused, kg_projected, kg_gate = self.kg_fusion(
            h_pair, kg_input, return_debug=True
        )
        mu, logvar = self.gaussian_parametrizer(h_fused)
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
            "Z_drug_KG": z_drug,
            "Z_side_KG": z_side,
            "drug_kg_mask": drug_mask,
            "side_kg_mask": side_mask,
            "KG_input": kg_input,
            "KG_projected": kg_projected,
            "KG_gate": kg_gate,
            "H_fused": h_fused,
            "latent": latent,
            "logits": logits,
        }
        return (*outputs, debug)
