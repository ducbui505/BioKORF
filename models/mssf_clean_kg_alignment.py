"""MSSF-clean with Drug-KG used only as auxiliary alignment supervision."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from models.kg_alignment import DEFAULT_KG_ARTIFACT, DrugKnowledgeAlignment
from models.mssf_clean import BRANCH_DIM, MSSFClean, MSSFCleanConfig


class BioKORFCleanDrugKGAlignment(MSSFClean):
    """Preserve CLEAN logits while exposing a shared drug representation."""

    def __init__(
        self,
        config: MSSFCleanConfig | Any | None = None,
        kg_artifact_path: str | Path = DEFAULT_KG_ARTIFACT,
    ) -> None:
        super().__init__(config)
        self.drug_kg_alignment = DrugKnowledgeAlignment(BRANCH_DIM, kg_artifact_path)

    def forward(
        self,
        drugs: Tensor,
        sides: Tensor,
        drug_index: Tensor | None = None,
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
        processed_drugs, processed_sides = self.preprocess(drugs, sides)
        # Each view is a trainable drug-only hidden state used directly by
        # crossProduction.  Their parameter-free mean is the shared H_drug.
        h_drug = torch.stack(processed_drugs, dim=0).mean(dim=0)
        h_cnn_im = self.crossProduction(processed_drugs, processed_sides)
        h_pair = torch.cat((h_en_con, h_en_add, h_cnn_im), dim=1)
        mu, logvar = self.gaussian_parametrizer(h_pair)
        latent = self.reparameterize(mu, logvar)
        logits = self.classifier(latent)

        if drug_index is None:
            projected = self.drug_kg_alignment.project(h_drug)
            alignment_loss = projected.sum() * 0.0
            mean_cosine = alignment_loss.detach()
            available_mask = torch.zeros(h_drug.shape[0], dtype=torch.bool, device=target_device)
            unique_indices = torch.empty(0, dtype=torch.long, device=target_device)
        else:
            projected, alignment_loss, mean_cosine, available_mask, unique_indices = (
                self.drug_kg_alignment(h_drug, drug_index.to(target_device))
            )

        outputs = (logits, rec_con, rec_add, mu, logvar)
        if not return_debug:
            return outputs
        debug = {
            "H_en_con": h_en_con,
            "H_en_add": h_en_add,
            "H_cnn_im": h_cnn_im,
            "H_drug": h_drug,
            "H_pair": h_pair,
            "latent": latent,
            "logits": logits,
            "projected_drug_representation": projected,
            "alignment_loss": alignment_loss,
            "mean_drug_kg_cosine": mean_cosine,
            "drug_kg_available_mask": available_mask,
            "unique_available_drug_indices": unique_indices,
        }
        return (*outputs, debug)


MSSFCleanDrugKGAlignment = BioKORFCleanDrugKGAlignment
