"""Clean MSSF backbone with a CORN conditional ordinal classifier."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from models.corn_ordinal_head import CORNOrdinalHead, CORNOrdinalLoss
from models.mssf_clean import MSSFClean, MSSFCleanConfig


class BioKORFCleanCORN(MSSFClean):
    """Replace only the clean categorical classifier and loss with CORN."""

    def __init__(
        self,
        config: MSSFCleanConfig | Any | None = None,
        pos_weights: Tensor | None = None,
    ) -> None:
        super().__init__(config)
        latent_dim = int(self.config.gp)
        del self.classifier
        del self.classification_loss
        self.corn_head = CORNOrdinalHead(latent_dim, hidden_dim=32, dropout=0.2)
        self.corn_loss = CORNOrdinalLoss(pos_weights)

    def frequency_classification_loss(
        self, conditional_logits: Tensor, frequency_label: Tensor
    ) -> Tensor:
        return self.corn_loss(conditional_logits, frequency_label)

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
        processed_drugs, processed_sides = self.preprocess(drugs, sides)
        h_cnn_im = self.crossProduction(processed_drugs, processed_sides)
        h_pair = torch.cat((h_en_con, h_en_add, h_cnn_im), dim=1)
        mu, logvar = self.gaussian_parametrizer(h_pair)
        latent = self.reparameterize(mu, logvar)
        conditional_logits = self.corn_head(latent)
        conditional, cumulative, class_probabilities = self.corn_head.probabilities(
            conditional_logits
        )
        predicted_class = self.corn_head.primary_decoder(class_probabilities)

        outputs = (conditional_logits, rec_con, rec_add, mu, logvar)
        if not return_debug:
            return outputs
        debug = {
            "H_en_con": h_en_con,
            "H_en_add": h_en_add,
            "H_cnn_im": h_cnn_im,
            "H_pair": h_pair,
            "latent": latent,
            "conditional_logits": conditional_logits,
            "conditional_probabilities": conditional,
            "cumulative_probabilities": cumulative,
            "class_probabilities": class_probabilities,
            "predicted_class": predicted_class,
        }
        return (*outputs, debug)


MSSFCleanCORN = BioKORFCleanCORN
