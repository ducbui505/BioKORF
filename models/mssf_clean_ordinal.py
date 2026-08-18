"""Clean MSSF backbone with a cumulative ordinal-logistic output head."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from models.mssf_clean import MSSFClean, MSSFCleanConfig
from models.ordinal_head import OrdinalCumulativeHead, OrdinalCumulativeLoss


class BioKORFCleanOrdinal(MSSFClean):
    """Replace only the clean model's categorical classifier/loss."""

    def __init__(self, config: MSSFCleanConfig | Any | None = None) -> None:
        super().__init__(config)
        latent_dim = int(self.config.gp)
        del self.classifier
        del self.classification_loss
        self.ordinal_head = OrdinalCumulativeHead(latent_dim)
        # The inspected BioKORF pipeline stores frequency labels as 1..5.
        self.ordinal_loss = OrdinalCumulativeLoss(label_base=1)

    def frequency_classification_loss(
        self, ordinal_logits: Tensor, frequency_label: Tensor
    ) -> Tensor:
        return self.ordinal_loss(ordinal_logits, frequency_label)

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

        ordinal_logits, severity_score, thresholds = self.ordinal_head(latent)
        cumulative_probabilities, class_probabilities = self.ordinal_head.probabilities(
            ordinal_logits
        )
        predicted_class = class_probabilities.argmax(dim=1)

        outputs = (ordinal_logits, rec_con, rec_add, mu, logvar)
        if not return_debug:
            return outputs
        debug = {
            "H_en_con": h_en_con,
            "H_en_add": h_en_add,
            "H_cnn_im": h_cnn_im,
            "H_pair": h_pair,
            "latent": latent,
            "severity_score": severity_score,
            "ordered_thresholds": thresholds,
            "ordinal_logits": ordinal_logits,
            "cumulative_probabilities": cumulative_probabilities,
            "class_probabilities": class_probabilities,
            # Existing evaluation metrics internally use zero-based classes.
            "predicted_class": predicted_class,
        }
        return (*outputs, debug)


MSSFCleanOrdinal = BioKORFCleanOrdinal
