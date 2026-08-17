"""No-training smoke test for frozen KG gated residual fusion."""

from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KG_ARTIFACT_PATH = (
    PROJECT_ROOT / "data_processed" / "kg_features" / "biokorf_kg_embeddings.pt"
)
REPORT_PATH = PROJECT_ROOT / "data_processed" / "architecture" / "kg_fusion_smoke_test.txt"
ORIGINAL_HASHES = {
    PROJECT_ROOT / "mssf.py": "4867fecd04beabb2d715b24073f82a46bd572c13294afa3565ddba99f963fdb1",
    PROJECT_ROOT / "model.py": "9c0d4bf17551a7d0f881a29e0f8e2727227f3561678064fec46f2848156a1e75",
    PROJECT_ROOT / "models" / "mssf_clean.py": "f2a0f68e062807cacc77540c14afd5bf0e66eb7571b76b99e095c4063b8dd6d2",
}
SEED = 42
BATCH_SIZE = 4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_index(mask: torch.Tensor, value: bool) -> int | None:
    matches = torch.nonzero(mask == value, as_tuple=False).flatten()
    return int(matches[0]) if matches.numel() else None


def main() -> None:
    for path in (*ORIGINAL_HASHES, KG_ARTIFACT_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required smoke-test input not found: {path}")
    hashes_before = {path: sha256(path) for path in ORIGINAL_HASHES}
    if hashes_before != ORIGINAL_HASHES:
        raise RuntimeError(f"Original-file hash mismatch before smoke test: {hashes_before}")
    kg_hash_before = sha256(KG_ARTIFACT_PATH)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    sys.path.insert(0, str(PROJECT_ROOT))
    from models.mssf_clean import (
        DRUG_COUNT,
        DRUG_VIEW_COUNT,
        SIDE_EFFECT_COUNT,
        SIDE_VIEW_COUNT,
        MSSFCleanConfig,
    )
    from models.mssf_clean_kg import BioKORFCleanKG

    raw_artifact = torch.load(KG_ARTIFACT_PATH, map_location="cpu", weights_only=False)
    drug_mask = raw_artifact["drug_available_mask"].to(torch.bool)
    side_mask = raw_artifact["side_available_mask"].to(torch.bool)
    drug_available = first_index(drug_mask, True)
    drug_missing = first_index(drug_mask, False)
    side_available = first_index(side_mask, True)
    side_missing = first_index(side_mask, False)
    requested = {
        "both drug and side have KG": (drug_available, side_available),
        "drug has KG; side does not": (drug_available, side_missing),
        "side has KG; drug does not": (drug_missing, side_available),
        "neither has KG": (drug_missing, side_missing),
    }
    unavailable_categories = [name for name, pair in requested.items() if None in pair]
    if unavailable_categories:
        valid_drug = drug_available if drug_available is not None else drug_missing
        valid_side = side_available if side_available is not None else side_missing
        if valid_drug is None or valid_side is None:
            raise RuntimeError("No valid drug/side KG rows are available")
        requested = {
            name: pair if None not in pair else (valid_drug, valid_side)
            for name, pair in requested.items()
        }

    selected_pairs = list(requested.values())
    batch_drug_index = torch.tensor([pair[0] for pair in selected_pairs], dtype=torch.long)
    batch_side_index = torch.tensor([pair[1] for pair in selected_pairs], dtype=torch.long)
    labels = torch.tensor([1, 2, 4, 5], dtype=torch.long)

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    shared_drug = torch.randn(1, DRUG_COUNT * DRUG_VIEW_COUNT, generator=generator)
    shared_side = torch.randn(1, SIDE_EFFECT_COUNT * SIDE_VIEW_COUNT, generator=generator)
    batch_a_drug = torch.cat(
        (shared_drug, torch.randn(3, DRUG_COUNT * DRUG_VIEW_COUNT, generator=generator)), dim=0
    )
    batch_a_side = torch.cat(
        (shared_side, torch.randn(3, SIDE_EFFECT_COUNT * SIDE_VIEW_COUNT, generator=generator)), dim=0
    )
    batch_b_drug = torch.cat(
        (shared_drug, torch.randn(3, DRUG_COUNT * DRUG_VIEW_COUNT, generator=generator)), dim=0
    )
    batch_b_side = torch.cat(
        (shared_side, torch.randn(3, SIDE_EFFECT_COUNT * SIDE_VIEW_COUNT, generator=generator)), dim=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BioKORFCleanKG(
        config=MSSFCleanConfig(dropout=0.5, gp=64),
        kg_artifact_path=KG_ARTIFACT_PATH,
    ).to(device).eval()

    with torch.inference_mode():
        logits_a, _, _, _, _, debug_a = model(
            batch_a_drug,
            batch_a_side,
            batch_drug_index,
            batch_side_index,
            device=device,
            return_debug=True,
        )
        _, _, _, _, _, debug_b = model(
            batch_b_drug,
            batch_b_side,
            batch_drug_index,
            batch_side_index,
            device=device,
            return_debug=True,
        )
        zero_kg_input = torch.zeros_like(debug_b["KG_input"])
        zero_h_fused, _, _ = model.kg_fusion(
            debug_b["H_pair"], zero_kg_input, return_debug=True
        )
        zero_mu, zero_logvar = model.gaussian_parametrizer(zero_h_fused)
        zero_latent = model.reparameterize(zero_mu, zero_logvar)
        zero_logits = model.classifier(zero_latent)
        loss = model.frequency_classification_loss(logits_a, labels)

    expected_shapes = {
        "H_en_con": (BATCH_SIZE, 128),
        "H_en_add": (BATCH_SIZE, 128),
        "H_cnn_im": (BATCH_SIZE, 128),
        "H_pair": (BATCH_SIZE, 384),
        "Z_drug_KG": (BATCH_SIZE, 128),
        "Z_side_KG": (BATCH_SIZE, 128),
        "drug_kg_mask": (BATCH_SIZE, 1),
        "side_kg_mask": (BATCH_SIZE, 1),
        "KG_input": (BATCH_SIZE, 258),
        "KG_projected": (BATCH_SIZE, 384),
        "KG_gate": (BATCH_SIZE, 384),
        "H_fused": (BATCH_SIZE, 384),
        "latent": (BATCH_SIZE, 64),
        "logits": (BATCH_SIZE, 5),
    }
    shape_check = all(tuple(debug_a[name].shape) == shape for name, shape in expected_shapes.items())

    kg_buffer_names = {name for name, _ in model.kg_features.named_buffers()}
    kg_parameter_names = {name for name, _ in model.kg_features.named_parameters()}
    frozen_kg_check = bool(
        kg_buffer_names
        == {"drug_embeddings", "side_embeddings", "drug_available_mask", "side_available_mask"}
        and not kg_parameter_names
        and all(not tensor.requires_grad for _, tensor in model.kg_features.named_buffers())
    )
    fusion_trainable_check = bool(
        list(model.kg_fusion.parameters())
        and all(parameter.requires_grad for parameter in model.kg_fusion.parameters())
    )
    gate = debug_a["KG_gate"]
    gate_check = bool(
        torch.isfinite(gate).all() and torch.all(gate >= 0.0) and torch.all(gate <= 1.0)
    )
    finite_check = bool(
        all(torch.isfinite(tensor).all() for tensor in debug_a.values())
        and torch.isfinite(zero_h_fused).all()
        and torch.isfinite(zero_logits).all()
        and torch.isfinite(loss)
    )
    batch_independence = bool(
        torch.allclose(debug_a["H_pair"][0], debug_b["H_pair"][0], rtol=1e-5, atol=1e-6)
    )
    exact_retrieval = bool(
        torch.equal(
            debug_a["Z_drug_KG"].cpu(),
            raw_artifact["drug_embeddings"].index_select(0, batch_drug_index),
        )
        and torch.equal(
            debug_a["Z_side_KG"].cpu(),
            raw_artifact["side_embeddings"].index_select(0, batch_side_index),
        )
        and torch.equal(
            debug_a["drug_kg_mask"].cpu().squeeze(1),
            raw_artifact["drug_available_mask"].index_select(0, batch_drug_index),
        )
        and torch.equal(
            debug_a["side_kg_mask"].cpu().squeeze(1),
            raw_artifact["side_available_mask"].index_select(0, batch_side_index),
        )
    )
    ordering_check = bool(
        torch.equal(raw_artifact["drug_matrix_indices"], torch.arange(DRUG_COUNT))
        and torch.equal(raw_artifact["side_matrix_indices"], torch.arange(SIDE_EFFECT_COUNT))
    )
    kg_effect_isolation = bool(
        batch_independence
        and not torch.allclose(
            debug_a["H_fused"][0], zero_h_fused[0], rtol=1e-5, atol=1e-6
        )
    )
    hashes_after = {path: sha256(path) for path in ORIGINAL_HASHES}
    original_file_safety = hashes_before == hashes_after == ORIGINAL_HASHES
    kg_artifact_unchanged = sha256(KG_ARTIFACT_PATH) == kg_hash_before

    checks = {
        "shape check": shape_check,
        "frozen KG check": frozen_kg_check,
        "fusion trainable check": fusion_trainable_check,
        "gate range check": gate_check,
        "finite-value check": finite_check,
        "batch-independence check": batch_independence,
        "exact KG row retrieval check": exact_retrieval,
        "matrix ordering check": ordering_check,
        "KG-effect isolation check": kg_effect_isolation,
        "original-file safety check": original_file_safety,
        "KG artifact unchanged check": kg_artifact_unchanged,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"KG fusion smoke test failed: {', '.join(failed)}")

    category_lines = [
        f"- {name}: drug_index={pair[0]}, side_effect_index={pair[1]}"
        for name, pair in requested.items()
    ]
    lines = [
        "BioKORF frozen-KG fusion smoke test",
        "====================================",
        f"Device: {device}",
        f"KG artifact path: {KG_ARTIFACT_PATH}",
        f"Frozen drug KG embeddings shape: {tuple(model.kg_features.drug_embeddings.shape)}",
        f"Frozen side KG embeddings shape: {tuple(model.kg_features.side_embeddings.shape)}",
        f"EN-con shape: {tuple(debug_a['H_en_con'].shape)}",
        f"EN-add shape: {tuple(debug_a['H_en_add'].shape)}",
        f"CNN-im shape: {tuple(debug_a['H_cnn_im'].shape)}",
        f"H_pair shape: {tuple(debug_a['H_pair'].shape)}",
        f"KG_input shape: {tuple(debug_a['KG_input'].shape)}",
        f"KG projection shape: {tuple(debug_a['KG_projected'].shape)}",
        f"Gate shape: {tuple(gate.shape)}",
        f"H_fused shape: {tuple(debug_a['H_fused'].shape)}",
        f"Latent shape: {tuple(debug_a['latent'].shape)}",
        f"Logits shape: {tuple(debug_a['logits'].shape)}",
        f"Gate mean/min/max: {gate.mean().item():.8f} / {gate.min().item():.8f} / {gate.max().item():.8f}",
        f"Frozen KG check: {'PASS' if frozen_kg_check else 'FAIL'}",
        f"Fusion parameters trainable: {'PASS' if fusion_trainable_check else 'FAIL'}",
        f"Exact KG row retrieval check: {'PASS' if exact_retrieval else 'FAIL'}",
        f"Matrix ordering check: {'PASS' if ordering_check else 'FAIL'}",
        f"KG-effect isolation check: {'PASS' if kg_effect_isolation else 'FAIL'}",
        f"Batch-independence check: {'PASS' if batch_independence else 'FAIL'}",
        f"Finite-value check: {'PASS' if finite_check else 'FAIL'}",
        f"Original-file safety check: {'PASS' if original_file_safety else 'FAIL'}",
        f"KG artifact unchanged check: {'PASS' if kg_artifact_unchanged else 'FAIL'}",
        f"Unavailable requested categories: {', '.join(unavailable_categories) if unavailable_categories else 'none'}",
        "Selected availability-category samples:",
        *category_lines,
        "Training performed: no",
        "Ordinal loss added: no",
        "Attention added: no",
        "R-GCN fine-tuned: no",
    ]
    report = "\n".join(lines) + "\n"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
