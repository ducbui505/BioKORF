"""No-training smoke test for the batch-independent clean MSSF backbone."""

from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "data_processed" / "architecture" / "mssf_clean_smoke_test.txt"
MSSF_PATH = PROJECT_ROOT / "mssf.py"
MODEL_PATH = PROJECT_ROOT / "model.py"
EXPECTED_HASHES = {
    MSSF_PATH: "4867fecd04beabb2d715b24073f82a46bd572c13294afa3565ddba99f963fdb1",
    MODEL_PATH: "9c0d4bf17551a7d0f881a29e0f8e2727227f3561678064fec46f2848156a1e75",
}
SEED = 42
BATCH_SIZE = 4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_unchanged(hashes_before: dict[Path, str]) -> bool:
    hashes_after = {path: sha256(path) for path in hashes_before}
    return hashes_before == hashes_after == EXPECTED_HASHES


def main() -> None:
    for path in EXPECTED_HASHES:
        if not path.is_file():
            raise FileNotFoundError(f"Required original MSSF file not found: {path}")
    hashes_before = {path: sha256(path) for path in EXPECTED_HASHES}
    if hashes_before != EXPECTED_HASHES:
        raise RuntimeError(
            "Original MSSF file hash differs from the pre-implementation baseline: "
            f"{hashes_before}"
        )

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
        MSSFClean,
        MSSFCleanConfig,
        build_indexed_pair_dataset,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MSSFClean(MSSFCleanConfig(dropout=0.5, gp=64)).to(device).eval()

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    shared_drug = torch.randn(1, DRUG_COUNT * DRUG_VIEW_COUNT, generator=generator)
    shared_side = torch.randn(1, SIDE_EFFECT_COUNT * SIDE_VIEW_COUNT, generator=generator)
    companions_a_drug = torch.randn(3, DRUG_COUNT * DRUG_VIEW_COUNT, generator=generator)
    companions_a_side = torch.randn(3, SIDE_EFFECT_COUNT * SIDE_VIEW_COUNT, generator=generator)
    companions_b_drug = torch.randn(3, DRUG_COUNT * DRUG_VIEW_COUNT, generator=generator)
    companions_b_side = torch.randn(3, SIDE_EFFECT_COUNT * SIDE_VIEW_COUNT, generator=generator)
    batch_a_drug = torch.cat((shared_drug, companions_a_drug), dim=0)
    batch_a_side = torch.cat((shared_side, companions_a_side), dim=0)
    batch_b_drug = torch.cat((shared_drug, companions_b_drug), dim=0)
    batch_b_side = torch.cat((shared_side, companions_b_side), dim=0)

    drug_index = torch.tensor([0, 1, 755, 756], dtype=torch.long)
    side_effect_index = torch.tensor([0, 1, 992, 993], dtype=torch.long)
    labels = torch.tensor([1, 2, 4, 5], dtype=torch.long)
    dataset = build_indexed_pair_dataset(
        batch_a_drug, batch_a_side, drug_index, side_effect_index, labels
    )
    sample_batch = tuple(torch.stack([dataset[i][field] for i in range(len(dataset))]) for field in range(5))
    _, _, preserved_drug_index, preserved_side_index, preserved_labels = sample_batch
    index_preservation = bool(
        torch.equal(preserved_drug_index, drug_index)
        and torch.equal(preserved_side_index, side_effect_index)
        and torch.equal(preserved_labels, labels)
        and preserved_drug_index.dtype == torch.long
        and preserved_side_index.dtype == torch.long
    )

    with torch.inference_mode():
        logits_a, _, _, _, _, debug_a = model(
            batch_a_drug, batch_a_side, device=device, return_debug=True
        )
        _, _, _, _, _, debug_b = model(
            batch_b_drug, batch_b_side, device=device, return_debug=True
        )
        # Exercise the preserved loss contract without training/backpropagation.
        loss = model.frequency_classification_loss(logits_a, labels)

    expected_shapes = {
        "H_en_con": (BATCH_SIZE, 128),
        "H_en_add": (BATCH_SIZE, 128),
        "H_cnn_im": (BATCH_SIZE, 128),
        "H_pair": (BATCH_SIZE, 384),
        "latent": (BATCH_SIZE, 64),
        "logits": (BATCH_SIZE, 5),
    }
    shape_check = all(tuple(debug_a[name].shape) == shape for name, shape in expected_shapes.items())
    batch_independence = bool(
        torch.allclose(debug_a["H_pair"][0], debug_b["H_pair"][0], rtol=1e-5, atol=1e-6)
    )
    finite_check = bool(
        all(torch.isfinite(tensor).all() for tensor in debug_a.values())
        and torch.isfinite(loss)
    )
    original_files_unchanged = require_unchanged(hashes_before)

    checks = {
        "branch and output shape check": shape_check,
        "index preservation check": index_preservation,
        "batch independence check": batch_independence,
        "finite tensor check": finite_check,
        "original MSSF files unchanged": original_files_unchanged,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"MSSF-clean smoke test failed: {', '.join(failed)}")

    lines = [
        "BioKORF MSSF-clean smoke test",
        "==============================",
        f"Device: {device}",
        "Training performed: no",
        "KG embeddings loaded: no",
        "New attention added: no",
        "Cross-minibatch attention present: no",
        f"EN-con shape: {tuple(debug_a['H_en_con'].shape)}",
        f"EN-add shape: {tuple(debug_a['H_en_add'].shape)}",
        f"CNN-im shape: {tuple(debug_a['H_cnn_im'].shape)}",
        f"H_pair shape: {tuple(debug_a['H_pair'].shape)}",
        f"Latent shape: {tuple(debug_a['latent'].shape)}",
        f"Logits shape: {tuple(debug_a['logits'].shape)}",
        f"drug_index shape: {tuple(preserved_drug_index.shape)}",
        f"side_effect_index shape: {tuple(preserved_side_index.shape)}",
        f"labels shape: {tuple(preserved_labels.shape)}",
        f"Index preservation check: {'PASS' if index_preservation else 'FAIL'}",
        f"Batch-independence check: {'PASS' if batch_independence else 'FAIL'}",
        f"Finite tensor check: {'PASS' if finite_check else 'FAIL'}",
        f"CrossEntropyLoss finite check: {'PASS' if torch.isfinite(loss) else 'FAIL'}",
        f"Original MSSF files unchanged: {'PASS' if original_files_unchanged else 'FAIL'}",
        f"mssf.py SHA256: {hashes_before[MSSF_PATH]}",
        f"model.py SHA256: {hashes_before[MODEL_PATH]}",
    ]
    report = "\n".join(lines) + "\n"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
