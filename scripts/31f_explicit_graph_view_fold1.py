"""Fold-1 MSSF-clean experiment with an appended explicit Drug graph view."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEP31C_PATH = PROJECT_ROOT / "scripts" / "31c_fold1_rewired_smd_experiment.py"
STEP31E_DIR = PROJECT_ROOT / "data_processed" / "rewiring" / "explicit_kg_task"
STEP31E_REPORT = STEP31E_DIR / "step31e_report.txt"
SPLIT_PATH = PROJECT_ROOT / "data_processed" / "experiments" / "pilot_fold1_split.npz"
CLEAN_RESULT_DIR = (
    PROJECT_ROOT
    / "data_processed"
    / "experiments"
    / "kg_alignment_fold1"
    / "clean_seed42_bs64"
)
OUTPUT_ROOT = (
    PROJECT_ROOT / "data_processed" / "experiments" / "explicit_graph_view_fold1"
)

VARIANT_FILES: dict[str, Path | None] = {
    "zero_control": None,
    "structure_only": STEP31E_DIR / "structure_only_matrix.npy",
    "task_only": STEP31E_DIR / "task_only_matrix.npy",
    "kg_task": STEP31E_DIR / "kg_task_explicit_matrix.npy",
}
DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 64
MAX_EPOCHS = 30
PATIENCE = 7
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
DROPOUT = 0.4
LATENT_DIM = 64
DRUG_COUNT = 757
SIDE_EFFECT_COUNT = 994
ORIGINAL_DRUG_VIEWS = 11
DRUG_VIEWS = 12
SIDE_VIEWS = 4
CLASS_LABELS = (1, 2, 3, 4, 5)

SOURCE_DATA_NAMES = (
    "drug_side.pkl",
    "Text_similarity_one.pkl",
    "Text_similarity_two.pkl",
    "Text_similarity_three.pkl",
    "Text_similarity_four.pkl",
    "Text_similarity_five.pkl",
    "drug_mol.pkl",
    "drug_target.pkl",
    "fingerprint_similarity.pkl",
    "drug_pathway_enzyme_similarity.pkl",
    "side_effect_semantic.pkl",
    "glove_wordEmbedding.pkl",
)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(PROJECT_ROOT))
step31c = load_module("biokorf_step31c_helpers_for_step31f", STEP31C_PATH)
pilot = step31c.pilot
from models.mssf_clean import MSSFCleanConfig
from models.mssf_clean_12view import MSSFClean12View


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matrix_sha256(matrix: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(matrix.dtype).encode("ascii"))
    digest.update(np.asarray(matrix.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(matrix).tobytes())
    return digest.hexdigest()


def require_files(paths: list[Path] | tuple[Path, ...]) -> None:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required Step 31F inputs are missing: {missing}")


def output_directory(variant: str, seed: int, batch_size: int) -> Path:
    return OUTPUT_ROOT / f"{variant}_seed{seed}_bs{batch_size}"


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "history": output_dir / "training_history.csv",
        "checkpoint": output_dir / "best_checkpoint.pt",
        "metrics": output_dir / "test_metrics.json",
        "confusion": output_dir / "confusion_matrix.csv",
        "per_class": output_dir / "per_class_metrics.csv",
        "report": output_dir / "experiment_report.txt",
    }


def protected_hashes(variant: str) -> dict[str, str]:
    paths = {
        "split": SPLIT_PATH,
        "step31e_report": STEP31E_REPORT,
        "mssf": PROJECT_ROOT / "mssf.py",
        "model": PROJECT_ROOT / "model.py",
        "mssf_clean": PROJECT_ROOT / "models" / "mssf_clean.py",
        "mssf_clean_12view": PROJECT_ROOT / "models" / "mssf_clean_12view.py",
        **{
            f"data:{name}": PROJECT_ROOT / "Datas" / name
            for name in SOURCE_DATA_NAMES
        },
    }
    selected = VARIANT_FILES[variant]
    if selected is not None:
        paths["explicit_matrix"] = selected
    require_files(tuple(paths.values()))
    return {name: sha256(path) for name, path in paths.items()}


def protected_unchanged(before: dict[str, str], variant: str) -> bool:
    return before == protected_hashes(variant)


def load_explicit_matrix(variant: str) -> tuple[np.ndarray, dict[str, Any]]:
    if variant == "zero_control":
        matrix = np.eye(DRUG_COUNT, dtype=np.float32)
        source = "generated identity control"
    else:
        path = VARIANT_FILES[variant]
        if path is None or not path.is_file():
            raise FileNotFoundError(f"Step 31E matrix not found: {path}")
        matrix = np.load(path, allow_pickle=False)
        source = str(path)
    valid = bool(
        isinstance(matrix, np.ndarray)
        and matrix.shape == (DRUG_COUNT, DRUG_COUNT)
        and matrix.dtype == np.float32
        and np.isfinite(matrix).all()
        and np.array_equal(np.diag(matrix), np.ones(DRUG_COUNT, dtype=np.float32))
        and np.all(matrix >= 0)
        and np.all(matrix <= 1)
    )
    if not valid:
        raise ValueError(f"Invalid explicit graph matrix contract for {variant}")
    offdiagonal = matrix.copy()
    np.fill_diagonal(offdiagonal, 0.0)
    zero_control_safe = bool(
        variant != "zero_control" or np.count_nonzero(offdiagonal) == 0
    )
    return matrix, {
        "source": source,
        "sha256": matrix_sha256(matrix),
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "minimum": float(matrix.min()),
        "maximum": float(matrix.max()),
        "nonzero_offdiagonal": int(np.count_nonzero(offdiagonal)),
        "zero_control_convention": (
            "identity matrix: diagonal self-similarity only; every off-diagonal is zero"
        ),
        "matrix_contract": valid,
        "zero_control_information": zero_control_safe,
    }


def explicit_graph_leakage_check() -> bool:
    if not STEP31E_REPORT.is_file():
        return False
    report = STEP31E_REPORT.read_text(encoding="utf-8")
    required = (
        "validation_positions_hidden: True",
        "test_positions_hidden: True",
        "TASK-AWARE LEAKAGE CHECK: PASS",
        "SOURCE FEATURE PRESERVATION CHECK: PASS",
        "EXPLICIT KG STRUCTURE CHECK: PASS",
        "Drug-Phenotype and adverse-drug-reaction relations are excluded.",
        "KG EMBEDDING COSINE EXCLUSION CHECK: PASS",
    )
    return all(marker in report for marker in required)


def load_experiment_data(
    variant: str,
) -> tuple[Any, Any, Any, dict[str, Any], dict[str, bool], dict[str, str]]:
    protected = protected_hashes(variant)
    frequency = np.asarray(pilot.load_pickle("drug_side.pkl"))
    samples = pilot.original_positive_sample_order(frequency)
    train_samples, validation_samples, test_samples = step31c.load_fixed_split(samples)
    hidden_samples = np.concatenate((validation_samples, test_samples), axis=0)
    original_drugs, side_features, label_safe = pilot.build_leakage_safe_features(
        frequency, hidden_samples
    )
    original_snapshot = original_drugs.clone()
    explicit_matrix, diagnostics = load_explicit_matrix(variant)
    drug_features = torch.cat(
        (original_drugs, torch.from_numpy(explicit_matrix)), dim=1
    )
    order_safe = bool(
        tuple(original_drugs.shape) == (DRUG_COUNT, DRUG_COUNT * ORIGINAL_DRUG_VIEWS)
        and torch.equal(drug_features[:, : original_drugs.shape[1]], original_snapshot)
        and torch.equal(
            drug_features[:, original_drugs.shape[1] :],
            torch.from_numpy(explicit_matrix),
        )
    )
    input_contract = bool(
        tuple(drug_features.shape) == (DRUG_COUNT, DRUG_COUNT * DRUG_VIEWS)
        and tuple(side_features.shape)
        == (SIDE_EFFECT_COUNT, SIDE_EFFECT_COUNT * SIDE_VIEWS)
    )
    finite = bool(
        torch.isfinite(drug_features).all() and torch.isfinite(side_features).all()
    )
    checks = {
        "label_derived_feature_leakage": bool(label_safe),
        "drug_phenotype_leakage": bool(pilot.scan_graph_leakage()),
        "explicit_graph_leakage": explicit_graph_leakage_check(),
        "source_feature_preservation": order_safe,
        "original_view_order_preservation": order_safe,
        "zero_control_information": diagnostics["zero_control_information"],
        "input_contract": input_contract,
        "finite_values": finite,
    }
    if not all(checks.values()):
        raise RuntimeError(f"A Step 31F data check failed: {checks}")
    datasets = tuple(
        pilot.IndexedPairDataset(part, drug_features, side_features)
        for part in (train_samples, validation_samples, test_samples)
    )
    diagnostics.update(
        {
            "variant": variant,
            "original_drug_view_count": ORIGINAL_DRUG_VIEWS,
            "final_drug_view_count": DRUG_VIEWS,
            "drug_input_dimension": int(drug_features.shape[1]),
            "side_input_dimension": int(side_features.shape[1]),
        }
    )
    return (*datasets, diagnostics, checks, protected)


def create_model(seed: int) -> MSSFClean12View:
    pilot.configure_reproducibility(seed)
    return MSSFClean12View(MSSFCleanConfig(dropout=DROPOUT, gp=LATENT_DIM))


def model_signature(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def parameter_fairness(seed: int) -> tuple[bool, int]:
    counts: dict[str, int] = {}
    signatures: dict[str, str] = {}
    for variant in VARIANT_FILES:
        model = create_model(seed)
        counts[variant] = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        signatures[variant] = model_signature(model)
    fair = len(set(counts.values())) == 1 and len(set(signatures.values())) == 1
    count = next(iter(counts.values()))
    print(f"Trainable parameter count: {count}")
    print(f"12-VIEW PARAMETER FAIRNESS CHECK: {'PASS' if fair else 'FAIL'}")
    return fair, count


def experiment_fairness(seed: int, batch_size: int) -> tuple[bool, int]:
    parameter_safe, count = parameter_fairness(seed)
    model = create_model(seed)
    fair = bool(
        parameter_safe
        and batch_size > 0
        and MAX_EPOCHS == 30
        and PATIENCE == 7
        and LEARNING_RATE == pilot.LEARNING_RATE
        and WEIGHT_DECAY == pilot.WEIGHT_DECAY
        and model.feature_nums == DRUG_VIEWS * SIDE_VIEWS
        and isinstance(model.classification_loss, nn.CrossEntropyLoss)
        and all(
            "attention" not in name.casefold() or isinstance(module, nn.Identity)
            for name, module in model.named_modules()
        )
        and not any("kg" in name.casefold() or "rgcn" in name.casefold() for name, _ in model.named_modules())
    )
    return fair, count


def composite_loss_12view(
    logits: Tensor,
    rec_con: Tensor,
    rec_add: Tensor,
    mu: Tensor,
    logvar: Tensor,
    labels: Tensor,
    drug_features: Tensor,
    side_features: Tensor,
    drug_view_count: int = DRUG_VIEWS,
) -> Tensor:
    """Legacy CLEAN composite loss with a validated Drug-view dimension."""
    if drug_features.ndim != 2:
        raise ValueError(
            f"drug_features must be two-dimensional; got {tuple(drug_features.shape)}"
        )
    if side_features.ndim != 2:
        raise ValueError(
            f"side_features must be two-dimensional; got {tuple(side_features.shape)}"
        )
    if drug_features.shape[1] != drug_view_count * DRUG_COUNT:
        raise ValueError(
            "drug_features width must equal "
            f"{drug_view_count} * {DRUG_COUNT} = {drug_view_count * DRUG_COUNT}; "
            f"got {drug_features.shape[1]}"
        )
    if side_features.shape[1] != SIDE_VIEWS * SIDE_EFFECT_COUNT:
        raise ValueError(
            "side_features width must equal "
            f"{SIDE_VIEWS} * {SIDE_EFFECT_COUNT} = {SIDE_VIEWS * SIDE_EFFECT_COUNT}; "
            f"got {side_features.shape[1]}"
        )
    if drug_features.shape[0] != side_features.shape[0]:
        raise ValueError("Drug and Side-effect feature batch sizes must match")

    batch_size = drug_features.shape[0]
    drug_views = drug_features.reshape(batch_size, drug_view_count, DRUG_COUNT)
    side_views = side_features.reshape(batch_size, SIDE_VIEWS, SIDE_EFFECT_COUNT)

    classification = nn.functional.cross_entropy(logits, labels.long() - 1)
    kl_divergence = (
        -0.5 * (1 + logvar - mu.square() - torch.exp(logvar))
    ).sum(1).mean()
    rec_connection_target = torch.cat((drug_features, side_features), dim=1)

    # Preserve the legacy stack-then-sum reduction exactly after safe reshape.
    drug_sum = torch.stack(drug_views.unbind(dim=1), dim=0).sum(dim=0)
    side_sum = torch.stack(side_views.unbind(dim=1), dim=0).sum(dim=0)
    rec_addition_target = torch.cat((drug_sum, side_sum), dim=1)
    rec_con_loss = nn.functional.mse_loss(
        rec_con, rec_connection_target, reduction="none"
    ).sum(dim=-1).mean()
    rec_add_loss = nn.functional.mse_loss(
        rec_add, rec_addition_target, reduction="none"
    ).sum(dim=-1).mean()
    return (
        classification
        + 0.001 * kl_divergence
        + 0.0001 * rec_con_loss
        + 0.0001 * rec_add_loss
    )


def prediction_loss(
    outputs: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    labels: Tensor,
    drugs: Tensor,
    sides: Tensor,
) -> Tensor:
    return composite_loss_12view(
        *outputs,
        labels,
        drugs,
        sides,
        drug_view_count=DRUG_VIEWS,
    )


def legacy_loss_equivalence_check(
    drugs_12view: Tensor,
    sides: Tensor,
    labels: Tensor,
) -> bool:
    batch_size = drugs_12view.shape[0]
    drugs_11view = drugs_12view[:, : ORIGINAL_DRUG_VIEWS * DRUG_COUNT]
    dtype = drugs_11view.dtype
    device = drugs_11view.device
    logits = torch.linspace(
        -0.25, 0.25, steps=batch_size * len(CLASS_LABELS), dtype=dtype, device=device
    ).reshape(batch_size, len(CLASS_LABELS))
    rec_con = torch.zeros(
        batch_size,
        ORIGINAL_DRUG_VIEWS * DRUG_COUNT + SIDE_VIEWS * SIDE_EFFECT_COUNT,
        dtype=dtype,
        device=device,
    )
    rec_add = torch.zeros(
        batch_size, DRUG_COUNT + SIDE_EFFECT_COUNT, dtype=dtype, device=device
    )
    mu = torch.linspace(
        -0.1, 0.1, steps=batch_size * LATENT_DIM, dtype=dtype, device=device
    ).reshape(batch_size, LATENT_DIM)
    logvar = torch.linspace(
        -0.05, 0.05, steps=batch_size * LATENT_DIM, dtype=dtype, device=device
    ).reshape(batch_size, LATENT_DIM)
    legacy = pilot.composite_loss(
        logits, rec_con, rec_add, mu, logvar, labels, drugs_11view, sides
    )
    local = composite_loss_12view(
        logits,
        rec_con,
        rec_add,
        mu,
        logvar,
        labels,
        drugs_11view,
        sides,
        drug_view_count=ORIGINAL_DRUG_VIEWS,
    )
    equivalent = bool(
        torch.isfinite(legacy)
        and torch.isfinite(local)
        and torch.allclose(legacy, local, rtol=1e-7, atol=1e-7)
    )
    print(f"LEGACY-LOSS EQUIVALENCE CHECK: {'PASS' if equivalent else 'FAIL'}")
    return equivalent


def smoke_contract(dataset: Any, seed: int, device: torch.device) -> bool:
    batch = [dataset[index] for index in range(4)]
    drugs = torch.stack([item[0] for item in batch]).to(device)
    sides = torch.stack([item[1] for item in batch]).to(device)
    labels = torch.stack([item[4] for item in batch]).to(device)
    model = create_model(seed).to(device).eval()
    with torch.inference_mode():
        outputs = model(drugs, sides, device=device, return_debug=True)
        loss = prediction_loss(outputs[:5], labels, drugs, sides)
        equivalence = legacy_loss_equivalence_check(drugs, sides, labels)
    logits, rec_con, rec_add, mu, logvar, debug = outputs
    tensors = (logits, rec_con, rec_add, mu, logvar, *debug.values())
    input_contract = bool(
        tuple(drugs.shape) == (4, 9084)
        and tuple(sides.shape) == (4, 3976)
        and tuple(debug["H_en_con"].shape) == (4, 128)
        and tuple(debug["H_en_add"].shape) == (4, 128)
        and tuple(debug["H_cnn_im"].shape) == (4, 128)
        and tuple(debug["H_pair"].shape) == (4, 384)
        and tuple(debug["latent"].shape) == (4, 64)
        and tuple(logits.shape) == (4, 5)
        and all(torch.isfinite(tensor).all() for tensor in tensors)
    )
    loss_contract = bool(
        tuple(drugs.reshape(4, DRUG_VIEWS, DRUG_COUNT).shape)
        == (4, DRUG_VIEWS, DRUG_COUNT)
        and tuple(sides.reshape(4, SIDE_VIEWS, SIDE_EFFECT_COUNT).shape)
        == (4, SIDE_VIEWS, SIDE_EFFECT_COUNT)
        and loss.ndim == 0
        and torch.isfinite(loss)
    )
    for label, shape in (
        ("Drug input", drugs.shape),
        ("Side input", sides.shape),
        ("EN-con", debug["H_en_con"].shape),
        ("EN-add", debug["H_en_add"].shape),
        ("CNN-im", debug["H_cnn_im"].shape),
        ("H_pair", debug["H_pair"].shape),
        ("latent", debug["latent"].shape),
        ("logits", logits.shape),
    ):
        print(f"{label}: {list(shape)}")
    print(f"Drug reshape: [4, 9084] -> {list(drugs.reshape(4, 12, 757).shape)}")
    print(f"Side reshape: [4, 3976] -> {list(sides.reshape(4, 4, 994).shape)}")
    print(f"12-VIEW INPUT CONTRACT CHECK: {'PASS' if input_contract else 'FAIL'}")
    print(f"12-VIEW LOSS CONTRACT CHECK: {'PASS' if loss_contract else 'FAIL'}")
    return bool(input_contract and loss_contract and equivalence)


SAFETY_LABELS = {
    "experiment_fairness": "EXPERIMENT FAIRNESS",
    "label_derived_feature_leakage": "LABEL-DERIVED FEATURE LEAKAGE",
    "drug_phenotype_leakage": "DRUG-PHENOTYPE LEAKAGE",
    "explicit_graph_leakage": "EXPLICIT GRAPH LEAKAGE",
    "source_feature_preservation": "SOURCE FEATURE PRESERVATION",
    "original_view_order_preservation": "ORIGINAL VIEW ORDER PRESERVATION",
    "parameter_fairness": "12-VIEW PARAMETER FAIRNESS",
    "finite_values": "FINITE-VALUE",
}


def print_checks(checks: dict[str, bool]) -> None:
    for key, label in SAFETY_LABELS.items():
        print(f"{label} CHECK: {'PASS' if checks[key] else 'FAIL'}")
    print(
        "ZERO CONTROL INFORMATION CHECK: "
        f"{'PASS' if checks.get('zero_control_information', False) else 'FAIL'}"
    )


def complete_checks(
    data_checks: dict[str, bool], fair: bool, protected_safe: bool, contract: bool
) -> dict[str, bool]:
    return {
        "experiment_fairness": fair,
        "label_derived_feature_leakage": data_checks["label_derived_feature_leakage"],
        "drug_phenotype_leakage": data_checks["drug_phenotype_leakage"],
        "explicit_graph_leakage": data_checks["explicit_graph_leakage"],
        "source_feature_preservation": bool(
            data_checks["source_feature_preservation"] and protected_safe
        ),
        "original_view_order_preservation": data_checks[
            "original_view_order_preservation"
        ],
        "parameter_fairness": fair,
        "finite_values": bool(data_checks["finite_values"] and contract),
        "zero_control_information": data_checks["zero_control_information"],
    }


def smoke_mode(variant: str, seed: int, batch_size: int) -> None:
    train_data, _validation, _test, diagnostics, data_checks, protected = (
        load_experiment_data(variant)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contract = smoke_contract(train_data, seed, device)
    fair, _count = experiment_fairness(seed, batch_size)
    checks = complete_checks(
        data_checks, fair, protected_unchanged(protected, variant), contract
    )
    print(f"Variant: {variant}")
    print(f"Zero control convention: {diagnostics['zero_control_convention']}")
    print_checks(checks)
    if not all(checks.values()):
        raise RuntimeError("A Step 31F smoke check failed")


def train_epoch(
    model: MSSFClean12View,
    dataset: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    seed: int,
    batch_size: int,
) -> float:
    model.train()
    total_loss, total_count = 0.0, 0
    for drugs, sides, _drug_index, _side_index, labels in step31c.make_loader(
        dataset, shuffle=True, seed=seed + epoch, batch_size=batch_size
    ):
        drugs, sides, labels = drugs.to(device), sides.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(drugs, sides, device=device)
        loss = prediction_loss(outputs, labels, drugs, sides)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        count = int(labels.shape[0])
        total_loss += float(loss.detach()) * count
        total_count += count
    return total_loss / total_count


def evaluate(
    model: MSSFClean12View,
    dataset: Any,
    device: torch.device,
    seed: int,
    batch_size: int,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    total_loss, total_count = 0.0, 0
    labels_all: list[np.ndarray] = []
    probabilities_all: list[np.ndarray] = []
    with torch.inference_mode():
        for drugs, sides, _drug_index, _side_index, labels in step31c.make_loader(
            dataset, shuffle=False, seed=seed, batch_size=batch_size
        ):
            drugs, sides = drugs.to(device), sides.to(device)
            outputs = model(drugs, sides, device=device)
            loss = prediction_loss(outputs, labels.to(device), drugs, sides)
            probabilities = torch.softmax(outputs[0], dim=1)
            if not torch.isfinite(loss) or not torch.isfinite(probabilities).all():
                raise FloatingPointError("Non-finite validation/test value")
            count = int(labels.shape[0])
            total_loss += float(loss) * count
            total_count += count
            labels_all.append((labels.numpy() - 1).astype(np.int64))
            probabilities_all.append(probabilities.cpu().numpy())
    metrics = pilot.classification_metrics(
        np.concatenate(labels_all), np.vstack(probabilities_all)
    )
    return total_loss / total_count, metrics


def train_mode(variant: str, seed: int, batch_size: int) -> dict[str, Any]:
    output_dir = output_directory(variant, seed, batch_size)
    paths = output_paths(output_dir)
    step31c.require_new([paths["history"], paths["checkpoint"]], "train")
    train_data, validation_data, _test, _diagnostics, data_checks, protected = (
        load_experiment_data(variant)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contract = smoke_contract(train_data, seed, device)
    fair, parameter_count = experiment_fairness(seed, batch_size)
    if not all((contract, fair, *data_checks.values())):
        raise RuntimeError("A required Step 31F pre-training check failed")
    output_dir.mkdir(parents=True, exist_ok=True)
    model = create_model(seed).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    history: list[dict[str, Any]] = []
    columns = [
        "epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1", "val_aupr"
    ]
    best_epoch, best_f1, stale = 0, -1.0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        pilot.configure_reproducibility(seed + epoch)
        started = time.perf_counter()
        train_loss = train_epoch(
            model, train_data, optimizer, device, epoch, seed, batch_size
        )
        val_loss, metrics = evaluate(
            model, validation_data, device, seed, batch_size
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": metrics["accuracy"],
            "val_macro_f1": metrics["macro_f1"],
            "val_aupr": metrics["aupr"],
        }
        if not all(np.isfinite(value) for value in row.values()):
            raise FloatingPointError("Non-finite training history value")
        history.append(row)
        step31c.write_csv(paths["history"], history, columns)
        if metrics["macro_f1"] > best_f1:
            best_epoch, best_f1, stale = epoch, float(metrics["macro_f1"]), 0
            temporary = paths["checkpoint"].with_suffix(".pt.tmp")
            torch.save(
                {
                    "model": "mssf_clean_12view",
                    "variant": variant,
                    "seed": seed,
                    "batch_size": batch_size,
                    "epoch": epoch,
                    "validation_macro_f1": best_f1,
                    "selection_metric": "validation_macro_f1",
                    "parameter_count": parameter_count,
                    "explicit_matrix_sha256": load_explicit_matrix(variant)[1]["sha256"],
                    "split_sha256": protected["split"],
                    "model_state_dict": step31c.checkpoint_state(model),
                },
                temporary,
            )
            temporary.replace(paths["checkpoint"])
        else:
            stale += 1
        print(
            f"variant={variant} epoch={epoch:02d} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} val_macro_f1={metrics['macro_f1']:.6f} "
            f"best={best_epoch} patience={stale}/{PATIENCE} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
        if stale >= PATIENCE:
            break
    checks = complete_checks(
        data_checks, fair, protected_unchanged(protected, variant), contract
    )
    print_checks(checks)
    if not all(checks.values()):
        raise RuntimeError("A Step 31F training safety check failed")
    return {"best_epoch": best_epoch, "best_validation_macro_f1": best_f1}


def test_mode(variant: str, seed: int, batch_size: int) -> dict[str, Any]:
    output_dir = output_directory(variant, seed, batch_size)
    paths = output_paths(output_dir)
    require_files((paths["history"], paths["checkpoint"]))
    step31c.require_new(
        [paths[key] for key in ("metrics", "confusion", "per_class", "report")],
        "test",
    )
    train_data, _validation, test_data, diagnostics, data_checks, protected = (
        load_experiment_data(variant)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contract = smoke_contract(train_data, seed, device)
    fair, parameter_count = experiment_fairness(seed, batch_size)
    checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
    expected = {
        "model": "mssf_clean_12view",
        "variant": variant,
        "seed": seed,
        "batch_size": batch_size,
        "parameter_count": parameter_count,
        "explicit_matrix_sha256": diagnostics["sha256"],
        "split_sha256": protected["split"],
    }
    if any(checkpoint.get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint metadata does not match this Step 31F run")
    model = create_model(seed)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)
    test_loss, metrics = evaluate(model, test_data, device, seed, batch_size)
    finite_metrics = all(
        np.isfinite(metrics[key])
        for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "micro_f1", "aupr")
    )
    checks = complete_checks(
        data_checks,
        fair,
        protected_unchanged(protected, variant),
        bool(contract and np.isfinite(test_loss) and finite_metrics),
    )
    metrics.update(
        {
            "model": "mssf_clean_12view",
            "variant": variant,
            "seed": seed,
            "batch_size": batch_size,
            "best_epoch": int(checkpoint["epoch"]),
            "best_validation_macro_f1": float(checkpoint["validation_macro_f1"]),
            "test_loss": float(test_loss),
            "parameter_count": parameter_count,
            "explicit_matrix_sha256": diagnostics["sha256"],
            "checks": checks,
        }
    )
    step31c.write_csv(
        paths["confusion"],
        [
            {
                "true_class": label,
                **{
                    f"predicted_{predicted}": row[predicted - 1]
                    for predicted in CLASS_LABELS
                },
            }
            for label, row in zip(CLASS_LABELS, metrics["confusion_matrix"], strict=True)
        ],
        ["true_class", *[f"predicted_{label}" for label in CLASS_LABELS]],
    )
    step31c.write_csv(
        paths["per_class"],
        [
            {"class": label, **metrics["per_class"][str(label)]}
            for label in CLASS_LABELS
        ],
        ["class", "precision", "recall", "f1", "support"],
    )
    step31c.write_json(paths["metrics"], metrics)
    report_lines = [
        "BioKORF Step 31F — Explicit Graph as Drug View #12",
        "===================================================",
        f"Variant: {variant}",
        f"Seed: {seed}",
        f"Batch size: {batch_size}",
        f"Trainable parameter count: {parameter_count}",
        f"Best epoch: {checkpoint['epoch']}",
        f"Best validation Macro-F1: {checkpoint['validation_macro_f1']:.8f}",
        f"Test Accuracy: {metrics['accuracy']:.8f}",
        f"Test Macro-F1: {metrics['macro_f1']:.8f}",
        f"Test AUPR: {metrics['aupr']:.8f}",
        "Original Drug views #1-#11 remain unchanged; the explicit matrix is view #12.",
        f"Zero control convention: {diagnostics['zero_control_convention']}",
        "",
        *[
            f"{SAFETY_LABELS[key]} CHECK: {'PASS' if checks[key] else 'FAIL'}"
            for key in SAFETY_LABELS
        ],
        f"ZERO CONTROL INFORMATION CHECK: {'PASS' if checks['zero_control_information'] else 'FAIL'}",
        "Training/testing performed: YES (explicit test mode).",
    ]
    paths["report"].write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines))
    print_checks(checks)
    if not all(checks.values()):
        raise RuntimeError("A Step 31F test safety check failed")
    return metrics


def train_test_mode(variant: str, seed: int, batch_size: int) -> None:
    paths = output_paths(output_directory(variant, seed, batch_size))
    step31c.require_new(list(paths.values()), "train_test")
    training = train_mode(variant, seed, batch_size)
    metrics = test_mode(variant, seed, batch_size)
    print("STEP 31F TRAIN_TEST SUMMARY")
    print(f"Variant: {variant}")
    print(f"Best epoch: {training['best_epoch']}")
    print(f"Best validation Macro-F1: {training['best_validation_macro_f1']:.8f}")
    print(f"Test Accuracy: {metrics['accuracy']:.8f}")
    print(f"Test Macro-F1: {metrics['macro_f1']:.8f}")
    print(f"Test AUPR: {metrics['aupr']:.8f}")


def read_result(path: Path, seed: int, batch_size: int) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if int(result.get("seed", -1)) != seed or int(result.get("batch_size", -1)) != batch_size:
        raise ValueError(f"Stored result settings mismatch: {path}")
    return result


def compare_mode(seed: int, batch_size: int) -> None:
    if seed != 42 or batch_size != 64:
        raise ValueError("The specified CLEAN comparison requires seed 42 and batch size 64")
    clean_path = CLEAN_RESULT_DIR / "test_metrics.json"
    require_files((clean_path,))
    results: dict[str, dict[str, Any]] = {
        "CLEAN_11_VIEW": read_result(clean_path, seed, batch_size)
    }
    names = {
        "zero_control": "ZERO_CONTROL_12_VIEW",
        "structure_only": "STRUCTURE_ONLY_12_VIEW",
        "task_only": "TASK_ONLY_12_VIEW",
        "kg_task": "KG_TASK_12_VIEW",
    }
    for variant, name in names.items():
        path = output_paths(output_directory(variant, seed, batch_size))["metrics"]
        if path.is_file():
            result = read_result(path, seed, batch_size)
            if result.get("variant") != variant:
                raise ValueError(f"Stored result variant mismatch: {path}")
            results[name] = result
    print("Model | Best epoch | Accuracy | Macro-F1 | AUPR")
    print("--- | ---: | ---: | ---: | ---:")
    for name, result in results.items():
        print(
            f"{name} | {result['best_epoch']} | {result['accuracy']:.8f} | "
            f"{result['macro_f1']:.8f} | {result['aupr']:.8f}"
        )
    deltas = (
        ("ZERO_CONTROL - CLEAN", "ZERO_CONTROL_12_VIEW", "CLEAN_11_VIEW"),
        ("STRUCTURE_ONLY - ZERO_CONTROL", "STRUCTURE_ONLY_12_VIEW", "ZERO_CONTROL_12_VIEW"),
        ("TASK_ONLY - ZERO_CONTROL", "TASK_ONLY_12_VIEW", "ZERO_CONTROL_12_VIEW"),
        ("KG_TASK - ZERO_CONTROL", "KG_TASK_12_VIEW", "ZERO_CONTROL_12_VIEW"),
        ("KG_TASK - CLEAN", "KG_TASK_12_VIEW", "CLEAN_11_VIEW"),
    )
    for label, left_name, right_name in deltas:
        if left_name not in results or right_name not in results:
            continue
        print(label)
        for metric in ("accuracy", "macro_f1", "aupr"):
            print(f"{metric}: {results[left_name][metric] - results[right_name][metric]:+.8f}")
    if "KG_TASK_12_VIEW" in results and "ZERO_CONTROL_12_VIEW" in results:
        positive = results["KG_TASK_12_VIEW"]["macro_f1"] > results["ZERO_CONTROL_12_VIEW"]["macro_f1"]
        print(f"EXPLICIT_KG_TASK_FEATURE_SIGNAL = {'POSITIVE' if positive else 'NEGATIVE'}")
    if "KG_TASK_12_VIEW" in results:
        improved = results["KG_TASK_12_VIEW"]["macro_f1"] > results["CLEAN_11_VIEW"]["macro_f1"]
        print(f"DOWNSTREAM_IMPROVEMENT_VS_CLEAN = {'YES' if improved else 'NO'}")
    print("Compare mode is read-only. These are decision signals, not publication claims.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(VARIANT_FILES), default="zero_control")
    parser.add_argument(
        "--mode", required=True, choices=("smoke", "train", "test", "train_test", "compare")
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def main() -> None:
    args = parse_args()
    print(
        f"Mode: {args.mode}; variant: {args.variant}; seed: {args.seed}; "
        f"batch size: {args.batch_size}"
    )
    require_files((STEP31E_REPORT, SPLIT_PATH, STEP31C_PATH))
    if args.mode == "smoke":
        smoke_mode(args.variant, args.seed, args.batch_size)
    elif args.mode == "train":
        train_mode(args.variant, args.seed, args.batch_size)
    elif args.mode == "test":
        test_mode(args.variant, args.seed, args.batch_size)
    elif args.mode == "train_test":
        train_test_mode(args.variant, args.seed, args.batch_size)
    else:
        compare_mode(args.seed, args.batch_size)


if __name__ == "__main__":
    main()
