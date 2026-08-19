"""Fold-1 residual view-aware enhancement experiment for BioKORF.

This module is inert on import. Training and test evaluation occur only when
the corresponding CLI mode is explicitly requested.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEP30A_SCRIPT = PROJECT_ROOT / "scripts" / "30a_viewaware_fold1.py"
SPLIT_PATH = PROJECT_ROOT / "data_processed" / "experiments" / "pilot_fold1_split.npz"
CLEAN_RESULT_DIR = (
    PROJECT_ROOT / "data_processed" / "experiments" / "kg_alignment_fold1"
    / "clean_seed42_bs64"
)
OUTPUT_ROOT = (
    PROJECT_ROOT / "data_processed" / "experiments" / "residual_viewaware_fold1"
)

DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 64
MAX_EPOCHS = 30
PATIENCE = 7
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
DROPOUT = 0.4
LATENT_DIM = 64
CLASS_LABELS = (1, 2, 3, 4, 5)
TOLERANCE = 1e-6
STRICT_EQUIVALENCE_TOLERANCE = 1e-7

PROTECTED_PATHS = (
    PROJECT_ROOT / "mssf.py",
    PROJECT_ROOT / "model.py",
    PROJECT_ROOT / "models" / "mssf_clean.py",
    PROJECT_ROOT / "models" / "view_aware_encoder.py",
    SPLIT_PATH,
)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


step30a = load_module("biokorf_step30a_helpers_for_30b", STEP30A_SCRIPT)
pilot = step30a.pilot
sys.path.insert(0, str(PROJECT_ROOT))
from models.mssf_clean import MSSFClean, MSSFCleanConfig
from models.mssf_residual_viewaware import BioKORFResidualViewAware


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protected_hashes() -> dict[Path, str]:
    for path in PROTECTED_PATHS:
        if not path.is_file():
            raise FileNotFoundError(f"Protected input not found: {path}")
    return {path: sha256(path) for path in PROTECTED_PATHS}


def unchanged(before: dict[Path, str]) -> bool:
    return before == {path: sha256(path) for path in before}


def output_directory(seed: int, batch_size: int) -> Path:
    return OUTPUT_ROOT / f"seed{seed}_bs{batch_size}"


def require_new(paths: list[Path], operation: str) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            f"Refusing to {operation}; output already exists: " + ", ".join(existing)
        )


def to_json_safe(value: Any) -> Any:
    """Recursively copy a value into JSON-native scalar/container types."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return to_json_safe(value.tolist())
    if isinstance(value, Tensor):
        tensor = value.detach().cpu()
        return to_json_safe(tensor.item() if tensor.ndim == 0 else tensor.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            to_json_safe(key): to_json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [to_json_safe(item) for item in value]
    raise TypeError(f"Unsupported JSON value type: {type(value).__name__}")


def create_clean(seed: int) -> MSSFClean:
    pilot.configure_reproducibility(seed)
    return MSSFClean(MSSFCleanConfig(dropout=DROPOUT, gp=LATENT_DIM))


def create_model(seed: int) -> BioKORFResidualViewAware:
    pilot.configure_reproducibility(seed)
    return BioKORFResidualViewAware(
        MSSFCleanConfig(dropout=DROPOUT, gp=LATENT_DIM)
    )


def copy_clean_weights(clean: MSSFClean, residual: BioKORFResidualViewAware) -> None:
    incompatible = residual.load_state_dict(clean.state_dict(), strict=False)
    allowed = (
        "drug_view_encoder.",
        "side_view_encoder.",
        "residual_view_enhancement.",
    )
    if incompatible.unexpected_keys or any(
        not key.startswith(allowed) for key in incompatible.missing_keys
    ):
        raise RuntimeError(
            "CLEAN weights were not compatible with the residual model: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )


def fairness_check(seed: int, batch_size: int) -> bool:
    clean = create_clean(seed)
    candidate = create_model(seed)
    clean_state = clean.state_dict()
    candidate_state = candidate.state_dict()
    clean_identical = all(
        name in candidate_state and torch.equal(value, candidate_state[name])
        for name, value in clean_state.items()
    )
    policy_identical = (
        LEARNING_RATE == pilot.LEARNING_RATE
        and WEIGHT_DECAY == pilot.WEIGHT_DECAY
        and batch_size > 0
    )
    return bool(clean_identical and policy_identical)


def stack_samples(dataset: Any, indices: list[int]) -> tuple[Tensor, Tensor, Tensor]:
    samples = [dataset[index] for index in indices]
    return (
        torch.stack([sample[0] for sample in samples]),
        torch.stack([sample[1] for sample in samples]),
        torch.stack([sample[4] for sample in samples]),
    )


def pooling_valid(weights: Tensor) -> bool:
    return bool(
        torch.isfinite(weights).all()
        and torch.all(weights >= -TOLERANCE)
        and torch.all(weights <= 1.0 + TOLERANCE)
        and torch.allclose(
            weights.sum(dim=1),
            torch.ones(weights.shape[0], device=weights.device),
            atol=TOLERANCE,
            rtol=0.0,
        )
    )


def module_has_finite_nonzero_gradient(module: nn.Module) -> bool:
    gradients = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    return bool(
        gradients
        and all(torch.isfinite(gradient).all() for gradient in gradients)
        and any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients)
    )


def smoke_checks(
    dataset: Any, seed: int, device: torch.device
) -> dict[str, bool]:
    if len(dataset) < 7:
        raise ValueError("At least seven training samples are required for smoke checks")
    drugs_a, sides_a, labels_a = stack_samples(dataset, [0, 1, 2, 3])
    drugs_b, sides_b, _labels_b = stack_samples(dataset, [0, 4, 5, 6])
    drugs_a, sides_a, labels_a = (
        drugs_a.to(device), sides_a.to(device), labels_a.to(device)
    )
    drugs_b, sides_b = drugs_b.to(device), sides_b.to(device)

    clean = create_clean(seed).to(device)
    residual = create_model(seed).to(device)
    copy_clean_weights(clean, residual)
    clean.eval()
    residual.eval()
    with torch.inference_mode():
        clean_outputs = clean(drugs_a, sides_a, device=device, return_debug=True)
        residual_outputs = residual(
            drugs_a, sides_a, device=device, return_debug=True
        )
        residual_other = residual(
            drugs_b, sides_b, device=device, return_debug=True
        )
    clean_debug = clean_outputs[-1]
    debug = residual_outputs[-1]
    other_debug = residual_other[-1]

    expected_debug_keys = {
        "H_en_con", "H_en_add", "H_cnn_im", "H_drug_view", "H_side_view",
        "drug_view_attention_weights", "side_view_attention_weights",
        "drug_view_pooling_weights", "side_view_pooling_weights", "V_residual",
        "residual_gate", "residual_correction", "H_en_add_enhanced",
        "H_pair_residual", "latent", "logits",
    }
    debug_contract = set(debug) == expected_debug_keys
    equivalence = all(
        torch.allclose(left, right, atol=STRICT_EQUIVALENCE_TOLERANCE, rtol=0.0)
        for left, right in (
            (debug["H_en_add_enhanced"], clean_debug["H_en_add"]),
            (debug["H_pair_residual"], clean_debug["H_pair"]),
            (residual_outputs[0], clean_outputs[0]),
        )
    )
    residual_zero = bool(
        debug["V_residual"].abs().mean() <= STRICT_EQUIVALENCE_TOLERANCE
        and debug["residual_correction"].abs().mean()
        <= STRICT_EQUIVALENCE_TOLERANCE
    )
    expected_gate = torch.full_like(debug["residual_gate"], 0.1)
    gate_initial = bool(
        torch.allclose(
            debug["residual_gate"], expected_gate,
            atol=STRICT_EQUIVALENCE_TOLERANCE, rtol=0.0,
        )
    )
    view_axis = (
        tuple(debug["drug_view_attention_weights"].shape) == (4, 4, 11, 11)
        and tuple(debug["side_view_attention_weights"].shape) == (4, 4, 4, 4)
    )
    view_pooling = pooling_valid(debug["drug_view_pooling_weights"]) and pooling_valid(
        debug["side_view_pooling_weights"]
    )
    batch_independence = all(
        torch.allclose(debug[key][0], other_debug[key][0], atol=TOLERANCE, rtol=0.0)
        for key in (
            "H_drug_view", "H_side_view", "H_en_add_enhanced", "H_pair_residual"
        )
    )

    # Disposable gradient diagnostic: no optimizer step and no state is saved.
    gradient_model = create_model(seed).to(device).eval()
    outputs = gradient_model(drugs_a, sides_a, device=device)
    first_loss = gradient_model.frequency_classification_loss(outputs[0], labels_a)
    first_loss.backward()
    final_projection = gradient_model.residual_view_enhancement.residual_projection[-1]
    first_backward = module_has_finite_nonzero_gradient(final_projection)
    clean_backward = all(
        module_has_finite_nonzero_gradient(module)
        for module in (
            gradient_model.encoderConnection,
            gradient_model.encoderAddition,
            gradient_model.crossProduction,
        )
    )

    gradient_model.zero_grad(set_to_none=True)
    with torch.no_grad():
        final_projection.weight.fill_(1e-4)
    outputs = gradient_model(drugs_a, sides_a, device=device)
    second_loss = gradient_model.frequency_classification_loss(outputs[0], labels_a)
    second_loss.backward()
    upstream = all(
        module_has_finite_nonzero_gradient(module)
        for module in (
            gradient_model.residual_view_enhancement.residual_projection[0],
            gradient_model.residual_view_enhancement.gate,
            gradient_model.drug_view_encoder,
            gradient_model.side_view_encoder,
        )
    )
    gradient_path = bool(first_backward and clean_backward and upstream)

    checks = {
        "debug_tensor_contract": debug_contract,
        "clean_equivalence_at_init": equivalence,
        "initial_residual_zero": residual_zero,
        "initial_gate": gate_initial,
        "residual_gradient_path": gradient_path,
        "view_axis_attention": view_axis,
        "view_batch_independence": batch_independence,
        "view_pooling": view_pooling,
    }
    print(f"Drug attention tensor shape: {tuple(debug['drug_view_attention_weights'].shape)}")
    print(f"Side attention tensor shape: {tuple(debug['side_view_attention_weights'].shape)}")
    print(f"Initial mean abs V_residual: {debug['V_residual'].abs().mean().item():.10f}")
    print(
        "Initial mean abs residual correction: "
        f"{debug['residual_correction'].abs().mean().item():.10f}"
    )
    print(f"Initial mean residual gate: {debug['residual_gate'].mean().item():.10f}")
    for name, value in checks.items():
        print(f"{name.replace('_', ' ').upper()} CHECK: {'PASS' if value else 'FAIL'}")
    return checks


def prediction_loss(
    logits: Tensor,
    rec_con: Tensor,
    rec_add: Tensor,
    mu: Tensor,
    logvar: Tensor,
    labels: Tensor,
    drugs: Tensor,
    sides: Tensor,
) -> Tensor:
    return pilot.composite_loss(
        logits, rec_con, rec_add, mu, logvar, labels, drugs, sides
    )


def correction_ratio(debug: dict[str, Tensor]) -> Tensor:
    numerator = torch.linalg.vector_norm(debug["residual_correction"], dim=1)
    denominator = torch.linalg.vector_norm(debug["H_en_add"], dim=1) + 1e-8
    return numerator / denominator


def train_epoch(
    model: BioKORFResidualViewAware,
    dataset: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    seed: int,
    batch_size: int,
) -> float:
    model.train()
    total, count_total = 0.0, 0
    for drugs, sides, _drug_index, _side_index, labels in step30a.make_loader(
        dataset, shuffle=True, seed=seed + epoch, batch_size=batch_size
    ):
        drugs, sides, labels = drugs.to(device), sides.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(drugs, sides, device=device)
        loss = prediction_loss(*outputs, labels, drugs, sides)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        count = int(labels.shape[0])
        total += float(loss.detach()) * count
        count_total += count
    return total / count_total


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def evaluate(
    model: BioKORFResidualViewAware,
    dataset: Any,
    device: torch.device,
    seed: int,
    batch_size: int,
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    model.eval()
    total, count_total = 0.0, 0
    labels_all: list[np.ndarray] = []
    probabilities_all: list[np.ndarray] = []
    gates: list[np.ndarray] = []
    ratios: list[np.ndarray] = []
    drug_weights: list[np.ndarray] = []
    side_weights: list[np.ndarray] = []
    with torch.inference_mode():
        for drugs, sides, _drug_index, _side_index, labels in step30a.make_loader(
            dataset, shuffle=False, seed=seed, batch_size=batch_size
        ):
            drugs, sides = drugs.to(device), sides.to(device)
            outputs = model(drugs, sides, device=device, return_debug=True)
            logits, rec_con, rec_add, mu, logvar, debug = outputs
            loss = prediction_loss(
                logits, rec_con, rec_add, mu, logvar, labels.to(device), drugs, sides
            )
            probabilities = torch.softmax(logits, dim=1)
            if not torch.isfinite(loss) or not torch.isfinite(probabilities).all():
                raise FloatingPointError("Non-finite validation/test value")
            if not pooling_valid(debug["drug_view_pooling_weights"]) or not pooling_valid(
                debug["side_view_pooling_weights"]
            ):
                raise RuntimeError("View pooling validation failed")
            count = int(labels.shape[0])
            total += float(loss) * count
            count_total += count
            labels_all.append((labels.numpy() - 1).astype(np.int64))
            probabilities_all.append(probabilities.cpu().numpy())
            gates.append(debug["residual_gate"].cpu().numpy())
            ratios.append(correction_ratio(debug).cpu().numpy())
            drug_weights.append(debug["drug_view_pooling_weights"].cpu().numpy())
            side_weights.append(debug["side_view_pooling_weights"].cpu().numpy())
    gate_values = np.concatenate([values.reshape(-1) for values in gates])
    ratio_values = np.concatenate(ratios)
    diagnostics = {
        "residual_gate": summarize(gate_values),
        "correction_ratio": summarize(ratio_values),
        "drug_pooling_weights": np.vstack(drug_weights),
        "side_pooling_weights": np.vstack(side_weights),
    }
    metrics = pilot.classification_metrics(
        np.concatenate(labels_all), np.vstack(probabilities_all)
    )
    return total / count_total, metrics, diagnostics


def checkpoint_state(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def print_safety(
    fair: bool,
    label_safe: bool,
    graph_safe: bool,
    finite: bool,
    smoke: dict[str, bool],
) -> None:
    print(f"EXPERIMENT FAIRNESS CHECK: {'PASS' if fair else 'FAIL'}")
    print(f"LABEL-DERIVED FEATURE LEAKAGE CHECK: {'PASS' if label_safe else 'FAIL'}")
    print(f"DRUG-PHENOTYPE LEAKAGE CHECK: {'PASS' if graph_safe else 'FAIL'}")
    print(f"FINITE-VALUE CHECK: {'PASS' if finite else 'FAIL'}")
    labels = {
        "clean_equivalence_at_init": "CLEAN-EQUIVALENCE-AT-INIT",
        "initial_residual_zero": "INITIAL RESIDUAL ZERO",
        "initial_gate": "INITIAL GATE",
        "residual_gradient_path": "RESIDUAL GRADIENT PATH",
        "view_axis_attention": "VIEW-AXIS ATTENTION",
        "view_batch_independence": "VIEW BATCH-INDEPENDENCE",
        "view_pooling": "VIEW POOLING",
    }
    for key, label in labels.items():
        print(f"{label} CHECK: {'PASS' if smoke[key] else 'FAIL'}")


def history_columns() -> list[str]:
    return [
        "epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1",
        "val_aupr", "mean_residual_gate", "std_residual_gate",
        "min_residual_gate", "max_residual_gate", "mean_correction_ratio",
        "std_correction_ratio",
        *[f"mean_drug_view_{index}_weight" for index in range(1, 12)],
        *[f"mean_side_view_{index}_weight" for index in range(1, 5)],
        "cuda_peak_allocated_mib", "cuda_peak_reserved_mib",
    ]


def train_mode(seed: int, batch_size: int) -> dict[str, Any]:
    before = protected_hashes()
    output_dir = output_directory(seed, batch_size)
    history_path = output_dir / "training_history.csv"
    checkpoint_path = output_dir / "best_checkpoint.pt"
    require_new([history_path, checkpoint_path], "train")
    train_data, validation_data, _test_data, label_safe, graph_safe = step30a.load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smoke = smoke_checks(train_data, seed, device)
    fair = fairness_check(seed, batch_size)
    if not all((fair, label_safe, graph_safe, *smoke.values())):
        raise RuntimeError("A required pre-training check failed")
    output_dir.mkdir(parents=True, exist_ok=True)
    model = create_model(seed).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    history: list[dict[str, Any]] = []
    best_epoch, best_f1, stale = 0, -1.0, 0
    finite = True
    for epoch in range(1, MAX_EPOCHS + 1):
        pilot.configure_reproducibility(seed + epoch)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        train_loss = train_epoch(
            model, train_data, optimizer, device, epoch, seed, batch_size
        )
        val_loss, metrics, diagnostic = evaluate(
            model, validation_data, device, seed, batch_size
        )
        gate = diagnostic["residual_gate"]
        ratio = diagnostic["correction_ratio"]
        drug_means = diagnostic["drug_pooling_weights"].mean(axis=0)
        side_means = diagnostic["side_pooling_weights"].mean(axis=0)
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": metrics["accuracy"],
            "val_macro_f1": metrics["macro_f1"],
            "val_aupr": metrics["aupr"],
            "mean_residual_gate": gate["mean"],
            "std_residual_gate": gate["std"],
            "min_residual_gate": gate["min"],
            "max_residual_gate": gate["max"],
            "mean_correction_ratio": ratio["mean"],
            "std_correction_ratio": ratio["std"],
            "cuda_peak_allocated_mib": (
                torch.cuda.max_memory_allocated(device) / 1024**2
                if device.type == "cuda" else 0.0
            ),
            "cuda_peak_reserved_mib": (
                torch.cuda.max_memory_reserved(device) / 1024**2
                if device.type == "cuda" else 0.0
            ),
        }
        row.update(
            {f"mean_drug_view_{i + 1}_weight": float(value) for i, value in enumerate(drug_means)}
        )
        row.update(
            {f"mean_side_view_{i + 1}_weight": float(value) for i, value in enumerate(side_means)}
        )
        finite = finite and all(
            np.isfinite(value) for value in row.values() if isinstance(value, (int, float))
        )
        history.append(row)
        step30a.write_csv(history_path, history, history_columns())
        if metrics["macro_f1"] > best_f1:
            best_epoch, best_f1, stale = epoch, float(metrics["macro_f1"]), 0
            temporary = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(
                {
                    "model": "residual_view_aware",
                    "seed": seed,
                    "batch_size": batch_size,
                    "epoch": epoch,
                    "validation_macro_f1": best_f1,
                    "selection_metric": "validation_macro_f1",
                    "model_state_dict": checkpoint_state(model),
                },
                temporary,
            )
            temporary.replace(checkpoint_path)
        else:
            stale += 1
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"val_macro_f1={metrics['macro_f1']:.6f} gate={gate['mean']:.6f} "
            f"correction_ratio={ratio['mean']:.6f} best={best_epoch} "
            f"patience={stale}/{PATIENCE} elapsed={time.perf_counter() - started:.1f}s "
            f"cuda_allocated={row['cuda_peak_allocated_mib']:.1f}MiB "
            f"cuda_reserved={row['cuda_peak_reserved_mib']:.1f}MiB",
            flush=True,
        )
        if stale >= PATIENCE:
            break
    fair = fair and unchanged(before)
    print_safety(fair, label_safe, graph_safe, finite, smoke)
    if not all((fair, label_safe, graph_safe, finite, *smoke.values())):
        raise RuntimeError("A required training safety check failed")
    return {"best_epoch": best_epoch, "best_validation_macro_f1": best_f1}


def write_weight_summary(path: Path, weights: np.ndarray) -> None:
    rows = [
        {
            "view_index": index + 1,
            "mean_weight": float(weights[:, index].mean()),
            "std_weight": float(weights[:, index].std()),
        }
        for index in range(weights.shape[1])
    ]
    step30a.write_csv(path, rows, ["view_index", "mean_weight", "std_weight"])


def expected_test_outputs(output_dir: Path) -> list[Path]:
    return [
        output_dir / "test_metrics.json",
        output_dir / "confusion_matrix.csv",
        output_dir / "per_class_metrics.csv",
        output_dir / "drug_view_weight_summary.csv",
        output_dir / "side_view_weight_summary.csv",
        output_dir / "residual_diagnostics.json",
        output_dir / "residual_viewaware_report.txt",
    ]


def test_mode(seed: int, batch_size: int) -> dict[str, Any]:
    before = protected_hashes()
    output_dir = output_directory(seed, batch_size)
    checkpoint_path = output_dir / "best_checkpoint.pt"
    history_path = output_dir / "training_history.csv"
    if not checkpoint_path.is_file() or not history_path.is_file():
        raise FileNotFoundError(f"Train mode must complete before test mode: {output_dir}")
    outputs = expected_test_outputs(output_dir)
    require_new(outputs, "test")
    train_data, _validation_data, test_data, label_safe, graph_safe = step30a.load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smoke = smoke_checks(train_data, seed, device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model") != "residual_view_aware":
        raise ValueError("Checkpoint is not a residual view-aware checkpoint")
    if int(checkpoint.get("seed", -1)) != seed or int(
        checkpoint.get("batch_size", -1)
    ) != batch_size:
        raise ValueError("Checkpoint seed or batch size differs from the requested run")
    model = create_model(seed)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)
    test_loss, metrics, diagnostic = evaluate(model, test_data, device, seed, batch_size)
    write_weight_summary(
        output_dir / "drug_view_weight_summary.csv",
        diagnostic["drug_pooling_weights"],
    )
    write_weight_summary(
        output_dir / "side_view_weight_summary.csv",
        diagnostic["side_pooling_weights"],
    )
    finite = all(
        np.isfinite(metrics[key])
        for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "micro_f1", "aupr")
    ) and np.isfinite(test_loss)
    fair = fairness_check(seed, batch_size) and unchanged(before)
    checks = {
        "experiment_fairness": fair,
        "label_derived_feature_leakage": label_safe,
        "drug_phenotype_leakage": graph_safe,
        "finite_values": finite,
        **smoke,
    }
    metrics.update(
        {
            "model": "residual_view_aware",
            "seed": seed,
            "batch_size": batch_size,
            "best_epoch": int(checkpoint["epoch"]),
            "best_validation_macro_f1": float(checkpoint["validation_macro_f1"]),
            "test_loss": float(test_loss),
            "test_mean_residual_gate": diagnostic["residual_gate"]["mean"],
            "test_mean_correction_ratio": diagnostic["correction_ratio"]["mean"],
            "checks": checks,
        }
    )
    step30a.write_csv(
        output_dir / "confusion_matrix.csv",
        [
            {
                "true_class": label,
                **{f"predicted_{p}": row[p - 1] for p in CLASS_LABELS},
            }
            for label, row in zip(CLASS_LABELS, metrics["confusion_matrix"])
        ],
        ["true_class", *[f"predicted_{label}" for label in CLASS_LABELS]],
    )
    step30a.write_csv(
        output_dir / "per_class_metrics.csv",
        [
            {"class": label, **metrics["per_class"][str(label)]}
            for label in CLASS_LABELS
        ],
        ["class", "precision", "recall", "f1", "support"],
    )
    serializable_diagnostics = {
        "residual_gate": diagnostic["residual_gate"],
        "correction_ratio": diagnostic["correction_ratio"],
        "checks": checks,
    }
    step30a.atomic_json(
        output_dir / "residual_diagnostics.json",
        to_json_safe(serializable_diagnostics),
    )
    step30a.atomic_json(output_dir / "test_metrics.json", to_json_safe(metrics))
    report_lines = [
        "BioKORF Fold-1 Residual View-Aware Experiment",
        "==============================================",
        f"Seed: {seed}",
        f"Batch size: {batch_size}",
        f"Best epoch: {checkpoint['epoch']}",
        f"Best validation Macro-F1: {checkpoint['validation_macro_f1']:.8f}",
        f"Test Accuracy: {metrics['accuracy']:.8f}",
        f"Test Macro Precision: {metrics['macro_precision']:.8f}",
        f"Test Macro Recall: {metrics['macro_recall']:.8f}",
        f"Test Macro-F1: {metrics['macro_f1']:.8f}",
        f"Test Micro-F1: {metrics['micro_f1']:.8f}",
        f"Test AUPR: {metrics['aupr']:.8f}",
        f"Test mean residual gate: {metrics['test_mean_residual_gate']:.8f}",
        f"Test mean correction ratio: {metrics['test_mean_correction_ratio']:.8f}",
        "",
        *[
            f"{name.replace('_', ' ').upper()} CHECK: {'PASS' if value else 'FAIL'}"
            for name, value in checks.items()
        ],
        "View pooling weights are descriptive and are not causal importance estimates.",
    ]
    report = "\n".join(report_lines) + "\n"
    (output_dir / "residual_viewaware_report.txt").write_text(report, encoding="utf-8")
    print(report, end="")
    print_safety(fair, label_safe, graph_safe, finite, smoke)
    if not all(checks.values()):
        raise RuntimeError("A required test safety check failed")
    return metrics


def print_clean_comparison(metrics: dict[str, Any], seed: int, batch_size: int) -> None:
    if seed != 42 or batch_size != 64:
        print("Stored CLEAN comparison is available only for seed 42, batch size 64.")
        return
    clean_path = CLEAN_RESULT_DIR / "test_metrics.json"
    if not clean_path.is_file():
        raise FileNotFoundError(f"Stored fair CLEAN result not found: {clean_path}")
    clean = json.loads(clean_path.read_text(encoding="utf-8"))
    if int(clean.get("seed", -1)) != seed or int(clean.get("batch_size", -1)) != batch_size:
        raise ValueError("Stored CLEAN result settings do not match this run")
    print("Model | Accuracy | Macro-F1 | AUPR")
    print("--- | ---: | ---: | ---:")
    print(f"CLEAN_BS64 | {clean['accuracy']:.8f} | {clean['macro_f1']:.8f} | {clean['aupr']:.8f}")
    print(
        f"RESIDUAL_VIEW_AWARE | {metrics['accuracy']:.8f} | "
        f"{metrics['macro_f1']:.8f} | {metrics['aupr']:.8f}"
    )
    print("Delta RESIDUAL_VIEW_AWARE - CLEAN")
    for key in ("accuracy", "macro_f1", "aupr"):
        print(f"{key}: {metrics[key] - clean[key]:+.8f}")


def smoke_mode(seed: int, batch_size: int) -> None:
    before = protected_hashes()
    train_data, _validation_data, _test_data, label_safe, graph_safe = step30a.load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smoke = smoke_checks(train_data, seed, device)
    fair = fairness_check(seed, batch_size) and unchanged(before)
    print_safety(fair, label_safe, graph_safe, True, smoke)
    if not all((fair, label_safe, graph_safe, *smoke.values())):
        raise RuntimeError("Residual view-aware smoke check failed")


def train_test_mode(seed: int, batch_size: int) -> None:
    output_dir = output_directory(seed, batch_size)
    require_new(
        [
            output_dir / "training_history.csv",
            output_dir / "best_checkpoint.pt",
            *expected_test_outputs(output_dir),
        ],
        "train_test",
    )
    training = train_mode(seed, batch_size)
    metrics = test_mode(seed, batch_size)
    print("\nRESIDUAL VIEW-AWARE TRAIN_TEST SUMMARY")
    print(f"Best epoch: {training['best_epoch']}")
    print(f"Best validation Macro-F1: {training['best_validation_macro_f1']:.8f}")
    print(f"Test Accuracy: {metrics['accuracy']:.8f}")
    print(f"Test Macro-F1: {metrics['macro_f1']:.8f}")
    print(f"Test AUPR: {metrics['aupr']:.8f}")
    print_clean_comparison(metrics, seed, batch_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", required=True, choices=("smoke", "train", "test", "train_test")
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
    print(f"Seed: {args.seed}; batch size: {args.batch_size}")
    if args.mode == "smoke":
        smoke_mode(args.seed, args.batch_size)
    elif args.mode == "train":
        train_mode(args.seed, args.batch_size)
    elif args.mode == "test":
        test_mode(args.seed, args.batch_size)
    else:
        train_test_mode(args.seed, args.batch_size)


if __name__ == "__main__":
    main()
