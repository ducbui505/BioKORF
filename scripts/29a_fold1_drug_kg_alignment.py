"""Fold-1 Drug-KG auxiliary alignment experiment for MSSF-clean."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PILOT_SCRIPT = PROJECT_ROOT / "scripts" / "26a_pilot_compare_clean_vs_kg.py"
SPLIT_PATH = PROJECT_ROOT / "data_processed" / "experiments" / "pilot_fold1_split.npz"
KG_ARTIFACT_PATH = PROJECT_ROOT / "data_processed" / "kg_features" / "biokorf_kg_embeddings.pt"
CLEAN_RESULTS_ROOT = PROJECT_ROOT / "data_processed" / "experiments" / "multiseed_fold1"
OUTPUT_ROOT = PROJECT_ROOT / "data_processed" / "experiments" / "kg_alignment_fold1"

DEFAULT_SEED = 42
MAX_EPOCHS = 30
PATIENCE = 7
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 128
DROPOUT = 0.4
LATENT_DIM = 64
DEFAULT_LAMBDA_ALIGN = 0.05
STEP29B_LAMBDAS = (0.005, 0.01, 0.02)
CLASS_LABELS = (1, 2, 3, 4, 5)

COMMON_PROTECTED_PATHS = (
    PROJECT_ROOT / "mssf.py",
    PROJECT_ROOT / "model.py",
    PROJECT_ROOT / "models" / "mssf_clean.py",
    SPLIT_PATH,
)
ALIGNMENT_PROTECTED_PATHS = (
    PROJECT_ROOT / "models" / "kg_fusion.py",
    PROJECT_ROOT / "models" / "mssf_clean_kg.py",
    PROJECT_ROOT / "models" / "kg_encoder.py",
    KG_ARTIFACT_PATH,
)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pilot = load_module("biokorf_pilot_helpers_29a", PILOT_SCRIPT)
sys.path.insert(0, str(PROJECT_ROOT))
from models.mssf_clean import MSSFClean, MSSFCleanConfig
from models.mssf_clean_kg_alignment import BioKORFCleanDrugKGAlignment


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protected_hashes(model_name: str) -> dict[Path, str]:
    paths = COMMON_PROTECTED_PATHS + (
        ALIGNMENT_PROTECTED_PATHS if model_name == "kg_alignment" else ()
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Protected experiment input not found: {path}")
    return {path: sha256(path) for path in paths}


def unchanged(before: dict[Path, str]) -> bool:
    return before == {path: sha256(path) for path in before}


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def lambda_folder_token(lambda_align: float) -> str:
    """Convert a finite non-negative lambda to a deterministic path token."""
    value = Decimal(str(lambda_align)).normalize()
    text = format(value, "f")
    return text.replace("-", "m").replace(".", "p")


def seed_directory(
    model_name: str, seed: int, batch_size: int, lambda_align: float
) -> Path:
    if model_name == "clean":
        return OUTPUT_ROOT / f"clean_seed{seed}_bs{batch_size}"
    # Preserve the completed Step 29A lambda=0.05 directory exactly.
    if Decimal(str(lambda_align)) == Decimal("0.05"):
        return OUTPUT_ROOT / f"seed{seed}"
    token = lambda_folder_token(lambda_align)
    return OUTPUT_ROOT / f"seed{seed}_lambda{token}_bs{batch_size}"


def require_new(paths: list[Path], operation: str) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            f"Refusing to {operation}; output already exists: " + ", ".join(existing)
        )


def load_fixed_split(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(SPLIT_PATH) as split:
        required = {
            "train_indices", "validation_indices", "test_indices", "train_samples",
            "validation_samples", "test_samples", "seed", "fold",
        }
        missing = required.difference(split.files)
        if missing:
            raise KeyError(f"Fixed split is missing arrays: {sorted(missing)}")
        if int(split["seed"]) != 42 or int(split["fold"]) != 1:
            raise ValueError("Fixed split metadata is not the established seed=42, fold=1")
        parts: list[np.ndarray] = []
        for name in ("train", "validation", "test"):
            saved = np.asarray(split[f"{name}_samples"])
            if not np.array_equal(samples[split[f"{name}_indices"]], saved):
                raise ValueError(f"Saved {name} samples do not match their indices")
            parts.append(saved.copy())
        combined = np.concatenate(
            (split["train_indices"], split["validation_indices"], split["test_indices"])
        )
        if len(combined) != len(samples) or len(np.unique(combined)) != len(samples):
            raise ValueError("Fixed split is not a complete disjoint partition")
    return tuple(parts)


def load_data() -> tuple[Any, Any, Any, bool, bool]:
    frequency_matrix = np.asarray(pilot.load_pickle("drug_side.pkl"))
    samples = pilot.original_positive_sample_order(frequency_matrix)
    train_samples, validation_samples, test_samples = load_fixed_split(samples)
    hidden = np.concatenate((validation_samples, test_samples), axis=0)
    drug_features, side_features, label_safe = pilot.build_leakage_safe_features(
        frequency_matrix, hidden
    )
    graph_safe = pilot.scan_graph_leakage()
    if not label_safe or not graph_safe:
        raise RuntimeError("A required leakage check failed")
    datasets = tuple(
        pilot.IndexedPairDataset(part, drug_features, side_features)
        for part in (train_samples, validation_samples, test_samples)
    )
    return (*datasets, bool(label_safe), bool(graph_safe))


def make_loader(
    dataset: Any, shuffle: bool, seed: int, batch_size: int
) -> torch.utils.data.DataLoader:
    """Build every experiment loader with the selected effective batch size."""
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def create_model(model_name: str, seed: int) -> nn.Module:
    pilot.configure_reproducibility(seed)
    config = MSSFCleanConfig(dropout=DROPOUT, gp=LATENT_DIM)
    if model_name == "clean":
        return MSSFClean(config)
    return BioKORFCleanDrugKGAlignment(
        config, KG_ARTIFACT_PATH
    )


def fairness_check(model_name: str, seed: int, batch_size: int) -> bool:
    config = MSSFCleanConfig(dropout=DROPOUT, gp=LATENT_DIM)
    pilot.configure_reproducibility(seed)
    clean = MSSFClean(config)
    candidate = create_model(model_name, seed)
    clean_state = clean.state_dict()
    same_clean_path = all(
        name in candidate.state_dict() and torch.equal(value, candidate.state_dict()[name])
        for name, value in clean_state.items()
    )
    same_policy = (
        LEARNING_RATE == pilot.LEARNING_RATE
        and WEIGHT_DECAY == pilot.WEIGHT_DECAY
        and batch_size > 0
    )
    frozen = model_name == "clean" or (
        not candidate.drug_kg_alignment.drug_embeddings.requires_grad
        and not candidate.drug_kg_alignment.drug_available_mask.requires_grad
    )
    return bool(same_clean_path and same_policy and frozen)


def finite_gradient(parameter: Tensor) -> bool:
    return parameter.grad is not None and bool(
        torch.isfinite(parameter.grad).all() and torch.any(parameter.grad != 0)
    )


def shared_representation_alignment_check(
    dataset: Any, seed: int, device: torch.device, batch_size: int
) -> bool:
    """Prove both losses reach the same prediction-path drug encoder."""
    probe = create_model("kg_alignment", seed).to(device)
    probe.eval()
    selected: tuple[Tensor, ...] | None = None
    for batch in make_loader(dataset, shuffle=False, seed=seed, batch_size=batch_size):
        candidate_indices = batch[2].to(device)
        if probe.drug_kg_alignment.drug_available_mask.index_select(
            0, candidate_indices
        ).any():
            selected = batch
            break
    if selected is None:
        raise RuntimeError("No KG-available training drug exists for gradient audit")
    drugs, sides, drug_index, _side_index, labels = selected
    drugs, sides = drugs.to(device), sides.to(device)
    drug_index, labels = drug_index.to(device), labels.to(device)
    shared_parameter = probe.preprocess.drug1_pre[0].weight
    projection_parameter = probe.drug_kg_alignment.projection[0].weight

    probe.zero_grad(set_to_none=True)
    _logits, _rec_con, _rec_add, _mu, _logvar, debug = probe(
        drugs, sides, drug_index, device=device, return_debug=True
    )
    debug["alignment_loss"].backward()
    alignment_reaches_projection = finite_gradient(projection_parameter)
    alignment_reaches_shared = finite_gradient(shared_parameter)
    kg_frozen = (
        probe.drug_kg_alignment.drug_embeddings.grad is None
        and probe.drug_kg_alignment.drug_available_mask.grad is None
        and not probe.drug_kg_alignment.drug_embeddings.requires_grad
    )

    probe.zero_grad(set_to_none=True)
    logits, rec_con, rec_add, mu, logvar = probe(drugs, sides, None, device=device)
    frequency_loss = pilot.composite_loss(
        logits, rec_con, rec_add, mu, logvar, labels, drugs, sides
    )
    frequency_loss.backward()
    frequency_reaches_shared = finite_gradient(shared_parameter)
    return bool(
        alignment_reaches_projection
        and alignment_reaches_shared
        and frequency_reaches_shared
        and kg_frozen
    )


def epoch_cosine(
    projected_by_drug: dict[int, list[np.ndarray]], kg_embeddings: np.ndarray
) -> tuple[float, int]:
    cosines: list[float] = []
    for drug_index, rows in projected_by_drug.items():
        projected = np.mean(np.vstack(rows), axis=0)
        target = kg_embeddings[drug_index]
        denominator = np.linalg.norm(projected) * np.linalg.norm(target)
        if denominator > 0:
            cosines.append(float(np.dot(projected, target) / denominator))
    return (float(np.mean(cosines)) if cosines else 0.0, len(cosines))


def collect_alignment_rows(
    store: dict[int, list[np.ndarray]], debug: dict[str, Tensor], drug_index: Tensor
) -> None:
    mask = debug["drug_kg_available_mask"].detach().cpu().numpy().astype(bool)
    projected = debug["projected_drug_representation"].detach().cpu().numpy()
    indices = drug_index.numpy()
    for index, row in zip(indices[mask], projected[mask]):
        store.setdefault(int(index), []).append(row)


def train_epoch(
    model: nn.Module,
    model_name: str,
    dataset: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    seed: int,
    lambda_align: float,
    batch_size: int,
) -> dict[str, float | int]:
    model.train()
    total, frequency_total, alignment_total, sample_count = 0.0, 0.0, 0.0, 0
    alignment_weight = 0
    projected_by_drug: dict[int, list[np.ndarray]] = {}
    for drugs, sides, drug_index, _side_index, labels in make_loader(
        dataset, shuffle=True, seed=seed + epoch, batch_size=batch_size
    ):
        drugs, sides, labels = drugs.to(device), sides.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        if model_name == "kg_alignment":
            logits, rec_con, rec_add, mu, logvar, debug = model(
                drugs, sides, drug_index.to(device), device=device, return_debug=True
            )
            alignment_loss = debug["alignment_loss"]
        else:
            logits, rec_con, rec_add, mu, logvar = model(drugs, sides, device=device)
            alignment_loss = logits.sum() * 0.0
        frequency_loss = pilot.composite_loss(
            logits, rec_con, rec_add, mu, logvar, labels, drugs, sides
        )
        loss = frequency_loss + lambda_align * alignment_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        count = int(labels.shape[0])
        unique_count = (
            int(debug["unique_available_drug_indices"].numel())
            if model_name == "kg_alignment"
            else 0
        )
        total += float(loss.detach()) * count
        frequency_total += float(frequency_loss.detach()) * count
        alignment_total += float(alignment_loss.detach()) * unique_count
        sample_count += count
        alignment_weight += unique_count
        if model_name == "kg_alignment":
            collect_alignment_rows(projected_by_drug, debug, drug_index)
    if model_name == "kg_alignment":
        kg = model.drug_kg_alignment.drug_embeddings.detach().cpu().numpy()
        cosine, unique_count = epoch_cosine(projected_by_drug, kg)
    else:
        cosine, unique_count = 0.0, 0
    return {
        "total_loss": total / sample_count,
        "frequency_loss": frequency_total / sample_count,
        "alignment_loss": alignment_total / max(alignment_weight, 1),
        "mean_cosine": cosine,
        "alignment_drug_count": unique_count,
    }


def evaluate(
    model: nn.Module,
    model_name: str,
    dataset: Any,
    device: torch.device,
    seed: int,
    batch_size: int,
    check_independence: bool = False,
) -> tuple[float, dict[str, Any], bool]:
    model.eval()
    loss_total, sample_count = 0.0, 0
    labels_all: list[np.ndarray] = []
    probabilities_all: list[np.ndarray] = []
    projected_by_drug: dict[int, list[np.ndarray]] = {}
    available_samples = 0
    independence_pass = True
    diagnostic_done = False
    with torch.inference_mode():
        for drugs, sides, drug_index, _side_index, labels in make_loader(
            dataset, shuffle=False, seed=seed, batch_size=batch_size
        ):
            drugs_device, sides_device = drugs.to(device), sides.to(device)
            if model_name == "kg_alignment":
                logits, rec_con, rec_add, mu, logvar, debug = model(
                    drugs_device,
                    sides_device,
                    drug_index.to(device),
                    device=device,
                    return_debug=True,
                )
            else:
                logits, rec_con, rec_add, mu, logvar = model(
                    drugs_device, sides_device, device=device
                )
            frequency_loss = pilot.composite_loss(
                logits,
                rec_con,
                rec_add,
                mu,
                logvar,
                labels.to(device),
                drugs_device,
                sides_device,
            )
            if not torch.isfinite(frequency_loss) or not torch.isfinite(logits).all():
                raise FloatingPointError("Non-finite validation/test value")
            probabilities = torch.softmax(logits, dim=1)
            if model_name == "kg_alignment" and check_independence and not diagnostic_done:
                original = model.drug_kg_alignment.drug_embeddings.clone()
                try:
                    model.drug_kg_alignment.drug_embeddings.zero_()
                    zero_logits, *_ = model(
                        drugs_device,
                        sides_device,
                        drug_index.to(device),
                        device=device,
                    )
                finally:
                    model.drug_kg_alignment.drug_embeddings.copy_(original)
                zero_probabilities = torch.softmax(zero_logits, dim=1)
                independence_pass = bool(
                    torch.equal(logits, zero_logits)
                    and torch.equal(probabilities, zero_probabilities)
                )
                diagnostic_done = True
            count = int(labels.shape[0])
            loss_total += float(frequency_loss) * count
            sample_count += count
            labels_all.append((labels.numpy() - 1).astype(np.int64))
            probabilities_all.append(probabilities.cpu().numpy())
            if model_name == "kg_alignment":
                available_samples += int(debug["drug_kg_available_mask"].sum())
                collect_alignment_rows(projected_by_drug, debug, drug_index)
    labels = np.concatenate(labels_all)
    probabilities_np = np.vstack(probabilities_all)
    metrics = pilot.classification_metrics(labels, probabilities_np)
    if model_name == "kg_alignment":
        kg = model.drug_kg_alignment.drug_embeddings.detach().cpu().numpy()
        cosine, unique_count = epoch_cosine(projected_by_drug, kg)
    else:
        cosine, unique_count = 0.0, 0
    metrics.update(
        {
            "mean_drug_kg_cosine": cosine,
            "unique_kg_available_drugs": unique_count,
            "kg_available_test_samples": available_samples,
            "kg_unavailable_test_samples": (
                sample_count - available_samples if model_name == "kg_alignment" else 0
            ),
        }
    )
    return loss_total / sample_count, metrics, independence_pass


def checkpoint_state(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def print_checks(
    fair: bool,
    label_safe: bool,
    graph_safe: bool,
    frozen: bool | None,
    shared: bool | None,
    independent: bool | None,
    finite: bool,
) -> None:
    print(f"EXPERIMENT FAIRNESS CHECK: {'PASS' if fair else 'FAIL'}")
    print(f"LABEL-DERIVED FEATURE LEAKAGE CHECK: {'PASS' if label_safe else 'FAIL'}")
    print(f"DRUG-PHENOTYPE LEAKAGE CHECK: {'PASS' if graph_safe else 'FAIL'}")
    print(f"FROZEN KG CHECK: {'N/A' if frozen is None else ('PASS' if frozen else 'FAIL')}")
    print(
        "SHARED-REPRESENTATION ALIGNMENT CHECK: "
        f"{'N/A' if shared is None else ('PASS' if shared else 'FAIL')}"
    )
    print(
        "KG-INFERENCE-INDEPENDENCE CHECK: "
        f"{'N/A' if independent is None else ('PASS' if independent else 'FAIL')}"
    )
    print(f"FINITE-VALUE CHECK: {'PASS' if finite else 'FAIL'}")


def train_mode(
    model_name: str, seed: int, lambda_align: float, batch_size: int
) -> dict[str, Any]:
    before = protected_hashes(model_name)
    output_dir = seed_directory(model_name, seed, batch_size, lambda_align)
    history_path = output_dir / "training_history.csv"
    checkpoint_path = output_dir / "best_checkpoint.pt"
    require_new([history_path, checkpoint_path], "train")
    train_data, validation_data, _test_data, label_safe, graph_safe = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fair = fairness_check(model_name, seed, batch_size)
    if model_name == "kg_alignment":
        shared: bool | None = shared_representation_alignment_check(
            train_data, seed, device, batch_size
        )
        frozen: bool | None = sha256(KG_ARTIFACT_PATH) == before[KG_ARTIFACT_PATH]
        effective_lambda = lambda_align
    else:
        shared, frozen, effective_lambda = None, None, 0.0
    required_checks = [fair, label_safe, graph_safe]
    if model_name == "kg_alignment":
        required_checks.extend((bool(shared), bool(frozen)))
    if not all(required_checks):
        raise RuntimeError("A required pre-training check failed")
    output_dir.mkdir(parents=True, exist_ok=True)
    model = create_model(model_name, seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    columns = [
        "epoch", "train_total_loss", "train_frequency_loss", "train_alignment_loss",
        "val_frequency_loss", "val_macro_f1", "val_accuracy", "val_aupr",
        "mean_train_drug_kg_cosine", "mean_val_drug_kg_cosine",
        "alignment_drug_count", "learning_rate",
        "cuda_peak_allocated_mib", "cuda_peak_reserved_mib",
    ]
    history: list[dict[str, Any]] = []
    best_epoch, best_f1, stale = 0, -1.0, 0
    finite = True
    for epoch in range(1, MAX_EPOCHS + 1):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        pilot.configure_reproducibility(seed + epoch)
        started = time.perf_counter()
        train_values = train_epoch(
            model,
            model_name,
            train_data,
            optimizer,
            device,
            epoch,
            seed,
            effective_lambda,
            batch_size,
        )
        val_loss, val_metrics, _unused = evaluate(
            model,
            model_name,
            validation_data,
            device,
            seed,
            batch_size,
            check_independence=False,
        )
        peak_allocated_mib = (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if torch.cuda.is_available()
            else 0.0
        )
        peak_reserved_mib = (
            torch.cuda.max_memory_reserved(device) / (1024**2)
            if torch.cuda.is_available()
            else 0.0
        )
        row = {
            "epoch": epoch,
            "train_total_loss": train_values["total_loss"],
            "train_frequency_loss": train_values["frequency_loss"],
            "train_alignment_loss": train_values["alignment_loss"],
            "val_frequency_loss": val_loss,
            "val_macro_f1": val_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_aupr": val_metrics["aupr"],
            "mean_train_drug_kg_cosine": train_values["mean_cosine"],
            "mean_val_drug_kg_cosine": val_metrics["mean_drug_kg_cosine"],
            "alignment_drug_count": train_values["alignment_drug_count"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "cuda_peak_allocated_mib": peak_allocated_mib,
            "cuda_peak_reserved_mib": peak_reserved_mib,
        }
        finite = finite and all(np.isfinite(value) for value in row.values())
        history.append(row)
        write_csv(history_path, history, columns)
        if val_metrics["macro_f1"] > best_f1:
            best_epoch, best_f1, stale = epoch, float(val_metrics["macro_f1"]), 0
            temporary = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(
                {
                    "model": (
                        "clean" if model_name == "clean" else "clean_drug_kg_alignment"
                    ),
                    "seed": seed,
                    "epoch": epoch,
                    "validation_macro_f1": best_f1,
                    "lambda_align": (
                        None if model_name == "clean" else effective_lambda
                    ),
                    "batch_size": batch_size,
                    "model_state_dict": checkpoint_state(model),
                    "selection_metric": "validation_macro_f1",
                },
                temporary,
            )
            temporary.replace(checkpoint_path)
        else:
            stale += 1
        print(
            f"epoch={epoch:02d} total={train_values['total_loss']:.6f} "
            f"frequency={train_values['frequency_loss']:.6f} "
            f"alignment={train_values['alignment_loss']:.6f} "
            f"val_macro_f1={val_metrics['macro_f1']:.6f} best={best_epoch} "
            f"patience={stale}/{PATIENCE} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
        if torch.cuda.is_available():
            print(
                f"CUDA peak allocated: {peak_allocated_mib:.2f} MiB; "
                f"peak reserved: {peak_reserved_mib:.2f} MiB",
                flush=True,
            )
        if stale >= PATIENCE:
            break
    fair = fair and unchanged(before)
    if model_name == "kg_alignment":
        frozen = bool(frozen) and sha256(KG_ARTIFACT_PATH) == before[KG_ARTIFACT_PATH]
    print_checks(
        fair,
        label_safe,
        graph_safe,
        frozen,
        shared,
        True if model_name == "kg_alignment" else None,
        finite,
    )
    final_checks = [fair, label_safe, graph_safe, finite]
    if model_name == "kg_alignment":
        final_checks.extend((bool(frozen), bool(shared)))
    if not all(final_checks):
        raise RuntimeError("A required training safety check failed")
    return {"best_epoch": best_epoch, "best_validation_macro_f1": best_f1}


def test_mode(
    model_name: str, seed: int, lambda_align: float, batch_size: int
) -> dict[str, Any]:
    before = protected_hashes(model_name)
    output_dir = seed_directory(model_name, seed, batch_size, lambda_align)
    checkpoint_path = output_dir / "best_checkpoint.pt"
    if not checkpoint_path.is_file() or not (output_dir / "training_history.csv").is_file():
        raise FileNotFoundError(f"Train mode must complete before test mode: {output_dir}")
    outputs = [
        output_dir / "test_metrics.json", output_dir / "confusion_matrix.csv",
        output_dir / "per_class_metrics.csv", output_dir / "alignment_diagnostics.json",
        output_dir / "kg_alignment_report.txt",
    ]
    require_new(outputs, "test")
    train_data, _validation_data, test_data, label_safe, graph_safe = load_data()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_checkpoint_model = (
        "clean" if model_name == "clean" else "clean_drug_kg_alignment"
    )
    if checkpoint.get("model") != expected_checkpoint_model or int(checkpoint.get("seed", -1)) != seed:
        raise ValueError("Checkpoint does not match the requested alignment run")
    if model_name == "kg_alignment" and abs(
        float(checkpoint["lambda_align"]) - lambda_align
    ) > 1e-12:
        raise ValueError("--lambda-align does not match the trained checkpoint")
    if int(checkpoint.get("batch_size", BATCH_SIZE)) != batch_size:
        raise ValueError("--batch-size does not match the trained checkpoint")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(model_name, seed)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)
    fair = fairness_check(model_name, seed, batch_size)
    shared = (
        shared_representation_alignment_check(train_data, seed, device, batch_size)
        if model_name == "kg_alignment"
        else None
    )
    _loss, metrics, independent = evaluate(
        model,
        model_name,
        test_data,
        device,
        seed,
        batch_size,
        check_independence=model_name == "kg_alignment",
    )
    if model_name == "kg_alignment":
        frozen: bool | None = sha256(KG_ARTIFACT_PATH) == before[KG_ARTIFACT_PATH] and all(
            buffer.grad is None and not buffer.requires_grad
            for buffer in (
                model.drug_kg_alignment.drug_embeddings,
                model.drug_kg_alignment.drug_available_mask,
            )
        )
    else:
        frozen, independent = None, None
    finite = all(
        np.isfinite(metrics[key])
        for key in ("accuracy", "macro_f1", "aupr", "mean_drug_kg_cosine")
    )
    checks = {
        "experiment_fairness": fair,
        "label_derived_feature_leakage": label_safe,
        "drug_phenotype_leakage": graph_safe,
        "finite_values": finite,
    }
    if model_name == "kg_alignment":
        checks.update(
            {
                "frozen_kg": frozen,
                "shared_representation_alignment": shared,
                "kg_inference_independence": independent,
            }
        )
    metrics.update(
        {
            "model": expected_checkpoint_model,
            "seed": seed,
            "best_epoch": int(checkpoint["epoch"]),
            "best_validation_macro_f1": float(checkpoint["validation_macro_f1"]),
            "lambda_align": None if model_name == "clean" else lambda_align,
            "batch_size": batch_size,
            "checks": checks,
        }
    )
    write_csv(
        output_dir / "confusion_matrix.csv",
        [{"true_class": label, **{f"predicted_{p}": row[p - 1] for p in CLASS_LABELS}} for label, row in zip(CLASS_LABELS, metrics["confusion_matrix"])],
        ["true_class", *[f"predicted_{label}" for label in CLASS_LABELS]],
    )
    write_csv(
        output_dir / "per_class_metrics.csv",
        [{"class": label, **metrics["per_class"][str(label)]} for label in CLASS_LABELS],
        ["class", "precision", "recall", "f1", "support"],
    )
    diagnostics = {
        "lambda_align": None if model_name == "clean" else lambda_align,
        "batch_size": batch_size,
        "mean_test_drug_kg_cosine": metrics["mean_drug_kg_cosine"],
        "unique_kg_available_test_drugs": metrics["unique_kg_available_drugs"],
        "kg_available_test_samples": metrics["kg_available_test_samples"],
        "kg_unavailable_test_samples": metrics["kg_unavailable_test_samples"],
        "checks": checks,
    }
    atomic_json(output_dir / "test_metrics.json", metrics)
    atomic_json(output_dir / "alignment_diagnostics.json", diagnostics)
    report_lines = [
        "BioKORF Fold-1 CLEAN / Drug Knowledge Alignment",
        "================================================",
        f"Model: {model_name}",
        f"Seed: {seed}",
        f"lambda_align: {'N/A' if model_name == 'clean' else lambda_align}",
        f"Batch size: {batch_size}",
        f"Best epoch: {checkpoint['epoch']}",
        f"Best validation Macro-F1: {checkpoint['validation_macro_f1']:.8f}",
        f"Accuracy: {metrics['accuracy']:.8f}",
        f"Macro-F1: {metrics['macro_f1']:.8f}",
        f"AUPR: {metrics['aupr']:.8f}",
        "",
        *[f"{name.replace('_', ' ').upper()} CHECK: {'PASS' if value else 'FAIL'}" for name, value in checks.items()],
        "No side-effect KG, direct KG fusion, ordinal learning, new attention, or R-GCN fine-tuning.",
    ]
    if model_name == "kg_alignment":
        report_lines[10:10] = [
            f"Mean test Drug-KG cosine: {metrics['mean_drug_kg_cosine']:.8f}",
            f"KG-available/unavailable test samples: {metrics['kg_available_test_samples']}/{metrics['kg_unavailable_test_samples']}",
        ]
    report = "\n".join(report_lines) + "\n"
    (output_dir / "kg_alignment_report.txt").write_text(report, encoding="utf-8")
    print(report, end="")
    print_checks(fair, label_safe, graph_safe, frozen, shared, independent, finite)
    if not all(checks.values()) or not unchanged(before):
        raise RuntimeError("A required test safety check failed")
    return metrics


def train_test_mode(
    model_name: str, seed: int, lambda_align: float, batch_size: int
) -> None:
    output_dir = seed_directory(model_name, seed, batch_size, lambda_align)
    require_new(
        [output_dir / name for name in ("training_history.csv", "best_checkpoint.pt", "test_metrics.json", "confusion_matrix.csv", "per_class_metrics.csv", "alignment_diagnostics.json", "kg_alignment_report.txt")],
        "train_test",
    )
    training = train_mode(model_name, seed, lambda_align, batch_size)
    metrics = test_mode(model_name, seed, lambda_align, batch_size)
    print("\nTRAIN_TEST SUMMARY")
    print(f"Model: {model_name}")
    print(f"Batch size: {batch_size}")
    print(f"Best epoch: {training['best_epoch']}")
    print(f"Best validation Macro-F1: {training['best_validation_macro_f1']:.8f}")
    print(f"Test Accuracy: {metrics['accuracy']:.8f}")
    print(f"Test Macro-F1: {metrics['macro_f1']:.8f}")
    print(f"Test AUPR: {metrics['aupr']:.8f}")
    if model_name == "kg_alignment":
        print(f"Mean test Drug-KG cosine: {metrics['mean_drug_kg_cosine']:.8f}")


def compare_mode(seed: int, batch_size: int, lambda_align: float) -> None:
    """Read and compare only runs with identical seed and batch size."""
    clean_path = (
        seed_directory("clean", seed, batch_size, lambda_align) / "test_metrics.json"
    )
    alignment_path = (
        seed_directory("kg_alignment", seed, batch_size, lambda_align)
        / "test_metrics.json"
    )
    for path in (clean_path, alignment_path):
        if not path.is_file():
            raise FileNotFoundError(f"Compare mode requires completed result: {path}")
    clean = json.loads(clean_path.read_text(encoding="utf-8"))
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    for name, result in (("CLEAN", clean), ("KG_ALIGNMENT", alignment)):
        if int(result.get("seed", -1)) != seed:
            raise ValueError(f"{name} result seed differs from requested seed {seed}")
        if int(result.get("batch_size", -1)) != batch_size:
            raise ValueError(
                f"{name} result batch size differs from requested batch size {batch_size}"
            )
    if clean.get("model") != "clean" or alignment.get("model") != "clean_drug_kg_alignment":
        raise ValueError("Compare inputs do not contain CLEAN and KG_ALIGNMENT respectively")
    print("Model | Accuracy | Macro-F1 | AUPR")
    print("--- | ---: | ---: | ---:")
    print(
        f"CLEAN | {clean['accuracy']:.8f} | {clean['macro_f1']:.8f} | "
        f"{clean['aupr']:.8f}"
    )
    print(
        f"KG_ALIGNMENT | {alignment['accuracy']:.8f} | "
        f"{alignment['macro_f1']:.8f} | {alignment['aupr']:.8f}"
    )
    print("Delta KG_ALIGNMENT - CLEAN")
    for metric in ("accuracy", "macro_f1", "aupr"):
        print(f"{metric}: {alignment[metric] - clean[metric]:+.8f}")
    print("Compare mode performed no training, testing, or file writes.")


def read_history(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Comparison requires training history: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Training history is empty: {path}")
    return rows


def compare_lambda_mode() -> None:
    """Read-only Step 29B comparison for seed 42 and batch size 64."""
    seed, batch_size = 42, 64
    clean_path = (
        seed_directory("clean", seed, batch_size, DEFAULT_LAMBDA_ALIGN)
        / "test_metrics.json"
    )
    if not clean_path.is_file():
        raise FileNotFoundError(f"CLEAN batch-64 result not found: {clean_path}")
    clean = json.loads(clean_path.read_text(encoding="utf-8"))
    if int(clean.get("seed", -1)) != seed or int(clean.get("batch_size", -1)) != batch_size:
        raise ValueError("Stored CLEAN result is not seed 42, batch size 64")

    lambda_values = (*STEP29B_LAMBDAS, DEFAULT_LAMBDA_ALIGN)
    alignment_results: list[dict[str, Any]] = []
    for lambda_value in lambda_values:
        directory = seed_directory("kg_alignment", seed, batch_size, lambda_value)
        metrics_path = directory / "test_metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(
                f"Lambda comparison requires completed result: {metrics_path}"
            )
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if int(metrics.get("seed", -1)) != seed:
            raise ValueError(f"lambda={lambda_value} result seed is not 42")
        if int(metrics.get("batch_size", -1)) != batch_size:
            raise ValueError(f"lambda={lambda_value} result batch size is not 64")
        if abs(float(metrics.get("lambda_align", -1)) - lambda_value) > 1e-12:
            raise ValueError(f"Stored lambda metadata differs for lambda={lambda_value}")
        if metrics.get("model") != "clean_drug_kg_alignment":
            raise ValueError(f"lambda={lambda_value} result is not KG_ALIGNMENT")
        if not all(metrics.get("checks", {}).values()):
            raise ValueError(f"lambda={lambda_value} has a failed saved safety check")
        history = read_history(directory / "training_history.csv")
        best_epoch = int(metrics["best_epoch"])
        matching = [row for row in history if int(row["epoch"]) == best_epoch]
        if len(matching) != 1:
            raise ValueError(
                f"lambda={lambda_value} history does not uniquely contain best epoch"
            )
        best_row, final_row = matching[0], history[-1]
        alignment_results.append(
            {
                "lambda": lambda_value,
                "metrics": metrics,
                "best_val_cosine": float(best_row["mean_val_drug_kg_cosine"]),
                "final_val_cosine": float(final_row["mean_val_drug_kg_cosine"]),
            }
        )

    print("Model/Lambda | Best epoch | Accuracy | Macro-F1 | AUPR | Mean KG cosine")
    print("--- | ---: | ---: | ---: | ---: | ---:")
    print(
        f"CLEAN | {clean['best_epoch']} | {clean['accuracy']:.8f} | "
        f"{clean['macro_f1']:.8f} | {clean['aupr']:.8f} | N/A"
    )
    for result in alignment_results:
        metrics = result["metrics"]
        print(
            f"lambda={result['lambda']} | {metrics['best_epoch']} | "
            f"{metrics['accuracy']:.8f} | {metrics['macro_f1']:.8f} | "
            f"{metrics['aupr']:.8f} | {metrics['mean_drug_kg_cosine']:.8f}"
        )

    print("\nDeltas relative to CLEAN:")
    for result in alignment_results:
        metrics = result["metrics"]
        print(
            f"lambda={result['lambda']}: "
            f"Accuracy={metrics['accuracy'] - clean['accuracy']:+.8f}, "
            f"Macro-F1={metrics['macro_f1'] - clean['macro_f1']:+.8f}, "
            f"AUPR={metrics['aupr'] - clean['aupr']:+.8f}"
        )

    print("\nAlignment-strength analysis (descriptive, not causal):")
    for result in alignment_results:
        metrics = result["metrics"]
        print(
            f"lambda={result['lambda']}: best-val cosine={result['best_val_cosine']:.8f}, "
            f"final-val cosine={result['final_val_cosine']:.8f}, "
            f"best validation Macro-F1={metrics['best_validation_macro_f1']:.8f}, "
            f"test Macro-F1={metrics['macro_f1']:.8f}"
        )

    best = max(alignment_results, key=lambda item: item["metrics"]["macro_f1"])
    any_beats_clean = any(
        item["metrics"]["macro_f1"] > clean["macro_f1"]
        for item in alignment_results
    )
    print(f"\nBest lambda by test Macro-F1: {best['lambda']}")
    print(f"Does any lambda beat CLEAN Macro-F1? {'YES' if any_beats_clean else 'NO'}")
    if not any_beats_clean:
        print("NO SMALL-LAMBDA ALIGNMENT IMPROVEMENT ON SEED 42")
    print("compare_lambda is read-only and makes no causal claim.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("train", "test", "train_test", "compare", "compare_lambda"),
    )
    parser.add_argument(
        "--model", choices=("clean", "kg_alignment"), default="kg_alignment"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--lambda-align", type=float, default=DEFAULT_LAMBDA_ALIGN)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.lambda_align < 0 or not np.isfinite(args.lambda_align):
        parser.error("--lambda-align must be finite and non-negative")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    is_step29b_lambda = any(
        Decimal(str(args.lambda_align)) == Decimal(str(value))
        for value in STEP29B_LAMBDAS
    )
    if (
        args.mode in ("train", "test", "train_test")
        and args.model == "kg_alignment"
        and is_step29b_lambda
        and (args.seed != 42 or args.batch_size != 64)
    ):
        parser.error("Step 29B lambdas require --seed 42 and --batch-size 64")
    return args


def main() -> None:
    args = parse_args()
    print(f"Model: {args.model}")
    print(f"Effective batch size: {args.batch_size}")
    print(
        "Effective lambda_align: "
        f"{'N/A' if args.model == 'clean' else args.lambda_align}"
    )
    if args.mode == "train":
        train_mode(args.model, args.seed, args.lambda_align, args.batch_size)
    elif args.mode == "test":
        test_mode(args.model, args.seed, args.lambda_align, args.batch_size)
    elif args.mode == "train_test":
        train_test_mode(args.model, args.seed, args.lambda_align, args.batch_size)
    elif args.mode == "compare":
        compare_mode(args.seed, args.batch_size, args.lambda_align)
    else:
        compare_lambda_mode()


if __name__ == "__main__":
    main()
