"""Fold-1 controlled experiment for sample-local view-aware MSSF."""

from __future__ import annotations

import argparse
import csv
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
PILOT_SCRIPT = PROJECT_ROOT / "scripts" / "26a_pilot_compare_clean_vs_kg.py"
SPLIT_PATH = PROJECT_ROOT / "data_processed" / "experiments" / "pilot_fold1_split.npz"
CLEAN_RESULT_DIR = (
    PROJECT_ROOT
    / "data_processed"
    / "experiments"
    / "kg_alignment_fold1"
    / "clean_seed42_bs64"
)
OUTPUT_ROOT = PROJECT_ROOT / "data_processed" / "experiments" / "viewaware_fold1"

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

PROTECTED_PATHS = (
    PROJECT_ROOT / "mssf.py",
    PROJECT_ROOT / "model.py",
    PROJECT_ROOT / "models" / "mssf_clean.py",
    SPLIT_PATH,
)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pilot = load_module("biokorf_pilot_helpers_30a", PILOT_SCRIPT)
sys.path.insert(0, str(PROJECT_ROOT))
from models.mssf_clean import MSSFClean, MSSFCleanConfig
from models.mssf_viewaware import BioKORFViewAware


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


def output_directory(seed: int, batch_size: int) -> Path:
    return OUTPUT_ROOT / f"seed{seed}_bs{batch_size}"


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
            raise ValueError("Fixed split metadata is not seed=42, fold=1")
        parts: list[np.ndarray] = []
        for name in ("train", "validation", "test"):
            saved = np.asarray(split[f"{name}_samples"])
            if not np.array_equal(samples[split[f"{name}_indices"]], saved):
                raise ValueError(f"Saved {name} samples do not match fixed indices")
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
    if tuple(drug_features.shape[1:]) != (757 * 11,):
        raise ValueError("Drug features do not encode 11 views of dimension 757")
    if tuple(side_features.shape[1:]) != (994 * 4,):
        raise ValueError("Side-effect features do not encode 4 views of dimension 994")
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
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def create_model(seed: int) -> BioKORFViewAware:
    pilot.configure_reproducibility(seed)
    return BioKORFViewAware(MSSFCleanConfig(dropout=DROPOUT, gp=LATENT_DIM))


def fairness_check(seed: int, batch_size: int) -> bool:
    config = MSSFCleanConfig(dropout=DROPOUT, gp=LATENT_DIM)
    pilot.configure_reproducibility(seed)
    clean = MSSFClean(config)
    candidate = create_model(seed)
    shared_prefixes = ("preprocess.", "crossProduction.", "gaussian_parametrizer.", "classifier.")
    clean_shared = {
        name: value
        for name, value in clean.state_dict().items()
        if name.startswith(shared_prefixes)
    }
    same_shared_modules = all(
        name in candidate.state_dict()
        and torch.equal(value, candidate.state_dict()[name])
        for name, value in clean_shared.items()
    )
    removed_old_branches = not hasattr(candidate, "encoderConnection") and not hasattr(
        candidate, "encoderAddition"
    )
    same_policy = (
        LEARNING_RATE == pilot.LEARNING_RATE
        and WEIGHT_DECAY == pilot.WEIGHT_DECAY
        and batch_size > 0
    )
    return bool(same_shared_modules and removed_old_branches and same_policy)


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


def debug_contracts(debug: dict[str, Tensor], batch_size: int) -> tuple[bool, bool]:
    drug_shape = tuple(debug["drug_view_attention_weights"].shape)
    side_shape = tuple(debug["side_view_attention_weights"].shape)
    print(f"Drug attention tensor shape: {drug_shape}")
    print(f"Side attention tensor shape: {side_shape}")
    axis_safe = drug_shape == (batch_size, 4, 11, 11) and side_shape == (
        batch_size, 4, 4, 4
    )
    pooling_safe = pooling_valid(debug["drug_view_pooling_weights"]) and pooling_valid(
        debug["side_view_pooling_weights"]
    )
    return axis_safe, pooling_safe


def stack_samples(dataset: Any, indices: list[int]) -> tuple[Tensor, Tensor]:
    samples = [dataset[index] for index in indices]
    return torch.stack([sample[0] for sample in samples]), torch.stack(
        [sample[1] for sample in samples]
    )


def smoke_checks(dataset: Any, seed: int, device: torch.device) -> tuple[bool, bool, bool]:
    if len(dataset) < 7:
        raise ValueError("At least seven training samples are required for smoke checks")
    model = create_model(seed).to(device).eval()
    drugs_a, sides_a = stack_samples(dataset, [0, 1, 2, 3])
    drugs_b, sides_b = stack_samples(dataset, [0, 4, 5, 6])
    with torch.inference_mode():
        _logits_a, _mu_a, _logvar_a, debug_a = model(
            drugs_a.to(device), sides_a.to(device), device=device, return_debug=True
        )
        _logits_b, _mu_b, _logvar_b, debug_b = model(
            drugs_b.to(device), sides_b.to(device), device=device, return_debug=True
        )
    axis_safe, pooling_safe = debug_contracts(debug_a, batch_size=4)
    batch_safe = all(
        torch.allclose(debug_a[key][0], debug_b[key][0], atol=TOLERANCE, rtol=0.0)
        for key in ("H_drug_view", "H_side_view", "H_pair_view")
    )
    print(f"VIEW-AXIS ATTENTION CHECK: {'PASS' if axis_safe else 'FAIL'}")
    print(f"VIEW BATCH-INDEPENDENCE CHECK: {'PASS' if batch_safe else 'FAIL'}")
    print(f"VIEW POOLING CHECK: {'PASS' if pooling_safe else 'FAIL'}")
    return axis_safe, batch_safe, pooling_safe


def prediction_loss(
    model: BioKORFViewAware, logits: Tensor, mu: Tensor, logvar: Tensor, labels: Tensor
) -> Tensor:
    classification = model.frequency_classification_loss(logits, labels)
    kl = (-0.5 * (1 + logvar - mu.square() - torch.exp(logvar))).sum(1).mean()
    return classification + 0.001 * kl


def train_epoch(
    model: BioKORFViewAware,
    dataset: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    seed: int,
    batch_size: int,
) -> float:
    model.train()
    total, count_total = 0.0, 0
    for drugs, sides, _drug_index, _side_index, labels in make_loader(
        dataset, shuffle=True, seed=seed + epoch, batch_size=batch_size
    ):
        drugs, sides, labels = drugs.to(device), sides.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, mu, logvar = model(drugs, sides, device=device)
        loss = prediction_loss(model, logits, mu, logvar, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        count = int(labels.shape[0])
        total += float(loss.detach()) * count
        count_total += count
    return total / count_total


def evaluate(
    model: BioKORFViewAware,
    dataset: Any,
    device: torch.device,
    seed: int,
    batch_size: int,
    collect_weights: bool = False,
) -> tuple[float, dict[str, Any], np.ndarray | None, np.ndarray | None]:
    model.eval()
    total, count_total = 0.0, 0
    labels_all: list[np.ndarray] = []
    probabilities_all: list[np.ndarray] = []
    drug_weights: list[np.ndarray] = []
    side_weights: list[np.ndarray] = []
    with torch.inference_mode():
        for drugs, sides, _drug_index, _side_index, labels in make_loader(
            dataset, shuffle=False, seed=seed, batch_size=batch_size
        ):
            drugs, sides = drugs.to(device), sides.to(device)
            logits, mu, logvar, debug = model(
                drugs, sides, device=device, return_debug=True
            )
            loss = prediction_loss(model, logits, mu, logvar, labels.to(device))
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
            if collect_weights:
                drug_weights.append(debug["drug_view_pooling_weights"].cpu().numpy())
                side_weights.append(debug["side_view_pooling_weights"].cpu().numpy())
    metrics = pilot.classification_metrics(
        np.concatenate(labels_all), np.vstack(probabilities_all)
    )
    return (
        total / count_total,
        metrics,
        np.vstack(drug_weights) if drug_weights else None,
        np.vstack(side_weights) if side_weights else None,
    )


def checkpoint_state(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def print_safety(
    fair: bool, label_safe: bool, graph_safe: bool, finite: bool,
    axis_safe: bool, batch_safe: bool, pooling_safe: bool,
) -> None:
    print(f"EXPERIMENT FAIRNESS CHECK: {'PASS' if fair else 'FAIL'}")
    print(f"LABEL-DERIVED FEATURE LEAKAGE CHECK: {'PASS' if label_safe else 'FAIL'}")
    print(f"DRUG-PHENOTYPE LEAKAGE CHECK: {'PASS' if graph_safe else 'FAIL'}")
    print(f"FINITE-VALUE CHECK: {'PASS' if finite else 'FAIL'}")
    print(f"VIEW-AXIS ATTENTION CHECK: {'PASS' if axis_safe else 'FAIL'}")
    print(f"VIEW BATCH-INDEPENDENCE CHECK: {'PASS' if batch_safe else 'FAIL'}")
    print(f"VIEW POOLING CHECK: {'PASS' if pooling_safe else 'FAIL'}")


def smoke_mode(seed: int, batch_size: int) -> None:
    before = protected_hashes()
    train_data, _validation_data, _test_data, label_safe, graph_safe = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    axis_safe, batch_safe, pooling_safe = smoke_checks(train_data, seed, device)
    fair = fairness_check(seed, batch_size) and unchanged(before)
    print_safety(fair, label_safe, graph_safe, True, axis_safe, batch_safe, pooling_safe)
    if not all((fair, label_safe, graph_safe, axis_safe, batch_safe, pooling_safe)):
        raise RuntimeError("View-aware smoke check failed")


def train_mode(seed: int, batch_size: int) -> dict[str, Any]:
    before = protected_hashes()
    output_dir = output_directory(seed, batch_size)
    history_path = output_dir / "training_history.csv"
    checkpoint_path = output_dir / "best_checkpoint.pt"
    require_new([history_path, checkpoint_path], "train")
    train_data, validation_data, _test_data, label_safe, graph_safe = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    axis_safe, batch_safe, pooling_safe = smoke_checks(train_data, seed, device)
    fair = fairness_check(seed, batch_size)
    if not all((fair, label_safe, graph_safe, axis_safe, batch_safe, pooling_safe)):
        raise RuntimeError("A required pre-training check failed")
    output_dir.mkdir(parents=True, exist_ok=True)
    model = create_model(seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    columns = ["epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1", "val_aupr"]
    history: list[dict[str, Any]] = []
    best_epoch, best_f1, stale = 0, -1.0, 0
    finite = True
    for epoch in range(1, MAX_EPOCHS + 1):
        pilot.configure_reproducibility(seed + epoch)
        started = time.perf_counter()
        train_loss = train_epoch(model, train_data, optimizer, device, epoch, seed, batch_size)
        val_loss, metrics, _drug_weights, _side_weights = evaluate(
            model, validation_data, device, seed, batch_size
        )
        row = {
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_accuracy": metrics["accuracy"], "val_macro_f1": metrics["macro_f1"],
            "val_aupr": metrics["aupr"],
        }
        finite = finite and all(np.isfinite(value) for value in row.values())
        history.append(row)
        write_csv(history_path, history, columns)
        if metrics["macro_f1"] > best_f1:
            best_epoch, best_f1, stale = epoch, float(metrics["macro_f1"]), 0
            temporary = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(
                {
                    "model": "view_aware", "seed": seed, "batch_size": batch_size,
                    "epoch": epoch, "validation_macro_f1": best_f1,
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
            f"val_macro_f1={metrics['macro_f1']:.6f} best={best_epoch} "
            f"patience={stale}/{PATIENCE} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
        if stale >= PATIENCE:
            break
    fair = fair and unchanged(before)
    print_safety(fair, label_safe, graph_safe, finite, axis_safe, batch_safe, pooling_safe)
    if not all((fair, label_safe, graph_safe, finite, axis_safe, batch_safe, pooling_safe)):
        raise RuntimeError("A required training safety check failed")
    return {"best_epoch": best_epoch, "best_validation_macro_f1": best_f1}


def weight_summary(path: Path, weights: np.ndarray, prefix: str) -> None:
    rows = [
        {
            "view_index": index,
            "view_name": f"{prefix}_view_{index + 1}",
            "mean_pooling_weight": float(weights[:, index].mean()),
            "std_pooling_weight": float(weights[:, index].std()),
        }
        for index in range(weights.shape[1])
    ]
    write_csv(
        path, rows,
        ["view_index", "view_name", "mean_pooling_weight", "std_pooling_weight"],
    )


def test_mode(seed: int, batch_size: int) -> dict[str, Any]:
    before = protected_hashes()
    output_dir = output_directory(seed, batch_size)
    checkpoint_path = output_dir / "best_checkpoint.pt"
    if not checkpoint_path.is_file() or not (output_dir / "training_history.csv").is_file():
        raise FileNotFoundError(f"Train mode must complete before test mode: {output_dir}")
    outputs = [
        output_dir / "test_metrics.json", output_dir / "confusion_matrix.csv",
        output_dir / "per_class_metrics.csv", output_dir / "drug_view_weight_summary.csv",
        output_dir / "side_view_weight_summary.csv", output_dir / "viewaware_report.txt",
    ]
    require_new(outputs, "test")
    train_data, _validation_data, test_data, label_safe, graph_safe = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    axis_safe, batch_safe, pooling_safe = smoke_checks(train_data, seed, device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model") != "view_aware" or int(checkpoint.get("seed", -1)) != seed:
        raise ValueError("Checkpoint does not match the requested view-aware run")
    if int(checkpoint.get("batch_size", -1)) != batch_size:
        raise ValueError("Checkpoint batch size differs from --batch-size")
    model = create_model(seed)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)
    _loss, metrics, drug_weights, side_weights = evaluate(
        model, test_data, device, seed, batch_size, collect_weights=True
    )
    if drug_weights is None or side_weights is None:
        raise RuntimeError("Test view weights were not collected")
    weight_summary(output_dir / "drug_view_weight_summary.csv", drug_weights, "drug")
    weight_summary(output_dir / "side_view_weight_summary.csv", side_weights, "side")
    finite = all(np.isfinite(metrics[key]) for key in ("accuracy", "macro_f1", "aupr"))
    fair = fairness_check(seed, batch_size) and unchanged(before)
    checks = {
        "experiment_fairness": fair,
        "label_derived_feature_leakage": label_safe,
        "drug_phenotype_leakage": graph_safe,
        "finite_values": finite,
        "view_axis_attention": axis_safe,
        "view_batch_independence": batch_safe,
        "view_pooling": pooling_safe,
    }
    metrics.update(
        {
            "model": "view_aware", "seed": seed, "batch_size": batch_size,
            "best_epoch": int(checkpoint["epoch"]),
            "best_validation_macro_f1": float(checkpoint["validation_macro_f1"]),
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
    atomic_json(output_dir / "test_metrics.json", metrics)
    report_lines = [
        "BioKORF Fold-1 View-Aware Experiment",
        "====================================",
        f"Seed: {seed}", f"Batch size: {batch_size}",
        f"Best epoch: {checkpoint['epoch']}",
        f"Best validation Macro-F1: {checkpoint['validation_macro_f1']:.8f}",
        f"Test Accuracy: {metrics['accuracy']:.8f}",
        f"Test Macro-F1: {metrics['macro_f1']:.8f}",
        f"Test AUPR: {metrics['aupr']:.8f}", "",
        *[f"{name.replace('_', ' ').upper()} CHECK: {'PASS' if value else 'FAIL'}" for name, value in checks.items()],
        "Pooling weights are descriptive and are not interpreted as causal importance.",
    ]
    report = "\n".join(report_lines) + "\n"
    (output_dir / "viewaware_report.txt").write_text(report, encoding="utf-8")
    print(report, end="")
    print_safety(fair, label_safe, graph_safe, finite, axis_safe, batch_safe, pooling_safe)
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
    print(f"VIEW_AWARE | {metrics['accuracy']:.8f} | {metrics['macro_f1']:.8f} | {metrics['aupr']:.8f}")
    print("Delta VIEW_AWARE - CLEAN")
    for key in ("accuracy", "macro_f1", "aupr"):
        print(f"{key}: {metrics[key] - clean[key]:+.8f}")


def train_test_mode(seed: int, batch_size: int) -> None:
    output_dir = output_directory(seed, batch_size)
    require_new(
        [output_dir / name for name in ("training_history.csv", "best_checkpoint.pt", "test_metrics.json", "confusion_matrix.csv", "per_class_metrics.csv", "drug_view_weight_summary.csv", "side_view_weight_summary.csv", "viewaware_report.txt")],
        "train_test",
    )
    training = train_mode(seed, batch_size)
    metrics = test_mode(seed, batch_size)
    print("\nVIEW_AWARE TRAIN_TEST SUMMARY")
    print(f"Best epoch: {training['best_epoch']}")
    print(f"Best validation Macro-F1: {training['best_validation_macro_f1']:.8f}")
    print(f"Test Accuracy: {metrics['accuracy']:.8f}")
    print(f"Test Macro-F1: {metrics['macro_f1']:.8f}")
    print(f"Test AUPR: {metrics['aupr']:.8f}")
    print_clean_comparison(metrics, seed, batch_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("smoke", "train", "test", "train_test"))
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
