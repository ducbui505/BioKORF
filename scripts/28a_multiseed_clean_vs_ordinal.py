"""Multi-seed Fold-1 comparison of MSSF CLEAN and CLEAN_ORDINAL.

Only the ordinal model is trained here.  CLEAN metrics are read from the
completed Step 27A runs, and compare mode never trains or evaluates a model.
"""

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
CLEAN_RESULTS_ROOT = PROJECT_ROOT / "data_processed" / "experiments" / "multiseed_fold1"
OUTPUT_ROOT = PROJECT_ROOT / "data_processed" / "experiments" / "ordinal_fold1"

SEEDS = (42, 123, 2026)
MAX_EPOCHS = 30
PATIENCE = 7
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 128
DROPOUT = 0.4
LATENT_DIM = 64
CLASS_LABELS = (1, 2, 3, 4, 5)
MONOTONICITY_TOLERANCE = 1e-6

PROTECTED_HASHES = {
    PROJECT_ROOT / "mssf.py": "4867fecd04beabb2d715b24073f82a46bd572c13294afa3565ddba99f963fdb1",
    PROJECT_ROOT / "model.py": "9c0d4bf17551a7d0f881a29e0f8e2727227f3561678064fec46f2848156a1e75",
    PROJECT_ROOT / "models" / "mssf_clean.py": "f2a0f68e062807cacc77540c14afd5bf0e66eb7571b76b99e095c4063b8dd6d2",
    PROJECT_ROOT / "models" / "mssf_clean_kg.py": "cbc505f64c718cb5ce861fd4eac1d4d5d7f6eaefb0b045059271aab79bf92b81",
    PROJECT_ROOT / "models" / "kg_fusion.py": "9c9ce093d5e86078e32cd11e8696d0dc37625f8980c3cdb66167b2a893ac7c0f",
    PROJECT_ROOT / "models" / "kg_encoder.py": "0e81eb5f31299f46ac64052ea929811dccb7fccb5e203e7f50f833fab375e464",
    SPLIT_PATH: "d6f4bb9854bca7296372ea580d07acc9dfe1e2bd3ed8fa6905a7bdb84a7e5575",
}


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pilot = load_module("biokorf_pilot_helpers_28a", PILOT_SCRIPT)
sys.path.insert(0, str(PROJECT_ROOT))
from models.mssf_clean import MSSFClean, MSSFCleanConfig
from models.mssf_clean_ordinal import BioKORFCleanOrdinal
from models.ordinal_head import OrdinalCumulativeHead


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_protected_inputs() -> dict[Path, str]:
    for path in (PILOT_SCRIPT, *PROTECTED_HASHES):
        if not path.is_file():
            raise FileNotFoundError(f"Required experiment input not found: {path}")
    observed = {path: sha256(path) for path in PROTECTED_HASHES}
    changed = [str(path) for path, expected in PROTECTED_HASHES.items() if observed[path] != expected]
    if changed:
        raise RuntimeError("Protected baseline differs from its established hash: " + ", ".join(changed))
    return observed


def validate_unchanged(before: dict[Path, str]) -> bool:
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


def seed_directory(seed: int) -> Path:
    return OUTPUT_ROOT / f"seed{seed}"


def require_new_outputs(paths: list[Path], operation: str) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            f"Refusing to {operation}; output already exists (remove it explicitly to rerun): "
            + ", ".join(existing)
        )


def load_fixed_split(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not SPLIT_PATH.is_file():
        raise FileNotFoundError(f"Required fixed Fold-1 split not found: {SPLIT_PATH}")
    with np.load(SPLIT_PATH) as split:
        required = {
            "train_indices", "validation_indices", "test_indices", "train_samples",
            "validation_samples", "test_samples", "seed", "fold",
        }
        missing = required.difference(split.files)
        if missing:
            raise KeyError(f"Saved split is missing arrays: {sorted(missing)}")
        if int(split["seed"]) != 42 or int(split["fold"]) != 1:
            raise ValueError("Saved split metadata is not the established seed=42, fold=1 split")
        for prefix in ("train", "validation", "test"):
            if not np.array_equal(samples[split[f"{prefix}_indices"]], split[f"{prefix}_samples"]):
                raise ValueError(f"Saved {prefix} samples do not match their fixed indices")
        all_indices = np.concatenate(
            (split["train_indices"], split["validation_indices"], split["test_indices"])
        )
        if len(all_indices) != len(samples) or len(np.unique(all_indices)) != len(samples):
            raise ValueError("Saved split is not a complete disjoint sample partition")
        return tuple(
            np.asarray(split[f"{prefix}_samples"]).copy()
            for prefix in ("train", "validation", "test")
        )


def validate_label_convention(samples: np.ndarray) -> None:
    labels = np.asarray(samples[:, 2])
    if not np.all(np.isfinite(labels)) or not np.all(labels == np.round(labels)):
        raise ValueError("Frequency labels must be finite integers")
    if labels.min() < 1 or labels.max() > 5:
        raise ValueError("Inspected BioKORF pipeline requires stored frequency labels in 1..5")


def load_experiment_data() -> tuple[Any, Any, Any, bool, bool]:
    frequency_matrix = np.asarray(pilot.load_pickle("drug_side.pkl"))
    samples = pilot.original_positive_sample_order(frequency_matrix)
    validate_label_convention(samples)
    train_samples, validation_samples, test_samples = load_fixed_split(samples)
    hidden_samples = np.concatenate((validation_samples, test_samples), axis=0)
    drug_features, side_features, label_safe = pilot.build_leakage_safe_features(
        frequency_matrix, hidden_samples
    )
    graph_safe = pilot.scan_graph_leakage()
    if not label_safe or not graph_safe:
        raise RuntimeError("A required leakage check failed")
    datasets = tuple(
        pilot.IndexedPairDataset(part, drug_features, side_features)
        for part in (train_samples, validation_samples, test_samples)
    )
    return (*datasets, bool(label_safe), bool(graph_safe))


def create_model(seed: int) -> BioKORFCleanOrdinal:
    pilot.configure_reproducibility(seed)
    return BioKORFCleanOrdinal(MSSFCleanConfig(dropout=DROPOUT, gp=LATENT_DIM))


def experiment_fairness_check(seed: int) -> bool:
    config = MSSFCleanConfig(dropout=DROPOUT, gp=LATENT_DIM)
    pilot.configure_reproducibility(seed)
    clean = MSSFClean(config)
    pilot.configure_reproducibility(seed)
    ordinal = BioKORFCleanOrdinal(config)
    clean_backbone = {
        name: value for name, value in clean.state_dict().items() if not name.startswith("classifier.")
    }
    same_backbone = all(
        name in ordinal.state_dict() and torch.equal(value, ordinal.state_dict()[name])
        for name, value in clean_backbone.items()
    )
    intended_heads = (
        any(name.startswith("classifier.") for name in clean.state_dict())
        and not any(name.startswith("classifier.") for name in ordinal.state_dict())
        and any(name.startswith("ordinal_head.") for name in ordinal.state_dict())
    )
    same_policy = (
        LEARNING_RATE == pilot.LEARNING_RATE
        and WEIGHT_DECAY == pilot.WEIGHT_DECAY
        and BATCH_SIZE == pilot.BATCH_SIZE
    )
    return bool(same_backbone and intended_heads and same_policy)


def ordinal_composite_loss(
    model: BioKORFCleanOrdinal,
    ordinal_logits: Tensor,
    rec_con: Tensor,
    rec_add: Tensor,
    mu: Tensor,
    logvar: Tensor,
    labels: Tensor,
    drug_features: Tensor,
    side_features: Tensor,
) -> Tensor:
    classification = model.frequency_classification_loss(ordinal_logits, labels)
    kl_divergence = (-0.5 * (1 + logvar - mu.square() - torch.exp(logvar))).sum(1).mean()
    rec_connection_target = torch.cat((drug_features, side_features), dim=1)
    drug_sum = torch.stack(drug_features.chunk(11, dim=1), dim=0).sum(dim=0)
    side_sum = torch.stack(side_features.chunk(4, dim=1), dim=0).sum(dim=0)
    rec_addition_target = torch.cat((drug_sum, side_sum), dim=1)
    rec_con_loss = nn.functional.mse_loss(
        rec_con, rec_connection_target, reduction="none"
    ).sum(dim=-1).mean()
    rec_add_loss = nn.functional.mse_loss(
        rec_add, rec_addition_target, reduction="none"
    ).sum(dim=-1).mean()
    return classification + 0.001 * kl_divergence + 0.0001 * rec_con_loss + 0.0001 * rec_add_loss


def train_one_epoch(
    model: BioKORFCleanOrdinal,
    dataset: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    seed: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for drugs, sides, _drug_index, _side_index, labels in pilot.make_loader(
        dataset, shuffle=True, seed=seed + epoch
    ):
        drugs = drugs.to(device, non_blocking=True)
        sides = sides.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        ordinal_logits, rec_con, rec_add, mu, logvar = model(drugs, sides, device=device)
        loss = ordinal_composite_loss(
            model, ordinal_logits, rec_con, rec_add, mu, logvar, labels, drugs, sides
        )
        loss.backward()
        optimizer.step()
        count = int(labels.shape[0])
        total_loss += float(loss.detach()) * count
        total_samples += count
    return total_loss / total_samples


def quadratic_weighted_kappa(labels: np.ndarray, predictions: np.ndarray) -> float:
    observed = np.zeros((5, 5), dtype=np.float64)
    np.add.at(observed, (labels, predictions), 1)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    indices = np.arange(5, dtype=np.float64)
    weights = ((indices[:, None] - indices[None, :]) ** 2) / 16.0
    denominator = float((weights * expected).sum())
    numerator = float((weights * observed).sum())
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else 0.0
    return 1.0 - numerator / denominator


def evaluate(
    model: BioKORFCleanOrdinal, dataset: Any, device: torch.device, seed: int
) -> tuple[float, dict[str, Any], bool]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    labels_all: list[np.ndarray] = []
    probabilities_all: list[np.ndarray] = []
    monotonicity_pass = True
    with torch.inference_mode():
        for drugs, sides, _drug_index, _side_index, labels in pilot.make_loader(
            dataset, shuffle=False, seed=seed
        ):
            drugs = drugs.to(device, non_blocking=True)
            sides = sides.to(device, non_blocking=True)
            labels_device = labels.to(device, non_blocking=True)
            ordinal_logits, rec_con, rec_add, mu, logvar, debug = model(
                drugs, sides, device=device, return_debug=True
            )
            cumulative = debug["cumulative_probabilities"]
            probabilities = debug["class_probabilities"]
            batch_valid = OrdinalCumulativeHead.validate_probabilities(
                cumulative, probabilities, MONOTONICITY_TOLERANCE
            )
            monotonicity_pass = monotonicity_pass and batch_valid
            loss = ordinal_composite_loss(
                model,
                ordinal_logits,
                rec_con,
                rec_add,
                mu,
                logvar,
                labels_device,
                drugs,
                sides,
            )
            count = int(labels.shape[0])
            total_loss += float(loss) * count
            total_samples += count
            labels_all.append((labels.numpy() - 1).astype(np.int64))
            probabilities_all.append(probabilities.cpu().numpy())
    labels_array = np.concatenate(labels_all)
    probabilities_array = np.vstack(probabilities_all)
    result = pilot.classification_metrics(labels_array, probabilities_array)
    predictions = probabilities_array.argmax(axis=1)
    result["mae"] = float(np.mean(np.abs(labels_array - predictions)))
    result["quadratic_weighted_kappa"] = quadratic_weighted_kappa(labels_array, predictions)
    result["ordinal_monotonicity"] = monotonicity_pass
    if not monotonicity_pass:
        raise RuntimeError("Ordinal monotonicity or class-probability validation failed")
    return total_loss / total_samples, result, monotonicity_pass


def cpu_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def print_checks(fair: bool, label_safe: bool, graph_safe: bool, monotonic: bool) -> None:
    print(f"EXPERIMENT FAIRNESS CHECK: {'PASS' if fair else 'FAIL'}")
    print(f"LABEL-DERIVED FEATURE LEAKAGE CHECK: {'PASS' if label_safe else 'FAIL'}")
    print(f"DRUG-PHENOTYPE LEAKAGE CHECK: {'PASS' if graph_safe else 'FAIL'}")
    print(f"ORDINAL MONOTONICITY CHECK: {'PASS' if monotonic else 'FAIL'}")


def train_mode(seed: int) -> dict[str, float | int]:
    before = validate_protected_inputs()
    output_dir = seed_directory(seed)
    history_path = output_dir / "training_history.csv"
    checkpoint_path = output_dir / "best_checkpoint.pt"
    require_new_outputs([history_path, checkpoint_path], "train")
    train_data, validation_data, _test_data, label_safe, graph_safe = load_experiment_data()
    fair = experiment_fairness_check(seed)
    if not fair:
        raise RuntimeError("Experiment fairness check failed before training")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(seed).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    history: list[dict[str, Any]] = []
    columns = ["epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1", "val_aupr", "ordinal_monotonicity"]
    best_epoch = 0
    best_macro_f1 = -1.0
    stale_epochs = 0
    all_monotonic = True
    print(f"Training CLEAN_ORDINAL seed={seed}, device={device}; fixed split={SPLIT_PATH}")
    for epoch in range(1, MAX_EPOCHS + 1):
        pilot.configure_reproducibility(seed + epoch)
        started = time.perf_counter()
        train_loss = train_one_epoch(model, train_data, optimizer, device, epoch, seed)
        val_loss, metrics, monotonic = evaluate(model, validation_data, device, seed)
        all_monotonic = all_monotonic and monotonic
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": metrics["accuracy"],
                "val_macro_f1": metrics["macro_f1"],
                "val_aupr": metrics["aupr"],
                "ordinal_monotonicity": monotonic,
            }
        )
        write_csv(history_path, history, columns)
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = float(metrics["macro_f1"])
            best_epoch = epoch
            stale_epochs = 0
            temporary = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(
                {
                    "model": "clean_ordinal",
                    "seed": seed,
                    "epoch": epoch,
                    "best_epoch": epoch,
                    "validation_macro_f1": best_macro_f1,
                    "model_state_dict": cpu_state_dict(model),
                    "selection_metric": "validation_macro_f1",
                    "optimizer": {"name": "Adam", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY},
                    "batch_size": BATCH_SIZE,
                },
                temporary,
            )
            temporary.replace(checkpoint_path)
        else:
            stale_epochs += 1
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"val_macro_f1={metrics['macro_f1']:.6f} best_epoch={best_epoch} "
            f"patience={stale_epochs}/{PATIENCE} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
        if stale_epochs >= PATIENCE:
            break
    originals_safe = validate_unchanged(before)
    fair = fair and originals_safe
    print_checks(fair, label_safe, graph_safe, all_monotonic)
    if not all((fair, label_safe, graph_safe, all_monotonic)):
        raise RuntimeError("A required final safety check failed")
    return {
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_macro_f1,
    }


def test_mode(seed: int) -> dict[str, Any]:
    before = validate_protected_inputs()
    output_dir = seed_directory(seed)
    checkpoint_path = output_dir / "best_checkpoint.pt"
    history_path = output_dir / "training_history.csv"
    if not checkpoint_path.is_file() or not history_path.is_file():
        raise FileNotFoundError(f"Train mode must complete before test mode: {output_dir}")
    metrics_path = output_dir / "test_metrics.json"
    confusion_path = output_dir / "confusion_matrix.csv"
    per_class_path = output_dir / "per_class_metrics.csv"
    require_new_outputs([metrics_path, confusion_path, per_class_path], "test")
    _train_data, _validation_data, test_data, label_safe, graph_safe = load_experiment_data()
    fair = experiment_fairness_check(seed)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model") != "clean_ordinal" or int(checkpoint.get("seed", -1)) != seed:
        raise ValueError("Checkpoint does not match the requested ordinal seed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(seed)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)
    _loss, metrics, monotonic = evaluate(model, test_data, device, seed)
    metrics.update(
        {
            "model": "clean_ordinal",
            "seed": seed,
            "best_epoch": int(checkpoint["epoch"]),
            "best_validation_macro_f1": float(checkpoint["validation_macro_f1"]),
            "selection_metric": "validation_macro_f1",
            "test_evaluation_policy": "one evaluation after validation-only model selection",
            "checks": {
                "experiment_fairness": fair,
                "label_derived_feature_leakage": label_safe,
                "drug_phenotype_leakage": graph_safe,
                "ordinal_monotonicity": monotonic,
            },
        }
    )
    write_csv(
        confusion_path,
        [
            {"true_class": true_label, **{f"predicted_{label}": row[label - 1] for label in CLASS_LABELS}}
            for true_label, row in zip(CLASS_LABELS, metrics["confusion_matrix"])
        ],
        ["true_class", *[f"predicted_{label}" for label in CLASS_LABELS]],
    )
    write_csv(
        per_class_path,
        [{"class": label, **metrics["per_class"][str(label)]} for label in CLASS_LABELS],
        ["class", "precision", "recall", "f1", "support"],
    )
    atomic_json(metrics_path, metrics)
    fair = fair and validate_unchanged(before)
    print(
        f"CLEAN_ORDINAL seed={seed}: Accuracy={metrics['accuracy']:.8f} "
        f"Macro-F1={metrics['macro_f1']:.8f} AUPR={metrics['aupr']:.8f} "
        f"MAE={metrics['mae']:.8f} QWK={metrics['quadratic_weighted_kappa']:.8f}"
    )
    print_checks(fair, label_safe, graph_safe, monotonic)
    if not all((fair, label_safe, graph_safe, monotonic)):
        raise RuntimeError("A required final safety check failed")
    return metrics


def train_test_mode(seed: int) -> None:
    """Finish validation-only training, then reload and test exactly once."""
    output_dir = seed_directory(seed)
    require_new_outputs(
        [
            output_dir / "training_history.csv",
            output_dir / "best_checkpoint.pt",
            output_dir / "test_metrics.json",
            output_dir / "confusion_matrix.csv",
            output_dir / "per_class_metrics.csv",
        ],
        "train_test",
    )

    # train_mode has no test-set evaluation.  Only after it has returned do we
    # enter test_mode, which reloads best_checkpoint.pt and evaluates once.
    training = train_mode(seed)
    test_metrics = test_mode(seed)
    print("\nCLEAN_ORDINAL TRAIN_TEST SUMMARY")
    print(f"Best epoch: {training['best_epoch']}")
    print(
        "Best validation Macro-F1: "
        f"{training['best_validation_macro_f1']:.8f}"
    )
    print(f"Test Accuracy: {test_metrics['accuracy']:.8f}")
    print(f"Test Macro-F1: {test_metrics['macro_f1']:.8f}")
    print(f"Test AUPR: {test_metrics['aupr']:.8f}")
    print(f"Test MAE: {test_metrics['mae']:.8f}")
    print(
        "Test Quadratic Weighted Kappa: "
        f"{test_metrics['quadratic_weighted_kappa']:.8f}"
    )


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1))


def compare_mode() -> None:
    clean: dict[int, dict[str, Any]] = {}
    ordinal: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        clean_path = CLEAN_RESULTS_ROOT / f"clean_seed{seed}" / "test_metrics.json"
        ordinal_path = seed_directory(seed) / "test_metrics.json"
        for path in (clean_path, ordinal_path):
            if not path.is_file():
                raise FileNotFoundError(f"Compare mode requires completed result: {path}")
        clean[seed] = json.loads(clean_path.read_text(encoding="utf-8"))
        ordinal[seed] = json.loads(ordinal_path.read_text(encoding="utf-8"))

    metric_names = ("accuracy", "macro_f1", "aupr")
    summary_rows: list[dict[str, Any]] = []
    report = [
        "BioKORF CLEAN versus CLEAN_ORDINAL multi-seed comparison",
        "========================================================",
        f"Seeds: {', '.join(map(str, SEEDS))}",
        "Model | Accuracy mean+/-std | Macro-F1 mean+/-std | AUPR mean+/-std",
        "--- | ---: | ---: | ---:",
    ]
    for model_name, results in (("clean", clean), ("clean_ordinal", ordinal)):
        aggregates = {
            metric: mean_std([float(results[seed][metric]) for seed in SEEDS])
            for metric in metric_names
        }
        summary_rows.append(
            {
                "row_type": "model_summary", "model": model_name, "seed": "",
                "accuracy": aggregates["accuracy"][0], "accuracy_std": aggregates["accuracy"][1],
                "macro_f1": aggregates["macro_f1"][0], "macro_f1_std": aggregates["macro_f1"][1],
                "aupr": aggregates["aupr"][0], "aupr_std": aggregates["aupr"][1],
                "mae": "" if model_name == "clean" else mean_std([float(results[s]["mae"]) for s in SEEDS])[0],
                "quadratic_weighted_kappa": "" if model_name == "clean" else mean_std([float(results[s]["quadratic_weighted_kappa"]) for s in SEEDS])[0],
                "ordinal_beats_clean_macro_f1": "",
            }
        )
        report.append(
            f"{model_name.upper()} | {aggregates['accuracy'][0]:.8f}+/-{aggregates['accuracy'][1]:.8f} | "
            f"{aggregates['macro_f1'][0]:.8f}+/-{aggregates['macro_f1'][1]:.8f} | "
            f"{aggregates['aupr'][0]:.8f}+/-{aggregates['aupr'][1]:.8f}"
        )

    report.extend(["", "Paired per-seed deltas (ORDINAL - CLEAN):", "Seed | Accuracy | Macro-F1 | AUPR", "---: | ---: | ---: | ---:"])
    wins = 0
    per_class_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        delta = {metric: float(ordinal[seed][metric]) - float(clean[seed][metric]) for metric in metric_names}
        win = int(delta["macro_f1"] > 0.0)
        wins += win
        summary_rows.append(
            {
                "row_type": "paired_delta", "model": "ordinal_minus_clean", "seed": seed,
                "accuracy": delta["accuracy"], "accuracy_std": "",
                "macro_f1": delta["macro_f1"], "macro_f1_std": "",
                "aupr": delta["aupr"], "aupr_std": "", "mae": "",
                "quadratic_weighted_kappa": "", "ordinal_beats_clean_macro_f1": win,
            }
        )
        report.append(f"{seed} | {delta['accuracy']:+.8f} | {delta['macro_f1']:+.8f} | {delta['aupr']:+.8f}")
        for label in CLASS_LABELS:
            clean_class = clean[seed]["per_class"][str(label)]
            ordinal_class = ordinal[seed]["per_class"][str(label)]
            per_class_rows.append(
                {
                    "seed": seed, "class": label, "support": clean_class["support"],
                    "clean_precision": clean_class["precision"], "ordinal_precision": ordinal_class["precision"],
                    "delta_precision": ordinal_class["precision"] - clean_class["precision"],
                    "clean_recall": clean_class["recall"], "ordinal_recall": ordinal_class["recall"],
                    "delta_recall": ordinal_class["recall"] - clean_class["recall"],
                    "clean_f1": clean_class["f1"], "ordinal_f1": ordinal_class["f1"],
                    "delta_f1": ordinal_class["f1"] - clean_class["f1"],
                }
            )
    mean_mae = float(np.mean([ordinal[seed]["mae"] for seed in SEEDS]))
    mean_qwk = float(np.mean([ordinal[seed]["quadratic_weighted_kappa"] for seed in SEEDS]))
    checks = [ordinal[seed].get("checks", {}) for seed in SEEDS]
    fair = all(item.get("experiment_fairness") for item in checks)
    label_safe = all(item.get("label_derived_feature_leakage") for item in checks)
    graph_safe = all(item.get("drug_phenotype_leakage") for item in checks)
    monotonic = all(item.get("ordinal_monotonicity") for item in checks)
    report.extend(
        [
            "", f"ORDINAL beats CLEAN on Macro-F1 for {wins} of {len(SEEDS)} seeds.",
            f"Ordinal mean MAE: {mean_mae:.8f}", f"Ordinal mean Quadratic Weighted Kappa: {mean_qwk:.8f}", "",
            f"EXPERIMENT FAIRNESS CHECK: {'PASS' if fair else 'FAIL'}",
            f"LABEL-DERIVED FEATURE LEAKAGE CHECK: {'PASS' if label_safe else 'FAIL'}",
            f"DRUG-PHENOTYPE LEAKAGE CHECK: {'PASS' if graph_safe else 'FAIL'}",
            f"ORDINAL MONOTONICITY CHECK: {'PASS' if monotonic else 'FAIL'}",
            "Compare mode performed no training or test evaluation.",
        ]
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(
        OUTPUT_ROOT / "ordinal_multiseed_summary.csv", summary_rows,
        ["row_type", "model", "seed", "accuracy", "accuracy_std", "macro_f1", "macro_f1_std", "aupr", "aupr_std", "mae", "quadratic_weighted_kappa", "ordinal_beats_clean_macro_f1"],
    )
    write_csv(
        OUTPUT_ROOT / "ordinal_per_class_comparison.csv", per_class_rows,
        ["seed", "class", "support", "clean_precision", "ordinal_precision", "delta_precision", "clean_recall", "ordinal_recall", "delta_recall", "clean_f1", "ordinal_f1", "delta_f1"],
    )
    text = "\n".join(report) + "\n"
    (OUTPUT_ROOT / "ordinal_multiseed_report.txt").write_text(text, encoding="utf-8")
    print(text, end="")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", required=True, choices=("train", "test", "train_test", "compare")
    )
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.mode in ("train", "test", "train_test") and args.seed is None:
        parser.error("--seed is required for train, test, and train_test modes")
    if args.mode == "compare" and args.seed is not None:
        parser.error("compare mode does not accept --seed")
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        train_mode(args.seed)
    elif args.mode == "test":
        test_mode(args.seed)
    elif args.mode == "train_test":
        train_test_mode(args.seed)
    else:
        compare_mode()


if __name__ == "__main__":
    main()
