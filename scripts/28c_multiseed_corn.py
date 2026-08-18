"""Multi-seed Fold-1 CLEAN versus CLEAN_CORN experiment."""

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
BASE_SCRIPT = PROJECT_ROOT / "scripts" / "28a_multiseed_clean_vs_ordinal.py"
SPLIT_PATH = PROJECT_ROOT / "data_processed" / "experiments" / "pilot_fold1_split.npz"
CLEAN_RESULTS_ROOT = PROJECT_ROOT / "data_processed" / "experiments" / "multiseed_fold1"
OUTPUT_ROOT = PROJECT_ROOT / "data_processed" / "experiments" / "corn_fold1"
SEEDS = (42, 123, 2026)
MAX_EPOCHS = 30
PATIENCE = 7
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 128
DROPOUT = 0.4
LATENT_DIM = 64
CLASS_LABELS = (1, 2, 3, 4, 5)
COLLAPSE_THRESHOLD = 0.80
MONOTONICITY_TOLERANCE = 1e-6

PROTECTED_MODEL_PATHS = (
    PROJECT_ROOT / "mssf.py",
    PROJECT_ROOT / "model.py",
    PROJECT_ROOT / "models" / "mssf_clean.py",
    PROJECT_ROOT / "models" / "ordinal_head.py",
    PROJECT_ROOT / "models" / "mssf_clean_ordinal.py",
    PROJECT_ROOT / "models" / "mssf_clean_kg.py",
    PROJECT_ROOT / "models" / "kg_fusion.py",
    PROJECT_ROOT / "models" / "kg_encoder.py",
    PROJECT_ROOT / "models" / "kg_pretraining.py",
)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_module("biokorf_step28a_helpers_for_corn", BASE_SCRIPT)
pilot = base.pilot
sys.path.insert(0, str(PROJECT_ROOT))
from models.corn_ordinal_head import (
    CORNOrdinalHead,
    corn_targets_and_masks,
    training_pos_weights,
)
from models.mssf_clean import MSSFClean, MSSFCleanConfig
from models.mssf_clean_corn import BioKORFCleanCORN


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protected_hashes() -> dict[Path, str]:
    for path in (*PROTECTED_MODEL_PATHS, SPLIT_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Protected input not found: {path}")
    return {path: sha256(path) for path in (*PROTECTED_MODEL_PATHS, SPLIT_PATH)}


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


def seed_directory(seed: int) -> Path:
    return OUTPUT_ROOT / f"seed{seed}"


def require_new(paths: list[Path], operation: str) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            f"Refusing to {operation}; output already exists: " + ", ".join(existing)
        )


def target_mask_check() -> bool:
    labels = torch.arange(1, 6)
    targets, masks = corn_targets_and_masks(labels)
    expected_targets = torch.tensor(
        [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]],
        dtype=torch.float32,
    )
    expected_masks = torch.tensor(
        [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1]],
        dtype=torch.bool,
    )
    return bool(torch.equal(targets, expected_targets) and torch.equal(masks, expected_masks))


def create_model(seed: int, pos_weights: Tensor) -> BioKORFCleanCORN:
    pilot.configure_reproducibility(seed)
    return BioKORFCleanCORN(
        MSSFCleanConfig(dropout=DROPOUT, gp=LATENT_DIM), pos_weights=pos_weights
    )


def fairness_check(seed: int, pos_weights: Tensor) -> bool:
    config = MSSFCleanConfig(dropout=DROPOUT, gp=LATENT_DIM)
    pilot.configure_reproducibility(seed)
    clean = MSSFClean(config)
    pilot.configure_reproducibility(seed)
    corn = BioKORFCleanCORN(config, pos_weights)
    clean_backbone = {
        name: value for name, value in clean.state_dict().items() if not name.startswith("classifier.")
    }
    same_backbone = all(
        name in corn.state_dict() and torch.equal(value, corn.state_dict()[name])
        for name, value in clean_backbone.items()
    )
    intended_head_only = (
        not any(name.startswith("classifier.") for name in corn.state_dict())
        and any(name.startswith("corn_head.") for name in corn.state_dict())
    )
    same_policy = (
        LEARNING_RATE == pilot.LEARNING_RATE
        and WEIGHT_DECAY == pilot.WEIGHT_DECAY
        and BATCH_SIZE == pilot.BATCH_SIZE
    )
    return bool(same_backbone and intended_head_only and same_policy)


def corn_composite_loss(
    model: BioKORFCleanCORN,
    logits: Tensor,
    rec_con: Tensor,
    rec_add: Tensor,
    mu: Tensor,
    logvar: Tensor,
    labels: Tensor,
    drug_features: Tensor,
    side_features: Tensor,
) -> Tensor:
    classification = model.frequency_classification_loss(logits, labels)
    kl = (-0.5 * (1 + logvar - mu.square() - torch.exp(logvar))).sum(1).mean()
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
    return classification + 0.001 * kl + 0.0001 * rec_con_loss + 0.0001 * rec_add_loss


def train_epoch(
    model: BioKORFCleanCORN,
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
        logits, rec_con, rec_add, mu, logvar = model(drugs, sides, device=device)
        loss = corn_composite_loss(
            model, logits, rec_con, rec_add, mu, logvar, labels, drugs, sides
        )
        loss.backward()
        optimizer.step()
        count = int(labels.shape[0])
        total_loss += float(loss.detach()) * count
        total_samples += count
    return total_loss / total_samples


def distribution(values: np.ndarray) -> dict[str, int]:
    return {str(label): int(np.sum(values == label)) for label in CLASS_LABELS}


def evaluate(
    model: BioKORFCleanCORN, dataset: Any, device: torch.device, seed: int
) -> tuple[float, dict[str, Any]]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    labels_all: list[np.ndarray] = []
    probabilities_all: list[np.ndarray] = []
    conditional_all: list[np.ndarray] = []
    cumulative_all: list[np.ndarray] = []
    monotonic = True
    with torch.inference_mode():
        for drugs, sides, _drug_index, _side_index, labels in pilot.make_loader(
            dataset, shuffle=False, seed=seed
        ):
            drugs = drugs.to(device, non_blocking=True)
            sides = sides.to(device, non_blocking=True)
            labels_device = labels.to(device, non_blocking=True)
            logits, rec_con, rec_add, mu, logvar, debug = model(
                drugs, sides, device=device, return_debug=True
            )
            valid = CORNOrdinalHead.validate_probabilities(
                debug["cumulative_probabilities"],
                debug["class_probabilities"],
                MONOTONICITY_TOLERANCE,
            )
            monotonic = monotonic and valid
            loss = corn_composite_loss(
                model, logits, rec_con, rec_add, mu, logvar, labels_device, drugs, sides
            )
            count = int(labels.shape[0])
            total_loss += float(loss) * count
            total_samples += count
            labels_all.append((labels.numpy() - 1).astype(np.int64))
            probabilities_all.append(debug["class_probabilities"].cpu().numpy())
            conditional_all.append(debug["conditional_probabilities"].cpu().numpy())
            cumulative_all.append(debug["cumulative_probabilities"].cpu().numpy())
    if not monotonic:
        raise RuntimeError("CORN monotonicity/class-probability check failed")
    labels = np.concatenate(labels_all)
    probabilities = np.vstack(probabilities_all)
    conditional = np.vstack(conditional_all)
    cumulative = np.vstack(cumulative_all)
    metrics = pilot.classification_metrics(labels, probabilities)
    predictions = probabilities.argmax(axis=1)
    metrics["mae"] = float(np.mean(np.abs(labels - predictions)))
    metrics["quadratic_weighted_kappa"] = base.quadratic_weighted_kappa(labels, predictions)
    metrics["true_distribution"] = distribution(labels + 1)
    metrics["predicted_distribution"] = distribution(predictions + 1)
    metrics["predicted_fractions"] = {
        label: count / len(predictions)
        for label, count in metrics["predicted_distribution"].items()
    }
    metrics["class_collapse"] = max(metrics["predicted_fractions"].values()) > COLLAPSE_THRESHOLD
    metrics["mean_conditional_probabilities"] = conditional.mean(axis=0).tolist()
    metrics["mean_cumulative_probabilities"] = cumulative.mean(axis=0).tolist()
    metrics["corn_monotonicity"] = monotonic
    return total_loss / total_samples, metrics


def checkpoint_state(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def print_checks(fair: bool, label_safe: bool, graph_safe: bool, target: bool, monotonic: bool, no_collapse: bool) -> None:
    print(f"EXPERIMENT FAIRNESS CHECK: {'PASS' if fair else 'FAIL'}")
    print(f"LABEL-DERIVED FEATURE LEAKAGE CHECK: {'PASS' if label_safe else 'FAIL'}")
    print(f"DRUG-PHENOTYPE LEAKAGE CHECK: {'PASS' if graph_safe else 'FAIL'}")
    print(f"CORN TARGET MASK CHECK: {'PASS' if target else 'FAIL'}")
    print(f"CORN MONOTONICITY CHECK: {'PASS' if monotonic else 'FAIL'}")
    print(f"CLASS COLLAPSE CHECK: {'PASS' if no_collapse else 'FAIL'}")


def train_mode(seed: int) -> dict[str, Any]:
    before = protected_hashes()
    output_dir = seed_directory(seed)
    history_path = output_dir / "training_history.csv"
    checkpoint_path = output_dir / "best_checkpoint.pt"
    require_new([history_path, checkpoint_path], "train")
    train_data, validation_data, _test_data, label_safe, graph_safe = base.load_experiment_data()
    pos_weights = training_pos_weights(train_data.labels, maximum=10.0)
    target_safe = target_mask_check()
    fair = fairness_check(seed, pos_weights)
    if not all((label_safe, graph_safe, target_safe, fair)):
        raise RuntimeError("Pre-training fairness/leakage/CORN target check failed")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(seed, pos_weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    history: list[dict[str, Any]] = []
    columns = [
        "epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1", "val_aupr",
        "predicted_class_1", "predicted_class_2", "predicted_class_3", "predicted_class_4",
        "predicted_class_5", "class_collapse", "corn_monotonicity",
    ]
    best_epoch, best_f1, stale = 0, -1.0, 0
    all_monotonic = True
    final_validation_no_collapse = True
    print(f"Training CLEAN_CORN seed={seed}; pos_weights={pos_weights.tolist()}; device={device}")
    for epoch in range(1, MAX_EPOCHS + 1):
        pilot.configure_reproducibility(seed + epoch)
        started = time.perf_counter()
        train_loss = train_epoch(model, train_data, optimizer, device, epoch, seed)
        val_loss, metrics = evaluate(model, validation_data, device, seed)
        all_monotonic = all_monotonic and metrics["corn_monotonicity"]
        final_validation_no_collapse = not metrics["class_collapse"]
        predicted = metrics["predicted_distribution"]
        row = {
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_accuracy": metrics["accuracy"], "val_macro_f1": metrics["macro_f1"],
            "val_aupr": metrics["aupr"], "class_collapse": metrics["class_collapse"],
            "corn_monotonicity": metrics["corn_monotonicity"],
            **{f"predicted_class_{label}": predicted[str(label)] for label in CLASS_LABELS},
        }
        history.append(row)
        write_csv(history_path, history, columns)
        print(f"epoch={epoch:02d} validation predicted distribution={predicted}")
        if metrics["macro_f1"] > best_f1:
            best_epoch, best_f1, stale = epoch, float(metrics["macro_f1"]), 0
            temporary = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(
                {
                    "model": "clean_corn", "seed": seed, "epoch": epoch,
                    "validation_macro_f1": best_f1, "model_state_dict": checkpoint_state(model),
                    "pos_weights": pos_weights.tolist(), "selection_metric": "validation_macro_f1",
                    "optimizer": {"name": "Adam", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY},
                    "batch_size": BATCH_SIZE,
                },
                temporary,
            )
            temporary.replace(checkpoint_path)
        else:
            stale += 1
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"val_macro_f1={metrics['macro_f1']:.6f} best_epoch={best_epoch} "
            f"patience={stale}/{PATIENCE} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
        if stale >= PATIENCE:
            break
    fair = fair and unchanged(before)
    print_checks(
        fair,
        label_safe,
        graph_safe,
        target_safe,
        all_monotonic,
        final_validation_no_collapse,
    )
    if not all((fair, label_safe, graph_safe, target_safe, all_monotonic)):
        raise RuntimeError("A required training safety check failed")
    return {"best_epoch": best_epoch, "best_validation_macro_f1": best_f1}


def test_mode(seed: int) -> dict[str, Any]:
    before = protected_hashes()
    output_dir = seed_directory(seed)
    checkpoint_path = output_dir / "best_checkpoint.pt"
    if not checkpoint_path.is_file() or not (output_dir / "training_history.csv").is_file():
        raise FileNotFoundError(f"Train mode must complete before test mode: {output_dir}")
    output_paths = [
        output_dir / "test_metrics.json", output_dir / "confusion_matrix.csv",
        output_dir / "per_class_metrics.csv", output_dir / "prediction_distribution.csv",
    ]
    require_new(output_paths, "test")
    train_data, _validation_data, test_data, label_safe, graph_safe = base.load_experiment_data()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model") != "clean_corn" or int(checkpoint.get("seed", -1)) != seed:
        raise ValueError("Checkpoint does not match the requested CLEAN_CORN seed")
    saved_weights = torch.tensor(checkpoint["pos_weights"], dtype=torch.float32)
    recomputed_weights = training_pos_weights(train_data.labels, maximum=10.0)
    weights_safe = torch.equal(saved_weights, recomputed_weights)
    target_safe = target_mask_check()
    fair = fairness_check(seed, saved_weights) and weights_safe
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(seed, saved_weights)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)
    _loss, metrics = evaluate(model, test_data, device, seed)
    metrics.update(
        {
            "model": "clean_corn", "seed": seed, "best_epoch": int(checkpoint["epoch"]),
            "best_validation_macro_f1": float(checkpoint["validation_macro_f1"]),
            "pos_weights": saved_weights.tolist(), "selection_metric": "validation_macro_f1",
            "checks": {
                "experiment_fairness": fair, "label_derived_feature_leakage": label_safe,
                "drug_phenotype_leakage": graph_safe, "corn_target_mask": target_safe,
                "corn_monotonicity": metrics["corn_monotonicity"],
                "class_collapse": not metrics["class_collapse"],
            },
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
    write_csv(
        output_dir / "prediction_distribution.csv",
        [{"class": label, "true_count": metrics["true_distribution"][str(label)], "predicted_count": metrics["predicted_distribution"][str(label)], "predicted_fraction": metrics["predicted_fractions"][str(label)]} for label in CLASS_LABELS],
        ["class", "true_count", "predicted_count", "predicted_fraction"],
    )
    atomic_json(output_dir / "test_metrics.json", metrics)
    print(f"True class distribution: {metrics['true_distribution']}")
    print(f"Predicted class distribution: {metrics['predicted_distribution']}")
    print(f"Predicted class fractions: {metrics['predicted_fractions']}")
    if metrics["class_collapse"]:
        print("CLASS COLLAPSE DETECTED")
    print(f"Mean conditional probabilities c1..c4: {metrics['mean_conditional_probabilities']}")
    print(f"Mean cumulative probabilities q1..q4: {metrics['mean_cumulative_probabilities']}")
    print(f"Task-specific pos_weight values: {saved_weights.tolist()}")
    fair = fair and unchanged(before)
    print_checks(fair, label_safe, graph_safe, target_safe, metrics["corn_monotonicity"], not metrics["class_collapse"])
    if not all((fair, label_safe, graph_safe, target_safe, metrics["corn_monotonicity"])):
        raise RuntimeError("A required test safety check failed")
    return metrics


def train_test_mode(seed: int) -> None:
    output_dir = seed_directory(seed)
    require_new(
        [output_dir / name for name in ("training_history.csv", "best_checkpoint.pt", "test_metrics.json", "confusion_matrix.csv", "per_class_metrics.csv", "prediction_distribution.csv")],
        "train_test",
    )
    training = train_mode(seed)
    metrics = test_mode(seed)
    print("\nCLEAN_CORN TRAIN_TEST SUMMARY")
    print(f"Best epoch: {training['best_epoch']}")
    print(f"Best validation Macro-F1: {training['best_validation_macro_f1']:.8f}")
    print(f"Test Accuracy: {metrics['accuracy']:.8f}")
    print(f"Test Macro-F1: {metrics['macro_f1']:.8f}")
    print(f"Test AUPR: {metrics['aupr']:.8f}")
    print(f"Test MAE: {metrics['mae']:.8f}")
    print(f"Test Quadratic Weighted Kappa: {metrics['quadratic_weighted_kappa']:.8f}")


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1))


def compare_mode() -> None:
    clean: dict[int, dict[str, Any]] = {}
    corn: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        clean_path = CLEAN_RESULTS_ROOT / f"clean_seed{seed}" / "test_metrics.json"
        corn_path = seed_directory(seed) / "test_metrics.json"
        for path in (clean_path, corn_path):
            if not path.is_file():
                raise FileNotFoundError(f"Compare mode requires completed result: {path}")
        clean[seed] = json.loads(clean_path.read_text(encoding="utf-8"))
        corn[seed] = json.loads(corn_path.read_text(encoding="utf-8"))
    metrics = ("accuracy", "macro_f1", "aupr")
    summary_rows: list[dict[str, Any]] = []
    report = [
        "BioKORF CLEAN versus CLEAN_CORN multi-seed comparison",
        "=====================================================",
        "Model | Accuracy mean+/-std | Macro-F1 mean+/-std | AUPR mean+/-std",
        "--- | ---: | ---: | ---:",
    ]
    for name, results in (("clean", clean), ("clean_corn", corn)):
        aggregate = {metric: mean_std([float(results[s][metric]) for s in SEEDS]) for metric in metrics}
        summary_rows.append({
            "row_type": "model_summary", "model": name, "seed": "",
            "accuracy": aggregate["accuracy"][0], "accuracy_std": aggregate["accuracy"][1],
            "macro_f1": aggregate["macro_f1"][0], "macro_f1_std": aggregate["macro_f1"][1],
            "aupr": aggregate["aupr"][0], "aupr_std": aggregate["aupr"][1],
            "mae": "" if name == "clean" else np.mean([results[s]["mae"] for s in SEEDS]),
            "quadratic_weighted_kappa": "" if name == "clean" else np.mean([results[s]["quadratic_weighted_kappa"] for s in SEEDS]),
            "corn_beats_clean_macro_f1": "",
        })
        report.append(f"{name.upper()} | {aggregate['accuracy'][0]:.8f}+/-{aggregate['accuracy'][1]:.8f} | {aggregate['macro_f1'][0]:.8f}+/-{aggregate['macro_f1'][1]:.8f} | {aggregate['aupr'][0]:.8f}+/-{aggregate['aupr'][1]:.8f}")
    report.extend(["", "Paired deltas (CORN - CLEAN):", "Seed | Accuracy | Macro-F1 | AUPR", "---: | ---: | ---: | ---:"])
    wins = 0
    per_class_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        delta = {metric: float(corn[seed][metric]) - float(clean[seed][metric]) for metric in metrics}
        win = int(delta["macro_f1"] > 0)
        wins += win
        summary_rows.append({"row_type": "paired_delta", "model": "corn_minus_clean", "seed": seed, "accuracy": delta["accuracy"], "accuracy_std": "", "macro_f1": delta["macro_f1"], "macro_f1_std": "", "aupr": delta["aupr"], "aupr_std": "", "mae": "", "quadratic_weighted_kappa": "", "corn_beats_clean_macro_f1": win})
        report.append(f"{seed} | {delta['accuracy']:+.8f} | {delta['macro_f1']:+.8f} | {delta['aupr']:+.8f}")
        for label in CLASS_LABELS:
            c, o = clean[seed]["per_class"][str(label)], corn[seed]["per_class"][str(label)]
            per_class_rows.append({"seed": seed, "class": label, "support": c["support"], "clean_precision": c["precision"], "corn_precision": o["precision"], "delta_precision": o["precision"] - c["precision"], "clean_recall": c["recall"], "corn_recall": o["recall"], "delta_recall": o["recall"] - c["recall"], "clean_f1": c["f1"], "corn_f1": o["f1"], "delta_f1": o["f1"] - c["f1"]})
    saved_checks = [corn[seed].get("checks", {}) for seed in SEEDS]
    fair = all(item.get("experiment_fairness") for item in saved_checks)
    label_safe = all(item.get("label_derived_feature_leakage") for item in saved_checks)
    graph_safe = all(item.get("drug_phenotype_leakage") for item in saved_checks)
    target_safe = all(item.get("corn_target_mask") for item in saved_checks)
    monotonic = all(item.get("corn_monotonicity") for item in saved_checks)
    no_collapse = all(item.get("class_collapse") for item in saved_checks)
    report.extend(
        [
            "",
            f"CORN beats CLEAN on Macro-F1 for {wins}/{len(SEEDS)} seeds.",
            f"CORN mean MAE: {np.mean([corn[s]['mae'] for s in SEEDS]):.8f}",
            f"CORN mean QWK: {np.mean([corn[s]['quadratic_weighted_kappa'] for s in SEEDS]):.8f}",
            "",
            f"EXPERIMENT FAIRNESS CHECK: {'PASS' if fair else 'FAIL'}",
            f"LABEL-DERIVED FEATURE LEAKAGE CHECK: {'PASS' if label_safe else 'FAIL'}",
            f"DRUG-PHENOTYPE LEAKAGE CHECK: {'PASS' if graph_safe else 'FAIL'}",
            f"CORN TARGET MASK CHECK: {'PASS' if target_safe else 'FAIL'}",
            f"CORN MONOTONICITY CHECK: {'PASS' if monotonic else 'FAIL'}",
            f"CLASS COLLAPSE CHECK: {'PASS' if no_collapse else 'FAIL'}",
            "Compare mode performed no training or test evaluation.",
        ]
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_ROOT / "corn_multiseed_summary.csv", summary_rows, ["row_type", "model", "seed", "accuracy", "accuracy_std", "macro_f1", "macro_f1_std", "aupr", "aupr_std", "mae", "quadratic_weighted_kappa", "corn_beats_clean_macro_f1"])
    write_csv(OUTPUT_ROOT / "corn_per_class_comparison.csv", per_class_rows, ["seed", "class", "support", "clean_precision", "corn_precision", "delta_precision", "clean_recall", "corn_recall", "delta_recall", "clean_f1", "corn_f1", "delta_f1"])
    text = "\n".join(report) + "\n"
    (OUTPUT_ROOT / "corn_multiseed_report.txt").write_text(text, encoding="utf-8")
    print(text, end="")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("train", "test", "train_test", "compare"))
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
