"""Audit Step 28A ordinal labels and decoding without training or weight updates."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PILOT_SCRIPT = PROJECT_ROOT / "scripts" / "26a_pilot_compare_clean_vs_kg.py"
SPLIT_PATH = PROJECT_ROOT / "data_processed" / "experiments" / "pilot_fold1_split.npz"
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "experiments"
    / "ordinal_fold1"
    / "seed42"
    / "best_checkpoint.pt"
)
OUTPUT_DIR = CHECKPOINT_PATH.parent
REPORT_PATH = OUTPUT_DIR / "ordinal_prediction_audit.txt"
COMPARISON_PATH = OUTPUT_DIR / "ordinal_decoder_comparison.csv"

SEED = 42
CLASS_COUNT = 5
TOLERANCE = 1e-6
COLLAPSE_FRACTION = 0.80
DRAMATIC_IMPROVEMENT = 0.10
THRESHOLD_DEGENERACY_TOLERANCE = 1e-3


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pilot = load_module("biokorf_pilot_helpers_28b", PILOT_SCRIPT)
sys.path.insert(0, str(PROJECT_ROOT))
from models.mssf_clean import MSSFCleanConfig
from models.mssf_clean_ordinal import BioKORFCleanOrdinal
from models.ordinal_head import ordered_class_targets


class AuditReport:
    """Mirror diagnostic text to the console and the final report file."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def print(self, value: object = "") -> None:
        line = str(value)
        self.lines.append(line)
        print(line)

    def save(self) -> None:
        temporary = REPORT_PATH.with_suffix(REPORT_PATH.suffix + ".tmp")
        temporary.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        temporary.replace(REPORT_PATH)


def atomic_csv(rows: list[dict[str, Any]], columns: list[str]) -> None:
    temporary = COMPARISON_PATH.with_suffix(COMPARISON_PATH.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(COMPARISON_PATH)


def load_raw_splits() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if not SPLIT_PATH.is_file():
        raise FileNotFoundError(f"Fixed Fold-1 split not found: {SPLIT_PATH}")
    frequency_matrix = np.asarray(pilot.load_pickle("drug_side.pkl"))
    all_samples = pilot.original_positive_sample_order(frequency_matrix)
    with np.load(SPLIT_PATH) as split:
        required = {
            "train_indices",
            "validation_indices",
            "test_indices",
            "train_samples",
            "validation_samples",
            "test_samples",
            "seed",
            "fold",
        }
        missing = required.difference(split.files)
        if missing:
            raise KeyError(f"Fixed split is missing arrays: {sorted(missing)}")
        if int(split["seed"]) != 42 or int(split["fold"]) != 1:
            raise ValueError("Fixed split metadata is not the established seed=42, fold=1")
        result: dict[str, np.ndarray] = {}
        for name in ("train", "validation", "test"):
            indices = np.asarray(split[f"{name}_indices"])
            saved_samples = np.asarray(split[f"{name}_samples"])
            if not np.array_equal(all_samples[indices], saved_samples):
                raise ValueError(f"Saved {name} samples do not match their fixed indices")
            result[name] = saved_samples.copy()
        combined_indices = np.concatenate(
            (split["train_indices"], split["validation_indices"], split["test_indices"])
        )
        if len(combined_indices) != len(all_samples) or len(np.unique(combined_indices)) != len(
            all_samples
        ):
            raise ValueError("Fixed split is not a complete disjoint sample partition")
    return frequency_matrix, result


def integral_labels(samples: np.ndarray, split_name: str) -> np.ndarray:
    raw = np.asarray(samples[:, 2])
    if not np.all(np.isfinite(raw)) or not np.all(raw == np.round(raw)):
        raise ValueError(f"{split_name} labels are not finite integers")
    return raw.astype(np.int64)


def determine_label_base(labels: np.ndarray) -> int:
    values = set(np.unique(labels).tolist())
    zero_based = values.issubset(set(range(5)))
    one_based = values.issubset(set(range(1, 6)))
    if 0 in values and zero_based:
        return 0
    if 5 in values and one_based:
        return 1
    if zero_based and not one_based:
        return 0
    if one_based and not zero_based:
        return 1
    raise ValueError(
        f"Cannot unambiguously determine 0..4 versus 1..5 convention from {sorted(values)}"
    )


def semantic_classes(raw_labels: np.ndarray, label_base: int) -> np.ndarray:
    classes = raw_labels + (1 if label_base == 0 else 0)
    if np.any(classes < 1) or np.any(classes > 5):
        raise ValueError("Interpreted semantic classes are outside 1..5")
    return classes


def format_distribution(values: np.ndarray) -> str:
    counts = Counter(int(value) for value in values)
    return ", ".join(f"class {label}={counts.get(label, 0)}" for label in range(1, 6))


def audit_targets(
    raw_labels: np.ndarray, label_base: int, report: AuditReport
) -> bool:
    report.print("PART B — TARGET CONSTRUCTION AUDIT")
    passed = True
    for raw_label in sorted(np.unique(raw_labels).tolist()):
        interpreted = int(raw_label) + (1 if label_base == 0 else 0)
        actual = (
            ordered_class_targets(torch.tensor([raw_label]), label_base=label_base)
            .squeeze(0)
            .to(torch.int64)
            .tolist()
        )
        expected = [int(interpreted > cut_point) for cut_point in range(1, 5)]
        row_pass = actual == expected
        passed = passed and row_pass
        report.print(
            f"raw label={raw_label}; interpreted ordinal class={interpreted}; "
            f"targets={actual}; expected={expected}; {'PASS' if row_pass else 'FAIL'}"
        )
    canonical = {
        class_value: [int(class_value > cut_point) for cut_point in range(1, 5)]
        for class_value in range(1, 6)
    }
    expected_canonical = {
        1: [0, 0, 0, 0],
        2: [1, 0, 0, 0],
        3: [1, 1, 0, 0],
        4: [1, 1, 1, 0],
        5: [1, 1, 1, 1],
    }
    passed = passed and canonical == expected_canonical
    report.print(f"ORDINAL TARGET CHECK: {'PASS' if passed else 'FAIL'}")
    report.print()
    return passed


def quadratic_weighted_kappa(labels: np.ndarray, predictions: np.ndarray) -> float:
    observed = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.float64)
    np.add.at(observed, (labels - 1, predictions - 1), 1)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    indices = np.arange(CLASS_COUNT, dtype=np.float64)
    weights = ((indices[:, None] - indices[None, :]) ** 2) / (CLASS_COUNT - 1) ** 2
    denominator = float((weights * expected).sum())
    numerator = float((weights * observed).sum())
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else 0.0
    return 1.0 - numerator / denominator


def decoder_metrics(true_classes: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    if np.any(predictions < 1) or np.any(predictions > 5):
        raise ValueError("Decoded semantic classes must be in 1..5")
    # The established metric helper uses zero-based indices internally.
    one_hot = np.eye(CLASS_COUNT, dtype=np.float64)[predictions - 1]
    metrics = pilot.classification_metrics(true_classes - 1, one_hot)
    return {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "mae": float(np.mean(np.abs(true_classes - predictions))),
        "quadratic_weighted_kappa": quadratic_weighted_kappa(true_classes, predictions),
        "confusion_matrix": metrics["confusion_matrix"],
        "distribution": {str(label): int(np.sum(predictions == label)) for label in range(1, 6)},
    }


def load_test_dataset(
    frequency_matrix: np.ndarray, splits: dict[str, np.ndarray]
) -> Any:
    hidden_samples = np.concatenate((splits["validation"], splits["test"]), axis=0)
    drug_features, side_features, label_safe = pilot.build_leakage_safe_features(
        frequency_matrix, hidden_samples
    )
    if not label_safe:
        raise RuntimeError("Label-derived feature leakage check failed")
    if not pilot.scan_graph_leakage():
        raise RuntimeError("Drug-Phenotype leakage check failed")
    return pilot.IndexedPairDataset(splits["test"], drug_features, side_features)


def infer_checkpoint(test_dataset: Any) -> dict[str, np.ndarray]:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Seed-42 ordinal checkpoint not found: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    if checkpoint.get("model") != "clean_ordinal" or int(checkpoint.get("seed", -1)) != SEED:
        raise ValueError("Checkpoint is not the expected seed-42 CLEAN_ORDINAL checkpoint")
    pilot.configure_reproducibility(SEED)
    model = BioKORFCleanOrdinal(MSSFCleanConfig(dropout=0.4, gp=64))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    cumulative_batches: list[np.ndarray] = []
    probability_batches: list[np.ndarray] = []
    severity_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    thresholds: np.ndarray | None = None
    with torch.inference_mode():
        for drugs, sides, _drug_index, _side_index, labels in pilot.make_loader(
            test_dataset, shuffle=False, seed=SEED
        ):
            _outputs, _rec_con, _rec_add, _mu, _logvar, debug = model(
                drugs.to(device, non_blocking=True),
                sides.to(device, non_blocking=True),
                device=device,
                return_debug=True,
            )
            batch_thresholds = debug["ordered_thresholds"].detach().cpu().numpy()
            if thresholds is None:
                thresholds = batch_thresholds
            elif not np.array_equal(thresholds, batch_thresholds):
                raise RuntimeError("Ordered thresholds changed between evaluation batches")
            cumulative_batches.append(debug["cumulative_probabilities"].cpu().numpy())
            probability_batches.append(debug["class_probabilities"].cpu().numpy())
            severity_batches.append(debug["severity_score"].squeeze(1).cpu().numpy())
            label_batches.append(labels.numpy().astype(np.int64))
    if thresholds is None:
        raise RuntimeError("Test dataset produced no batches")
    return {
        "thresholds": thresholds,
        "cumulative": np.vstack(cumulative_batches),
        "probabilities": np.vstack(probability_batches),
        "severity": np.concatenate(severity_batches),
        "raw_labels": np.concatenate(label_batches),
        "best_epoch": np.asarray(int(checkpoint["epoch"])),
        "best_validation_macro_f1": np.asarray(float(checkpoint["validation_macro_f1"])),
    }


def print_confusion_matrix(
    report: AuditReport, decoder: str, matrix: list[list[int]]
) -> None:
    report.print(f"{decoder} confusion matrix (rows=true 1..5, columns=predicted 1..5):")
    report.print("true\\pred | 1 | 2 | 3 | 4 | 5")
    for label, row in enumerate(matrix, start=1):
        report.print(f"{label} | " + " | ".join(str(value) for value in row))


def main() -> None:
    report = AuditReport()
    report.print("BioKORF Step 28A ordinal prediction audit")
    report.print("==========================================")
    report.print("Checkpoint-only evaluation; no training and no weight updates.")
    report.print(f"Fixed split: {SPLIT_PATH}")
    report.print(f"Checkpoint: {CHECKPOINT_PATH}")
    report.print()

    frequency_matrix, splits = load_raw_splits()
    split_labels = {
        name: integral_labels(samples, name) for name, samples in splits.items()
    }
    all_raw_labels = np.concatenate(tuple(split_labels.values()))
    label_base = determine_label_base(all_raw_labels)

    report.print("PART A — RAW LABEL AUDIT")
    for name in ("train", "validation", "test"):
        counts = Counter(split_labels[name].tolist())
        report.print(
            f"{name}: "
            + ", ".join(f"{label}={counts[label]}" for label in sorted(counts))
        )
    report.print(
        f"Detected raw label convention: {'0..4' if label_base == 0 else '1..5'}"
    )
    report.print("Determination used observed values in the saved split, not source comments.")
    report.print()

    target_check = audit_targets(all_raw_labels, label_base, report)
    test_dataset = load_test_dataset(frequency_matrix, splits)
    inference = infer_checkpoint(test_dataset)
    if not np.array_equal(inference["raw_labels"], split_labels["test"]):
        raise RuntimeError("Inference loader changed test row or label order")
    true_classes = semantic_classes(inference["raw_labels"], label_base)

    thresholds = inference["thresholds"]
    differences = np.diff(thresholds)
    severity = inference["severity"]
    report.print("PART C — LOAD CHECKPOINT ONLY")
    report.print(f"Best epoch: {int(inference['best_epoch'])}")
    report.print(
        f"Best validation Macro-F1: {float(inference['best_validation_macro_f1']):.8f}"
    )
    report.print(f"Ordered thresholds: {thresholds.tolist()}")
    report.print(f"Threshold differences: {differences.tolist()}")
    report.print(f"Mean severity score: {severity.mean():.8f}")
    report.print(
        f"Severity min/max/std: {severity.min():.8f} / {severity.max():.8f} / "
        f"{severity.std():.8f}"
    )
    report.print()

    cumulative = inference["cumulative"]
    monotonic = bool(np.all(cumulative[:, :-1] + TOLERANCE >= cumulative[:, 1:]))
    report.print("PART D — CUMULATIVE PROBABILITY AUDIT")
    for index in range(4):
        values = cumulative[:, index]
        report.print(
            f"q{index + 1}=P(y>{index + 1}): mean={values.mean():.8f}, "
            f"std={values.std():.8f}, min={values.min():.8f}, max={values.max():.8f}"
        )
    report.print(f"CUMULATIVE MONOTONICITY CHECK: {'PASS' if monotonic else 'FAIL'}")
    report.print()

    q1, q2, q3, q4 = cumulative.T
    reconstructed = np.column_stack((1.0 - q1, q1 - q2, q2 - q3, q3 - q4, q4))
    stored_probabilities = inference["probabilities"]
    reconstruction_match = bool(
        np.allclose(reconstructed, stored_probabilities, atol=TOLERANCE, rtol=0.0)
    )
    finite = bool(np.isfinite(reconstructed).all())
    nonnegative = bool(np.all(reconstructed >= -TOLERANCE))
    normalized = bool(
        np.allclose(reconstructed.sum(axis=1), 1.0, atol=TOLERANCE, rtol=0.0)
    )
    probability_check = reconstruction_match and finite and nonnegative and normalized
    report.print("PART E — CLASS PROBABILITY AUDIT")
    report.print(f"Reconstruction matches model output: {'PASS' if reconstruction_match else 'FAIL'}")
    report.print(f"No materially negative probabilities: {'PASS' if nonnegative else 'FAIL'}")
    report.print(f"Rows sum to one: {'PASS' if normalized else 'FAIL'}")
    report.print(f"All values finite: {'PASS' if finite else 'FAIL'}")
    for label, mean_probability in enumerate(reconstructed.mean(axis=0), start=1):
        report.print(f"Mean predicted probability class {label}: {mean_probability:.8f}")
    report.print()

    decoders = {
        "Decoder A (class-probability argmax)": reconstructed.argmax(axis=1) + 1,
        "Decoder B (0.5 threshold count)": 1 + (cumulative >= 0.5).sum(axis=1),
        "Decoder C (rounded expected value)": np.clip(
            np.rint(reconstructed @ np.arange(1, 6)), 1, 5
        ).astype(np.int64),
    }
    decoder_results = {
        name: decoder_metrics(true_classes, predictions)
        for name, predictions in decoders.items()
    }
    report.print("PART F — THREE DECODERS, SAME CHECKPOINT")
    for name, metrics in decoder_results.items():
        report.print(name)
        report.print(
            f"Accuracy={metrics['accuracy']:.8f}; Macro-F1={metrics['macro_f1']:.8f}; "
            f"MAE={metrics['mae']:.8f}; QWK={metrics['quadratic_weighted_kappa']:.8f}"
        )
        report.print("Predicted distribution: " + format_distribution(decoders[name]))
    report.print()

    current_predictions = decoders["Decoder A (class-probability argmax)"]
    current_metrics = decoder_results["Decoder A (class-probability argmax)"]
    shift_metrics: dict[int, dict[str, Any]] = {}
    report.print("PART G — OFF-BY-ONE DIAGNOSTIC (NOT A FINAL DECODER)")
    possible_index_bug = label_base != 1
    for shift in (-1, 1):
        shifted = np.clip(current_predictions + shift, 1, 5)
        metrics = decoder_metrics(true_classes, shifted)
        shift_metrics[shift] = metrics
        dramatic = (
            metrics["accuracy"] - current_metrics["accuracy"] >= DRAMATIC_IMPROVEMENT
            or metrics["macro_f1"] - current_metrics["macro_f1"] >= DRAMATIC_IMPROVEMENT
        )
        possible_index_bug = possible_index_bug or dramatic
        report.print(
            f"shift {shift:+d} (clamped 1..5): Accuracy={metrics['accuracy']:.8f}; "
            f"Macro-F1={metrics['macro_f1']:.8f}; dramatic={dramatic}"
        )
    if possible_index_bug:
        report.print("POSSIBLE LABEL INDEXING BUG")
    report.print()

    report.print("PART H — CLASS COLLAPSE")
    report.print("True test class distribution: " + format_distribution(true_classes))
    collapse_detected = False
    collapsed_decoders: list[str] = []
    for name, predictions in decoders.items():
        counts = Counter(predictions.tolist())
        majority_class, majority_count = counts.most_common(1)[0]
        fraction = majority_count / len(predictions)
        report.print(f"{name}: {format_distribution(predictions)}")
        report.print(
            f"Majority predicted class={majority_class}; fraction={fraction:.6f}"
        )
        if fraction > COLLAPSE_FRACTION:
            collapse_detected = True
            collapsed_decoders.append(name)
            report.print("CLASS COLLAPSE DETECTED")
    report.print()

    report.print("PART I — CONFUSION MATRICES")
    comparison_rows: list[dict[str, Any]] = []
    for name, metrics in decoder_results.items():
        print_confusion_matrix(report, name, metrics["confusion_matrix"])
        comparison_rows.append(
            {
                "decoder": name,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "mae": metrics["mae"],
                "quadratic_weighted_kappa": metrics["quadratic_weighted_kappa"],
                "predicted_class_distribution": json.dumps(metrics["distribution"]),
                "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
            }
        )
    atomic_csv(
        comparison_rows,
        [
            "decoder",
            "accuracy",
            "macro_f1",
            "mae",
            "quadratic_weighted_kappa",
            "predicted_class_distribution",
            "confusion_matrix",
        ],
    )
    report.print()

    implementation_label_base = 1
    label_mismatch = label_base != implementation_label_base or possible_index_bug
    target_error = not target_check
    decoder_a = decoder_results["Decoder A (class-probability argmax)"]
    alternative_improvement = max(
        max(
            result["accuracy"] - decoder_a["accuracy"],
            result["macro_f1"] - decoder_a["macro_f1"],
        )
        for name, result in decoder_results.items()
        if name != "Decoder A (class-probability argmax)"
    )
    decoding_error = alternative_improvement >= DRAMATIC_IMPROVEMENT or not probability_check
    threshold_degeneracy = bool(
        not np.isfinite(thresholds).all()
        or np.any(differences <= THRESHOLD_DEGENERACY_TOLERANCE)
        or np.all(cumulative.std(axis=0) <= THRESHOLD_DEGENERACY_TOLERANCE)
    )
    diagnoses: list[str] = []
    if label_mismatch:
        diagnoses.append("LABEL_INDEX_MISMATCH")
    if target_error:
        diagnoses.append("TARGET_CONSTRUCTION_ERROR")
    if decoding_error:
        diagnoses.append("DECODING_ERROR")
    if collapse_detected:
        diagnoses.append("CLASS_COLLAPSE")
    if threshold_degeneracy:
        diagnoses.append("THRESHOLD_DEGENERACY")
    if not diagnoses:
        diagnoses.append("NO_IMPLEMENTATION_ERROR_FOUND")

    report.print("PART J — FINAL DIAGNOSIS")
    report.print("Diagnosis is based only on the diagnostics above:")
    for diagnosis in diagnoses:
        report.print(f"- {diagnosis}")
    report.print(
        "Criteria: dramatic decoder/shift improvement is >=0.10 absolute Accuracy or "
        "Macro-F1; collapse is >80% in one class; threshold degeneracy is a non-finite "
        "threshold, increment <=1e-3, or all cumulative-probability std values <=1e-3."
    )
    if diagnoses == ["NO_IMPLEMENTATION_ERROR_FOUND"]:
        report.print(
            "No audited implementation error was detected; genuinely poor learned "
            "representations/optimization remains a possible explanation."
        )
    report.print("No code or weights were changed, and no training was performed.")
    report.save()
    report.print(f"Saved report: {REPORT_PATH}")
    report.print(f"Saved decoder comparison: {COMPARISON_PATH}")


if __name__ == "__main__":
    main()
