"""Controlled fold-1 pilot comparing MSSF-clean with frozen-KG fusion."""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Datas"
EXPERIMENTS_DIR = PROJECT_ROOT / "data_processed" / "experiments"
OUTPUT_DIR = EXPERIMENTS_DIR / "pilot_fold1"
SPLIT_PATH = EXPERIMENTS_DIR / "pilot_fold1_split.npz"
KG_ARTIFACT_PATH = (
    PROJECT_ROOT / "data_processed" / "kg_features" / "biokorf_kg_embeddings.pt"
)
GRAPH_EDGE_PATH = PROJECT_ROOT / "data_processed" / "rgcn" / "edges_with_inverse.csv"
REPORT_PATH = OUTPUT_DIR / "pilot_report.txt"

SEED = 42
FOLD = 1
EPOCHS = 5
VALIDATION_FRACTION = 0.10
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
DROPOUT = 0.4
LATENT_DIM = 64
CLASS_COUNT = 5

PROTECTED_HASHES = {
    PROJECT_ROOT / "mssf.py": "4867fecd04beabb2d715b24073f82a46bd572c13294afa3565ddba99f963fdb1",
    PROJECT_ROOT / "model.py": "9c0d4bf17551a7d0f881a29e0f8e2727227f3561678064fec46f2848156a1e75",
    PROJECT_ROOT / "models" / "mssf_clean.py": "f2a0f68e062807cacc77540c14afd5bf0e66eb7571b76b99e095c4063b8dd6d2",
    PROJECT_ROOT / "models" / "mssf_clean_kg.py": "cbc505f64c718cb5ce861fd4eac1d4d5d7f6eaefb0b045059271aab79bf92b81",
    PROJECT_ROOT / "models" / "kg_fusion.py": "9c9ce093d5e86078e32cd11e8696d0dc37625f8980c3cdb66167b2a893ac7c0f",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_pickle(name: str) -> Any:
    path = DATA_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Required MSSF data file not found: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def original_positive_sample_order(frequency_matrix: np.ndarray) -> np.ndarray:
    """Reproduce Extract_positive_negative_samples ordering exactly."""
    interaction_target = np.zeros((frequency_matrix.size, 3), dtype=int)
    cursor = 0
    for drug_index in range(frequency_matrix.shape[0]):
        for side_index in range(frequency_matrix.shape[1]):
            interaction_target[cursor] = (
                drug_index,
                side_index,
                frequency_matrix[drug_index, side_index],
            )
            cursor += 1
    sorted_rows = interaction_target[interaction_target[:, 2].argsort()]
    positive_count = len(np.nonzero(sorted_rows[:, 2])[0])
    return sorted_rows[interaction_target.shape[0] - positive_count :]


def cosine_similarity(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)
    return normalized @ normalized.T


def stratified_outer_first_fold(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NumPy equivalent of shuffled StratifiedKFold's first fold."""
    _, encoded = np.unique(labels, return_inverse=True)
    class_count = int(encoded.max()) + 1
    allocation = np.asarray(
        [
            np.bincount(np.sort(encoded)[fold::10], minlength=class_count)
            for fold in range(10)
        ]
    )
    rng = np.random.RandomState(SEED)
    fold_assignment = np.empty(len(labels), dtype=np.int64)
    for class_index in range(class_count):
        class_folds = np.arange(10).repeat(allocation[:, class_index])
        rng.shuffle(class_folds)
        fold_assignment[encoded == class_index] = class_folds
    test_indices = np.flatnonzero(fold_assignment == 0)
    development_indices = np.flatnonzero(fold_assignment != 0)
    return development_indices, test_indices


def stratified_validation_split(
    development_indices: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(SEED)
    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []
    development_labels = labels[development_indices]
    for class_label in sorted(np.unique(development_labels)):
        class_indices = development_indices[development_labels == class_label].copy()
        rng.shuffle(class_indices)
        validation_count = max(1, int(round(len(class_indices) * VALIDATION_FRACTION)))
        validation_parts.append(class_indices[:validation_count])
        train_parts.append(class_indices[validation_count:])
    train_indices = np.sort(np.concatenate(train_parts))
    validation_indices = np.sort(np.concatenate(validation_parts))
    return train_indices, validation_indices


def create_or_validate_split(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = samples[:, 2].astype(np.int64)
    development_indices, test_indices = stratified_outer_first_fold(labels)
    train_indices, validation_indices = stratified_validation_split(
        development_indices, labels
    )

    if SPLIT_PATH.exists():
        saved = np.load(SPLIT_PATH)
        for name, generated in (
            ("train_indices", train_indices),
            ("validation_indices", validation_indices),
            ("test_indices", test_indices),
        ):
            if name not in saved or not np.array_equal(saved[name], generated):
                raise ValueError(f"Existing pilot split does not match deterministic {name}")
    else:
        SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            SPLIT_PATH,
            train_indices=train_indices,
            validation_indices=validation_indices,
            test_indices=test_indices,
            train_samples=samples[train_indices],
            validation_samples=samples[validation_indices],
            test_samples=samples[test_indices],
            seed=np.array(SEED),
            fold=np.array(FOLD),
        )
    all_indices = np.concatenate((train_indices, validation_indices, test_indices))
    if len(np.unique(all_indices)) != len(samples) or len(all_indices) != len(samples):
        raise AssertionError("Train/validation/test split is not a complete disjoint partition")
    return train_indices, validation_indices, test_indices


def build_leakage_safe_features(
    frequency_matrix: np.ndarray, hidden_samples: np.ndarray
) -> tuple[Tensor, Tensor, bool]:
    masked_frequency = np.array(frequency_matrix, copy=True)
    hidden_drug = hidden_samples[:, 0].astype(np.int64)
    hidden_side = hidden_samples[:, 1].astype(np.int64)
    masked_frequency[hidden_drug, hidden_side] = 0
    leakage_safe = bool(np.all(masked_frequency[hidden_drug, hidden_side] == 0))
    if not leakage_safe:
        raise RuntimeError("Could not hide validation/test labels from frequency-derived features")
    binary_matrix = (masked_frequency > 0).astype(np.float32)

    drug_features = [
        load_pickle("Text_similarity_one.pkl"),
        load_pickle("Text_similarity_two.pkl"),
        load_pickle("Text_similarity_three.pkl"),
        load_pickle("Text_similarity_four.pkl"),
        load_pickle("Text_similarity_five.pkl"),
        cosine_similarity(load_pickle("drug_mol.pkl")),
        cosine_similarity(load_pickle("drug_target.pkl")),
        load_pickle("fingerprint_similarity.pkl"),
        cosine_similarity(masked_frequency),
        cosine_similarity(binary_matrix),
        load_pickle("drug_pathway_enzyme_similarity.pkl"),
    ]
    side_features = [
        load_pickle("side_effect_semantic.pkl"),
        cosine_similarity(load_pickle("glove_wordEmbedding.pkl")),
        cosine_similarity(masked_frequency.T),
        cosine_similarity(binary_matrix.T),
    ]
    drug_matrix = np.hstack(drug_features).astype(np.float32, copy=False)
    side_matrix = np.hstack(side_features).astype(np.float32, copy=False)
    if drug_matrix.shape != (757, 8327) or side_matrix.shape != (994, 3976):
        raise ValueError(
            f"Unexpected MSSF feature shapes: drug={drug_matrix.shape}, side={side_matrix.shape}"
        )
    if not np.isfinite(drug_matrix).all() or not np.isfinite(side_matrix).all():
        raise ValueError("MSSF feature matrices contain non-finite values")
    return torch.from_numpy(drug_matrix), torch.from_numpy(side_matrix), leakage_safe


class IndexedPairDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]):
    def __init__(self, samples: np.ndarray, drug_features: Tensor, side_features: Tensor) -> None:
        self.drug_index = torch.from_numpy(samples[:, 0].astype(np.int64, copy=True))
        self.side_index = torch.from_numpy(samples[:, 1].astype(np.int64, copy=True))
        self.labels = torch.from_numpy(samples[:, 2].astype(np.int64, copy=True))
        self.drug_features = drug_features
        self.side_features = side_features

    def __len__(self) -> int:
        return self.labels.numel()

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        drug_index = self.drug_index[index]
        side_index = self.side_index[index]
        return (
            self.drug_features[drug_index],
            self.side_features[side_index],
            drug_index,
            side_index,
            self.labels[index],
        )


def make_loader(dataset: Dataset, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def composite_loss(
    logits: Tensor,
    rec_con: Tensor,
    rec_add: Tensor,
    mu: Tensor,
    logvar: Tensor,
    labels: Tensor,
    drug_features: Tensor,
    side_features: Tensor,
) -> Tensor:
    classification = nn.functional.cross_entropy(logits, labels.long() - 1)
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


def train_epoch(
    model: nn.Module,
    dataset: Dataset,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_kg: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    loader = make_loader(dataset, shuffle=True, seed=SEED + epoch)
    for drugs, sides, drug_index, side_index, labels in loader:
        drugs = drugs.to(device, non_blocking=True)
        sides = sides.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_kg:
            outputs = model(
                drugs,
                sides,
                drug_index.to(device, non_blocking=True),
                side_index.to(device, non_blocking=True),
                device=device,
            )
        else:
            outputs = model(drugs, sides, device=device)
        logits, rec_con, rec_add, mu, logvar = outputs
        loss = composite_loss(
            logits, rec_con, rec_add, mu, logvar, labels, drugs, sides
        )
        loss.backward()
        optimizer.step()
        batch_count = labels.shape[0]
        total_loss += float(loss.detach()) * batch_count
        total_samples += batch_count
    return total_loss / total_samples


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    matrix = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    true_positive = np.diag(matrix).astype(np.float64)
    predicted_count = matrix.sum(axis=0).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.int64)
    precision = np.divide(
        true_positive, predicted_count, out=np.zeros(CLASS_COUNT), where=predicted_count != 0
    )
    recall = np.divide(
        true_positive, support, out=np.zeros(CLASS_COUNT), where=support != 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(CLASS_COUNT),
        where=(precision + recall) != 0,
    )
    average_precisions: list[float] = []
    for class_index in range(CLASS_COUNT):
        binary = (labels == class_index).astype(np.int64)
        order = np.argsort(-probabilities[:, class_index], kind="mergesort")
        ranked = binary[order]
        positive_count = int(ranked.sum())
        if positive_count == 0:
            average_precisions.append(0.0)
            continue
        cumulative = np.cumsum(ranked)
        positive_positions = np.flatnonzero(ranked) + 1
        average_precisions.append(
            float(np.mean(cumulative[positive_positions - 1] / positive_positions))
        )
    accuracy = float(true_positive.sum() / len(labels))
    return {
        "accuracy": accuracy,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "micro_f1": accuracy,
        "aupr": float(np.mean(average_precisions)),
        "confusion_matrix": matrix.tolist(),
        "per_class": {
            str(index + 1): {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in range(CLASS_COUNT)
        },
    }


def evaluate(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    use_kg: bool,
    collect_gate_categories: bool = False,
) -> tuple[float, dict[str, Any], dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    labels_all: list[np.ndarray] = []
    probabilities_all: list[np.ndarray] = []
    gates: list[Tensor] = []
    category_gate_values: dict[str, list[Tensor]] = {
        "both_available": [],
        "drug_only": [],
        "side_only": [],
        "neither_available": [],
    }
    with torch.inference_mode():
        for drugs, sides, drug_index, side_index, labels in make_loader(
            dataset, shuffle=False, seed=SEED
        ):
            drugs = drugs.to(device, non_blocking=True)
            sides = sides.to(device, non_blocking=True)
            labels_device = labels.to(device, non_blocking=True)
            if use_kg:
                logits, rec_con, rec_add, mu, logvar, debug = model(
                    drugs,
                    sides,
                    drug_index.to(device, non_blocking=True),
                    side_index.to(device, non_blocking=True),
                    device=device,
                    return_debug=True,
                )
                gate = debug["KG_gate"].detach().cpu()
                gates.append(gate.reshape(-1))
                if collect_gate_categories:
                    drug_available = debug["drug_kg_mask"].squeeze(1).cpu()
                    side_available = debug["side_kg_mask"].squeeze(1).cpu()
                    category_masks = {
                        "both_available": drug_available & side_available,
                        "drug_only": drug_available & ~side_available,
                        "side_only": ~drug_available & side_available,
                        "neither_available": ~drug_available & ~side_available,
                    }
                    for name, mask in category_masks.items():
                        if mask.any():
                            category_gate_values[name].append(gate[mask].reshape(-1))
            else:
                logits, rec_con, rec_add, mu, logvar = model(
                    drugs, sides, device=device
                )
            loss = composite_loss(
                logits,
                rec_con,
                rec_add,
                mu,
                logvar,
                labels_device,
                drugs,
                sides,
            )
            batch_count = labels.shape[0]
            total_loss += float(loss) * batch_count
            total_samples += batch_count
            labels_all.append((labels.numpy() - 1).astype(np.int64))
            probabilities_all.append(torch.softmax(logits, dim=1).cpu().numpy())
    metrics = classification_metrics(np.concatenate(labels_all), np.vstack(probabilities_all))
    gate_stats: dict[str, float] = {}
    if gates:
        all_gates = torch.cat(gates)
        gate_stats = {
            "gate_mean": float(all_gates.mean()),
            "gate_std": float(all_gates.std(unbiased=False)),
            "gate_min": float(all_gates.min()),
            "gate_max": float(all_gates.max()),
        }
        if collect_gate_categories:
            for name, values in category_gate_values.items():
                gate_stats[f"{name}_gate_mean"] = (
                    float(torch.cat(values).mean()) if values else float("nan")
                )
    return total_loss / total_samples, metrics, gate_stats


def cpu_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_confusion(path: Path, matrix: list[list[int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_class", *[f"predicted_{index}" for index in range(1, 6)]])
        for index, row in enumerate(matrix, start=1):
            writer.writerow([index, *row])


def scan_graph_leakage() -> bool:
    with GRAPH_EDGE_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            endpoint_types = {row["source_type"], row["target_type"]}
            if (
                endpoint_types == {"DRUG", "PHENOTYPE"}
                or row["relation"].upper() == "ADVERSE_DRUG_REACTION"
                or endpoint_types == {"BIOKORF_DRUG", "BIOKORF_SIDE"}
            ):
                return False
    return True


def train_model(
    name: str,
    model: nn.Module,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    device: torch.device,
    use_kg: bool,
) -> tuple[list[dict[str, Any]], dict[str, Tensor], int, float]:
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_macro_f1 = -1.0
    best_state: dict[str, Tensor] | None = None
    for epoch in range(1, EPOCHS + 1):
        configure_reproducibility(SEED + epoch)
        started = time.perf_counter()
        train_loss = train_epoch(
            model, train_dataset, optimizer, device, epoch, use_kg
        )
        validation_loss, validation_metrics, gate_stats = evaluate(
            model, validation_dataset, device, use_kg
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_precision": validation_metrics["macro_precision"],
            "validation_macro_recall": validation_metrics["macro_recall"],
            "validation_macro_f1": validation_metrics["macro_f1"],
            "validation_micro_f1": validation_metrics["micro_f1"],
            "validation_aupr": validation_metrics["aupr"],
            **gate_stats,
        }
        history.append(row)
        if validation_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation_metrics["macro_f1"]
            best_epoch = epoch
            best_state = cpu_state_dict(model)
        gate_text = (
            f" gate={gate_stats['gate_mean']:.6f}±{gate_stats['gate_std']:.6f} "
            f"[{gate_stats['gate_min']:.6f},{gate_stats['gate_max']:.6f}]"
            if gate_stats
            else ""
        )
        print(
            f"{name} epoch {epoch}/{EPOCHS}: train_loss={train_loss:.6f} "
            f"val_loss={validation_loss:.6f} val_macro_f1={validation_metrics['macro_f1']:.6f}"
            f"{gate_text} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    if best_state is None:
        raise RuntimeError(f"No best state captured for {name}")
    return history, best_state, best_epoch, best_macro_f1


def main() -> None:
    required_paths = (*PROTECTED_HASHES, KG_ARTIFACT_PATH, GRAPH_EDGE_PATH)
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required experiment input not found: {path}")
    protected_before = {path: sha256(path) for path in PROTECTED_HASHES}
    if protected_before != PROTECTED_HASHES:
        raise RuntimeError(f"Protected-file hash mismatch before experiment: {protected_before}")
    kg_hash_before = sha256(KG_ARTIFACT_PATH)
    configure_reproducibility(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frequency_matrix = np.asarray(load_pickle("drug_side.pkl"))
    if frequency_matrix.shape != (757, 994):
        raise ValueError(f"Expected frequency matrix [757,994], found {frequency_matrix.shape}")
    samples = original_positive_sample_order(frequency_matrix)
    train_indices, validation_indices, test_indices = create_or_validate_split(samples)
    hidden_samples = samples[np.concatenate((validation_indices, test_indices))]
    drug_features, side_features, label_feature_leakage_safe = build_leakage_safe_features(
        frequency_matrix, hidden_samples
    )
    graph_leakage_safe = scan_graph_leakage()
    leakage_check = label_feature_leakage_safe and graph_leakage_safe
    print(
        f"Samples: total={len(samples)} train={len(train_indices)} "
        f"validation={len(validation_indices)} test={len(test_indices)}"
    )
    print(
        f"LABEL-DERIVED FEATURE LEAKAGE CHECK: {'PASS' if label_feature_leakage_safe else 'FAIL'}"
    )
    if not leakage_check:
        raise RuntimeError("Leakage checks failed before model training")

    train_dataset = IndexedPairDataset(samples[train_indices], drug_features, side_features)
    validation_dataset = IndexedPairDataset(
        samples[validation_indices], drug_features, side_features
    )
    test_dataset = IndexedPairDataset(samples[test_indices], drug_features, side_features)

    sys.path.insert(0, str(PROJECT_ROOT))
    from models.mssf_clean import MSSFClean, MSSFCleanConfig
    from models.mssf_clean_kg import BioKORFCleanKG

    config = MSSFCleanConfig(dropout=DROPOUT, gp=LATENT_DIM)
    configure_reproducibility(SEED)
    clean_model = MSSFClean(config)
    configure_reproducibility(SEED)
    kg_model = BioKORFCleanKG(config, KG_ARTIFACT_PATH)
    common_initialization = all(
        torch.equal(value, kg_model.state_dict()[name])
        for name, value in clean_model.state_dict().items()
    )
    if not common_initialization:
        raise AssertionError("Common MSSF modules were not initialized identically")
    clean_trainable = sum(p.numel() for p in clean_model.parameters() if p.requires_grad)
    kg_trainable = sum(p.numel() for p in kg_model.parameters() if p.requires_grad)
    clean_total = sum(p.numel() for p in clean_model.parameters())
    kg_total = sum(p.numel() for p in kg_model.parameters())
    frozen_kg_before = bool(
        not list(kg_model.kg_features.parameters())
        and all(not value.requires_grad for _, value in kg_model.kg_features.named_buffers())
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean_model = clean_model.to(device)
    kg_model = kg_model.to(device)

    print(f"Device: {device}")
    print(f"Clean parameters: trainable={clean_trainable} total={clean_total}")
    print(f"Clean+KG parameters: trainable={kg_trainable} total={kg_total}")
    print("Evaluation BVI policy: model.eval() uses latent=mu; no latent sampling")

    clean_history, clean_best_state, clean_best_epoch, clean_best_f1 = train_model(
        "clean", clean_model, train_dataset, validation_dataset, device, use_kg=False
    )
    write_history(OUTPUT_DIR / "clean_history.csv", clean_history)
    del clean_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    kg_history, kg_best_state, kg_best_epoch, kg_best_f1 = train_model(
        "clean_kg", kg_model, train_dataset, validation_dataset, device, use_kg=True
    )
    write_history(OUTPUT_DIR / "kg_history.csv", kg_history)

    # Test is evaluated exactly once per model, after validation-only selection.
    configure_reproducibility(SEED)
    clean_best_model = MSSFClean(config).to(device)
    clean_best_model.load_state_dict(clean_best_state, strict=True)
    _, clean_test_metrics, _ = evaluate(
        clean_best_model, test_dataset, device, use_kg=False
    )
    kg_model.load_state_dict(kg_best_state, strict=True)
    _, kg_test_metrics, final_gate_stats = evaluate(
        kg_model, test_dataset, device, use_kg=True, collect_gate_categories=True
    )

    frozen_kg_after = bool(
        not list(kg_model.kg_features.parameters())
        and all(not value.requires_grad for _, value in kg_model.kg_features.named_buffers())
        and sha256(KG_ARTIFACT_PATH) == kg_hash_before
    )
    frozen_kg_check = frozen_kg_before and frozen_kg_after
    protected_after = {path: sha256(path) for path in PROTECTED_HASHES}
    protected_files_safe = protected_before == protected_after == PROTECTED_HASHES

    identical_split = bool(
        np.array_equal(train_dataset.drug_index.numpy(), samples[train_indices, 0])
        and np.array_equal(validation_dataset.side_index.numpy(), samples[validation_indices, 1])
        and np.array_equal(test_dataset.labels.numpy(), samples[test_indices, 2])
    )
    fairness_check = bool(
        identical_split
        and common_initialization
        and LATENT_DIM == 64
        and isinstance(clean_best_model.classification_loss, nn.CrossEntropyLoss)
        and isinstance(kg_model.classification_loss, nn.CrossEntropyLoss)
        and protected_files_safe
        and frozen_kg_check
    )

    clean_metrics_path = OUTPUT_DIR / "clean_test_metrics.json"
    kg_metrics_path = OUTPUT_DIR / "kg_test_metrics.json"
    clean_metrics_path.write_text(
        json.dumps(
            {**clean_test_metrics, "selected_epoch": clean_best_epoch}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    kg_metrics_path.write_text(
        json.dumps(
            {
                **kg_test_metrics,
                "selected_epoch": kg_best_epoch,
                "gate_diagnostics": final_gate_stats,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    write_confusion(
        OUTPUT_DIR / "clean_confusion_matrix.csv", clean_test_metrics["confusion_matrix"]
    )
    write_confusion(
        OUTPUT_DIR / "kg_confusion_matrix.csv", kg_test_metrics["confusion_matrix"]
    )

    comparison = {
        "clean": {
            key: clean_test_metrics[key] for key in ("accuracy", "macro_f1", "aupr")
        },
        "clean_kg": {
            key: kg_test_metrics[key] for key in ("accuracy", "macro_f1", "aupr")
        },
        "delta_kg_minus_clean": {
            key: kg_test_metrics[key] - clean_test_metrics[key]
            for key in ("accuracy", "macro_f1", "aupr")
        },
    }
    (OUTPUT_DIR / "pilot_comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )

    report_lines = [
        "BioKORF fold-1 clean versus frozen-KG pilot",
        "============================================",
        f"Fold: {FOLD}",
        f"Seed: {SEED}",
        f"Epochs: {EPOCHS}",
        f"Device: {device}",
        f"Samples total/train/validation/test: {len(samples)}/{len(train_indices)}/{len(validation_indices)}/{len(test_indices)}",
        f"Split file: {SPLIT_PATH}",
        "Outer split: original StratifiedKFold(10, shuffle=True, random_state=42), first fold as test",
        f"Validation split: {VALIDATION_FRACTION:.0%} stratified from non-test development data, random_state=42",
        f"Optimizer both models: Adam(lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})",
        "Training objective both models: CrossEntropy + 0.001*KL + 0.0001*EN-con reconstruction + 0.0001*EN-add reconstruction",
        "Evaluation BVI policy both models: eval mode returns latent=mu deterministically; no sampling",
        "Model selection: maximum validation Macro-F1 within epochs 1..5; test never consulted",
        f"Clean parameters trainable/total: {clean_trainable}/{clean_total}",
        f"Clean+KG parameters trainable/total: {kg_trainable}/{kg_total}",
        f"Common MSSF initialization identical: {'PASS' if common_initialization else 'FAIL'}",
        f"Clean best epoch/validation Macro-F1: {clean_best_epoch}/{clean_best_f1:.8f}",
        f"Clean+KG best epoch/validation Macro-F1: {kg_best_epoch}/{kg_best_f1:.8f}",
        f"Clean test accuracy/Macro-F1/AUPR: {clean_test_metrics['accuracy']:.8f}/{clean_test_metrics['macro_f1']:.8f}/{clean_test_metrics['aupr']:.8f}",
        f"Clean+KG test accuracy/Macro-F1/AUPR: {kg_test_metrics['accuracy']:.8f}/{kg_test_metrics['macro_f1']:.8f}/{kg_test_metrics['aupr']:.8f}",
        f"Delta KG-clean accuracy/Macro-F1/AUPR: {comparison['delta_kg_minus_clean']['accuracy']:.8f}/{comparison['delta_kg_minus_clean']['macro_f1']:.8f}/{comparison['delta_kg_minus_clean']['aupr']:.8f}",
        "Final clean+KG test gate category means:",
        f"- drug available, side available: {final_gate_stats['both_available_gate_mean']:.8f}",
        f"- drug available, side unavailable: {final_gate_stats['drug_only_gate_mean']:.8f}",
        f"- drug unavailable, side available: {final_gate_stats['side_only_gate_mean']:.8f}",
        f"- neither available: {final_gate_stats['neither_available_gate_mean']:.8f}",
        f"FROZEN KG CHECK: {'PASS' if frozen_kg_check else 'FAIL'}",
        f"LABEL-DERIVED FEATURE LEAKAGE CHECK: {'PASS' if label_feature_leakage_safe else 'FAIL'}",
        f"Drug-Phenotype graph leakage check: {'PASS' if graph_leakage_safe else 'FAIL'}",
        f"EXPERIMENT FAIRNESS CHECK: {'PASS' if fairness_check else 'FAIL'}",
        f"LEAKAGE CHECK: {'PASS' if leakage_check else 'FAIL'}",
        f"Protected-file safety check: {'PASS' if protected_files_safe else 'FAIL'}",
        "Test evaluations performed after selection: exactly one per model",
        "Ordinal learning/new attention/alignment/R-GCN fine-tuning/KG rewiring: none",
    ]
    report = "\n".join(report_lines) + "\n"
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report, end="")
    if not frozen_kg_check or not fairness_check or not leakage_check:
        raise RuntimeError("One or more final experiment checks failed")


if __name__ == "__main__":
    main()
