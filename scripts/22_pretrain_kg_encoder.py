"""Reproducible leakage-safe self-supervised pretraining for the BioKORF R-GCN."""

from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
from array import array
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RGCN_DIR = PROJECT_ROOT / "data_processed" / "rgcn"
NODE_INDEX_PATH = RGCN_DIR / "node_index.csv"
RELATION_INDEX_PATH = RGCN_DIR / "relation_index.csv"
ORIGINAL_EDGES_PATH = RGCN_DIR / "edges_indexed.csv"
FULL_MESSAGE_EDGES_PATH = RGCN_DIR / "edges_with_inverse.csv"
FEATURE_PATH = RGCN_DIR / "node_feature_metadata.csv"
METADATA_PATH = RGCN_DIR / "graph_metadata.json"
TRAIN_EDGES_PATH = RGCN_DIR / "pretraining_train_edges.csv"
VAL_EDGES_PATH = RGCN_DIR / "pretraining_val_edges.csv"
HISTORY_PATH = RGCN_DIR / "kg_pretraining_history.csv"
REPORT_PATH = RGCN_DIR / "kg_pretraining_report.txt"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "kg_encoder_best.pt"

SEED = 42
TRAIN_FRACTION = 0.90
MAX_RELATION_SAMPLES = 5_000
MAX_EPOCHS = 60
PATIENCE = 8
HIDDEN_DIM = 128
OUTPUT_DIM = 128
NUM_BASES = 8
DROPOUT = 0.2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
MAX_GRAD_NORM = 1.0
EXCLUDED_RELATIONS = {"MAPS_TO_DRUG", "MAPS_TO_PHENOTYPE"}
EDGE_COLUMNS = (
    "source_index", "target_index", "relation_index", "source_node_id",
    "target_node_id", "relation", "source_type", "target_type",
)
HISTORY_COLUMNS = (
    "epoch", "train_loss", "val_loss", "train_positive_score",
    "train_negative_score", "val_positive_score", "val_negative_score",
    "val_roc_auc", "val_average_precision", "sampled_positive_edges",
    "learning_rate", "peak_cuda_allocated_mib",
)


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required graph artifact not found: {path}")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def configure_reproducibility() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def validate_original_edge(row: dict[str, str]) -> None:
    relation = row["relation"]
    types = {row["source_type"], row["target_type"]}
    if types == {"DRUG", "PHENOTYPE"}:
        raise ValueError(f"Drug-Phenotype edge detected: {row}")
    if relation.upper() == "ADVERSE_DRUG_REACTION":
        raise ValueError(f"ADVERSE_DRUG_REACTION relation detected: {row}")
    if types == {"BIOKORF_DRUG", "BIOKORF_SIDE"}:
        raise ValueError(f"Direct BioKORF drug-side anchor edge detected: {row}")


def stratified_split(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, dict[str, int]]]:
    by_relation: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        validate_original_edge(row)
        if row["relation"] not in EXCLUDED_RELATIONS:
            by_relation[row["relation"]].append(row)
    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    split_stats: dict[str, dict[str, int]] = {}
    for relation in sorted(by_relation):
        relation_rows = list(by_relation[relation])
        relation_index = int(relation_rows[0]["relation_index"])
        rng = random.Random(SEED + relation_index)
        rng.shuffle(relation_rows)
        count = len(relation_rows)
        if count == 1:
            val_count = 0
        else:
            val_count = min(count - 1, max(1, int(round(count * (1.0 - TRAIN_FRACTION)))))
        relation_val = relation_rows[:val_count]
        relation_train = relation_rows[val_count:]
        if not relation_train:
            raise AssertionError(f"No training edge remains for relation {relation}")
        train_rows.extend(relation_train)
        val_rows.extend(relation_val)
        split_stats[relation] = {
            "total": count, "train": len(relation_train), "validation": len(relation_val)
        }
    train_rows.sort(key=lambda row: (
        int(row["relation_index"]), int(row["source_index"]), int(row["target_index"])
    ))
    val_rows.sort(key=lambda row: (
        int(row["relation_index"]), int(row["source_index"]), int(row["target_index"])
    ))
    return train_rows, val_rows, split_stats


def write_edge_split(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EDGE_COLUMNS)
        writer.writeheader()
        writer.writerows({column: row[column] for column in EDGE_COLUMNS} for row in rows)


def build_training_message_graph(
    validation_rows: list[dict[str, str]],
    relation_name_to_index: dict[str, int],
) -> tuple[array, array, array, int]:
    forbidden: set[tuple[int, int, int]] = set()
    for row in validation_rows:
        source = int(row["source_index"])
        target = int(row["target_index"])
        relation = row["relation"]
        relation_index = int(row["relation_index"])
        inverse_name = f"{relation}__INV"
        if inverse_name not in relation_name_to_index:
            raise ValueError(f"Missing inverse relation for {relation}")
        forbidden.add((source, relation_index, target))
        forbidden.add((target, relation_name_to_index[inverse_name], source))

    sources = array("q")
    targets = array("q")
    relations = array("q")
    removed = 0
    with FULL_MESSAGE_EDGES_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            triple = (
                int(row["source_index"]),
                int(row["relation_index"]),
                int(row["target_index"]),
            )
            if triple in forbidden:
                removed += 1
                continue
            sources.append(triple[0])
            relations.append(triple[1])
            targets.append(triple[2])
    if removed != len(forbidden):
        raise ValueError(
            f"Expected to remove {len(forbidden)} validation/inverse edges; removed {removed}"
        )
    return sources, targets, relations, removed


def compact_decoder_relations(
    train_rows: list[dict[str, str]], val_rows: list[dict[str, str]]
) -> tuple[dict[str, int], dict[int, int]]:
    original_indices = {
        row["relation"]: int(row["relation_index"])
        for row in [*train_rows, *val_rows]
    }
    ordered = sorted(original_indices, key=lambda relation: original_indices[relation])
    name_to_decoder = {relation: index for index, relation in enumerate(ordered)}
    original_to_decoder = {
        original_indices[relation]: name_to_decoder[relation] for relation in ordered
    }
    return name_to_decoder, original_to_decoder


def biological_positive_set(rows: list[dict[str, str]]) -> set[tuple[int, int, int]]:
    return {
        (int(row["source_index"]), int(row["relation_index"]), int(row["target_index"]))
        for row in rows
        if row["relation"] not in EXCLUDED_RELATIONS
    }


def nodes_by_type(node_rows: list[dict[str, str]]) -> dict[str, list[int]]:
    pools: dict[str, list[int]] = defaultdict(list)
    for row in node_rows:
        pools[row["node_type"]].append(int(row["node_index"]))
    return pools


def type_aware_negatives(
    positives: list[dict[str, str]],
    all_biological_positives: set[tuple[int, int, int]],
    pools: dict[str, list[int]],
    rng: random.Random,
) -> list[tuple[int, int, int]]:
    negatives: list[tuple[int, int, int]] = []
    for row in positives:
        source = int(row["source_index"])
        target = int(row["target_index"])
        relation = int(row["relation_index"])
        for _attempt in range(10_000):
            if rng.randrange(2) == 0:
                candidate = (rng.choice(pools[row["source_type"]]), relation, target)
            else:
                candidate = (source, relation, rng.choice(pools[row["target_type"]]))
            if candidate not in all_biological_positives:
                negatives.append(candidate)
                break
        else:
            raise RuntimeError(f"Could not sample a valid negative for {row}")
    return negatives


def relation_aware_epoch_sample(
    train_by_relation: dict[str, list[dict[str, str]]], epoch: int
) -> tuple[list[dict[str, str]], dict[str, int]]:
    rng = random.Random(SEED + epoch)
    sampled: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    for relation in sorted(train_by_relation):
        rows = train_by_relation[relation]
        selected = rows if len(rows) <= MAX_RELATION_SAMPLES else rng.sample(rows, MAX_RELATION_SAMPLES)
        sampled.extend(selected)
        counts[relation] = len(selected)
    rng.shuffle(sampled)
    return sampled, counts


def triple_tensors(
    rows: list[dict[str, str]] | list[tuple[int, int, int]],
    original_to_decoder: dict[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if rows and isinstance(rows[0], dict):
        triples = [
            (
                int(row["source_index"]),
                original_to_decoder[int(row["relation_index"])],
                int(row["target_index"]),
            )
            for row in rows
        ]
    else:
        triples = [
            (source, original_to_decoder[relation], target)
            for source, relation, target in rows
        ]
    return (
        torch.tensor([triple[0] for triple in triples], dtype=torch.long, device=device),
        torch.tensor([triple[1] for triple in triples], dtype=torch.long, device=device),
        torch.tensor([triple[2] for triple in triples], dtype=torch.long, device=device),
    )


def scores_and_loss(
    decoder: Any,
    embeddings: torch.Tensor,
    positive_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    negative_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    loss_function: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positive_scores = decoder(embeddings, *positive_tensors)
    negative_scores = decoder(embeddings, *negative_tensors)
    logits = torch.cat((positive_scores, negative_scores))
    labels = torch.cat((torch.ones_like(positive_scores), torch.zeros_like(negative_scores)))
    return loss_function(logits, labels), positive_scores, negative_scores


def recursive_tensors_finite(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(recursive_tensors_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(recursive_tensors_finite(item) for item in value)
    return True


def save_checkpoint_atomically(checkpoint: dict[str, Any], epoch: int) -> None:
    """Avoid Windows in-place overwrite failures for serialized tensor files."""
    temporary_path = CHECKPOINT_PATH.with_name(
        f".{CHECKPOINT_PATH.stem}.epoch{epoch}.tmp{CHECKPOINT_PATH.suffix}"
    )
    if temporary_path.exists():
        temporary_path.unlink()
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, CHECKPOINT_PATH)


def binary_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute binary ROC-AUC by average ranks, including score ties."""
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positive_count = int(labels.sum())
    negative_count = int(labels.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        raise ValueError("ROC-AUC requires both positive and negative examples")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = ((start + 1) + end) / 2.0
        start = end
    positive_rank_sum = float(ranks[labels == 1].sum())
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def binary_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute non-interpolated binary average precision."""
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positive_count = int(labels.sum())
    if positive_count == 0:
        raise ValueError("Average precision requires at least one positive example")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    cumulative_positives = np.cumsum(sorted_labels)
    positive_positions = np.flatnonzero(sorted_labels == 1)
    precisions = cumulative_positives[positive_positions] / (positive_positions + 1)
    return float(precisions.mean())


def main() -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    from models.kg_encoder import BioKORFKGEncoder
    from models.kg_pretraining import BioKORFDistMultDecoder

    for path in (
        NODE_INDEX_PATH, RELATION_INDEX_PATH, ORIGINAL_EDGES_PATH,
        FULL_MESSAGE_EDGES_PATH, FEATURE_PATH, METADATA_PATH,
    ):
        require_file(path)
    configure_reproducibility()
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if metadata.get("leakage_check") != "PASS":
        raise ValueError("Input graph leakage check is not PASS")

    node_rows = read_rows(NODE_INDEX_PATH)
    feature_rows = read_rows(FEATURE_PATH)
    relation_rows = read_rows(RELATION_INDEX_PATH)
    original_rows = read_rows(ORIGINAL_EDGES_PATH)
    if [int(row["node_index"]) for row in node_rows] != list(range(len(node_rows))):
        raise ValueError("node_index is not continuous")
    if [int(row["node_index"]) for row in feature_rows] != list(range(len(feature_rows))):
        raise ValueError("node feature ordering does not match node_index")
    for row in original_rows:
        validate_original_edge(row)

    # Resume uses the persisted split verbatim; it must never be regenerated here.
    require_file(TRAIN_EDGES_PATH)
    require_file(VAL_EDGES_PATH)
    require_file(CHECKPOINT_PATH)
    train_rows = read_rows(TRAIN_EDGES_PATH)
    val_rows = read_rows(VAL_EDGES_PATH)
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for partition, rows in (("train", train_rows), ("validation", val_rows)):
        for row in rows:
            validate_original_edge(row)
            if row["relation"] in EXCLUDED_RELATIONS:
                raise ValueError(f"Excluded mapping relation found in persisted {partition} split")
            split_counts[row["relation"]][partition] += 1
    split_stats = {
        relation: {
            "total": counts["train"] + counts["validation"],
            "train": counts["train"],
            "validation": counts["validation"],
        }
        for relation, counts in sorted(split_counts.items())
    }
    relation_name_to_index = {row["relation"]: int(row["relation_index"]) for row in relation_rows}
    message_sources, message_targets, message_relations, removed_count = build_training_message_graph(
        val_rows, relation_name_to_index
    )
    name_to_decoder, original_to_decoder = compact_decoder_relations(train_rows, val_rows)
    all_biological_positives = biological_positive_set(original_rows)
    pools = nodes_by_type(node_rows)

    fixed_val_rng = random.Random(SEED + 1_000_000)
    fixed_val_negatives = type_aware_negatives(
        val_rows, all_biological_positives, pools, fixed_val_rng
    )
    if len(fixed_val_negatives) != len(val_rows):
        raise AssertionError("Validation positive/negative counts differ")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_type_index = torch.tensor(
        [int(row["type_index"]) for row in feature_rows], dtype=torch.long, device=device
    )
    edge_index = torch.stack((
        torch.tensor(message_sources, dtype=torch.long),
        torch.tensor(message_targets, dtype=torch.long),
    )).to(device)
    edge_type = torch.tensor(message_relations, dtype=torch.long, device=device)
    val_positive_tensors = triple_tensors(val_rows, original_to_decoder, device)
    val_negative_tensors = triple_tensors(fixed_val_negatives, original_to_decoder, device)

    encoder = BioKORFKGEncoder(
        num_node_types=len(set(int(row["type_index"]) for row in feature_rows)),
        num_relations=len(relation_rows),
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        num_bases=NUM_BASES,
        dropout=DROPOUT,
    ).to(device)
    decoder = BioKORFDistMultDecoder(
        num_relations=len(name_to_decoder), embedding_dim=OUTPUT_DIM
    ).to(device)
    optimizer = torch.optim.AdamW(
        [*encoder.parameters(), *decoder.parameters()],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    resumed_checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    encoder.load_state_dict(resumed_checkpoint["kg_encoder_state_dict"])
    decoder.load_state_dict(resumed_checkpoint["distmult_decoder_state_dict"])
    optimizer.load_state_dict(resumed_checkpoint["optimizer_state_dict"])
    previous_best_epoch = int(resumed_checkpoint["epoch"])
    previous_best_ap = float(resumed_checkpoint["validation_average_precision"])
    resumed_start_epoch = previous_best_epoch + 1
    if resumed_start_epoch > MAX_EPOCHS:
        raise ValueError(
            f"Checkpoint epoch {previous_best_epoch} is already at or beyond total epoch {MAX_EPOCHS}"
        )
    loss_function = torch.nn.BCEWithLogitsLoss()

    train_by_relation: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in train_rows:
        train_by_relation[row["relation"]].append(row)

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing_history = read_rows(HISTORY_PATH) if HISTORY_PATH.is_file() else []
    # Rows after the checkpoint do not belong to the resumed checkpoint lineage.
    history: list[dict[str, Any]] = [
        row for row in existing_history if int(row["epoch"]) <= previous_best_epoch
    ]
    if len({int(row["epoch"]) for row in history}) != len(history):
        raise ValueError("Existing history contains duplicate epoch rows")
    checkpoint_history = next(
        (row for row in history if int(row["epoch"]) == previous_best_epoch), None
    )
    best_ap = previous_best_ap
    best_record: dict[str, Any] = (
        {
            "epoch": previous_best_epoch,
            "val_loss": float(checkpoint_history["val_loss"]),
            "val_roc_auc": float(checkpoint_history["val_roc_auc"]),
            "val_average_precision": float(checkpoint_history["val_average_precision"]),
            "val_positive_score": float(checkpoint_history["val_positive_score"]),
            "val_negative_score": float(checkpoint_history["val_negative_score"]),
        }
        if checkpoint_history is not None
        else {
            "epoch": previous_best_epoch,
            "val_loss": float(resumed_checkpoint["validation_loss"]),
            "val_roc_auc": float(resumed_checkpoint["validation_roc_auc"]),
            "val_average_precision": previous_best_ap,
            "val_positive_score": math.nan,
            "val_negative_score": math.nan,
        }
    )
    epochs_without_improvement = 0
    checkpoint_improved = False
    global_peak_mib = 0.0
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Training message graph edges: {len(message_sources)}")
    print(f"Train biological edges: {len(train_rows)}")
    print(f"Validation biological edges: {len(val_rows)}")
    print(f"Validation and inverse message edges removed: {removed_count}")
    print(f"Supervised decoder relations: {len(name_to_decoder)}")
    print(f"Previous best epoch: {previous_best_epoch}")
    print(f"Previous best AP: {previous_best_ap:.8f}")
    print(f"Resumed starting epoch: {resumed_start_epoch}")

    for epoch in range(resumed_start_epoch, MAX_EPOCHS + 1):
        torch.manual_seed(SEED + epoch)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED + epoch)
        sampled_rows, sample_counts = relation_aware_epoch_sample(train_by_relation, epoch)
        if epoch == resumed_start_epoch:
            print(
                f"Resume epoch {epoch} relation sample counts: "
                f"{dict(sorted(sample_counts.items()))}"
            )
        train_rng = random.Random(SEED + epoch)
        train_negatives = type_aware_negatives(
            sampled_rows, all_biological_positives, pools, train_rng
        )
        train_positive_tensors = triple_tensors(sampled_rows, original_to_decoder, device)
        train_negative_tensors = triple_tensors(train_negatives, original_to_decoder, device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        encoder.train(); decoder.train()
        optimizer.zero_grad(set_to_none=True)
        embeddings = encoder(node_type_index, edge_index, edge_type)
        train_loss, train_positive_scores, train_negative_scores = scores_and_loss(
            decoder, embeddings, train_positive_tensors, train_negative_tensors, loss_function
        )
        if not torch.isfinite(train_loss):
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [*encoder.parameters(), *decoder.parameters()], max_norm=MAX_GRAD_NORM
        )
        optimizer.step()
        del embeddings

        encoder.eval(); decoder.eval()
        with torch.no_grad():
            validation_embeddings = encoder(node_type_index, edge_index, edge_type)
            val_loss, val_positive_scores, val_negative_scores = scores_and_loss(
                decoder, validation_embeddings, val_positive_tensors,
                val_negative_tensors, loss_function
            )
            validation_logits = torch.cat((val_positive_scores, val_negative_scores))
            validation_labels = np.concatenate((
                np.ones(len(val_rows), dtype=np.int8),
                np.zeros(len(val_rows), dtype=np.int8),
            ))
            validation_probabilities = torch.sigmoid(validation_logits).cpu().numpy()
            val_roc_auc = binary_roc_auc(validation_labels, validation_probabilities)
            val_ap = binary_average_precision(validation_labels, validation_probabilities)

        peak_mib = (
            torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
        )
        global_peak_mib = max(global_peak_mib, peak_mib)
        record = {
            "epoch": epoch,
            "train_loss": float(train_loss.detach().item()),
            "val_loss": float(val_loss.detach().item()),
            "train_positive_score": float(train_positive_scores.detach().mean().item()),
            "train_negative_score": float(train_negative_scores.detach().mean().item()),
            "val_positive_score": float(val_positive_scores.detach().mean().item()),
            "val_negative_score": float(val_negative_scores.detach().mean().item()),
            "val_roc_auc": val_roc_auc,
            "val_average_precision": val_ap,
            "sampled_positive_edges": len(sampled_rows),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "peak_cuda_allocated_mib": peak_mib,
        }
        history.append(record)
        with HISTORY_PATH.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS)
            writer.writeheader(); writer.writerows(history)

        improved = val_ap > best_ap + 1e-12
        if improved:
            best_ap = val_ap
            best_record = dict(record)
            epochs_without_improvement = 0
            checkpoint_improved = True
            checkpoint = {
                "kg_encoder_state_dict": encoder.state_dict(),
                "distmult_decoder_state_dict": decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "validation_loss": record["val_loss"],
                "validation_roc_auc": record["val_roc_auc"],
                "validation_average_precision": record["val_average_precision"],
                "model_hyperparameters": {
                    "num_node_types": encoder.num_node_types,
                    "num_message_relations": encoder.num_relations,
                    "num_supervised_relations": len(name_to_decoder),
                    "supervised_relation_to_decoder_index": name_to_decoder,
                    "hidden_dim": HIDDEN_DIM, "output_dim": OUTPUT_DIM,
                    "num_bases": NUM_BASES, "dropout": DROPOUT,
                    "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
                    "max_grad_norm": MAX_GRAD_NORM,
                },
                "random_seed": SEED,
            }
            save_checkpoint_atomically(checkpoint, epoch)
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:02d} | train_loss={record['train_loss']:.6f} "
            f"val_loss={record['val_loss']:.6f} val_auc={val_roc_auc:.6f} "
            f"val_ap={val_ap:.6f} pos={record['val_positive_score']:.6f} "
            f"neg={record['val_negative_score']:.6f} peak={peak_mib:.2f} MiB"
        )
        del validation_embeddings, validation_logits, validation_probabilities
        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch} after {PATIENCE} epochs without AP improvement")
            break

    if not CHECKPOINT_PATH.is_file():
        raise AssertionError("Best checkpoint was not created")
    checkpoint_cpu = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    if not recursive_tensors_finite(checkpoint_cpu):
        raise FloatingPointError("Checkpoint contains a non-finite tensor")

    validation_forbidden = {
        (int(row["source_index"]), int(row["relation_index"]), int(row["target_index"]))
        for row in val_rows
    }
    validation_inverse_forbidden = {
        (
            int(row["target_index"]),
            relation_name_to_index[f"{row['relation']}__INV"],
            int(row["source_index"]),
        )
        for row in val_rows
    }
    retained_message = set(zip(message_sources, message_relations, message_targets))
    if retained_message.intersection(validation_forbidden):
        raise AssertionError("A validation edge remains in the training message graph")
    if retained_message.intersection(validation_inverse_forbidden):
        raise AssertionError("A validation inverse remains in the training message graph")

    final_epoch = int(history[-1]["epoch"])
    score_warning = best_record["val_positive_score"] <= best_record["val_negative_score"]
    report_lines = [
        "BioKORF KG encoder pretraining report",
        "=" * 38,
        f"Device: {device}",
        f"Epochs completed: {len(history)}",
        f"Previous best epoch: {previous_best_epoch}",
        f"Previous best AP: {previous_best_ap:.8f}",
        f"Resumed starting epoch: {resumed_start_epoch}",
        f"Final epoch: {final_epoch}",
        f"Checkpoint improved during resume: {checkpoint_improved}",
        f"Best epoch: {best_record['epoch']}",
        f"Best validation loss: {best_record['val_loss']:.8f}",
        f"Best validation ROC-AUC: {best_record['val_roc_auc']:.8f}",
        f"Best validation AP: {best_record['val_average_precision']:.8f}",
        f"Best mean positive score: {best_record['val_positive_score']:.8f}",
        f"Best mean negative score: {best_record['val_negative_score']:.8f}",
        f"Positive-score sanity check: {'WARNING: positive mean is not greater than negative mean' if score_warning else 'PASS'}",
        f"Training message graph edge count: {len(message_sources)}",
        f"Train biological edge count: {len(train_rows)}",
        f"Validation biological edge count: {len(val_rows)}",
        f"Validation/inverse edges removed: {removed_count}",
        f"Relation-wise split statistics: {json.dumps(split_stats, sort_keys=True)}",
        f"Peak CUDA memory: {global_peak_mib:.2f} MiB",
        f"Checkpoint path: {CHECKPOINT_PATH}",
        "Checkpoint tensor finite check: PASS",
        "Validation edges absent from message graph: PASS",
        "Validation inverse edges absent from message graph: PASS",
        "No Drug-Phenotype relation: PASS",
        "No ADVERSE_DRUG_REACTION relation: PASS",
        "No drug-side-effect frequency labels loaded: PASS",
        "LEAKAGE CHECK: PASS",
        "KG embeddings exported: no",
        "MSSF integration performed: no",
    ]
    report_text = "\n".join(report_lines) + "\n"
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(report_text, end="")


if __name__ == "__main__":
    main()
