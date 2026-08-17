"""Export deterministic frozen BioKORF KG anchor embeddings."""

from __future__ import annotations

import csv
import os
import random
import sys
from array import array
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RGCN_DIR = PROJECT_ROOT / "data_processed" / "rgcn"
OUTPUT_DIR = PROJECT_ROOT / "data_processed" / "kg_features"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "kg_encoder_best.pt"
NODE_INDEX_PATH = RGCN_DIR / "node_index.csv"
RELATION_INDEX_PATH = RGCN_DIR / "relation_index.csv"
EDGE_PATH = RGCN_DIR / "edges_with_inverse.csv"
FEATURE_PATH = RGCN_DIR / "node_feature_metadata.csv"
DRUG_ANCHOR_PATH = RGCN_DIR / "drug_anchor_indices.csv"
SIDE_ANCHOR_PATH = RGCN_DIR / "side_anchor_indices.csv"
DRUG_MAPPING_PATH = (
    PROJECT_ROOT / "data_processed" / "optimuskg" / "final_drug_anchor_mapping.csv"
)
SIDE_MAPPING_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "optimuskg"
    / "final_side_effect_anchor_mapping.csv"
)

PT_OUTPUT_PATH = OUTPUT_DIR / "biokorf_kg_embeddings.pt"
DRUG_INDEX_OUTPUT_PATH = OUTPUT_DIR / "drug_kg_embedding_index.csv"
SIDE_INDEX_OUTPUT_PATH = OUTPUT_DIR / "side_kg_embedding_index.csv"
DRUG_NPY_OUTPUT_PATH = OUTPUT_DIR / "drug_kg_embeddings.npy"
SIDE_NPY_OUTPUT_PATH = OUTPUT_DIR / "side_kg_embeddings.npy"
DRUG_MASK_NPY_OUTPUT_PATH = OUTPUT_DIR / "drug_kg_available_mask.npy"
SIDE_MASK_NPY_OUTPUT_PATH = OUTPUT_DIR / "side_kg_available_mask.npy"
REPORT_PATH = OUTPUT_DIR / "kg_embedding_export_report.txt"

EXPECTED_NODES = 21_829
EXPECTED_EDGES = 1_048_598
EXPECTED_RELATIONS = 40
EXPECTED_DRUGS = 757
EXPECTED_SIDES = 994
EXPECTED_DIM = 128
EXPECTED_DRUG_CONNECTED = 730
EXPECTED_SIDE_CONNECTED = 319
SEED = 42


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact not found: {path}")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_bool(value: str, field: str) -> bool:
    normalized = value.strip().casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Invalid Boolean value for {field}: {value!r}")


def configure_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_graph_inputs() -> tuple[list[int], array, array, array, int, int]:
    node_rows = read_rows(NODE_INDEX_PATH)
    feature_rows = read_rows(FEATURE_PATH)
    relation_rows = read_rows(RELATION_INDEX_PATH)

    if len(node_rows) != EXPECTED_NODES or len(feature_rows) != EXPECTED_NODES:
        raise ValueError(
            f"Expected {EXPECTED_NODES} node/index rows; got "
            f"{len(node_rows)} and {len(feature_rows)}"
        )
    node_indices = [int(row["node_index"]) for row in node_rows]
    feature_indices = [int(row["node_index"]) for row in feature_rows]
    expected_nodes = list(range(EXPECTED_NODES))
    if node_indices != expected_nodes or feature_indices != expected_nodes:
        raise ValueError("Node indices are not continuous and identically ordered")
    for node_row, feature_row in zip(node_rows, feature_rows, strict=True):
        if (
            node_row["node_id"] != feature_row["node_id"]
            or node_row["node_type"] != feature_row["node_type"]
        ):
            raise ValueError("node_index.csv and node_feature_metadata.csv are misaligned")

    relation_indices = [int(row["relation_index"]) for row in relation_rows]
    if relation_indices != list(range(len(relation_rows))):
        raise ValueError("Relation indices are not continuous and ordered")
    if len(relation_rows) != EXPECTED_RELATIONS:
        raise ValueError(
            f"Expected {EXPECTED_RELATIONS} relations, found {len(relation_rows)}"
        )

    node_type_index = [int(row["type_index"]) for row in feature_rows]
    sources = array("q")
    targets = array("q")
    relations = array("q")
    leakage_count = 0
    with EDGE_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            source = int(row["source_index"])
            target = int(row["target_index"])
            relation = int(row["relation_index"])
            if not (0 <= source < EXPECTED_NODES and 0 <= target < EXPECTED_NODES):
                raise ValueError(f"Out-of-range endpoint at edge row {row_number}")
            if not 0 <= relation < len(relation_rows):
                raise ValueError(f"Out-of-range relation at edge row {row_number}")
            endpoint_types = {row["source_type"], row["target_type"]}
            relation_name = row["relation"].upper()
            forbidden = (
                endpoint_types == {"DRUG", "PHENOTYPE"}
                or relation_name == "ADVERSE_DRUG_REACTION"
                or endpoint_types == {"BIOKORF_DRUG", "BIOKORF_SIDE"}
            )
            if forbidden:
                leakage_count += 1
            sources.append(source)
            targets.append(target)
            relations.append(relation)

    if len(sources) != EXPECTED_EDGES:
        raise ValueError(f"Expected {EXPECTED_EDGES} edges, found {len(sources)}")
    if leakage_count:
        raise ValueError(f"Leakage check failed: found {leakage_count} forbidden edges")
    return node_type_index, sources, targets, relations, len(relation_rows), leakage_count


def load_anchors(
    path: Path,
    expected_count: int,
    id_column: str,
    id_prefix: str,
    expected_connected: int,
) -> tuple[list[dict[str, str]], list[int], list[bool]]:
    rows = read_rows(path)
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} anchors in {path}, found {len(rows)}")
    matrix_indices = [int(row["matrix_index"]) for row in rows]
    if matrix_indices != list(range(expected_count)):
        raise ValueError(f"matrix_index is not continuous and ordered in {path}")
    expected_ids = [f"{id_prefix}{index:03d}" for index in range(expected_count)]
    if [row[id_column] for row in rows] != expected_ids:
        raise ValueError(f"Anchor IDs do not correspond exactly to matrix_index in {path}")
    graph_indices = [int(row["graph_node_index"]) for row in rows]
    if any(index < 0 or index >= EXPECTED_NODES for index in graph_indices):
        raise ValueError(f"Anchor graph index outside valid range in {path}")
    masks = [parse_bool(row["has_kg_connection"], "has_kg_connection") for row in rows]
    if sum(masks) != expected_connected:
        raise ValueError(
            f"Expected {expected_connected} connected anchors in {path}, found {sum(masks)}"
        )
    return rows, graph_indices, masks


def load_names(
    path: Path,
    expected_count: int,
    id_column: str,
    name_column: str,
    expected_ids: list[str],
) -> list[str]:
    rows = read_rows(path)
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} mapping rows in {path}, found {len(rows)}")
    if [int(row["matrix_index"]) for row in rows] != list(range(expected_count)):
        raise ValueError(f"Mapping rows are not in matrix order in {path}")
    if [row[id_column] for row in rows] != expected_ids:
        raise ValueError(f"Mapping anchor IDs do not match graph anchors in {path}")
    return [row[name_column] for row in rows]


def mean_norm(embeddings: torch.Tensor, mask: torch.Tensor) -> float:
    subset = embeddings[mask]
    if subset.shape[0] == 0:
        return float("nan")
    return float(torch.linalg.vector_norm(subset, dim=1).mean().item())


def save_csv_atomic(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    required = (
        CHECKPOINT_PATH,
        NODE_INDEX_PATH,
        RELATION_INDEX_PATH,
        EDGE_PATH,
        FEATURE_PATH,
        DRUG_ANCHOR_PATH,
        SIDE_ANCHOR_PATH,
        DRUG_MAPPING_PATH,
        SIDE_MAPPING_PATH,
    )
    for path in required:
        require_file(path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    required_checkpoint_keys = {
        "kg_encoder_state_dict",
        "epoch",
        "validation_average_precision",
        "validation_roc_auc",
        "model_hyperparameters",
    }
    missing_checkpoint_keys = required_checkpoint_keys.difference(checkpoint)
    if missing_checkpoint_keys:
        raise KeyError(f"Checkpoint is missing keys: {sorted(missing_checkpoint_keys)}")
    hyperparameters = checkpoint["model_hyperparameters"]
    checkpoint_seed = int(checkpoint.get("random_seed", SEED))
    configure_determinism(checkpoint_seed)

    sys.path.insert(0, str(PROJECT_ROOT))
    from models.kg_encoder import BioKORFKGEncoder

    encoder = BioKORFKGEncoder(
        num_node_types=int(hyperparameters["num_node_types"]),
        num_relations=int(hyperparameters["num_message_relations"]),
        hidden_dim=int(hyperparameters["hidden_dim"]),
        output_dim=int(hyperparameters["output_dim"]),
        num_bases=int(hyperparameters["num_bases"]),
        dropout=float(hyperparameters["dropout"]),
    )
    encoder.load_state_dict(checkpoint["kg_encoder_state_dict"], strict=True)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise AssertionError("Encoder parameters were not fully frozen")

    print("Loading and leakage-checking the full biomedical graph...")
    node_types, sources, targets, relations, num_relations, leakage_count = (
        load_graph_inputs()
    )
    if num_relations != int(hyperparameters["num_message_relations"]):
        raise ValueError("Graph relation count does not match the checkpoint")

    drug_rows, drug_graph_indices, drug_mask_values = load_anchors(
        DRUG_ANCHOR_PATH,
        EXPECTED_DRUGS,
        "biokorf_drug_id",
        "BIOKORF_DRUG_",
        EXPECTED_DRUG_CONNECTED,
    )
    side_rows, side_graph_indices, side_mask_values = load_anchors(
        SIDE_ANCHOR_PATH,
        EXPECTED_SIDES,
        "biokorf_side_id",
        "BIOKORF_SIDE_",
        EXPECTED_SIDE_CONNECTED,
    )
    drug_names = load_names(
        DRUG_MAPPING_PATH,
        EXPECTED_DRUGS,
        "biokorf_drug_id",
        "drug_name",
        [row["biokorf_drug_id"] for row in drug_rows],
    )
    side_names = load_names(
        SIDE_MAPPING_PATH,
        EXPECTED_SIDES,
        "biokorf_side_id",
        "side_effect_name",
        [row["biokorf_side_id"] for row in side_rows],
    )
    if drug_names[0].casefold() != "lepirudin":
        raise ValueError("BIOKORF_DRUG_000 does not correspond to lepirudin")
    if side_names[0].casefold() != "abdominal discomfort":
        raise ValueError(
            "BIOKORF_SIDE_000 does not correspond to abdominal discomfort"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device).eval()
    node_type_tensor = torch.tensor(node_types, dtype=torch.long, device=device)
    source_tensor = torch.frombuffer(sources, dtype=torch.int64).clone()
    target_tensor = torch.frombuffer(targets, dtype=torch.int64).clone()
    edge_index = torch.stack((source_tensor, target_tensor), dim=0).to(device)
    edge_type = torch.frombuffer(relations, dtype=torch.int64).clone().to(device)
    drug_graph_tensor = torch.tensor(drug_graph_indices, dtype=torch.long, device=device)
    side_graph_tensor = torch.tensor(side_graph_indices, dtype=torch.long, device=device)
    drug_mask = torch.tensor(drug_mask_values, dtype=torch.bool, device=device)
    side_mask = torch.tensor(side_mask_values, dtype=torch.bool, device=device)

    print(f"Running two frozen eval passes on {device} for determinism validation...")
    with torch.no_grad():
        first_node_embeddings = encoder(node_type_tensor, edge_index, edge_type)
        if tuple(first_node_embeddings.shape) != (EXPECTED_NODES, EXPECTED_DIM):
            raise ValueError(
                f"Unexpected node embedding shape: {tuple(first_node_embeddings.shape)}"
            )
        first_drug, first_drug_mask = encoder.extract_drug_anchor_embeddings(
            first_node_embeddings, drug_graph_tensor, drug_mask
        )
        first_side, first_side_mask = encoder.extract_side_anchor_embeddings(
            first_node_embeddings, side_graph_tensor, side_mask
        )
        drug_embeddings = first_drug.detach().cpu().to(torch.float32).contiguous()
        side_embeddings = first_side.detach().cpu().to(torch.float32).contiguous()
        saved_drug_mask = first_drug_mask.detach().cpu().contiguous()
        saved_side_mask = first_side_mask.detach().cpu().contiguous()
        del first_node_embeddings, first_drug, first_side

        second_node_embeddings = encoder(node_type_tensor, edge_index, edge_type)
        second_drug, _ = encoder.extract_drug_anchor_embeddings(
            second_node_embeddings, drug_graph_tensor, drug_mask
        )
        second_side, _ = encoder.extract_side_anchor_embeddings(
            second_node_embeddings, side_graph_tensor, side_mask
        )
        deterministic_drug = torch.allclose(drug_embeddings, second_drug.cpu())
        deterministic_side = torch.allclose(side_embeddings, second_side.cpu())
        del second_node_embeddings, second_drug, second_side

    deterministic = bool(deterministic_drug and deterministic_side)
    finite = bool(
        torch.isfinite(drug_embeddings).all()
        and torch.isfinite(side_embeddings).all()
    )
    if tuple(drug_embeddings.shape) != (EXPECTED_DRUGS, EXPECTED_DIM):
        raise ValueError(f"Unexpected drug embedding shape: {tuple(drug_embeddings.shape)}")
    if tuple(side_embeddings.shape) != (EXPECTED_SIDES, EXPECTED_DIM):
        raise ValueError(f"Unexpected side embedding shape: {tuple(side_embeddings.shape)}")
    if tuple(saved_drug_mask.shape) != (EXPECTED_DRUGS,):
        raise ValueError(f"Unexpected drug mask shape: {tuple(saved_drug_mask.shape)}")
    if tuple(saved_side_mask.shape) != (EXPECTED_SIDES,):
        raise ValueError(f"Unexpected side mask shape: {tuple(saved_side_mask.shape)}")
    if not deterministic:
        raise RuntimeError("Deterministic export check failed")
    if not finite:
        raise RuntimeError("Embedding finite-value check failed")

    frozen_encoder = encoder.cpu()
    missing_drug = ~saved_drug_mask
    missing_side = ~saved_side_mask
    expected_drug_fallback = frozen_encoder.missing_drug_kg_embedding.detach().expand(
        int(missing_drug.sum()), -1
    )
    expected_side_fallback = frozen_encoder.missing_side_kg_embedding.detach().expand(
        int(missing_side.sum()), -1
    )
    missing_drug_fallback_ok = bool(
        torch.allclose(drug_embeddings[missing_drug], expected_drug_fallback)
    )
    missing_side_fallback_ok = bool(
        torch.allclose(side_embeddings[missing_side], expected_side_fallback)
    )
    if not missing_drug_fallback_ok or not missing_side_fallback_ok:
        raise RuntimeError("Missing-anchor fallback representation check failed")

    drug_connected_norm = mean_norm(drug_embeddings, saved_drug_mask)
    drug_missing_norm = mean_norm(drug_embeddings, missing_drug)
    side_connected_norm = mean_norm(side_embeddings, saved_side_mask)
    side_missing_norm = mean_norm(side_embeddings, missing_side)

    artifact = {
        "drug_embeddings": drug_embeddings,
        "side_embeddings": side_embeddings,
        "drug_available_mask": saved_drug_mask,
        "side_available_mask": saved_side_mask,
        "drug_matrix_indices": torch.arange(EXPECTED_DRUGS, dtype=torch.long),
        "side_matrix_indices": torch.arange(EXPECTED_SIDES, dtype=torch.long),
        "embedding_dim": EXPECTED_DIM,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_validation_ap": float(checkpoint["validation_average_precision"]),
        "checkpoint_validation_roc_auc": float(checkpoint["validation_roc_auc"]),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "graph_num_nodes": EXPECTED_NODES,
        "graph_num_edges": len(sources),
        "num_relations": num_relations,
        "encoder_frozen": True,
    }
    temporary_pt = PT_OUTPUT_PATH.with_suffix(".pt.tmp")
    torch.save(artifact, temporary_pt)
    os.replace(temporary_pt, PT_OUTPUT_PATH)

    save_csv_atomic(
        DRUG_INDEX_OUTPUT_PATH,
        ("matrix_index", "biokorf_drug_id", "drug_name", "kg_available", "embedding_row"),
        [
            {
                "matrix_index": index,
                "biokorf_drug_id": row["biokorf_drug_id"],
                "drug_name": drug_names[index],
                "kg_available": bool(saved_drug_mask[index]),
                "embedding_row": index,
            }
            for index, row in enumerate(drug_rows)
        ],
    )
    save_csv_atomic(
        SIDE_INDEX_OUTPUT_PATH,
        ("matrix_index", "biokorf_side_id", "side_effect_name", "kg_available", "embedding_row"),
        [
            {
                "matrix_index": index,
                "biokorf_side_id": row["biokorf_side_id"],
                "side_effect_name": side_names[index],
                "kg_available": bool(saved_side_mask[index]),
                "embedding_row": index,
            }
            for index, row in enumerate(side_rows)
        ],
    )
    np.save(DRUG_NPY_OUTPUT_PATH, drug_embeddings.numpy(), allow_pickle=False)
    np.save(SIDE_NPY_OUTPUT_PATH, side_embeddings.numpy(), allow_pickle=False)
    np.save(DRUG_MASK_NPY_OUTPUT_PATH, saved_drug_mask.numpy(), allow_pickle=False)
    np.save(SIDE_MASK_NPY_OUTPUT_PATH, saved_side_mask.numpy(), allow_pickle=False)

    drug_connected = int(saved_drug_mask.sum().item())
    side_connected = int(saved_side_mask.sum().item())
    report_lines = [
        "BioKORF pretrained KG embedding export",
        "======================================",
        f"Checkpoint epoch: {checkpoint['epoch']}",
        f"Validation AP: {float(checkpoint['validation_average_precision']):.12f}",
        f"Validation ROC-AUC: {float(checkpoint['validation_roc_auc']):.12f}",
        f"Device used for export: {device}",
        f"Full graph nodes: {EXPECTED_NODES}",
        f"Full graph edges: {len(sources)}",
        f"Number of relations: {num_relations}",
        f"Embedding dimension: {EXPECTED_DIM}",
        f"Node embeddings shape: ({EXPECTED_NODES}, {EXPECTED_DIM})",
        f"Drug embeddings shape: {tuple(drug_embeddings.shape)}",
        f"Side embeddings shape: {tuple(side_embeddings.shape)}",
        f"Drug KG coverage: {drug_connected}/{EXPECTED_DRUGS} ({100.0 * drug_connected / EXPECTED_DRUGS:.2f}%)",
        f"Side KG coverage: {side_connected}/{EXPECTED_SIDES} ({100.0 * side_connected / EXPECTED_SIDES:.2f}%)",
        f"Drug connected/missing: {drug_connected}/{EXPECTED_DRUGS - drug_connected}",
        f"Side connected/missing: {side_connected}/{EXPECTED_SIDES - side_connected}",
        f"Deterministic export check: {'PASS' if deterministic else 'FAIL'}",
        f"Finite-value check: {'PASS' if finite else 'FAIL'}",
        f"Leakage check: {'PASS' if leakage_count == 0 else 'FAIL'}",
        "Leakage details: no Drug-Phenotype edges; no ADVERSE_DRUG_REACTION relation; no direct BIOKORF_DRUG-BIOKORF_SIDE edges; no frequency labels used",
        f"Mean embedding norm, connected drug anchors: {drug_connected_norm:.8f}",
        f"Mean embedding norm, connected side anchors: {side_connected_norm:.8f}",
        f"Mean embedding norm, missing drug anchors: {drug_missing_norm:.8f}",
        f"Mean embedding norm, missing side anchors: {side_missing_norm:.8f}",
        f"Missing drug fallback check: {'PASS' if missing_drug_fallback_ok else 'FAIL'}",
        f"Missing side fallback check: {'PASS' if missing_side_fallback_ok else 'FAIL'}",
        "Encoder frozen: True",
        "Inference passes: 2 (first exported; second used only for determinism validation)",
        "Output paths:",
        *[f"- {path}" for path in (
            PT_OUTPUT_PATH,
            DRUG_INDEX_OUTPUT_PATH,
            SIDE_INDEX_OUTPUT_PATH,
            DRUG_NPY_OUTPUT_PATH,
            SIDE_NPY_OUTPUT_PATH,
            DRUG_MASK_NPY_OUTPUT_PATH,
            SIDE_MASK_NPY_OUTPUT_PATH,
            REPORT_PATH,
        )],
    ]
    report = "\n".join(report_lines) + "\n"
    temporary_report = REPORT_PATH.with_suffix(".txt.tmp")
    temporary_report.write_text(report, encoding="utf-8")
    os.replace(temporary_report, REPORT_PATH)
    print(report, end="")


if __name__ == "__main__":
    main()
