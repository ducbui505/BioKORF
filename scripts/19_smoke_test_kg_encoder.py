"""Run one untrained forward-pass smoke test of the BioKORF R-GCN encoder."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from array import array
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RGCN_DIR = PROJECT_ROOT / "data_processed" / "rgcn"
NODE_INDEX_PATH = RGCN_DIR / "node_index.csv"
RELATION_INDEX_PATH = RGCN_DIR / "relation_index.csv"
EDGES_PATH = RGCN_DIR / "edges_with_inverse.csv"
DRUG_ANCHOR_PATH = RGCN_DIR / "drug_anchor_indices.csv"
SIDE_ANCHOR_PATH = RGCN_DIR / "side_anchor_indices.csv"
FEATURE_PATH = RGCN_DIR / "node_feature_metadata.csv"
METADATA_PATH = RGCN_DIR / "graph_metadata.json"
REPORT_PATH = RGCN_DIR / "kg_encoder_smoke_test.txt"
EXPECTED_NODES = 21829
EXPECTED_EDGES = 1048598
EXPECTED_DRUG_ANCHORS = 757
EXPECTED_SIDE_ANCHORS = 994


def emit_report(lines: list[str]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines).rstrip() + "\n"
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(text, end="")


def dependency_check() -> tuple[bool, list[str]]:
    missing = [
        package
        for package in ("torch", "torch_geometric")
        if importlib.util.find_spec(package) is None
    ]
    if not missing:
        return True, []
    lines = [
        "BioKORF KG encoder smoke test: DEPENDENCY MISSING",
        "=" * 52,
        f"Missing Python package(s): {', '.join(missing)}",
        "Installation required before the smoke test can run:",
        "  pip install torch torch-geometric",
        "No model was instantiated, no forward pass was run, and no training occurred.",
    ]
    return False, lines


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required R-GCN artifact not found: {path}")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_bool(value: str) -> bool:
    normalized = value.strip().casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Invalid Boolean value: {value!r}")


def load_anchor_table(
    path: Path,
    expected_count: int,
    id_column: str,
    expected_prefix: str,
) -> tuple[list[int], list[bool]]:
    rows = read_rows(path)
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} rows in {path}; found {len(rows)}")
    indices: list[int] = []
    masks: list[bool] = []
    for expected_matrix_index, row in enumerate(rows):
        if int(row["matrix_index"]) != expected_matrix_index:
            raise ValueError(f"matrix_index ordering is invalid in {path}")
        if row[id_column] != f"{expected_prefix}{expected_matrix_index:03d}":
            raise ValueError(f"Anchor ordering is invalid in {path}: {row[id_column]}")
        indices.append(int(row["graph_node_index"]))
        masks.append(parse_bool(row["has_kg_connection"]))
    return indices, masks


def main() -> int:
    dependencies_ok, dependency_report = dependency_check()
    if not dependencies_ok:
        emit_report(dependency_report)
        return 0

    import torch
    import torch_geometric

    sys.path.insert(0, str(PROJECT_ROOT))
    from models.kg_encoder import BioKORFKGEncoder

    for path in (
        NODE_INDEX_PATH, RELATION_INDEX_PATH, EDGES_PATH, DRUG_ANCHOR_PATH,
        SIDE_ANCHOR_PATH, FEATURE_PATH, METADATA_PATH,
    ):
        require_file(path)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    node_rows = read_rows(NODE_INDEX_PATH)
    feature_rows = read_rows(FEATURE_PATH)
    relation_rows = read_rows(RELATION_INDEX_PATH)
    if len(node_rows) != EXPECTED_NODES or len(feature_rows) != EXPECTED_NODES:
        raise ValueError("Node index and feature metadata must both contain 21,829 rows")
    if [int(row["node_index"]) for row in node_rows] != list(range(EXPECTED_NODES)):
        raise ValueError("node_index ordering is not exactly continuous")
    if [int(row["node_index"]) for row in feature_rows] != list(range(EXPECTED_NODES)):
        raise ValueError("Feature metadata ordering does not match node_index")
    if [int(row["relation_index"]) for row in relation_rows] != list(range(len(relation_rows))):
        raise ValueError("relation_index ordering is not exactly continuous")

    node_type_values = [int(row["type_index"]) for row in feature_rows]
    num_node_types = len(set(node_type_values))
    num_relations = len(relation_rows)
    source_values = array("q")
    target_values = array("q")
    relation_values = array("q")
    with EDGES_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            source_values.append(int(row["source_index"]))
            target_values.append(int(row["target_index"]))
            relation_values.append(int(row["relation_index"]))
    if len(source_values) != EXPECTED_EDGES:
        raise ValueError(f"Expected {EXPECTED_EDGES} inverse-expanded edges; found {len(source_values)}")

    drug_indices, drug_mask_values = load_anchor_table(
        DRUG_ANCHOR_PATH, EXPECTED_DRUG_ANCHORS, "biokorf_drug_id", "BIOKORF_DRUG_"
    )
    side_indices, side_mask_values = load_anchor_table(
        SIDE_ANCHOR_PATH, EXPECTED_SIDE_ANCHORS, "biokorf_side_id", "BIOKORF_SIDE_"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_type_index = torch.tensor(node_type_values, dtype=torch.long, device=device)
    edge_index = torch.stack(
        (
            torch.tensor(source_values, dtype=torch.long),
            torch.tensor(target_values, dtype=torch.long),
        )
    ).to(device)
    edge_type = torch.tensor(relation_values, dtype=torch.long, device=device)
    drug_graph_index = torch.tensor(drug_indices, dtype=torch.long, device=device)
    side_graph_index = torch.tensor(side_indices, dtype=torch.long, device=device)
    drug_mask = torch.tensor(drug_mask_values, dtype=torch.bool, device=device)
    side_mask = torch.tensor(side_mask_values, dtype=torch.bool, device=device)

    torch.manual_seed(17)
    model = BioKORFKGEncoder(
        num_node_types=num_node_types,
        num_relations=num_relations,
        hidden_dim=128,
        output_dim=128,
        num_bases=8,
        dropout=0.2,
    ).to(device)
    model.eval()
    with torch.no_grad():
        node_embeddings = model(node_type_index, edge_index, edge_type)
        drug_embeddings, returned_drug_mask = model.extract_drug_anchor_embeddings(
            node_embeddings, drug_graph_index, drug_mask
        )
        side_embeddings, returned_side_mask = model.extract_side_anchor_embeddings(
            node_embeddings, side_graph_index, side_mask
        )

    expected_shapes = {
        "node embeddings": (EXPECTED_NODES, 128),
        "drug anchor embeddings": (EXPECTED_DRUG_ANCHORS, 128),
        "side anchor embeddings": (EXPECTED_SIDE_ANCHORS, 128),
        "drug availability mask": (EXPECTED_DRUG_ANCHORS,),
        "side availability mask": (EXPECTED_SIDE_ANCHORS,),
    }
    actual_tensors = {
        "node embeddings": node_embeddings,
        "drug anchor embeddings": drug_embeddings,
        "side anchor embeddings": side_embeddings,
        "drug availability mask": returned_drug_mask,
        "side availability mask": returned_side_mask,
    }
    for label, expected_shape in expected_shapes.items():
        if tuple(actual_tensors[label].shape) != expected_shape:
            raise AssertionError(f"{label} has shape {tuple(actual_tensors[label].shape)}, expected {expected_shape}")
    if not all(torch.isfinite(tensor).all() for tensor in (node_embeddings, drug_embeddings, side_embeddings)):
        raise AssertionError("A model output contains NaN or infinity")

    missing_drug_count = int((~drug_mask).sum().item())
    missing_side_count = int((~side_mask).sum().item())
    missing_drug_expected = model.missing_drug_kg_embedding.unsqueeze(0).expand(missing_drug_count, -1)
    missing_side_expected = model.missing_side_kg_embedding.unsqueeze(0).expand(missing_side_count, -1)
    if not torch.equal(drug_embeddings[~drug_mask], missing_drug_expected):
        raise AssertionError("An isolated drug anchor did not use the drug fallback embedding")
    if not torch.equal(side_embeddings[~side_mask], missing_side_expected):
        raise AssertionError("An isolated side anchor did not use the side fallback embedding")
    connected_drug_count = int(drug_mask.sum().item())
    connected_side_count = int(side_mask.sum().item())
    if connected_drug_count and torch.any(torch.all(
        drug_embeddings[drug_mask]
        == model.missing_drug_kg_embedding.unsqueeze(0).expand(connected_drug_count, -1),
        dim=1,
    )):
        raise AssertionError("A connected drug anchor was replaced by the fallback")
    if connected_side_count and torch.any(torch.all(
        side_embeddings[side_mask]
        == model.missing_side_kg_embedding.unsqueeze(0).expand(connected_side_count, -1),
        dim=1,
    )):
        raise AssertionError("A connected side anchor was replaced by the fallback")

    total_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    lines = [
        "BioKORF KG encoder smoke test: PASS",
        "=" * 38,
        f"PyTorch version: {torch.__version__}",
        f"PyG version: {torch_geometric.__version__}",
        f"Device: {device}",
        f"Num nodes: {node_type_index.numel()}",
        f"Num edges: {edge_type.numel()}",
        f"Num relations: {num_relations}",
        f"Num node types: {num_node_types}",
        "Hidden dimension: 128",
        "Output dimension: 128",
        "Num bases: 8",
        f"Total trainable parameters: {total_parameters}",
        f"R-GCN parameters: {model.rgcn_parameter_count()}",
        f"Node embeddings shape: {list(node_embeddings.shape)}",
        f"Drug anchor embeddings shape: {list(drug_embeddings.shape)}",
        f"Side anchor embeddings shape: {list(side_embeddings.shape)}",
        f"Drug availability mask shape: {list(drug_mask.shape)}",
        f"Side availability mask shape: {list(side_mask.shape)}",
        f"Drug anchors connected / missing: {int(drug_mask.sum())} / {int((~drug_mask).sum())}",
        f"Side anchors connected / missing: {int(side_mask.sum())} / {int((~side_mask).sum())}",
        "Finite-value check: PASS",
        "Fallback replacement check: PASS",
        "Connected-anchor preservation check: PASS",
        "Matrix ordering check: PASS",
        f"Graph leakage metadata: {metadata.get('leakage_check')}",
        "Training steps performed: 0",
        "Embeddings saved: no",
    ]
    emit_report(lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
