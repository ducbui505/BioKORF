"""Prepare deterministic indexed BioKORF graph artifacts for a future R-GCN."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NODES_INPUT = PROJECT_ROOT / "data_processed" / "biomedical_kg" / "nodes.csv"
EDGES_INPUT = PROJECT_ROOT / "data_processed" / "biomedical_kg" / "edges.csv"
STATS_INPUT = PROJECT_ROOT / "data_processed" / "biomedical_kg" / "subgraph_stats.json"
DRUG_ANCHORS_INPUT = PROJECT_ROOT / "data_processed" / "optimuskg" / "final_drug_anchor_mapping.csv"
SIDE_ANCHORS_INPUT = PROJECT_ROOT / "data_processed" / "optimuskg" / "final_side_effect_anchor_mapping.csv"
OUTPUT_DIR = PROJECT_ROOT / "data_processed" / "rgcn"
NODE_INDEX_PATH = OUTPUT_DIR / "node_index.csv"
RELATION_INDEX_PATH = OUTPUT_DIR / "relation_index.csv"
EDGES_INDEXED_PATH = OUTPUT_DIR / "edges_indexed.csv"
EDGES_INVERSE_PATH = OUTPUT_DIR / "edges_with_inverse.csv"
DRUG_ANCHOR_INDEX_PATH = OUTPUT_DIR / "drug_anchor_indices.csv"
SIDE_ANCHOR_INDEX_PATH = OUTPUT_DIR / "side_anchor_indices.csv"
FEATURE_METADATA_PATH = OUTPUT_DIR / "node_feature_metadata.csv"
GRAPH_METADATA_PATH = OUTPUT_DIR / "graph_metadata.json"

EXPECTED_NODES = 21829
EXPECTED_EDGES = 524299
EXPECTED_DRUG_ANCHORS = 757
EXPECTED_SIDE_ANCHORS = 994
NODE_TYPE_ORDER = (
    "BIOKORF_DRUG", "BIOKORF_SIDE", "DRUG", "PHENOTYPE", "GENE", "PATHWAY"
)
TYPE_INDEX = {node_type: index for index, node_type in enumerate(NODE_TYPE_ORDER)}
NODE_INDEX_COLUMNS = ("node_index", "node_id", "node_type", "name", "source")
INDEXED_EDGE_COLUMNS = (
    "source_index", "target_index", "relation_index", "source_node_id",
    "target_node_id", "relation", "source_type", "target_type",
)


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_and_index_nodes() -> tuple[list[dict[str, Any]], dict[str, int], Counter[str]]:
    rows = read_csv_rows(NODES_INPUT)
    if len(rows) != EXPECTED_NODES:
        raise ValueError(f"Expected {EXPECTED_NODES} subgraph nodes; found {len(rows)}")
    node_ids = [row["node_id"] for row in rows]
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("nodes.csv contains duplicate node IDs")
    unknown_types = {row["node_type"] for row in rows}.difference(TYPE_INDEX)
    if unknown_types:
        raise ValueError(f"Unknown node types: {sorted(unknown_types)}")
    rows.sort(key=lambda row: (TYPE_INDEX[row["node_type"]], row["node_id"]))
    indexed: list[dict[str, Any]] = []
    node_to_index: dict[str, int] = {}
    for index, row in enumerate(rows):
        node_to_index[row["node_id"]] = index
        indexed.append({"node_index": index, **row})
    return indexed, node_to_index, Counter(row["node_type"] for row in rows)


def scan_original_edges(
    node_to_index: dict[str, int],
) -> tuple[Counter[str], set[str], set[str], set[str], int, int]:
    relation_counts: Counter[str] = Counter()
    nodes_with_neighbors: set[str] = set()
    drug_anchors_connected: set[str] = set()
    side_anchors_connected: set[str] = set()
    edge_count = 0
    preexisting_self_edges = 0
    with EDGES_INPUT.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            edge_count += 1
            source = row["source"]
            target = row["target"]
            relation = row["relation"]
            if source not in node_to_index or target not in node_to_index:
                raise ValueError(f"Unresolved edge endpoint: {source} -> {target}")
            if {row["source_type"], row["target_type"]} == {"DRUG", "PHENOTYPE"}:
                raise ValueError(f"Direct Drug-Phenotype edge detected: {row}")
            if relation.upper() == "ADVERSE_DRUG_REACTION":
                raise ValueError(f"Prohibited ADVERSE_DRUG_REACTION edge detected: {row}")
            if {row["source_type"], row["target_type"]} == {"BIOKORF_DRUG", "BIOKORF_SIDE"}:
                raise ValueError(f"Direct BioKORF drug-side anchor edge detected: {row}")
            relation_counts[relation] += 1
            preexisting_self_edges += source == target
            nodes_with_neighbors.update((source, target))
            if relation == "MAPS_TO_DRUG" and row["source_type"] == "BIOKORF_DRUG":
                drug_anchors_connected.add(source)
            if relation == "MAPS_TO_PHENOTYPE" and row["source_type"] == "BIOKORF_SIDE":
                side_anchors_connected.add(source)
    if edge_count != EXPECTED_EDGES:
        raise ValueError(f"Expected {EXPECTED_EDGES} original edges; found {edge_count}")
    return (
        relation_counts, nodes_with_neighbors, drug_anchors_connected,
        side_anchors_connected, edge_count, preexisting_self_edges,
    )


def write_index_tables(
    indexed_nodes: list[dict[str, Any]],
    original_relations: list[str],
) -> dict[str, int]:
    with NODE_INDEX_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=NODE_INDEX_COLUMNS)
        writer.writeheader(); writer.writerows(indexed_nodes)
    all_relations = original_relations + [f"{relation}__INV" for relation in original_relations]
    relation_to_index = {relation: index for index, relation in enumerate(all_relations)}
    with RELATION_INDEX_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("relation_index", "relation"))
        writer.writeheader()
        writer.writerows(
            {"relation_index": index, "relation": relation}
            for relation, index in relation_to_index.items()
        )
    return relation_to_index


def indexed_edge(row: dict[str, str], node_to_index: dict[str, int], relation_to_index: dict[str, int]) -> dict[str, Any]:
    return {
        "source_index": node_to_index[row["source"]],
        "target_index": node_to_index[row["target"]],
        "relation_index": relation_to_index[row["relation"]],
        "source_node_id": row["source"],
        "target_node_id": row["target"],
        "relation": row["relation"],
        "source_type": row["source_type"],
        "target_type": row["target_type"],
    }


def write_edges(node_to_index: dict[str, int], relation_to_index: dict[str, int]) -> tuple[int, int]:
    original_count = 0
    inverse_count = 0
    with (
        EDGES_INPUT.open("r", encoding="utf-8-sig", newline="") as source_handle,
        EDGES_INDEXED_PATH.open("w", encoding="utf-8", newline="") as indexed_handle,
        EDGES_INVERSE_PATH.open("w", encoding="utf-8", newline="") as inverse_handle,
    ):
        indexed_writer = csv.DictWriter(indexed_handle, fieldnames=INDEXED_EDGE_COLUMNS)
        inverse_writer = csv.DictWriter(inverse_handle, fieldnames=(*INDEXED_EDGE_COLUMNS, "is_inverse"))
        indexed_writer.writeheader(); inverse_writer.writeheader()
        for row in csv.DictReader(source_handle):
            edge = indexed_edge(row, node_to_index, relation_to_index)
            indexed_writer.writerow(edge)
            inverse_writer.writerow({**edge, "is_inverse": False})
            original_count += 1
            inverse_relation = f"{row['relation']}__INV"
            inverse = {
                "source_index": edge["target_index"],
                "target_index": edge["source_index"],
                "relation_index": relation_to_index[inverse_relation],
                "source_node_id": edge["target_node_id"],
                "target_node_id": edge["source_node_id"],
                "relation": inverse_relation,
                "source_type": edge["target_type"],
                "target_type": edge["source_type"],
                "is_inverse": True,
            }
            inverse_writer.writerow(inverse)
            inverse_count += 1
    return original_count, inverse_count


def validate_and_write_anchor_indices(
    anchor_path: Path,
    output_path: Path,
    expected_count: int,
    id_column: str,
    expected_prefix: str,
    node_to_index: dict[str, int],
    connected: set[str],
) -> None:
    rows = read_csv_rows(anchor_path)
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} anchors in {anchor_path}; found {len(rows)}")
    if [int(row["matrix_index"]) for row in rows] != list(range(expected_count)):
        raise ValueError(f"matrix_index ordering is invalid in {anchor_path}")
    output_rows = []
    for matrix_index, row in enumerate(rows):
        anchor_id = row[id_column]
        expected_id = f"{expected_prefix}{matrix_index:03d}"
        if anchor_id != expected_id or anchor_id not in node_to_index:
            raise ValueError(f"Invalid or missing graph anchor at row {matrix_index}: {anchor_id}")
        output_rows.append({
            "matrix_index": matrix_index,
            id_column: anchor_id,
            "graph_node_index": node_to_index[anchor_id],
            "has_kg_connection": anchor_id in connected,
        })
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("matrix_index", id_column, "graph_node_index", "has_kg_connection"),
        )
        writer.writeheader(); writer.writerows(output_rows)


def main() -> None:
    for path in (NODES_INPUT, EDGES_INPUT, STATS_INPUT, DRUG_ANCHORS_INPUT, SIDE_ANCHORS_INPUT):
        require_file(path)
    source_stats = json.loads(STATS_INPUT.read_text(encoding="utf-8"))
    if source_stats.get("total_nodes") != EXPECTED_NODES or source_stats.get("total_edges") != EXPECTED_EDGES:
        raise ValueError("subgraph_stats.json does not contain the expected graph dimensions")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    indexed_nodes, node_to_index, node_type_counts = load_and_index_nodes()
    (
        relation_counts, nodes_with_neighbors, connected_drugs, connected_sides,
        scanned_edges, preexisting_self_edges,
    ) = scan_original_edges(node_to_index)
    original_relations = sorted(relation_counts)
    relation_to_index = write_index_tables(indexed_nodes, original_relations)
    original_count, inverse_count = write_edges(node_to_index, relation_to_index)
    if original_count != scanned_edges or inverse_count != original_count:
        raise ValueError("Original/inverse edge counts do not reconcile")

    validate_and_write_anchor_indices(
        DRUG_ANCHORS_INPUT, DRUG_ANCHOR_INDEX_PATH, EXPECTED_DRUG_ANCHORS,
        "biokorf_drug_id", "BIOKORF_DRUG_", node_to_index, connected_drugs,
    )
    validate_and_write_anchor_indices(
        SIDE_ANCHORS_INPUT, SIDE_ANCHOR_INDEX_PATH, EXPECTED_SIDE_ANCHORS,
        "biokorf_side_id", "BIOKORF_SIDE_", node_to_index, connected_sides,
    )

    with FEATURE_METADATA_PATH.open("w", encoding="utf-8", newline="") as handle:
        columns = (
            "node_index", "node_id", "node_type", "type_index",
            "is_drug_anchor", "is_side_anchor", "has_neighbor",
        )
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in indexed_nodes:
            node_type = row["node_type"]
            writer.writerow({
                "node_index": row["node_index"], "node_id": row["node_id"],
                "node_type": node_type, "type_index": TYPE_INDEX[node_type],
                "is_drug_anchor": node_type == "BIOKORF_DRUG",
                "is_side_anchor": node_type == "BIOKORF_SIDE",
                "has_neighbor": row["node_id"] in nodes_with_neighbors,
            })

    isolated_drugs = EXPECTED_DRUG_ANCHORS - len(connected_drugs)
    isolated_sides = EXPECTED_SIDE_ANCHORS - len(connected_sides)
    inverse_relation_counts = {f"{relation}__INV": count for relation, count in relation_counts.items()}
    metadata = {
        "num_nodes": len(indexed_nodes),
        "num_original_edges": original_count,
        "num_edges_with_inverse": original_count + inverse_count,
        "num_relations_original": len(original_relations),
        "num_relations_with_inverse": len(relation_to_index),
        "node_type_counts": dict(sorted(node_type_counts.items())),
        "relation_counts": dict(sorted(relation_counts.items())),
        "inverse_relation_counts": dict(sorted(inverse_relation_counts.items())),
        "drug_anchors_confirmed": len([row for row in indexed_nodes if row["node_type"] == "BIOKORF_DRUG"]),
        "side_anchors_confirmed": len([row for row in indexed_nodes if row["node_type"] == "BIOKORF_SIDE"]),
        "isolated_drug_anchor_count": isolated_drugs,
        "isolated_side_anchor_count": isolated_sides,
        "leakage_check": "PASS",
        "preexisting_source_self_edge_count": preexisting_self_edges,
        "self_loop_edges_added_by_preprocessing": 0,
        "self_loop_policy": "Do not materialize self-loop edges; use a learnable self-loop/root transformation in the future R-GCN.",
        "node_ordering": "node_type order followed by lexicographic node_id",
        "node_type_index": TYPE_INDEX,
    }
    GRAPH_METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Nodes indexed: {len(indexed_nodes)}")
    print(f"Original edges indexed: {original_count}")
    print(f"Inverse edges created: {inverse_count}")
    print(f"Edges with inverse: {original_count + inverse_count}")
    print(f"Original relations: {len(original_relations)}")
    print(f"Relations with inverse: {len(relation_to_index)}")
    print("Detected original relations:")
    for relation in original_relations:
        print(f"  {relation_to_index[relation]}: {relation} ({relation_counts[relation]} edges)")
    print("Inverse relations:")
    for relation in original_relations:
        inverse = f"{relation}__INV"
        print(f"  {relation_to_index[inverse]}: {inverse} ({relation_counts[relation]} edges)")
    print(f"Drug anchors confirmed: {EXPECTED_DRUG_ANCHORS}")
    print(f"Side anchors confirmed: {EXPECTED_SIDE_ANCHORS}")
    print(f"Isolated drug anchors: {isolated_drugs}")
    print(f"Isolated side anchors: {isolated_sides}")
    print(f"Pre-existing source self-edges retained: {preexisting_self_edges}")
    print("New self-loop edges materialized by preprocessing: 0")
    print("Future self-loop policy: learnable self-loop/root transformation")
    print("Leakage check: PASS")
    for path in (
        NODE_INDEX_PATH, RELATION_INDEX_PATH, EDGES_INDEXED_PATH, EDGES_INVERSE_PATH,
        DRUG_ANCHOR_INDEX_PATH, SIDE_ANCHOR_INDEX_PATH, FEATURE_METADATA_PATH,
        GRAPH_METADATA_PATH,
    ):
        print(f"Created: {path}")


if __name__ == "__main__":
    main()
