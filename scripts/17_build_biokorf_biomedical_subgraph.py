"""Build a leakage-safe BioKORF biomedical knowledge-graph subgraph."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRUG_ANCHORS_PATH = PROJECT_ROOT / "data_processed" / "optimuskg" / "final_drug_anchor_mapping.csv"
SIDE_ANCHORS_PATH = PROJECT_ROOT / "data_processed" / "optimuskg" / "final_side_effect_anchor_mapping.csv"
NODE_PATHS = {
    "DRUG": PROJECT_ROOT / "kg" / "optimuskg" / "nodes" / "drug.parquet",
    "GENE": PROJECT_ROOT / "kg" / "optimuskg" / "nodes" / "gene.parquet",
    "PATHWAY": PROJECT_ROOT / "kg" / "optimuskg" / "nodes" / "pathway.parquet",
    "PHENOTYPE": PROJECT_ROOT / "kg" / "optimuskg" / "nodes" / "phenotype.parquet",
}
EDGE_PATHS = {
    "drug_gene.parquet": PROJECT_ROOT / "kg" / "optimuskg" / "edges" / "drug_gene.parquet",
    "gene_gene.parquet": PROJECT_ROOT / "kg" / "optimuskg" / "edges" / "gene_gene.parquet",
    "pathway_gene.parquet": PROJECT_ROOT / "kg" / "optimuskg" / "edges" / "pathway_gene.parquet",
    "phenotype_gene.parquet": PROJECT_ROOT / "kg" / "optimuskg" / "edges" / "phenotype_gene.parquet",
}
OUTPUT_DIR = PROJECT_ROOT / "data_processed" / "biomedical_kg"
NODES_PATH = OUTPUT_DIR / "nodes.csv"
EDGES_PATH = OUTPUT_DIR / "edges.csv"
STATS_PATH = OUTPUT_DIR / "subgraph_stats.json"
LEAKAGE_PATH = OUTPUT_DIR / "leakage_check.txt"

NODE_COLUMNS = ("node_id", "node_type", "name", "source")
EDGE_COLUMNS = ("source", "target", "relation", "source_type", "target_type", "original_source")
EXPECTED_DRUG_ANCHORS = 757
EXPECTED_SIDE_ANCHORS = 994


def require_files(paths: Iterable[Path]) -> None:
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required local input not found: {path}")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_json_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON list, found {value!r}")
    return [str(item) for item in parsed if item is not None and str(item).strip()]


def validate_anchor_inputs(
    drug_anchors: list[dict[str, str]], side_anchors: list[dict[str, str]]
) -> None:
    if len(drug_anchors) != EXPECTED_DRUG_ANCHORS:
        raise ValueError(f"Expected 757 drug anchors; found {len(drug_anchors)}")
    if len(side_anchors) != EXPECTED_SIDE_ANCHORS:
        raise ValueError(f"Expected 994 side-effect anchors; found {len(side_anchors)}")
    if [int(row["matrix_index"]) for row in drug_anchors] != list(range(EXPECTED_DRUG_ANCHORS)):
        raise ValueError("Drug anchor matrix_index must be exactly 0 through 756")
    if [int(row["matrix_index"]) for row in side_anchors] != list(range(EXPECTED_SIDE_ANCHORS)):
        raise ValueError("Side anchor matrix_index must be exactly 0 through 993")


def inspect_edge_schema(name: str, frame: pl.DataFrame) -> None:
    required = {"from", "to", "relation"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
    relation_values = frame["relation"].drop_nulls().unique().sort().to_list()
    label_values = frame["label"].drop_nulls().unique().sort().to_list() if "label" in frame.columns else []
    print(f"Edge table {name} shape: {frame.shape}")
    print(f"Edge table {name} columns: {frame.columns}")
    print(f"Edge table {name} relation values: {relation_values}")
    if label_values:
        print(f"Edge table {name} label/type values: {label_values}")


def property_name(node_type: str, properties: dict[str, Any]) -> str:
    if node_type == "GENE":
        return properties.get("symbol") or properties.get("name") or ""
    return properties.get("name") or ""


def node_index(node_type: str, frame: pl.DataFrame) -> dict[str, dict[str, str]]:
    if missing := {"id", "properties"}.difference(frame.columns):
        raise ValueError(f"{node_type} node table is missing columns: {sorted(missing)}")
    result: dict[str, dict[str, str]] = {}
    for row in frame.iter_rows(named=True):
        node_id = row["id"]
        if not isinstance(node_id, str) or not node_id:
            continue
        properties = row["properties"] or {}
        result[node_id] = {
            "node_id": node_id,
            "node_type": node_type,
            "name": property_name(node_type, properties),
            "source": "OptimusKG",
        }
    return result


def source_provenance(properties: Any, fallback: str) -> str:
    if not isinstance(properties, dict):
        return fallback
    sources = properties.get("sources") or {}
    if not isinstance(sources, dict):
        return fallback
    direct = sources.get("direct") or []
    indirect = sources.get("indirect") or []
    values = sorted({str(value) for value in [*direct, *indirect] if value})
    return f"{fallback}:{';'.join(values)}" if values else fallback


def convert_edges(
    frame: pl.DataFrame,
    source_type: str,
    target_type: str,
    original_file: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in frame.iter_rows(named=True):
        rows.append({
            "source": row["from"],
            "target": row["to"],
            "relation": row["relation"] or row.get("label") or "RELATED_TO",
            "source_type": source_type,
            "target_type": target_type,
            "original_source": source_provenance(row.get("properties"), original_file),
        })
    return rows


def deduplicate_dict_rows(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> list[dict[str, Any]]:
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        unique[tuple(row[column] for column in columns)] = row
    return [unique[key] for key in sorted(unique)]


def deduplicate_edges(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    structural_columns = ("source", "target", "relation", "source_type", "target_type")
    grouped: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for row in rows:
        key = tuple(row[column] for column in structural_columns)
        grouped[key].update(value for value in row["original_source"].split("|") if value)
    return [
        {
            **dict(zip(structural_columns, key)),
            "original_source": "|".join(sorted(grouped[key])),
        }
        for key in sorted(grouped)
    ]


def main() -> None:
    require_files([DRUG_ANCHORS_PATH, SIDE_ANCHORS_PATH, *NODE_PATHS.values(), *EDGE_PATHS.values()])
    drug_anchors = read_csv_rows(DRUG_ANCHORS_PATH)
    side_anchors = read_csv_rows(SIDE_ANCHORS_PATH)
    validate_anchor_inputs(drug_anchors, side_anchors)

    node_frames = {node_type: pl.read_parquet(path) for node_type, path in NODE_PATHS.items()}
    edge_frames = {name: pl.read_parquet(path) for name, path in EDGE_PATHS.items()}
    for node_type, frame in node_frames.items():
        print(f"Node table {node_type} shape: {frame.shape}")
    for name, frame in edge_frames.items():
        inspect_edge_schema(name, frame)

    indexes = {node_type: node_index(node_type, frame) for node_type, frame in node_frames.items()}
    requested_drug_ids = {
        node_id for row in drug_anchors if row["kg_mapping_status"] in {"mapped_single", "mapped_multi"}
        for node_id in parse_json_list(row["optimuskg_ids"])
    }
    requested_phenotype_ids = {
        node_id for row in side_anchors
        for node_id in ([row["canonical_optimuskg_id"]] if row["canonical_optimuskg_id"] else [])
        + parse_json_list(row["alias_optimuskg_ids"])
    }
    mapped_drug_ids = requested_drug_ids.intersection(indexes["DRUG"])
    mapped_phenotype_ids = requested_phenotype_ids.intersection(indexes["PHENOTYPE"])

    drug_gene = edge_frames["drug_gene.parquet"].filter(pl.col("from").is_in(mapped_drug_ids))
    phenotype_gene = edge_frames["phenotype_gene.parquet"].filter(pl.col("from").is_in(mapped_phenotype_ids))
    connected_gene_ids = set(drug_gene["to"].to_list()) | set(phenotype_gene["to"].to_list())
    gene_gene = edge_frames["gene_gene.parquet"].filter(
        pl.col("from").is_in(connected_gene_ids) & pl.col("to").is_in(connected_gene_ids)
    )
    pathway_gene = edge_frames["pathway_gene.parquet"].filter(pl.col("to").is_in(connected_gene_ids))
    connected_pathway_ids = set(pathway_gene["from"].to_list())

    missing_genes = connected_gene_ids.difference(indexes["GENE"])
    missing_pathways = connected_pathway_ids.difference(indexes["PATHWAY"])
    if missing_genes or missing_pathways:
        raise ValueError(
            f"Selected edges reference missing nodes; genes={sorted(missing_genes)[:10]}, "
            f"pathways={sorted(missing_pathways)[:10]}"
        )

    nodes: list[dict[str, str]] = []
    nodes.extend({"node_id": row["biokorf_drug_id"], "node_type": "BIOKORF_DRUG", "name": row["drug_name"], "source": "BioKORF"} for row in drug_anchors)
    nodes.extend({"node_id": row["biokorf_side_id"], "node_type": "BIOKORF_SIDE", "name": row["side_effect_name"], "source": "BioKORF"} for row in side_anchors)
    for node_type, selected_ids in (
        ("DRUG", mapped_drug_ids), ("PHENOTYPE", mapped_phenotype_ids),
        ("GENE", connected_gene_ids), ("PATHWAY", connected_pathway_ids),
    ):
        nodes.extend(indexes[node_type][node_id] for node_id in selected_ids)
    nodes = deduplicate_dict_rows(nodes, NODE_COLUMNS)
    node_id_to_type: dict[str, str] = {}
    for node in nodes:
        prior = node_id_to_type.setdefault(node["node_id"], node["node_type"])
        if prior != node["node_type"]:
            raise ValueError(f"Node ID {node['node_id']} has conflicting types: {prior}, {node['node_type']}")

    edges: list[dict[str, str]] = []
    edges.extend(convert_edges(drug_gene, "DRUG", "GENE", "drug_gene.parquet"))
    edges.extend(convert_edges(phenotype_gene, "PHENOTYPE", "GENE", "phenotype_gene.parquet"))
    edges.extend(convert_edges(gene_gene, "GENE", "GENE", "gene_gene.parquet"))
    edges.extend(convert_edges(pathway_gene, "PATHWAY", "GENE", "pathway_gene.parquet"))

    drug_anchor_edges: list[dict[str, str]] = []
    for row in drug_anchors:
        for node_id in parse_json_list(row["optimuskg_ids"]):
            if node_id in mapped_drug_ids:
                drug_anchor_edges.append({
                    "source": row["biokorf_drug_id"], "target": node_id,
                    "relation": "MAPS_TO_DRUG", "source_type": "BIOKORF_DRUG",
                    "target_type": "DRUG", "original_source": "final_drug_anchor_mapping.csv",
                })
    side_anchor_edges: list[dict[str, str]] = []
    for row in side_anchors:
        ids = ([row["canonical_optimuskg_id"]] if row["canonical_optimuskg_id"] else []) + parse_json_list(row["alias_optimuskg_ids"])
        for node_id in sorted(set(ids)):
            if node_id in mapped_phenotype_ids:
                side_anchor_edges.append({
                    "source": row["biokorf_side_id"], "target": node_id,
                    "relation": "MAPS_TO_PHENOTYPE", "source_type": "BIOKORF_SIDE",
                    "target_type": "PHENOTYPE", "original_source": "final_side_effect_anchor_mapping.csv",
                })
    edges.extend(drug_anchor_edges)
    edges.extend(side_anchor_edges)
    edges = deduplicate_edges(edges)

    node_ids = set(node_id_to_type)
    if any(edge["source"] not in node_ids or edge["target"] not in node_ids for edge in edges):
        raise ValueError("At least one edge endpoint is missing from nodes.csv")
    direct_drug_phenotype = [
        edge for edge in edges
        if {edge["source_type"], edge["target_type"]} == {"DRUG", "PHENOTYPE"}
    ]
    adverse_edges = [edge for edge in edges if edge["relation"].upper() == "ADVERSE_DRUG_REACTION"]
    direct_anchor_cross_edges = [
        edge for edge in edges
        if {edge["source_type"], edge["target_type"]} == {"BIOKORF_DRUG", "BIOKORF_SIDE"}
    ]
    leakage_pass = not direct_drug_phenotype and not adverse_edges and not direct_anchor_cross_edges
    if not leakage_pass:
        raise ValueError("Leakage validation failed")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with NODES_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=NODE_COLUMNS)
        writer.writeheader(); writer.writerows(nodes)
    with EDGES_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EDGE_COLUMNS)
        writer.writeheader(); writer.writerows(edges)

    node_counts = Counter(node["node_type"] for node in nodes)
    relation_counts = Counter(edge["relation"] for edge in edges)
    connected_drug_anchors = {edge["source"] for edge in drug_anchor_edges}
    connected_side_anchors = {edge["source"] for edge in side_anchor_edges}
    isolated_count = (EXPECTED_DRUG_ANCHORS - len(connected_drug_anchors)) + (EXPECTED_SIDE_ANCHORS - len(connected_side_anchors))
    stats = {
        "total_nodes": len(nodes), "total_edges": len(edges),
        "count_by_node_type": dict(sorted(node_counts.items())),
        "count_by_relation": dict(sorted(relation_counts.items())),
        "biokorf_drug_anchors_with_kg_connection": len(connected_drug_anchors),
        "biokorf_side_anchors_with_kg_connection": len(connected_side_anchors),
        "unique_connected_genes": len(connected_gene_ids),
        "unique_connected_pathways": len(connected_pathway_ids),
        "isolated_biokorf_anchors": isolated_count,
        "drug_anchor_kg_coverage_percentage": round(len(connected_drug_anchors) / EXPECTED_DRUG_ANCHORS * 100, 2),
        "side_effect_anchor_kg_coverage_percentage": round(len(connected_side_anchors) / EXPECTED_SIDE_ANCHORS * 100, 2),
    }
    STATS_PATH.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    leakage_lines = [
        "BioKORF biomedical subgraph leakage check",
        "==========================================",
        f"No direct Drug -> Phenotype edge: {'PASS' if not direct_drug_phenotype else 'FAIL'}",
        f"No ADVERSE_DRUG_REACTION relation: {'PASS' if not adverse_edges else 'FAIL'}",
        "No edge file named drug_phenotype.parquet was used: PASS",
        f"No BioKORF drug anchor directly connected to a BioKORF side anchor: {'PASS' if not direct_anchor_cross_edges else 'FAIL'}",
        f"Overall leakage check: {'PASS' if leakage_pass else 'FAIL'}",
    ]
    LEAKAGE_PATH.write_text("\n".join(leakage_lines) + "\n", encoding="utf-8")

    dg_edges = convert_edges(drug_gene, "DRUG", "GENE", "drug_gene.parquet")
    pg_edges = convert_edges(phenotype_gene, "PHENOTYPE", "GENE", "phenotype_gene.parquet")
    print(f"757 BioKORF drug anchors: {len(drug_anchors)}")
    print(f"994 BioKORF side-effect anchors: {len(side_anchors)}")
    print(f"Mapped OptimusKG drug nodes: {len(mapped_drug_ids)}")
    print(f"Mapped OptimusKG phenotype nodes: {len(mapped_phenotype_ids)}")
    print(f"Connected genes: {len(connected_gene_ids)}")
    print(f"Connected pathways: {len(connected_pathway_ids)}")
    print(f"Drug-Gene edges: {len(deduplicate_edges(dg_edges))}")
    print(f"Phenotype-Gene edges: {len(deduplicate_edges(pg_edges))}")
    print(f"Gene-Gene edges: {len(deduplicate_edges(convert_edges(gene_gene, 'GENE', 'GENE', 'gene_gene.parquet')))}")
    print(f"Pathway-Gene edges: {len(deduplicate_edges(convert_edges(pathway_gene, 'PATHWAY', 'GENE', 'pathway_gene.parquet')))}")
    print(f"Anchor mapping edges: {len(deduplicate_edges(drug_anchor_edges + side_anchor_edges))}")
    print(f"Total nodes: {len(nodes)}")
    print(f"Total edges: {len(edges)}")
    print(f"Drug anchor KG coverage: {stats['drug_anchor_kg_coverage_percentage']:.2f}%")
    print(f"Side-effect anchor KG coverage: {stats['side_effect_anchor_kg_coverage_percentage']:.2f}%")
    print(f"Leakage check: {'PASS' if leakage_pass else 'FAIL'}")

    drug_node_to_anchors: dict[str, list[str]] = defaultdict(list)
    for edge in drug_anchor_edges: drug_node_to_anchors[edge["target"]].append(edge["source"])
    phenotype_node_to_anchors: dict[str, list[str]] = defaultdict(list)
    for edge in side_anchor_edges: phenotype_node_to_anchors[edge["target"]].append(edge["source"])
    gene_to_phenotypes: dict[str, list[str]] = defaultdict(list)
    for edge in pg_edges: gene_to_phenotypes[edge["target"]].append(edge["source"])
    examples: list[str] = []
    for edge in dg_edges:
        for drug_anchor in sorted(drug_node_to_anchors.get(edge["source"], [])):
            for phenotype_id in sorted(gene_to_phenotypes.get(edge["target"], [])):
                for side_anchor in sorted(phenotype_node_to_anchors.get(phenotype_id, [])):
                    examples.append(f"{drug_anchor} -> {edge['source']} -> {edge['target']} -> {phenotype_id} -> {side_anchor}")
                    if len(examples) == 5: break
                if len(examples) == 5: break
            if len(examples) == 5: break
        if len(examples) == 5: break
    print("Example Drug-Gene-Phenotype anchor paths:")
    if examples:
        for example in examples: print(f"  {example}")
    else:
        print("  None found")
    print(f"Nodes CSV: {NODES_PATH}")
    print(f"Edges CSV: {EDGES_PATH}")
    print(f"Stats JSON: {STATS_PATH}")
    print(f"Leakage report: {LEAKAGE_PATH}")


if __name__ == "__main__":
    main()
