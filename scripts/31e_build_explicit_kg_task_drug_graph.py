"""Build explicit-structure, task-only, and KG-task Drug graph channels.

This Step 31E script performs graph construction only. It does not load or use
pretrained KG embeddings, train a model, test a model, or modify MSSF inputs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KG_EDGES_PATH = PROJECT_ROOT / "data_processed" / "biomedical_kg" / "edges.csv"
DRUG_ANCHOR_PATH = (
    PROJECT_ROOT / "data_processed" / "optimuskg" / "final_drug_anchor_mapping.csv"
)
SPLIT_PATH = PROJECT_ROOT / "data_processed" / "experiments" / "pilot_fold1_split.npz"
FREQUENCY_PATH = PROJECT_ROOT / "Datas" / "drug_side.pkl"
AUDIT_PATH = PROJECT_ROOT / "data_processed" / "rewiring" / "mssf_similarity_audit.csv"
OUTPUT_DIR = PROJECT_ROOT / "data_processed" / "rewiring" / "explicit_kg_task"

STRUCTURE_MATRIX_PATH = OUTPUT_DIR / "structure_only_matrix.npy"
TASK_MATRIX_PATH = OUTPUT_DIR / "task_only_matrix.npy"
KG_TASK_MATRIX_PATH = OUTPUT_DIR / "kg_task_explicit_matrix.npy"
STRUCTURE_EDGES_PATH = OUTPUT_DIR / "structure_only_edges.csv"
TASK_EDGES_PATH = OUTPUT_DIR / "task_only_edges.csv"
KG_TASK_EDGES_PATH = OUTPUT_DIR / "kg_task_explicit_edges.csv"
METADATA_PATH = OUTPUT_DIR / "drug_explicit_kg_metadata.csv"
DIAGNOSTICS_PATH = OUTPUT_DIR / "graph_diagnostics.csv"
COMPLEMENTARITY_PATH = OUTPUT_DIR / "mssf_view_complementarity.csv"
REPORT_PATH = OUTPUT_DIR / "step31e_report.txt"

DRUG_COUNT = 757
SIDE_EFFECT_COUNT = 994
TOP_K = 10
MIN_TASK_OVERLAP = 3
RANDOM_SEED = 42
KG_WEIGHT = 0.5
TASK_WEIGHT = 0.5

INPUT_PATHS = (
    KG_EDGES_PATH,
    DRUG_ANCHOR_PATH,
    SPLIT_PATH,
    FREQUENCY_PATH,
    AUDIT_PATH,
)

EDGE_COLUMNS = [
    "source_drug_index",
    "target_drug_index",
    "edge_value",
    "kg_structure_score",
    "gene_jaccard",
    "pathway_jaccard",
    "task_similarity",
    "normalized_task_similarity",
    "combined_score",
    "task_overlap_count",
    "edge_origin",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_inputs() -> None:
    missing = [path for path in INPUT_PATHS if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required Step 31E inputs are missing: {missing}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_anchor_metadata() -> tuple[list[str], list[str]]:
    rows = read_csv(DRUG_ANCHOR_PATH)
    if len(rows) != DRUG_COUNT:
        raise ValueError("Final Drug anchor mapping must contain exactly 757 rows")
    indices = [int(row["matrix_index"]) for row in rows]
    if indices != list(range(DRUG_COUNT)):
        raise ValueError("Drug anchor matrix_index order must be exactly 0..756")
    anchors = [row["biokorf_drug_id"] for row in rows]
    expected = [f"BIOKORF_DRUG_{index:03d}" for index in range(DRUG_COUNT)]
    if anchors != expected:
        raise ValueError("BioKORF Drug anchor IDs do not correspond to matrix_index")
    return [row["drug_name"] for row in rows], anchors


def build_explicit_sets(
    anchors: list[str],
) -> tuple[list[set[str]], list[set[str]], dict[str, Any]]:
    anchor_to_drugs: dict[str, set[str]] = defaultdict(set)
    drug_to_genes: dict[str, set[str]] = defaultdict(set)
    gene_to_pathways: dict[str, set[str]] = defaultdict(set)
    used_relations: set[str] = set()
    ignored_drug_phenotype_edges = 0
    ignored_adverse_edges = 0
    with KG_EDGES_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"source", "target", "relation", "source_type", "target_type"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise KeyError(f"Biomedical KG edges are missing columns: {sorted(missing)}")
        for row in reader:
            source = row["source"]
            target = row["target"]
            relation = row["relation"]
            source_type = row["source_type"]
            target_type = row["target_type"]
            endpoint_types = {source_type, target_type}
            if relation.upper() == "ADVERSE_DRUG_REACTION":
                ignored_adverse_edges += 1
                continue
            if endpoint_types == {"DRUG", "PHENOTYPE"}:
                ignored_drug_phenotype_edges += 1
                continue
            if (
                relation == "MAPS_TO_DRUG"
                and source_type == "BIOKORF_DRUG"
                and target_type == "DRUG"
            ):
                anchor_to_drugs[source].add(target)
                used_relations.add(relation)
            elif endpoint_types == {"DRUG", "GENE"}:
                drug = source if source_type == "DRUG" else target
                gene = target if target_type == "GENE" else source
                drug_to_genes[drug].add(gene)
                used_relations.add(relation)
            elif endpoint_types == {"PATHWAY", "GENE"}:
                pathway = source if source_type == "PATHWAY" else target
                gene = target if target_type == "GENE" else source
                gene_to_pathways[gene].add(pathway)
                used_relations.add(relation)
    genes_by_index: list[set[str]] = []
    pathways_by_index: list[set[str]] = []
    for anchor in anchors:
        identities = anchor_to_drugs.get(anchor, set())
        genes = set().union(*(drug_to_genes.get(identity, set()) for identity in identities))
        pathways = set().union(*(gene_to_pathways.get(gene, set()) for gene in genes))
        genes_by_index.append(genes)
        pathways_by_index.append(pathways)
    checks = {
        "only_typed_maps_drug_gene_pathway_gene_used": True,
        "drug_phenotype_relations_used": False,
        "adverse_drug_reaction_relations_used": False,
        "used_relations": sorted(used_relations),
        "ignored_drug_phenotype_edge_count": ignored_drug_phenotype_edges,
        "ignored_adverse_edge_count": ignored_adverse_edges,
        "anchors_with_mapped_drug": int(
            sum(bool(anchor_to_drugs.get(anchor)) for anchor in anchors)
        ),
        "anchors_with_gene_evidence": int(sum(bool(values) for values in genes_by_index)),
        "anchors_with_pathway_evidence": int(
            sum(bool(values) for values in pathways_by_index)
        ),
    }
    return genes_by_index, pathways_by_index, checks


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else math.nan


def build_explicit_similarity(
    genes: list[set[str]], pathways: list[set[str]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gene_similarity = np.full((DRUG_COUNT, DRUG_COUNT), np.nan, dtype=np.float64)
    pathway_similarity = np.full((DRUG_COUNT, DRUG_COUNT), np.nan, dtype=np.float64)
    structure = np.full((DRUG_COUNT, DRUG_COUNT), np.nan, dtype=np.float64)
    for source in range(DRUG_COUNT):
        for target in range(source, DRUG_COUNT):
            gene_value = jaccard(genes[source], genes[target])
            pathway_value = jaccard(pathways[source], pathways[target])
            values = [value for value in (gene_value, pathway_value) if np.isfinite(value)]
            structure_value = float(np.mean(values)) if values else math.nan
            gene_similarity[source, target] = gene_similarity[target, source] = gene_value
            pathway_similarity[source, target] = pathway_similarity[target, source] = pathway_value
            structure[source, target] = structure[target, source] = structure_value
    return gene_similarity, pathway_similarity, structure


def load_train_only_profiles() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with FREQUENCY_PATH.open("rb") as handle:
        full_frequency = np.asarray(pickle.load(handle))
    if full_frequency.shape != (DRUG_COUNT, SIDE_EFFECT_COUNT):
        raise ValueError("drug_side.pkl must have shape [757,994]")
    with np.load(SPLIT_PATH) as split:
        required = {"train_samples", "validation_samples", "test_samples", "seed", "fold"}
        missing = required.difference(split.files)
        if missing:
            raise KeyError(f"Fixed split is missing arrays: {sorted(missing)}")
        if int(split["seed"]) != RANDOM_SEED or int(split["fold"]) != 1:
            raise ValueError("Expected fixed Fold-1 split with seed 42")
        train = np.asarray(split["train_samples"])
        validation_pairs = np.asarray(split["validation_samples"])[:, :2].astype(np.int64)
        test_pairs = np.asarray(split["test_samples"])[:, :2].astype(np.int64)
    train_drug = train[:, 0].astype(np.int64)
    train_side = train[:, 1].astype(np.int64)
    train_label = train[:, 2].astype(np.int64)
    if not set(np.unique(train_label)).issubset({1, 2, 3, 4, 5}):
        raise ValueError("Fold-1 training frequency labels must be within 1..5")
    if not np.array_equal(full_frequency[train_drug, train_side].astype(np.int64), train_label):
        raise ValueError("Fold-1 training labels disagree with drug_side.pkl")
    observed = np.zeros((DRUG_COUNT, SIDE_EFFECT_COUNT), dtype=bool)
    centered = np.zeros((DRUG_COUNT, SIDE_EFFECT_COUNT), dtype=np.float64)
    observed[train_drug, train_side] = True
    centered[train_drug, train_side] = train_label.astype(np.float64) - 3.0
    validation_hidden = not observed[validation_pairs[:, 0], validation_pairs[:, 1]].any()
    test_hidden = not observed[test_pairs[:, 0], test_pairs[:, 1]].any()
    if not validation_hidden or not test_hidden:
        raise RuntimeError("Validation/test entries leaked into train-only profiles")
    diagnostics = {
        "train_observation_count": int(observed.sum()),
        "validation_positions_hidden": bool(validation_hidden),
        "test_positions_hidden": bool(test_hidden),
        "minimum_shared_observations": MIN_TASK_OVERLAP,
        "frequency_centering": {"1": -2, "2": -1, "3": 0, "4": 1, "5": 2},
        "missing_entries_remain_unobserved": True,
    }
    return observed, centered, diagnostics


def build_task_similarity(
    observed: np.ndarray, centered: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    overlap = observed.astype(np.int16) @ observed.astype(np.int16).T
    task = np.full((DRUG_COUNT, DRUG_COUNT), np.nan, dtype=np.float64)
    degenerate = 0
    for source in range(DRUG_COUNT):
        for target in range(source + 1, DRUG_COUNT):
            if overlap[source, target] < MIN_TASK_OVERLAP:
                continue
            common = observed[source] & observed[target]
            left = centered[source, common]
            right = centered[target, common]
            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
            if denominator == 0:
                degenerate += 1
                continue
            value = float(np.dot(left, right) / denominator)
            task[source, target] = task[target, source] = value
    upper = np.triu_indices(DRUG_COUNT, k=1)
    diagnostics = {
        "valid_task_pair_count": int(np.isfinite(task[upper]).sum()),
        "pairs_meeting_overlap_threshold": int(
            np.count_nonzero(overlap[upper] >= MIN_TASK_OVERLAP)
        ),
        "degenerate_zero_norm_pair_count": degenerate,
        "mean_pair_overlap": float(overlap[upper].mean()),
        "median_pair_overlap": float(np.median(overlap[upper])),
    }
    return task, overlap, diagnostics


def deterministic_rank(values: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=np.int64)
    order = np.lexsort((candidates, -values[candidates]))
    return candidates[order]


def edge_row(
    source: int,
    target: int,
    edge_value: float,
    origin: str,
    structure: np.ndarray,
    gene: np.ndarray,
    pathway: np.ndarray,
    task: np.ndarray,
    overlap: np.ndarray,
    combined: float = math.nan,
) -> dict[str, Any]:
    task_value = float(task[source, target])
    return {
        "source_drug_index": source,
        "target_drug_index": target,
        "edge_value": edge_value,
        "kg_structure_score": float(structure[source, target]),
        "gene_jaccard": float(gene[source, target]),
        "pathway_jaccard": float(pathway[source, target]),
        "task_similarity": task_value,
        "normalized_task_similarity": (
            (task_value + 1.0) / 2.0 if np.isfinite(task_value) else math.nan
        ),
        "combined_score": combined,
        "task_overlap_count": int(overlap[source, target]),
        "edge_origin": origin,
    }


def build_graphs(
    structure: np.ndarray,
    gene: np.ndarray,
    pathway: np.ndarray,
    task: np.ndarray,
    overlap: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, list[dict[str, Any]]]]:
    matrices = {
        "STRUCTURE_ONLY": np.eye(DRUG_COUNT, dtype=np.float32),
        "TASK_ONLY_EXPLICIT": np.eye(DRUG_COUNT, dtype=np.float32),
        "KG_TASK_EXPLICIT": np.eye(DRUG_COUNT, dtype=np.float32),
    }
    edges: dict[str, list[dict[str, Any]]] = {name: [] for name in matrices}
    all_indices = np.arange(DRUG_COUNT, dtype=np.int64)
    for source in range(DRUG_COUNT):
        not_self = all_indices != source

        structure_valid = not_self & np.isfinite(structure[source]) & (structure[source] > 0)
        structure_candidates = deterministic_rank(
            structure[source], all_indices[structure_valid]
        )[:TOP_K]
        for target_value in structure_candidates:
            target = int(target_value)
            value = float(structure[source, target])
            matrices["STRUCTURE_ONLY"][source, target] = value
            edges["STRUCTURE_ONLY"].append(
                edge_row(
                    source, target, value, "STRUCTURE_ONLY", structure,
                    gene, pathway, task, overlap,
                )
            )

        task_valid = not_self & np.isfinite(task[source])
        normalized_task = (task[source] + 1.0) / 2.0
        task_candidates = deterministic_rank(
            normalized_task, all_indices[task_valid]
        )[:TOP_K]
        for target_value in task_candidates:
            target = int(target_value)
            value = float(normalized_task[target])
            matrices["TASK_ONLY_EXPLICIT"][source, target] = value
            edges["TASK_ONLY_EXPLICIT"].append(
                edge_row(
                    source, target, value, "TASK_ONLY_EXPLICIT", structure,
                    gene, pathway, task, overlap,
                )
            )

        combined_valid = structure_valid & np.isfinite(task[source])
        combined_values = np.full(DRUG_COUNT, np.nan, dtype=np.float64)
        combined_values[combined_valid] = (
            KG_WEIGHT * structure[source, combined_valid]
            + TASK_WEIGHT * normalized_task[combined_valid]
        )
        combined_candidates = deterministic_rank(
            combined_values, all_indices[combined_valid]
        )[:TOP_K]
        for target_value in combined_candidates:
            target = int(target_value)
            value = float(combined_values[target])
            matrices["KG_TASK_EXPLICIT"][source, target] = value
            edges["KG_TASK_EXPLICIT"].append(
                edge_row(
                    source, target, value, "KG_TASK_EXPLICIT", structure,
                    gene, pathway, task, overlap, combined=value,
                )
            )
    return matrices, edges


def deterministic_random_task_baseline(task: np.ndarray) -> tuple[float, int]:
    upper = np.triu_indices(DRUG_COUNT, k=1)
    valid = np.flatnonzero(np.isfinite(task[upper]))
    count = min(10000, len(valid))
    if count == 0:
        return math.nan, 0
    rng = np.random.default_rng(RANDOM_SEED)
    selected = rng.choice(valid, size=count, replace=False)
    return float(task[upper[0][selected], upper[1][selected]].mean()), count


def finite_values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    return values[np.isfinite(values)]


def graph_diagnostics(
    name: str, rows: list[dict[str, Any]], random_task_baseline: float
) -> dict[str, Any]:
    degree = np.bincount(
        [row["source_drug_index"] for row in rows], minlength=DRUG_COUNT
    )
    edge_values = finite_values(rows, "edge_value")
    task_values = finite_values(rows, "task_similarity")
    gene_values = finite_values(rows, "gene_jaccard")
    pathway_values = finite_values(rows, "pathway_jaccard")
    return {
        "graph": name,
        "directed_nonself_edge_count": len(rows),
        "source_drug_coverage_count": int(np.count_nonzero(degree)),
        "source_drug_coverage_percentage": 100.0 * np.count_nonzero(degree) / DRUG_COUNT,
        "mean_degree": float(degree.mean()),
        "isolated_drug_count": int(np.count_nonzero(degree == 0)),
        "mean_edge_value": float(edge_values.mean()),
        "median_edge_value": float(np.median(edge_values)),
        "evaluable_task_edge_count": int(task_values.size),
        "mean_task_similarity": (
            float(task_values.mean()) if task_values.size else math.nan
        ),
        "median_task_similarity": (
            float(np.median(task_values)) if task_values.size else math.nan
        ),
        "random_task_baseline": random_task_baseline,
        "task_enrichment": (
            float(task_values.mean() - random_task_baseline)
            if task_values.size else math.nan
        ),
        "mean_gene_jaccard": float(gene_values.mean()) if gene_values.size else math.nan,
        "mean_pathway_jaccard": (
            float(pathway_values.mean()) if pathway_values.size else math.nan
        ),
        "fraction_with_shared_gene": float(np.mean(gene_values > 0)) if gene_values.size else math.nan,
        "fraction_with_shared_pathway": (
            float(np.mean(pathway_values > 0)) if pathway_values.size else math.nan
        ),
    }


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 2:
        return math.nan
    left = left[valid] - left[valid].mean()
    right = right[valid] - right[valid].mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator > 0 else math.nan


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    return pearson(average_ranks(left[valid]), average_ranks(right[valid]))


def cosine_similarity(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)
    return normalized @ normalized.T


def load_external_views() -> tuple[list[dict[str, Any]], Path, str]:
    audit_rows = read_csv(AUDIT_PATH)
    source_matches = [
        row for row in audit_rows
        if row["entity_type"] == "DRUG"
        and row["view_index"] == "3"
        and row["view_name"] == "SMDdatabase"
    ]
    if len(source_matches) != 1:
        raise ValueError("Could not identify exactly one audited SMDdatabase view")
    smd_path = (PROJECT_ROOT / source_matches[0]["source_file"]).resolve()
    if not smd_path.is_file():
        raise FileNotFoundError(f"Audited SMDdatabase source is missing: {smd_path}")
    smd_hash = sha256(smd_path)

    # Step 31A contains nine external-static Drug views, but exactly eight are
    # actionable safe rewiring candidates; drug_f_sim is NEEDS_MORE_EVIDENCE.
    selected = [
        row for row in audit_rows
        if row["entity_type"] == "DRUG"
        and row["provenance_class"] == "EXTERNAL_STATIC"
        and row["leakage_status"] == "SAFE_EXTERNAL"
        and row["rewiring_priority"] != "NEEDS_MORE_EVIDENCE"
    ]
    selected.sort(key=lambda row: int(row["view_index"]))
    if len(selected) != 8:
        raise ValueError(
            f"Expected eight actionable SAFE_EXTERNAL Drug views; found {len(selected)}"
        )
    views: list[dict[str, Any]] = []
    for row in selected:
        source_path = (PROJECT_ROOT / row["source_file"]).resolve()
        with source_path.open("rb") as handle:
            raw = np.asarray(pickle.load(handle), dtype=np.float64)
        matrix = raw if raw.shape == (DRUG_COUNT, DRUG_COUNT) else cosine_similarity(raw)
        if matrix.shape != (DRUG_COUNT, DRUG_COUNT) or not np.isfinite(matrix).all():
            raise ValueError(f"Invalid external Drug view: {row['view_name']} {matrix.shape}")
        views.append(
            {
                "view_index": int(row["view_index"]),
                "view_name": row["view_name"],
                "source_file": row["source_file"],
                "matrix": matrix,
            }
        )
    return views, smd_path, smd_hash


def deterministic_top10(matrix: np.ndarray) -> list[set[int]]:
    indices = np.arange(DRUG_COUNT, dtype=np.int64)
    output: list[set[int]] = []
    for source in range(DRUG_COUNT):
        candidates = indices[indices != source]
        order = np.lexsort((candidates, -matrix[source, candidates]))
        output.append(set(map(int, candidates[order[:TOP_K]])))
    return output


def complementarity(
    kg_task_matrix: np.ndarray,
    kg_task_edges: list[dict[str, Any]],
    views: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    upper = np.triu_indices(DRUG_COUNT, k=1)
    kg_values = kg_task_matrix[upper].astype(np.float64)
    kg_neighbors = [set() for _ in range(DRUG_COUNT)]
    for row in kg_task_edges:
        kg_neighbors[row["source_drug_index"]].add(row["target_drug_index"])
    rows = []
    for view in views:
        matrix = view["matrix"]
        view_neighbors = deterministic_top10(matrix)
        jaccards = []
        for source in range(DRUG_COUNT):
            union = kg_neighbors[source] | view_neighbors[source]
            jaccards.append(
                len(kg_neighbors[source] & view_neighbors[source]) / len(union)
                if union else 0.0
            )
        rows.append(
            {
                "entity_type": "DRUG",
                "view_index": view["view_index"],
                "view_name": view["view_name"],
                "source_file": view["source_file"],
                "pearson_off_diagonal": pearson(kg_values, matrix[upper]),
                "spearman_off_diagonal": spearman(kg_values, matrix[upper]),
                "mean_top10_neighbor_jaccard": float(np.mean(jaccards)),
                "comparison_selection_rule": (
                    "EXTERNAL_STATIC + SAFE_EXTERNAL + rewiring_priority != NEEDS_MORE_EVIDENCE"
                ),
            }
        )
    return rows


def validate_matrices(matrices: dict[str, np.ndarray]) -> bool:
    return all(
        matrix.shape == (DRUG_COUNT, DRUG_COUNT)
        and matrix.dtype == np.float32
        and np.isfinite(matrix).all()
        and np.array_equal(np.diag(matrix), np.ones(DRUG_COUNT, dtype=np.float32))
        and np.all(matrix >= 0)
        and np.all(matrix <= 1)
        for matrix in matrices.values()
    )


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    raise TypeError(f"Unsupported JSON type: {type(value).__name__}")


def csv_safe(value: Any) -> Any:
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return ""
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            {column: csv_safe(row.get(column, "")) for column in columns}
            for row in rows
        )
    temporary.replace(path)


def write_npy(path: Path, matrix: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, matrix, allow_pickle=False)
    temporary.replace(path)


def format_metric(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return "NA" if not np.isfinite(value) else f"{float(value):.8f}"
    return str(value)


def build_report(
    graph_rows: list[dict[str, Any]],
    complementarity_rows: list[dict[str, Any]],
    metadata_summary: dict[str, Any],
    task_checks: dict[str, Any],
    checks: dict[str, bool],
) -> str:
    graph_by_name = {row["graph"]: row for row in graph_rows}
    complement_by_name = {row["view_name"]: row for row in complementarity_rows}
    lines = [
        "BioKORF Step 31E — Explicit KG + Task Drug Graphs",
        "=================================================",
        "Graph construction only; no KG embedding cosine, model training, or model testing.",
        "",
        "Parameters",
        "----------",
        f"top_k = {TOP_K}",
        f"minimum task overlap = {MIN_TASK_OVERLAP}",
        f"seed = {RANDOM_SEED}",
        f"KG/task combined weights = {KG_WEIGHT:.1f}/{TASK_WEIGHT:.1f}",
        "",
        "Explicit KG sets",
        "----------------",
        *[f"{key}: {format_metric(value)}" for key, value in metadata_summary.items()],
        "Drug genes use only anchor MAPS_TO_DRUG and typed DRUG-GENE edges.",
        "Drug pathways use only those genes and typed PATHWAY-GENE edges.",
        "Drug-Phenotype and adverse-drug-reaction relations are excluded.",
        "",
        "Train-only task construction",
        "----------------------------",
        *[f"{key}: {format_metric(value)}" for key, value in task_checks.items()],
        "",
        "Graph diagnostics",
        "-----------------",
    ]
    for row in graph_rows:
        lines.append(
            f"{row['graph']}: edges={row['directed_nonself_edge_count']}; "
            f"coverage={row['source_drug_coverage_percentage']:.4f}%; "
            f"mean_degree={row['mean_degree']:.4f}; isolated={row['isolated_drug_count']}; "
            f"edge mean/median={row['mean_edge_value']:.8f}/{row['median_edge_value']:.8f}; "
            f"task mean/median={format_metric(row['mean_task_similarity'])}/"
            f"{format_metric(row['median_task_similarity'])}; "
            f"enrichment={format_metric(row['task_enrichment'])}; "
            f"geneJ={format_metric(row['mean_gene_jaccard'])}; "
            f"pathwayJ={format_metric(row['mean_pathway_jaccard'])}"
        )
    lines.extend(
        [
            "",
            "Complementarity with eight actionable SAFE_EXTERNAL Drug views",
            "---------------------------------------------------------------",
            "DIPA and DIPF are excluded. The fingerprint view is SAFE_EXTERNAL but excluded "
            "because Step 31A marked it NEEDS_MORE_EVIDENCE, leaving the requested eight.",
        ]
    )
    for row in complementarity_rows:
        lines.append(
            f"view {row['view_index']} {row['view_name']}: Pearson={row['pearson_off_diagonal']:.8f}; "
            f"Spearman={row['spearman_off_diagonal']:.8f}; "
            f"top10 Jaccard={row['mean_top10_neighbor_jaccard']:.8f}"
        )
    lines.extend(
        [
            "",
            "Residual-channel policy",
            "-----------------------",
            "The original dense SMDdatabase remains unchanged. kg_task_explicit_matrix.npy is "
            "saved as a separate prospective feature channel; the two are not combined numerically.",
            "",
            "Final summary",
            "-------------",
            "Graph | Edge count | Drug coverage | Mean task similarity | Task enrichment",
        ]
    )
    for row in graph_rows:
        lines.append(
            f"{row['graph']} | {row['directed_nonself_edge_count']} | "
            f"{row['source_drug_coverage_percentage']:.4f}% | "
            f"{format_metric(row['mean_task_similarity'])} | {format_metric(row['task_enrichment'])}"
        )
    kg_task = graph_by_name["KG_TASK_EXPLICIT"]
    lines.extend(
        [
            f"KG_TASK_EXPLICIT mean gene Jaccard: {kg_task['mean_gene_jaccard']:.8f}",
            f"KG_TASK_EXPLICIT mean pathway Jaccard: {kg_task['mean_pathway_jaccard']:.8f}",
            "KG_TASK_EXPLICIT top10 overlap with SMDdatabase: "
            f"{complement_by_name['SMDdatabase']['mean_top10_neighbor_jaccard']:.8f}",
            "KG_TASK_EXPLICIT top10 overlap with drug_target_sim: "
            f"{complement_by_name['drug_target_sim']['mean_top10_neighbor_jaccard']:.8f}",
            "KG_TASK_EXPLICIT top10 overlap with drug_p_e_sim: "
            f"{complement_by_name['drug_p_e_sim']['mean_top10_neighbor_jaccard']:.8f}",
            "",
            f"EXPLICIT KG STRUCTURE CHECK: {'PASS' if checks['explicit_kg_structure'] else 'FAIL'}",
            f"TASK-AWARE LEAKAGE CHECK: {'PASS' if checks['task_aware_leakage'] else 'FAIL'}",
            f"SOURCE FEATURE PRESERVATION CHECK: {'PASS' if checks['source_feature_preservation'] else 'FAIL'}",
            f"KG EMBEDDING COSINE EXCLUSION CHECK: {'PASS' if checks['kg_embedding_cosine_exclusion'] else 'FAIL'}",
            "Training/testing performed: NO",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    require_inputs()
    hashes_before = {str(path): sha256(path) for path in INPUT_PATHS}
    names, anchors = load_anchor_metadata()
    genes, pathways, structure_checks = build_explicit_sets(anchors)
    gene_similarity, pathway_similarity, structure_similarity = (
        build_explicit_similarity(genes, pathways)
    )
    observed, centered, train_checks = load_train_only_profiles()
    task_similarity, overlap, task_diagnostics = build_task_similarity(observed, centered)
    matrices, edges = build_graphs(
        structure_similarity,
        gene_similarity,
        pathway_similarity,
        task_similarity,
        overlap,
    )
    if not validate_matrices(matrices):
        raise RuntimeError("A Step 31E matrix contract check failed")
    random_task_baseline, random_pair_count = deterministic_random_task_baseline(
        task_similarity
    )
    graph_rows = [
        graph_diagnostics(name, edges[name], random_task_baseline)
        for name in ("STRUCTURE_ONLY", "TASK_ONLY_EXPLICIT", "KG_TASK_EXPLICIT")
    ]
    external_views, smd_path, smd_hash = load_external_views()
    complementarity_rows = complementarity(
        matrices["KG_TASK_EXPLICIT"], edges["KG_TASK_EXPLICIT"], external_views
    )
    metadata_rows = [
        {
            "matrix_index": index,
            "biokorf_drug_id": anchors[index],
            "drug_name": names[index],
            "gene_count": len(genes[index]),
            "pathway_count": len(pathways[index]),
            "has_gene_evidence": bool(genes[index]),
            "has_pathway_evidence": bool(pathways[index]),
        }
        for index in range(DRUG_COUNT)
    ]
    metadata_summary = {
        "drug_count": DRUG_COUNT,
        "drugs_with_gene_evidence": int(sum(bool(values) for values in genes)),
        "drugs_with_pathway_evidence": int(sum(bool(values) for values in pathways)),
        "mean_gene_count": float(np.mean([len(values) for values in genes])),
        "mean_pathway_count": float(np.mean([len(values) for values in pathways])),
        "used_relations": structure_checks["used_relations"],
    }
    task_checks = {
        **train_checks,
        **task_diagnostics,
        "deterministic_random_pair_count": random_pair_count,
        "deterministic_random_task_baseline": random_task_baseline,
    }
    hashes_after = {str(path): sha256(path) for path in INPUT_PATHS}
    source_feature_safe = bool(
        hashes_before == hashes_after
        and smd_hash == sha256(smd_path)
        and hashes_before[str(FREQUENCY_PATH)] == sha256(FREQUENCY_PATH)
        and hashes_before[str(SPLIT_PATH)] == sha256(SPLIT_PATH)
    )
    embedding_exclusion = bool(
        not any(path.name == "biokorf_kg_embeddings.pt" for path in INPUT_PATHS)
        and "torch" not in globals()
    )
    checks = {
        "explicit_kg_structure": bool(
            structure_checks["only_typed_maps_drug_gene_pathway_gene_used"]
            and not structure_checks["drug_phenotype_relations_used"]
            and not structure_checks["adverse_drug_reaction_relations_used"]
            and validate_matrices(matrices)
        ),
        "task_aware_leakage": bool(
            train_checks["validation_positions_hidden"]
            and train_checks["test_positions_hidden"]
        ),
        "source_feature_preservation": source_feature_safe,
        "kg_embedding_cosine_exclusion": embedding_exclusion,
    }
    if not all(checks.values()):
        raise RuntimeError(f"A Step 31E safety check failed: {checks}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_npy(STRUCTURE_MATRIX_PATH, matrices["STRUCTURE_ONLY"])
    write_npy(TASK_MATRIX_PATH, matrices["TASK_ONLY_EXPLICIT"])
    write_npy(KG_TASK_MATRIX_PATH, matrices["KG_TASK_EXPLICIT"])
    write_csv(STRUCTURE_EDGES_PATH, edges["STRUCTURE_ONLY"], EDGE_COLUMNS)
    write_csv(TASK_EDGES_PATH, edges["TASK_ONLY_EXPLICIT"], EDGE_COLUMNS)
    write_csv(KG_TASK_EDGES_PATH, edges["KG_TASK_EXPLICIT"], EDGE_COLUMNS)
    write_csv(
        METADATA_PATH,
        metadata_rows,
        [
            "matrix_index", "biokorf_drug_id", "drug_name", "gene_count",
            "pathway_count", "has_gene_evidence", "has_pathway_evidence",
        ],
    )
    diagnostics_columns = [
        "graph", "directed_nonself_edge_count", "source_drug_coverage_count",
        "source_drug_coverage_percentage", "mean_degree", "isolated_drug_count",
        "mean_edge_value", "median_edge_value", "evaluable_task_edge_count",
        "mean_task_similarity", "median_task_similarity", "random_task_baseline",
        "task_enrichment", "mean_gene_jaccard", "mean_pathway_jaccard",
        "fraction_with_shared_gene", "fraction_with_shared_pathway",
    ]
    write_csv(DIAGNOSTICS_PATH, graph_rows, diagnostics_columns)
    write_csv(
        COMPLEMENTARITY_PATH,
        complementarity_rows,
        [
            "entity_type", "view_index", "view_name", "source_file",
            "pearson_off_diagonal", "spearman_off_diagonal",
            "mean_top10_neighbor_jaccard", "comparison_selection_rule",
        ],
    )
    report = build_report(
        graph_rows, complementarity_rows, metadata_summary, task_checks, checks
    )
    temporary_report = REPORT_PATH.with_suffix(REPORT_PATH.suffix + ".tmp")
    temporary_report.write_text(report, encoding="utf-8")
    temporary_report.replace(REPORT_PATH)

    final_hashes = {str(path): sha256(path) for path in INPUT_PATHS}
    if final_hashes != hashes_before or sha256(smd_path) != smd_hash:
        raise RuntimeError("An input/source artifact changed while Step 31E outputs were written")
    print("Graph | Edge count | Drug coverage | Mean task similarity | Task enrichment")
    for row in graph_rows:
        print(
            f"{row['graph']} | {row['directed_nonself_edge_count']} | "
            f"{row['source_drug_coverage_percentage']:.4f}% | "
            f"{format_metric(row['mean_task_similarity'])} | "
            f"{format_metric(row['task_enrichment'])}"
        )
    kg_task = next(row for row in graph_rows if row["graph"] == "KG_TASK_EXPLICIT")
    complement_by_name = {row["view_name"]: row for row in complementarity_rows}
    print(f"KG_TASK_EXPLICIT mean gene Jaccard: {kg_task['mean_gene_jaccard']:.8f}")
    print(f"KG_TASK_EXPLICIT mean pathway Jaccard: {kg_task['mean_pathway_jaccard']:.8f}")
    print(
        "KG_TASK_EXPLICIT top10 overlap with SMDdatabase: "
        f"{complement_by_name['SMDdatabase']['mean_top10_neighbor_jaccard']:.8f}"
    )
    print(
        "KG_TASK_EXPLICIT top10 overlap with drug_target_sim: "
        f"{complement_by_name['drug_target_sim']['mean_top10_neighbor_jaccard']:.8f}"
    )
    print(
        "KG_TASK_EXPLICIT top10 overlap with drug_p_e_sim: "
        f"{complement_by_name['drug_p_e_sim']['mean_top10_neighbor_jaccard']:.8f}"
    )
    print(f"EXPLICIT KG STRUCTURE CHECK: {'PASS' if checks['explicit_kg_structure'] else 'FAIL'}")
    print(f"TASK-AWARE LEAKAGE CHECK: {'PASS' if checks['task_aware_leakage'] else 'FAIL'}")
    print(f"SOURCE FEATURE PRESERVATION CHECK: {'PASS' if checks['source_feature_preservation'] else 'FAIL'}")
    print(f"KG EMBEDDING COSINE EXCLUSION CHECK: {'PASS' if checks['kg_embedding_cosine_exclusion'] else 'FAIL'}")
    print("Training/testing performed: NO")


if __name__ == "__main__":
    main()
