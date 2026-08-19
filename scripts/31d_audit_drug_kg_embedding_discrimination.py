"""Audit discrimination of frozen BioKORF Drug KG embeddings.

This is a read-only structural/embedding audit. It does not load frequency
labels, train a model, test a model, or alter any artifact.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KG_EMBEDDING_PATH = (
    PROJECT_ROOT / "data_processed" / "kg_features" / "biokorf_kg_embeddings.pt"
)
KG_INDEX_PATH = (
    PROJECT_ROOT / "data_processed" / "kg_features" / "drug_kg_embedding_index.csv"
)
DRUG_ANCHOR_PATH = (
    PROJECT_ROOT / "data_processed" / "optimuskg" / "final_drug_anchor_mapping.csv"
)
KG_EDGES_PATH = PROJECT_ROOT / "data_processed" / "biomedical_kg" / "edges.csv"
STEP31B_ADDED_PATH = (
    PROJECT_ROOT / "data_processed" / "rewiring" / "smd_database"
    / "added_edges_kg_task.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "data_processed" / "rewiring" / "kg_embedding_audit"
STATISTICS_PATH = OUTPUT_DIR / "drug_kg_embedding_statistics.json"
COSINE_PATH = OUTPUT_DIR / "drug_kg_cosine_distribution.csv"
NEIGHBOR_PATH = OUTPUT_DIR / "drug_kg_neighbor_diagnostics.csv"
DUPLICATE_PATH = OUTPUT_DIR / "drug_kg_duplicate_groups.csv"
BIOLOGICAL_SUPPORT_PATH = OUTPUT_DIR / "step31b_edge_biological_support.csv"
REPORT_PATH = OUTPUT_DIR / "drug_kg_embedding_audit_report.txt"

DRUG_COUNT = 757
EMBEDDING_DIM = 128
EXPECTED_AVAILABLE_COUNT = 730
EXPECTED_STEP31B_EDGE_COUNT = 1911
NEAR_ZERO_VARIANCE_THRESHOLD = 1e-6
RANDOM_SEED = 42
TOP_K = 20
ROUNDING_PRECISIONS = (6, 4, 3)

INPUT_PATHS = (
    KG_EMBEDDING_PATH,
    KG_INDEX_PATH,
    DRUG_ANCHOR_PATH,
    KG_EDGES_PATH,
    STEP31B_ADDED_PATH,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_inputs() -> None:
    missing = [path for path in INPUT_PATHS if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required KG audit inputs are missing: {missing}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_bool(value: str) -> bool:
    normalized = value.strip().casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Could not parse Boolean value: {value!r}")


def load_drug_metadata() -> tuple[list[str], list[str], np.ndarray]:
    index_rows = read_csv(KG_INDEX_PATH)
    anchor_rows = read_csv(DRUG_ANCHOR_PATH)
    if len(index_rows) != DRUG_COUNT or len(anchor_rows) != DRUG_COUNT:
        raise ValueError("Drug index and anchor mapping must each contain 757 rows")
    index_values = [int(row["matrix_index"]) for row in index_rows]
    anchor_values = [int(row["matrix_index"]) for row in anchor_rows]
    embedding_rows = [int(row["embedding_row"]) for row in index_rows]
    expected = list(range(DRUG_COUNT))
    if index_values != expected or anchor_values != expected or embedding_rows != expected:
        raise ValueError("Drug metadata order is not exactly matrix_index/embedding_row 0..756")
    names = [row["drug_name"] for row in index_rows]
    anchor_names = [row["drug_name"] for row in anchor_rows]
    anchors = [row["biokorf_drug_id"] for row in anchor_rows]
    expected_anchors = [f"BIOKORF_DRUG_{index:03d}" for index in expected]
    if names != anchor_names or anchors != expected_anchors:
        raise ValueError("Embedding index and final anchor mapping order/names disagree")
    csv_mask = np.asarray([parse_bool(row["kg_available"]) for row in index_rows])
    return names, anchors, csv_mask


def load_embeddings(csv_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    artifact = torch.load(KG_EMBEDDING_PATH, map_location="cpu", weights_only=False)
    required = {"drug_embeddings", "drug_available_mask", "drug_matrix_indices"}
    missing = required.difference(artifact)
    if missing:
        raise KeyError(f"KG embedding artifact is missing fields: {sorted(missing)}")
    embeddings = artifact["drug_embeddings"].detach().cpu().numpy().astype(np.float64)
    mask = artifact["drug_available_mask"].detach().cpu().numpy().astype(bool)
    indices = artifact["drug_matrix_indices"].detach().cpu().numpy().astype(np.int64)
    if embeddings.shape != (DRUG_COUNT, EMBEDDING_DIM):
        raise ValueError(f"Expected drug_embeddings [757,128], found {embeddings.shape}")
    if mask.shape != (DRUG_COUNT,) or indices.shape != (DRUG_COUNT,):
        raise ValueError("Unexpected Drug KG mask/index shape")
    if not np.array_equal(indices, np.arange(DRUG_COUNT)):
        raise ValueError("Artifact drug_matrix_indices are not exactly 0..756")
    if not np.array_equal(mask, csv_mask):
        raise ValueError("Artifact availability mask differs from drug_kg_embedding_index.csv")
    if not np.isfinite(embeddings).all():
        raise ValueError("Drug KG embeddings contain non-finite values")
    available_count = int(mask.sum())
    if available_count != EXPECTED_AVAILABLE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_AVAILABLE_COUNT} KG-available drugs; found {available_count}"
        )
    diagnostics = {
        "embedding_shape": list(embeddings.shape),
        "mask_shape": list(mask.shape),
        "available_count": available_count,
        "unavailable_count": int((~mask).sum()),
        "ordering_check": True,
        "finite_value_check": True,
        "mask_matches_index_csv": True,
    }
    return embeddings, mask, diagnostics


def embedding_variance_and_rank(available: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    norms = np.linalg.norm(available, axis=1)
    centered = available - available.mean(axis=0, keepdims=True)
    dimension_std = centered.std(axis=0)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    energy = singular_values**2
    energy_total = float(energy.sum())
    explained = np.cumsum(energy) / energy_total if energy_total > 0 else np.zeros_like(energy)
    tolerance = (
        float(singular_values[0]) * max(centered.shape) * np.finfo(np.float64).eps
        if singular_values.size else 0.0
    )
    numerical_rank = int(np.count_nonzero(singular_values > tolerance))
    probabilities = energy / energy_total if energy_total > 0 else np.zeros_like(energy)
    positive = probabilities > 0
    entropy = float(-np.sum(probabilities[positive] * np.log(probabilities[positive])))
    effective_rank = float(np.exp(entropy))
    statistics = {
        "l2_norm": {
            "mean": float(norms.mean()),
            "std": float(norms.std()),
            "min": float(norms.min()),
            "max": float(norms.max()),
        },
        "centering": "feature-wise mean subtraction over KG-available drugs",
        "per_dimension_standard_deviation": dimension_std.tolist(),
        "mean_per_dimension_standard_deviation": float(dimension_std.mean()),
        "minimum_per_dimension_standard_deviation": float(dimension_std.min()),
        "maximum_per_dimension_standard_deviation": float(dimension_std.max()),
        "near_zero_variance_threshold": NEAR_ZERO_VARIANCE_THRESHOLD,
        "near_zero_variance_dimension_count": int(
            np.count_nonzero(dimension_std <= NEAR_ZERO_VARIANCE_THRESHOLD)
        ),
        "first_20_singular_values": singular_values[:20].tolist(),
        "variance_explained": {
            str(count): float(explained[count - 1])
            for count in (1, 2, 5, 10, 20)
        },
        "numerical_rank": numerical_rank,
        "numerical_rank_tolerance": tolerance,
        "numerical_rank_tolerance_definition": (
            "largest_singular_value * max(n_samples, n_features) * float64_epsilon"
        ),
        "effective_rank": effective_rank,
        "effective_rank_definition": "exp(entropy(normalized squared singular values))",
        "effective_rank_fraction_of_128": effective_rank / EMBEDDING_DIM,
    }
    return statistics, centered


def l2_normalize_available(
    embeddings: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    available_indices = np.flatnonzero(mask)
    available = embeddings[available_indices]
    norms = np.linalg.norm(available, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("A KG-available Drug embedding has zero L2 norm")
    return available_indices, available / norms


def distribution(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Cosine distribution must be finite and non-empty")
    percentiles = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
    return {
        "pair_count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p01": float(percentiles[0]),
        "p05": float(percentiles[1]),
        "p25": float(percentiles[2]),
        "median": float(percentiles[3]),
        "p75": float(percentiles[4]),
        "p95": float(percentiles[5]),
        "p99": float(percentiles[6]),
        "max": float(values.max()),
        "fraction_ge_0_90": float(np.mean(values >= 0.90)),
        "fraction_ge_0_95": float(np.mean(values >= 0.95)),
        "fraction_ge_0_99": float(np.mean(values >= 0.99)),
        "fraction_ge_0_999": float(np.mean(values >= 0.999)),
        "fraction_approximately_1_atol_1e_6": float(
            np.mean(np.isclose(values, 1.0, atol=1e-6, rtol=0.0))
        ),
    }


def deterministic_top_neighbors(
    cosine: np.ndarray, available_indices: np.ndarray, k: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    local_neighbors: list[np.ndarray] = []
    neighbor_values: list[np.ndarray] = []
    local = np.arange(len(available_indices), dtype=np.int64)
    for source in range(len(available_indices)):
        candidates = local[local != source]
        values = cosine[source, candidates]
        order = np.lexsort((available_indices[candidates], -values))
        selected = candidates[order[:k]]
        local_neighbors.append(selected)
        neighbor_values.append(cosine[source, selected])
    return local_neighbors, neighbor_values


def neighbor_diagnostics(
    cosine: np.ndarray,
    available_indices: np.ndarray,
    names: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray]:
    neighbors, values = deterministic_top_neighbors(cosine, available_indices, TOP_K)
    rows: list[dict[str, Any]] = []
    for local_source, (selected, selected_values) in enumerate(zip(neighbors, values)):
        matrix_index = int(available_indices[local_source])
        rows.append(
            {
                "matrix_index": matrix_index,
                "drug_name": names[matrix_index],
                "top1_neighbor_index": int(available_indices[selected[0]]),
                "top1_neighbor_name": names[int(available_indices[selected[0]])],
                "top1_nonself_cosine": float(selected_values[0]),
                "top5_mean_cosine": float(selected_values[:5].mean()),
                "top10_mean_cosine": float(selected_values[:10].mean()),
                "tenth_neighbor_cosine": float(selected_values[9]),
                "top1_minus_tenth_similarity_margin": float(
                    selected_values[0] - selected_values[9]
                ),
                "top1_minus_top10_mean_margin": float(
                    selected_values[0] - selected_values[:10].mean()
                ),
            }
        )
    metrics = (
        "top1_nonself_cosine",
        "top5_mean_cosine",
        "top10_mean_cosine",
        "top1_minus_tenth_similarity_margin",
        "top1_minus_top10_mean_margin",
    )
    summary = {
        metric: distribution(np.asarray([row[metric] for row in rows]))
        for metric in metrics
    }
    margins = np.asarray(
        [row["top1_minus_tenth_similarity_margin"] for row in rows], dtype=np.float64
    )
    summary["weak_nearest_neighbor_discrimination"] = bool(
        margins.mean() < 0.01 or np.median(margins) < 0.005
    )
    return rows, summary, np.concatenate(values)


def duplicate_groups(
    available_embeddings: np.ndarray,
    available_indices: np.ndarray,
    names: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for decimals in ROUNDING_PRECISIONS:
        rounded = np.round(available_embeddings, decimals=decimals)
        _unique, inverse, counts = np.unique(
            rounded, axis=0, return_inverse=True, return_counts=True
        )
        groups: list[tuple[int, list[int]]] = []
        for group_id, count in enumerate(counts):
            if count <= 1:
                continue
            members = available_indices[np.flatnonzero(inverse == group_id)].astype(int).tolist()
            groups.append((int(count), members))
        groups.sort(key=lambda item: (-item[0], item[1]))
        summary[str(decimals)] = {
            "unique_representation_count": int(len(counts)),
            "duplicate_group_count": len(groups),
            "duplicate_embedding_count": int(sum(size for size, _members in groups)),
            "largest_duplicate_group_size": max((size for size, _members in groups), default=1),
            "duplicate_group_sizes": [size for size, _members in groups],
        }
        for rank, (size, members) in enumerate(groups, start=1):
            rows.append(
                {
                    "rounding_decimals": decimals,
                    "group_rank_within_precision": rank,
                    "group_size": size,
                    "member_matrix_indices": json.dumps(members),
                    "member_drug_names": json.dumps(
                        [names[index] for index in members], ensure_ascii=False
                    ),
                }
            )
    rows.sort(
        key=lambda row: (
            -int(row["group_size"]),
            int(row["rounding_decimals"]),
            int(row["group_rank_within_precision"]),
        )
    )
    return rows, summary


def load_step31b_edges(
    cosine: np.ndarray,
    available_indices: np.ndarray,
    mask: np.ndarray,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    rows = read_csv(STEP31B_ADDED_PATH)
    if len(rows) != EXPECTED_STEP31B_EDGE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_STEP31B_EDGE_COUNT} Step-31B added edges; found {len(rows)}"
        )
    matrix_to_local = {int(index): local for local, index in enumerate(available_indices)}
    values: list[float] = []
    parsed: list[dict[str, Any]] = []
    for row in rows:
        source = int(row["source_drug_index"])
        target = int(row["target_drug_index"])
        if not mask[source] or not mask[target]:
            raise ValueError("A Step-31B KG_TASK edge contains a KG-unavailable endpoint")
        calculated = float(cosine[matrix_to_local[source], matrix_to_local[target]])
        recorded = float(row["kg_similarity"])
        if not math.isclose(calculated, recorded, rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError(
                f"Step-31B recorded KG similarity differs at edge {(source, target)}"
            )
        values.append(calculated)
        parsed.append(
            {
                "source": source,
                "target": target,
                "kg_similarity": calculated,
            }
        )
    return parsed, np.asarray(values, dtype=np.float64)


def deterministic_random_pairs(
    available_indices: np.ndarray, count: int
) -> list[tuple[int, int]]:
    row, column = np.triu_indices(len(available_indices), k=1)
    if count > len(row):
        raise ValueError("Requested random pair sample exceeds all unique available pairs")
    rng = np.random.default_rng(RANDOM_SEED)
    selected = rng.choice(len(row), size=count, replace=False)
    return [
        (int(available_indices[row[position]]), int(available_indices[column[position]]))
        for position in selected
    ]


def pair_cosines(
    pairs: list[tuple[int, int]],
    cosine: np.ndarray,
    available_indices: np.ndarray,
) -> np.ndarray:
    matrix_to_local = {int(index): local for local, index in enumerate(available_indices)}
    return np.asarray(
        [cosine[matrix_to_local[left], matrix_to_local[right]] for left, right in pairs],
        dtype=np.float64,
    )


def load_explicit_structure(
    anchors: list[str],
) -> tuple[list[set[str]], list[set[str]], dict[str, Any]]:
    anchor_to_drugs: dict[str, set[str]] = defaultdict(set)
    drug_to_genes: dict[str, set[str]] = defaultdict(set)
    gene_to_pathways: dict[str, set[str]] = defaultdict(set)
    relation_counts: dict[str, int] = defaultdict(int)
    with KG_EDGES_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"source", "target", "relation", "source_type", "target_type"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise KeyError(f"Biomedical KG edges are missing columns: {sorted(missing)}")
        for row in reader:
            source = row["source"]
            target = row["target"]
            source_type = row["source_type"]
            target_type = row["target_type"]
            relation_counts[row["relation"]] += 1
            if (
                row["relation"] == "MAPS_TO_DRUG"
                and source_type == "BIOKORF_DRUG"
                and target_type == "DRUG"
            ):
                anchor_to_drugs[source].add(target)
            elif {source_type, target_type} == {"DRUG", "GENE"}:
                drug = source if source_type == "DRUG" else target
                gene = target if target_type == "GENE" else source
                drug_to_genes[drug].add(gene)
            elif {source_type, target_type} == {"PATHWAY", "GENE"}:
                pathway = source if source_type == "PATHWAY" else target
                gene = target if target_type == "GENE" else source
                gene_to_pathways[gene].add(pathway)
    genes_by_index: list[set[str]] = []
    pathways_by_index: list[set[str]] = []
    for anchor in anchors:
        identities = anchor_to_drugs.get(anchor, set())
        genes = set().union(*(drug_to_genes.get(identity, set()) for identity in identities))
        pathways = set().union(*(gene_to_pathways.get(gene, set()) for gene in genes))
        genes_by_index.append(genes)
        pathways_by_index.append(pathways)
    diagnostics = {
        "anchor_count": len(anchors),
        "anchors_with_mapped_drug_identity": int(
            sum(bool(anchor_to_drugs.get(anchor)) for anchor in anchors)
        ),
        "anchors_with_direct_genes": int(sum(bool(values) for values in genes_by_index)),
        "anchors_with_reachable_pathways": int(
            sum(bool(values) for values in pathways_by_index)
        ),
        "relation_counts": dict(sorted(relation_counts.items())),
        "structure_definition": (
            "BIOKORF_DRUG --MAPS_TO_DRUG--> DRUG; direct typed DRUG-GENE edges; "
            "PATHWAY reached through typed PATHWAY-GENE edges"
        ),
    }
    return genes_by_index, pathways_by_index, diagnostics


def set_support(left: set[str], right: set[str]) -> tuple[int, float]:
    shared = len(left & right)
    union = len(left | right)
    return shared, shared / union if union else 0.0


def biological_support_rows(
    pair_group: str,
    pairs: list[tuple[int, int]],
    names: list[str],
    genes: list[set[str]],
    pathways: list[set[str]],
    cosine: np.ndarray,
    available_indices: np.ndarray,
) -> list[dict[str, Any]]:
    cosine_values = pair_cosines(pairs, cosine, available_indices)
    rows: list[dict[str, Any]] = []
    for (source, target), cosine_value in zip(pairs, cosine_values):
        shared_genes, gene_jaccard = set_support(genes[source], genes[target])
        shared_pathways, pathway_jaccard = set_support(
            pathways[source], pathways[target]
        )
        rows.append(
            {
                "pair_group": pair_group,
                "source_drug_index": source,
                "source_drug_name": names[source],
                "target_drug_index": target,
                "target_drug_name": names[target],
                "kg_embedding_cosine": float(cosine_value),
                "source_gene_count": len(genes[source]),
                "target_gene_count": len(genes[target]),
                "shared_gene_count": shared_genes,
                "gene_jaccard": gene_jaccard,
                "source_pathway_count": len(pathways[source]),
                "target_pathway_count": len(pathways[target]),
                "shared_pathway_count": shared_pathways,
                "pathway_jaccard": pathway_jaccard,
            }
        )
    return rows


def support_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pair_count": len(rows),
        "mean_shared_gene_count": float(
            np.mean([row["shared_gene_count"] for row in rows])
        ),
        "fraction_with_shared_gene": float(
            np.mean([row["shared_gene_count"] > 0 for row in rows])
        ),
        "mean_gene_jaccard": float(np.mean([row["gene_jaccard"] for row in rows])),
        "mean_shared_pathway_count": float(
            np.mean([row["shared_pathway_count"] for row in rows])
        ),
        "fraction_with_shared_pathway": float(
            np.mean([row["shared_pathway_count"] > 0 for row in rows])
        ),
        "mean_pathway_jaccard": float(
            np.mean([row["pathway_jaccard"] for row in rows])
        ),
    }


def edge_distinguishability(
    added: dict[str, Any],
    global_pairs: dict[str, Any],
    top20: dict[str, Any],
) -> dict[str, Any]:
    global_standardized = (
        (added["mean"] - global_pairs["mean"]) / global_pairs["std"]
        if global_pairs["std"] > 0 else math.nan
    )
    top20_standardized = (
        (added["mean"] - top20["mean"]) / top20["std"]
        if top20["std"] > 0 else math.nan
    )
    global_close = bool(
        abs(global_standardized) < 0.10
        and abs(added["median"] - global_pairs["median"]) < 0.01
        and abs(added["fraction_ge_0_99"] - global_pairs["fraction_ge_0_99"]) < 0.05
    )
    top20_close = bool(
        abs(top20_standardized) < 0.10
        and abs(added["median"] - top20["median"]) < 0.01
        and abs(added["fraction_ge_0_99"] - top20["fraction_ge_0_99"]) < 0.05
    )
    if global_close:
        verdict = "NOT_DISTINGUISHABLE_FROM_GLOBAL_COSINE_DISTRIBUTION"
    elif top20_close:
        verdict = "DISTINGUISHABLE_FROM_GLOBAL_BUT_NOT_FROM_TOP20_NEIGHBORS"
    elif abs(global_standardized) < 0.25:
        verdict = "WEAKLY_DISTINGUISHABLE_FROM_GLOBAL_COSINE_DISTRIBUTION"
    else:
        verdict = "DISTINGUISHABLE_FROM_GLOBAL_COSINE_DISTRIBUTION"
    return {
        "mean_delta_vs_all_available_pairs": added["mean"] - global_pairs["mean"],
        "median_delta_vs_all_available_pairs": added["median"] - global_pairs["median"],
        "standardized_mean_delta_vs_all_available_pairs": global_standardized,
        "mean_delta_vs_top20_neighbors": added["mean"] - top20["mean"],
        "median_delta_vs_top20_neighbors": added["median"] - top20["median"],
        "standardized_mean_delta_vs_top20_neighbors": top20_standardized,
        "verdict": verdict,
        "rule": (
            "not distinguishable requires |standardized mean delta|<0.10, "
            "|median delta|<0.01, and |fraction cosine>=0.99 delta|<0.05"
        ),
    }


def collapse_diagnosis(
    pairwise: dict[str, Any],
    rank_stats: dict[str, Any],
    duplicate_stats: dict[str, Any],
    neighbor_stats: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    compression = bool(
        pairwise["median"] > 0.90 or pairwise["fraction_ge_0_95"] > 0.50
    )
    strong_indicators = {
        "majority_pairs_ge_0_99": pairwise["fraction_ge_0_99"] > 0.50,
        "effective_rank_below_10_percent_of_128": (
            rank_stats["effective_rank"] < 0.10 * EMBEDDING_DIM
        ),
        "large_near_duplicate_groups": bool(
            duplicate_stats["3"]["largest_duplicate_group_size"] >= 10
            or duplicate_stats["3"]["unique_representation_count"]
            < 0.75 * EXPECTED_AVAILABLE_COUNT
        ),
        "weak_nearest_neighbor_margin": bool(
            neighbor_stats["weak_nearest_neighbor_discrimination"]
        ),
    }
    strong_count = sum(strong_indicators.values())
    if strong_count >= 2:
        classification = "STRONG_REPRESENTATION_COLLAPSE"
    elif compression:
        classification = "HIGH_SIMILARITY_COMPRESSION"
    elif (
        pairwise["median"] < 0.75
        and pairwise["fraction_ge_0_95"] < 0.10
        and rank_stats["effective_rank"] >= 0.25 * EMBEDDING_DIM
        and not strong_indicators["large_near_duplicate_groups"]
    ):
        classification = "HEALTHY_DISCRIMINATION"
    else:
        classification = "INCONCLUSIVE"
    rules = {
        "high_similarity_compression_rule": (
            "median pairwise cosine > 0.90 OR fraction cosine>=0.95 > 0.50"
        ),
        "strong_collapse_rule": (
            "at least two of: majority cosine>=0.99, effective rank <12.8, "
            "large rounded duplicate groups, weak nearest-neighbor margin"
        ),
        "compression_indicator": compression,
        "strong_indicators": strong_indicators,
        "strong_indicator_count": strong_count,
    }
    return classification, rules


def recommendation(
    diagnosis: str,
    added_support: dict[str, Any],
    random_support: dict[str, Any],
) -> tuple[str, str]:
    explicit_signal = bool(
        added_support["mean_gene_jaccard"]
        > random_support["mean_gene_jaccard"] + 0.01
        or added_support["mean_pathway_jaccard"]
        > random_support["mean_pathway_jaccard"] + 0.01
        or added_support["fraction_with_shared_gene"]
        > random_support["fraction_with_shared_gene"] + 0.05
        or added_support["fraction_with_shared_pathway"]
        > random_support["fraction_with_shared_pathway"] + 0.05
    )
    if diagnosis == "HEALTHY_DISCRIMINATION":
        return (
            "KEEP_KG_EMBEDDING_COSINE_FOR_REWIRING",
            "Pairwise spread, rank, duplicate groups, and nearest-neighbor margins meet the healthy heuristic.",
        )
    if diagnosis in {
        "STRONG_REPRESENTATION_COLLAPSE", "HIGH_SIMILARITY_COMPRESSION"
    } and explicit_signal:
        return (
            "REPLACE_EMBEDDING_COSINE_WITH_EXPLICIT_KG_STRUCTURE",
            "Embedding similarity is compressed while explicit shared-gene/pathway overlap distinguishes Step-31B edges from random pairs.",
        )
    if diagnosis == "STRONG_REPRESENTATION_COLLAPSE":
        return (
            "RETRAIN_KG_ENCODER_WITH_MORE_DISCRIMINATIVE_NODE_FEATURES",
            "Multiple collapse indicators are present and explicit overlap does not provide a sufficiently clear replacement signal.",
        )
    return (
        "NEEDS_MORE_EVIDENCE",
        "The discrimination and explicit-structure diagnostics do not jointly support one replacement strategy.",
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


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def format_metric(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return "NA" if not np.isfinite(value) else f"{float(value):.8f}"
    return str(value)


def distribution_line(label: str, stats: dict[str, Any]) -> str:
    return (
        f"{label}: n={stats['pair_count']}; mean={stats['mean']:.8f}; "
        f"std={stats['std']:.8f}; min/median/max={stats['min']:.8f}/"
        f"{stats['median']:.8f}/{stats['max']:.8f}; p95={stats['p95']:.8f}; "
        f"p99={stats['p99']:.8f}; frac>=.95={stats['fraction_ge_0_95']:.6f}; "
        f"frac>=.99={stats['fraction_ge_0_99']:.6f}"
    )


def build_report(statistics: dict[str, Any], duplicate_rows: list[dict[str, Any]]) -> str:
    variance = statistics["embedding_variance_and_rank"]
    pairwise = statistics["cosine_distributions"]["ALL_AVAILABLE_UNIQUE_PAIRS"]
    neighbor = statistics["nearest_neighbor_discrimination"]
    support = statistics["biological_support"]
    lines = [
        "BioKORF Step 31D — Drug KG Embedding Discrimination Audit",
        "==========================================================",
        "Read-only audit; no frequency/fold labels, training, or testing.",
        "",
        "A. Embedding validation",
        "-----------------------",
        *[
            f"{key}: {format_metric(value)}"
            for key, value in statistics["embedding_validation"].items()
        ],
        "",
        "B. Embedding variance",
        "---------------------",
        f"Mean L2 norm: {variance['l2_norm']['mean']:.8f}",
        f"Std L2 norm: {variance['l2_norm']['std']:.8f}",
        f"Mean dimension std: {variance['mean_per_dimension_standard_deviation']:.10f}",
        f"Min dimension std: {variance['minimum_per_dimension_standard_deviation']:.10f}",
        f"Max dimension std: {variance['maximum_per_dimension_standard_deviation']:.10f}",
        f"Near-zero dimensions (<=1e-6): {variance['near_zero_variance_dimension_count']}",
        "",
        "C. SVD / effective rank",
        "-----------------------",
        "First 20 singular values: " + json.dumps(variance["first_20_singular_values"]),
        "Variance explained: " + json.dumps(variance["variance_explained"]),
        f"Numerical rank: {variance['numerical_rank']} (tolerance={variance['numerical_rank_tolerance']:.12g})",
        f"Effective rank: {variance['effective_rank']:.8f} / {EMBEDDING_DIM}",
        "",
        "D. Pairwise cosine distribution",
        "--------------------------------",
        distribution_line("ALL_AVAILABLE_UNIQUE_PAIRS", pairwise),
        f"p01={pairwise['p01']:.8f}; p05={pairwise['p05']:.8f}; p25={pairwise['p25']:.8f}; "
        f"p75={pairwise['p75']:.8f}",
        f"fraction >=0.90: {pairwise['fraction_ge_0_90']:.8f}",
        f"fraction >=0.95: {pairwise['fraction_ge_0_95']:.8f}",
        f"fraction >=0.99: {pairwise['fraction_ge_0_99']:.8f}",
        f"fraction >=0.999: {pairwise['fraction_ge_0_999']:.8f}",
        f"fraction approximately 1 (atol=1e-6): {pairwise['fraction_approximately_1_atol_1e_6']:.8f}",
        "",
        "E. Nearest-neighbor discrimination",
        "------------------------------------",
    ]
    for key, value in neighbor.items():
        if isinstance(value, dict):
            lines.append(distribution_line(key, value))
        else:
            lines.append(f"{key}: {value}")
    lines.extend(["", "F. Unique representations", "----------------------------"])
    for decimals, summary in statistics["duplicate_summary"].items():
        lines.append(
            f"Rounded {decimals} decimals: unique={summary['unique_representation_count']}; "
            f"duplicate_groups={summary['duplicate_group_count']}; "
            f"largest_group={summary['largest_duplicate_group_size']}"
        )
    lines.append("Largest 20 duplicate/near-duplicate groups:")
    for row in duplicate_rows[:20]:
        lines.append(
            f"  decimals={row['rounding_decimals']} size={row['group_size']} "
            f"indices={row['member_matrix_indices']} names={row['member_drug_names']}"
        )
    lines.extend(["", "G. Step-31B cosine edge audit", "--------------------------------"])
    for cohort, values in statistics["cosine_distributions"].items():
        lines.append(distribution_line(cohort, values))
    lines.append(
        "Step-31B distinguishability: "
        + json.dumps(statistics["step31b_edge_distinguishability"], sort_keys=True)
    )
    lines.extend(
        [
            "",
            "H. Explicit KG structural evidence",
            "----------------------------------",
            "Only existing MAPS_TO_DRUG, typed DRUG-GENE, and typed PATHWAY-GENE edges are used.",
            "STEP31B_KG_TASK_ADDED: " + json.dumps(support["STEP31B_KG_TASK_ADDED"], sort_keys=True),
            "DETERMINISTIC_RANDOM_AVAILABLE_PAIRS: "
            + json.dumps(support["DETERMINISTIC_RANDOM_AVAILABLE_PAIRS"], sort_keys=True),
            "",
            "I. Collapse diagnosis",
            "---------------------",
            f"Classification: {statistics['collapse_diagnosis']}",
            "Rules/results: " + json.dumps(statistics["collapse_rules"], sort_keys=True),
            "",
            "J. Recommendation",
            "-----------------",
            f"Recommendation: {statistics['recommendation']}",
            f"Reason: {statistics['recommendation_reason']}",
        ]
    )
    if statistics["recommendation"] == "REPLACE_EMBEDDING_COSINE_WITH_EXPLICIT_KG_STRUCTURE":
        lines.append(
            "Future rewiring should use explicit shared genes and shared pathways, not frozen embedding cosine."
        )
    lines.extend(
        [
            "",
            f"KG EMBEDDING ARTIFACT SAFETY CHECK: {'PASS' if statistics['checks']['kg_embedding_artifact_safety'] else 'FAIL'}",
            f"FREQUENCY-LABEL INDEPENDENCE CHECK: {'PASS' if statistics['checks']['frequency_label_independence'] else 'FAIL'}",
            "Training/testing performed: NO",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    require_inputs()
    hashes_before = {str(path): sha256(path) for path in INPUT_PATHS}
    names, anchors, csv_mask = load_drug_metadata()
    embeddings, mask, validation = load_embeddings(csv_mask)
    available_indices, normalized = l2_normalize_available(embeddings, mask)
    available_embeddings = embeddings[available_indices]
    rank_stats, _centered = embedding_variance_and_rank(available_embeddings)
    cosine = normalized @ normalized.T
    upper = np.triu_indices(len(available_indices), k=1)
    all_pair_values = cosine[upper]
    neighbor_rows, neighbor_summary, top20_values = neighbor_diagnostics(
        cosine, available_indices, names
    )
    duplicate_rows, duplicate_summary = duplicate_groups(
        available_embeddings, available_indices, names
    )
    step31b_edges, step31b_values = load_step31b_edges(
        cosine, available_indices, mask
    )
    random_pairs = deterministic_random_pairs(
        available_indices, EXPECTED_STEP31B_EDGE_COUNT
    )
    random_values = pair_cosines(random_pairs, cosine, available_indices)
    cosine_distributions = {
        "ALL_AVAILABLE_UNIQUE_PAIRS": distribution(all_pair_values),
        "DETERMINISTIC_RANDOM_AVAILABLE_PAIRS": distribution(random_values),
        "TOP20_DIRECTED_NEIGHBORS": distribution(top20_values),
        "STEP31B_KG_TASK_ADDED": distribution(step31b_values),
    }
    distinguishability = edge_distinguishability(
        cosine_distributions["STEP31B_KG_TASK_ADDED"],
        cosine_distributions["ALL_AVAILABLE_UNIQUE_PAIRS"],
        cosine_distributions["TOP20_DIRECTED_NEIGHBORS"],
    )

    genes, pathways, structure_diagnostics = load_explicit_structure(anchors)
    step31b_pairs = [(row["source"], row["target"]) for row in step31b_edges]
    added_support_rows = biological_support_rows(
        "STEP31B_KG_TASK_ADDED",
        step31b_pairs,
        names,
        genes,
        pathways,
        cosine,
        available_indices,
    )
    random_support_rows = biological_support_rows(
        "DETERMINISTIC_RANDOM_AVAILABLE_PAIRS",
        random_pairs,
        names,
        genes,
        pathways,
        cosine,
        available_indices,
    )
    biological_support = {
        "STEP31B_KG_TASK_ADDED": support_summary(added_support_rows),
        "DETERMINISTIC_RANDOM_AVAILABLE_PAIRS": support_summary(random_support_rows),
    }
    diagnosis, collapse_rules = collapse_diagnosis(
        cosine_distributions["ALL_AVAILABLE_UNIQUE_PAIRS"],
        rank_stats,
        duplicate_summary,
        neighbor_summary,
    )
    recommendation_value, recommendation_reason = recommendation(
        diagnosis,
        biological_support["STEP31B_KG_TASK_ADDED"],
        biological_support["DETERMINISTIC_RANDOM_AVAILABLE_PAIRS"],
    )
    hashes_after = {str(path): sha256(path) for path in INPUT_PATHS}
    forbidden_label_inputs = {"drug_side.pkl", "pilot_fold1_split.npz"}
    frequency_independent = bool(
        not any(path.name in forbidden_label_inputs for path in INPUT_PATHS)
        and structure_diagnostics["structure_definition"].startswith("BIOKORF_DRUG")
    )
    checks = {
        "kg_embedding_artifact_safety": hashes_before == hashes_after,
        "frequency_label_independence": frequency_independent,
    }
    if not all(checks.values()):
        raise RuntimeError(f"A KG embedding audit safety check failed: {checks}")

    statistics = {
        "parameters": {
            "near_zero_variance_threshold": NEAR_ZERO_VARIANCE_THRESHOLD,
            "random_seed": RANDOM_SEED,
            "nearest_neighbor_k": TOP_K,
            "rounding_precisions": list(ROUNDING_PRECISIONS),
        },
        "embedding_validation": validation,
        "embedding_variance_and_rank": rank_stats,
        "cosine_distributions": cosine_distributions,
        "step31b_edge_distinguishability": distinguishability,
        "nearest_neighbor_discrimination": neighbor_summary,
        "duplicate_summary": duplicate_summary,
        "step31b_edge_count": len(step31b_edges),
        "explicit_structure": structure_diagnostics,
        "biological_support": biological_support,
        "collapse_diagnosis": diagnosis,
        "collapse_rules": collapse_rules,
        "recommendation": recommendation_value,
        "recommendation_reason": recommendation_reason,
        "input_hashes_before": hashes_before,
        "input_hashes_after": hashes_after,
        "checks": checks,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(STATISTICS_PATH, statistics)
    distribution_rows = [
        {"cohort": cohort, **values}
        for cohort, values in cosine_distributions.items()
    ]
    distribution_columns = [
        "cohort", "pair_count", "mean", "std", "min", "p01", "p05", "p25",
        "median", "p75", "p95", "p99", "max", "fraction_ge_0_90",
        "fraction_ge_0_95", "fraction_ge_0_99", "fraction_ge_0_999",
        "fraction_approximately_1_atol_1e_6",
    ]
    write_csv(COSINE_PATH, distribution_rows, distribution_columns)
    write_csv(
        NEIGHBOR_PATH,
        neighbor_rows,
        [
            "matrix_index", "drug_name", "top1_neighbor_index", "top1_neighbor_name",
            "top1_nonself_cosine", "top5_mean_cosine", "top10_mean_cosine",
            "tenth_neighbor_cosine", "top1_minus_tenth_similarity_margin",
            "top1_minus_top10_mean_margin",
        ],
    )
    write_csv(
        DUPLICATE_PATH,
        duplicate_rows,
        [
            "rounding_decimals", "group_rank_within_precision", "group_size",
            "member_matrix_indices", "member_drug_names",
        ],
    )
    support_rows = added_support_rows + random_support_rows
    write_csv(
        BIOLOGICAL_SUPPORT_PATH,
        support_rows,
        [
            "pair_group", "source_drug_index", "source_drug_name",
            "target_drug_index", "target_drug_name", "kg_embedding_cosine",
            "source_gene_count", "target_gene_count", "shared_gene_count", "gene_jaccard",
            "source_pathway_count", "target_pathway_count", "shared_pathway_count",
            "pathway_jaccard",
        ],
    )
    report = build_report(statistics, duplicate_rows)
    temporary_report = REPORT_PATH.with_suffix(REPORT_PATH.suffix + ".tmp")
    temporary_report.write_text(report, encoding="utf-8")
    temporary_report.replace(REPORT_PATH)

    final_hashes = {str(path): sha256(path) for path in INPUT_PATHS}
    if final_hashes != hashes_before:
        raise RuntimeError("An input artifact changed while audit outputs were written")
    print(report, end="")


if __name__ == "__main__":
    main()
