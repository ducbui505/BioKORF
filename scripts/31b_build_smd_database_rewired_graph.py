"""Build additive SMDdatabase adjacency controls for BioKORF Step 31B.

The original similarity matrix is read-only. This script creates edge-list
adjacencies only and never trains or evaluates a prediction model.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Datas"
AUDIT_PATH = PROJECT_ROOT / "data_processed" / "rewiring" / "mssf_similarity_audit.csv"
SPLIT_PATH = PROJECT_ROOT / "data_processed" / "experiments" / "pilot_fold1_split.npz"
FREQUENCY_PATH = DATA_DIR / "drug_side.pkl"
KG_PATH = PROJECT_ROOT / "data_processed" / "kg_features" / "biokorf_kg_embeddings.pt"
MSSF_PATH = PROJECT_ROOT / "mssf.py"
MODEL_PATH = PROJECT_ROOT / "model.py"
CLEAN_MODEL_PATH = PROJECT_ROOT / "models" / "mssf_clean.py"
OUTPUT_DIR = PROJECT_ROOT / "data_processed" / "rewiring" / "smd_database"

DRUG_COUNT = 757
SIDE_EFFECT_COUNT = 994
K_ORIGINAL = 10
KG_CANDIDATE_K = 20
ADD_K = 3
MIN_TASK_OVERLAP = 3
SEED = 42
KG_WEIGHT = 0.5
TASK_WEIGHT = 0.5

ORIGINAL_OUTPUT = OUTPUT_DIR / "original_top10_edges.csv"
KG_ONLY_OUTPUT = OUTPUT_DIR / "kg_only_edges.csv"
TASK_ONLY_OUTPUT = OUTPUT_DIR / "task_only_edges.csv"
KG_TASK_OUTPUT = OUTPUT_DIR / "kg_task_edges.csv"
ADDED_KG_OUTPUT = OUTPUT_DIR / "added_edges_kg_only.csv"
ADDED_TASK_OUTPUT = OUTPUT_DIR / "added_edges_task_only.csv"
ADDED_KG_TASK_OUTPUT = OUTPUT_DIR / "added_edges_kg_task.csv"
COMPARISON_OUTPUT = OUTPUT_DIR / "graph_comparison.csv"
DIAGNOSTICS_OUTPUT = OUTPUT_DIR / "rewiring_diagnostics.json"
REPORT_OUTPUT = OUTPUT_DIR / "step31b_report.txt"

ORIGINAL_COLUMNS = [
    "source_drug_index",
    "target_drug_index",
    "original_similarity",
    "task_similarity",
    "kg_similarity",
    "edge_origin",
]
DETAILED_EDGE_COLUMNS = [
    "source_drug_index",
    "target_drug_index",
    "original_similarity",
    "task_similarity",
    "kg_similarity",
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


def require_files(paths: Iterable[Path]) -> None:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required Step 31B inputs are missing: {missing}")


def read_audit_source() -> tuple[dict[str, str], Path]:
    if not AUDIT_PATH.is_file():
        raise FileNotFoundError(f"Step 31A audit not found: {AUDIT_PATH}")
    with AUDIT_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        matches = [
            row
            for row in csv.DictReader(handle)
            if row.get("entity_type") == "DRUG"
            and row.get("view_index") == "3"
            and row.get("view_name") == "SMDdatabase"
        ]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one DRUG/view_index=3/SMDdatabase row in Step 31A audit; "
            f"found {len(matches)}"
        )
    row = matches[0]
    if row.get("provenance_class") != "EXTERNAL_STATIC":
        raise ValueError("Step 31A does not classify SMDdatabase as EXTERNAL_STATIC")
    if row.get("leakage_status") != "SAFE_EXTERNAL":
        raise ValueError("Step 31A does not classify SMDdatabase as SAFE_EXTERNAL")
    source = (PROJECT_ROOT / row["source_file"]).resolve()
    try:
        source.relative_to(DATA_DIR.resolve())
    except ValueError as error:
        raise ValueError(f"Audit source_file resolves outside Datas/: {source}") from error
    if not source.is_file():
        raise FileNotFoundError(f"Audited SMDdatabase source not found: {source}")
    return row, source


def load_pickle_matrix(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        matrix = np.asarray(pickle.load(handle), dtype=np.float64)
    return matrix


def load_source_matrix(path: Path) -> np.ndarray:
    matrix = load_pickle_matrix(path)
    if matrix.shape != (DRUG_COUNT, DRUG_COUNT):
        raise ValueError(
            f"SMDdatabase must have shape {(DRUG_COUNT, DRUG_COUNT)}; found {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("SMDdatabase contains NaN or infinite values")
    matrix.setflags(write=False)
    return matrix


def load_train_only_profiles() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    full_frequency = load_pickle_matrix(FREQUENCY_PATH)
    if full_frequency.shape != (DRUG_COUNT, SIDE_EFFECT_COUNT):
        raise ValueError(
            f"drug_side.pkl must have shape {(DRUG_COUNT, SIDE_EFFECT_COUNT)}"
        )
    with np.load(SPLIT_PATH) as split:
        required = {"train_samples", "validation_samples", "test_samples", "seed", "fold"}
        missing = required.difference(split.files)
        if missing:
            raise KeyError(f"Fixed split is missing arrays: {sorted(missing)}")
        if int(split["seed"]) != SEED or int(split["fold"]) != 1:
            raise ValueError("Expected fixed Fold-1 split with seed 42")
        train = np.asarray(split["train_samples"])
        validation_pairs = np.asarray(split["validation_samples"])[:, :2].astype(np.int64)
        test_pairs = np.asarray(split["test_samples"])[:, :2].astype(np.int64)
    if train.ndim != 2 or train.shape[1] != 3:
        raise ValueError("train_samples must contain drug, side-effect, and frequency columns")
    train_drug = train[:, 0].astype(np.int64)
    train_side = train[:, 1].astype(np.int64)
    train_labels = train[:, 2].astype(np.int64)
    if not set(np.unique(train_labels)).issubset({1, 2, 3, 4, 5}):
        raise ValueError("Training frequencies must be ordered labels in 1..5")
    if not np.array_equal(full_frequency[train_drug, train_side].astype(np.int64), train_labels):
        raise ValueError("Fold-1 training labels do not match drug_side.pkl at training positions")

    observed = np.zeros((DRUG_COUNT, SIDE_EFFECT_COUNT), dtype=bool)
    centered = np.zeros((DRUG_COUNT, SIDE_EFFECT_COUNT), dtype=np.float64)
    observed[train_drug, train_side] = True
    centered[train_drug, train_side] = train_labels.astype(np.float64) - 3.0

    validation_hidden = bool(
        not observed[validation_pairs[:, 0], validation_pairs[:, 1]].any()
    )
    test_hidden = bool(not observed[test_pairs[:, 0], test_pairs[:, 1]].any())
    leakage_safe = validation_hidden and test_hidden
    if not leakage_safe:
        raise RuntimeError("Validation or test positions entered the train-only task profiles")
    diagnostics = {
        "training_observation_count": int(observed.sum()),
        "validation_position_count": int(len(validation_pairs)),
        "test_position_count": int(len(test_pairs)),
        "validation_positions_hidden": validation_hidden,
        "test_positions_hidden": test_hidden,
        "missing_entries_are_unobserved": True,
        "centered_frequency_mapping": {"1": -2, "2": -1, "3": 0, "4": 1, "5": 2},
    }
    return observed, centered, train_labels, diagnostics


def build_masked_task_similarity(
    observed: np.ndarray, centered: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    overlap = observed.astype(np.int16) @ observed.astype(np.int16).T
    similarity = np.full((DRUG_COUNT, DRUG_COUNT), np.nan, dtype=np.float64)
    degenerate_count = 0
    for source in range(DRUG_COUNT):
        similarity[source, source] = 1.0 if overlap[source, source] >= MIN_TASK_OVERLAP else np.nan
        for target in range(source + 1, DRUG_COUNT):
            shared = int(overlap[source, target])
            if shared < MIN_TASK_OVERLAP:
                continue
            common = observed[source] & observed[target]
            left = centered[source, common]
            right = centered[target, common]
            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
            if denominator == 0.0:
                degenerate_count += 1
                continue
            value = float(np.dot(left, right) / denominator)
            similarity[source, target] = value
            similarity[target, source] = value
    upper = np.triu_indices(DRUG_COUNT, k=1)
    pair_overlaps = overlap[upper]
    valid = np.isfinite(similarity[upper])
    diagnostics = {
        "minimum_required_overlap": MIN_TASK_OVERLAP,
        "all_pair_overlap_min": int(pair_overlaps.min()),
        "all_pair_overlap_max": int(pair_overlaps.max()),
        "all_pair_overlap_mean": float(pair_overlaps.mean()),
        "all_pair_overlap_median": float(np.median(pair_overlaps)),
        "pairs_meeting_overlap_threshold": int(np.count_nonzero(pair_overlaps >= MIN_TASK_OVERLAP)),
        "pairs_with_valid_task_similarity": int(valid.sum()),
        "degenerate_zero_norm_pairs": int(degenerate_count),
        "unavailable_task_similarity_is_nan": True,
    }
    return similarity, overlap, diagnostics


def deterministic_rank(values: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=np.int64)
    scores = np.asarray(values[candidates], dtype=np.float64)
    finite_score = np.nan_to_num(scores, nan=-np.inf)
    return candidates[np.lexsort((candidates, -finite_score))]


def top_neighbors(
    matrix: np.ndarray,
    k: int,
    available: np.ndarray | None = None,
) -> list[np.ndarray]:
    entity_indices = np.arange(DRUG_COUNT, dtype=np.int64)
    permitted = (
        np.ones(DRUG_COUNT, dtype=bool)
        if available is None
        else np.asarray(available, dtype=bool)
    )
    neighbors: list[np.ndarray] = []
    for source in range(DRUG_COUNT):
        candidate_mask = permitted.copy()
        candidate_mask[source] = False
        candidates = entity_indices[candidate_mask]
        ranked = deterministic_rank(matrix[source], candidates)
        neighbors.append(ranked[: min(k, len(ranked))])
    return neighbors


def load_kg() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    artifact = torch.load(KG_PATH, map_location="cpu", weights_only=False)
    required = {"drug_embeddings", "drug_available_mask"}
    missing = required.difference(artifact)
    if missing:
        raise KeyError(f"KG artifact is missing fields: {sorted(missing)}")
    embeddings = artifact["drug_embeddings"].detach().cpu().numpy().astype(np.float64)
    available = artifact["drug_available_mask"].detach().cpu().numpy().astype(bool)
    if embeddings.shape != (DRUG_COUNT, 128) or available.shape != (DRUG_COUNT,):
        raise ValueError(
            f"Unexpected KG drug shapes: embeddings={embeddings.shape}, mask={available.shape}"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("KG drug embeddings contain non-finite values")
    normalized = np.zeros_like(embeddings)
    norms = np.linalg.norm(embeddings[available], axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("An available KG drug embedding has zero norm")
    normalized[available] = embeddings[available] / norms
    similarity = normalized @ normalized.T
    similarity[~available, :] = np.nan
    similarity[:, ~available] = np.nan
    return embeddings, available, similarity


def edge_row(
    source: int,
    target: int,
    origin: str,
    original: np.ndarray,
    task: np.ndarray,
    kg: np.ndarray,
    overlap: np.ndarray,
    combined_score: float = math.nan,
) -> dict[str, Any]:
    return {
        "source_drug_index": int(source),
        "target_drug_index": int(target),
        "original_similarity": float(original[source, target]),
        "task_similarity": float(task[source, target]),
        "kg_similarity": float(kg[source, target]),
        "combined_score": float(combined_score),
        "task_overlap_count": int(overlap[source, target]),
        "edge_origin": origin,
    }


def build_original_edges(
    original: np.ndarray,
    task: np.ndarray,
    kg: np.ndarray,
    overlap: np.ndarray,
) -> tuple[list[dict[str, Any]], list[set[int]]]:
    neighbors = top_neighbors(original, K_ORIGINAL)
    neighbor_sets = [set(map(int, row)) for row in neighbors]
    rows = [
        edge_row(source, int(target), "ORIGINAL", original, task, kg, overlap)
        for source, selected in enumerate(neighbors)
        for target in selected
    ]
    if len(rows) != DRUG_COUNT * K_ORIGINAL:
        raise RuntimeError("Original directed top-10 graph has an unexpected edge count")
    return rows, neighbor_sets


def build_kg_only_added(
    original_neighbors: list[set[int]],
    kg_neighbors: list[np.ndarray],
    available: np.ndarray,
    original: np.ndarray,
    task: np.ndarray,
    kg: np.ndarray,
    overlap: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in range(DRUG_COUNT):
        if not available[source]:
            continue
        eligible = [
            int(target) for target in kg_neighbors[source]
            if int(target) not in original_neighbors[source]
        ]
        for target in eligible[:ADD_K]:
            rows.append(
                edge_row(source, target, "KG_ONLY_ADDED", original, task, kg, overlap)
            )
    return rows


def build_task_only_added(
    original_neighbors: list[set[int]],
    original: np.ndarray,
    task: np.ndarray,
    kg: np.ndarray,
    overlap: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_indices = np.arange(DRUG_COUNT, dtype=np.int64)
    for source in range(DRUG_COUNT):
        valid = np.isfinite(task[source])
        valid[source] = False
        if original_neighbors[source]:
            valid[np.fromiter(original_neighbors[source], dtype=np.int64)] = False
        candidates = all_indices[valid]
        ranked = deterministic_rank(task[source], candidates)
        for target in ranked[:ADD_K]:
            rows.append(
                edge_row(
                    source, int(target), "TASK_ONLY_ADDED", original, task, kg, overlap
                )
            )
    return rows


def build_kg_task_added(
    original_neighbors: list[set[int]],
    kg_neighbors: list[np.ndarray],
    available: np.ndarray,
    original: np.ndarray,
    task: np.ndarray,
    kg: np.ndarray,
    overlap: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[int, float | None]]:
    rows: list[dict[str, Any]] = []
    baselines: dict[int, float | None] = {}
    for source in range(DRUG_COUNT):
        original_task_values = np.asarray(
            [task[source, target] for target in original_neighbors[source]],
            dtype=np.float64,
        )
        finite_baseline = original_task_values[np.isfinite(original_task_values)]
        baseline = float(finite_baseline.mean()) if finite_baseline.size else None
        baselines[source] = baseline
        if not available[source]:
            continue
        eligible: list[tuple[float, int]] = []
        for target_value in kg_neighbors[source]:
            target = int(target_value)
            if target in original_neighbors[source]:
                continue
            task_value = task[source, target]
            kg_value = kg[source, target]
            if not np.isfinite(task_value) or not np.isfinite(kg_value) or kg_value <= 0:
                continue
            threshold = baseline if baseline is not None else 0.0
            if task_value <= threshold:
                continue
            normalized_kg = (float(kg_value) + 1.0) / 2.0
            normalized_task = (float(task_value) + 1.0) / 2.0
            combined = KG_WEIGHT * normalized_kg + TASK_WEIGHT * normalized_task
            eligible.append((combined, target))
        eligible.sort(key=lambda item: (-item[0], item[1]))
        for combined, target in eligible[:ADD_K]:
            rows.append(
                edge_row(
                    source,
                    target,
                    "KG_TASK_ADDED",
                    original,
                    task,
                    kg,
                    overlap,
                    combined_score=combined,
                )
            )
    return rows, baselines


def combine_edges(
    original_rows: list[dict[str, Any]], added_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    combined = [dict(row) for row in original_rows] + [dict(row) for row in added_rows]
    pairs = [
        (row["source_drug_index"], row["target_drug_index"])
        for row in combined
    ]
    if len(pairs) != len(set(pairs)):
        raise RuntimeError("An additive graph contains duplicate directed edges")
    return combined


def connected_components(rows: list[dict[str, Any]]) -> tuple[int, int]:
    adjacency = [set() for _ in range(DRUG_COUNT)]
    for row in rows:
        source = int(row["source_drug_index"])
        target = int(row["target_drug_index"])
        adjacency[source].add(target)
        adjacency[target].add(source)
    unseen = set(range(DRUG_COUNT))
    sizes: list[int] = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack = [start]
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            next_nodes = adjacency[node].intersection(unseen)
            unseen.difference_update(next_nodes)
            stack.extend(sorted(next_nodes, reverse=True))
        sizes.append(size)
    return len(sizes), max(sizes, default=0)


def deterministic_random_task_baseline(
    task: np.ndarray, sample_size: int = 10000
) -> tuple[float, int]:
    upper = np.triu_indices(DRUG_COUNT, k=1)
    valid_positions = np.flatnonzero(np.isfinite(task[upper]))
    count = min(sample_size, len(valid_positions))
    if count == 0:
        return math.nan, 0
    rng = np.random.default_rng(SEED)
    selected = rng.choice(valid_positions, size=count, replace=False)
    return float(task[upper[0][selected], upper[1][selected]].mean()), count


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else math.nan


def graph_statistics(
    name: str,
    rows: list[dict[str, Any]],
    random_baseline: float,
    original_mean_task: float | None = None,
) -> dict[str, Any]:
    degrees = np.zeros(DRUG_COUNT, dtype=np.int64)
    pairs: set[tuple[int, int]] = set()
    task_values: list[float] = []
    per_source_tasks: list[list[float]] = [[] for _ in range(DRUG_COUNT)]
    added_rows = [row for row in rows if row["edge_origin"] != "ORIGINAL"]
    retained_original = [row for row in rows if row["edge_origin"] == "ORIGINAL"]
    for row in rows:
        source = int(row["source_drug_index"])
        target = int(row["target_drug_index"])
        degrees[source] += 1
        pairs.add((source, target))
        value = float(row["task_similarity"])
        if np.isfinite(value):
            task_values.append(value)
            per_source_tasks[source].append(value)
    reciprocal = sum((target, source) in pairs for source, target in pairs)
    components, largest = connected_components(rows)
    source_means = [float(np.mean(values)) for values in per_source_tasks if values]
    mean_task = float(np.mean(task_values)) if task_values else math.nan
    median_task = float(np.median(task_values)) if task_values else math.nan
    return {
        "graph": name,
        "original_edges": len(retained_original),
        "added_edges": len(added_rows),
        "total_directed_edges": len(rows),
        "mean_degree": float(degrees.mean()),
        "min_degree": int(degrees.min()),
        "max_degree": int(degrees.max()),
        "isolated_nodes": int(np.count_nonzero(degrees == 0)),
        "reciprocal_edge_percentage": 100.0 * reciprocal / len(pairs) if pairs else 0.0,
        "connected_component_count": components,
        "largest_component_size": largest,
        "mean_original_similarity_of_retained_original_edges": finite_mean(
            row["original_similarity"] for row in retained_original
        ),
        "added_mean_kg_similarity": finite_mean(row["kg_similarity"] for row in added_rows),
        "added_mean_task_similarity": finite_mean(row["task_similarity"] for row in added_rows),
        "added_mean_task_overlap_count": finite_mean(row["task_overlap_count"] for row in added_rows),
        "added_evaluable_task_edge_count": int(
            sum(np.isfinite(float(row["task_similarity"])) for row in added_rows)
        ),
        "mean_task_similarity": mean_task,
        "median_task_similarity": median_task,
        "top_neighborhood_task_similarity": (
            float(np.mean(source_means)) if source_means else math.nan
        ),
        "random_task_baseline": random_baseline,
        "task_enrichment": mean_task - random_baseline,
        "delta_mean_task_vs_original": (
            0.0 if original_mean_task is None else mean_task - original_mean_task
        ),
    }


def set_diagnostics(
    task_added: list[dict[str, Any]], kg_task_added: list[dict[str, Any]]
) -> dict[str, Any]:
    task_pairs = {
        (int(row["source_drug_index"]), int(row["target_drug_index"]))
        for row in task_added
    }
    kg_task_pairs = {
        (int(row["source_drug_index"]), int(row["target_drug_index"]))
        for row in kg_task_added
    }
    shared = task_pairs & kg_task_pairs
    union = task_pairs | kg_task_pairs
    return {
        "shared_added_edges": len(shared),
        "unique_to_task_only": len(task_pairs - kg_task_pairs),
        "unique_to_kg_task": len(kg_task_pairs - task_pairs),
        "jaccard_overlap": len(shared) / len(union) if union else 0.0,
        "kg_task_fraction_also_selected_by_task_only": (
            len(shared) / len(kg_task_pairs) if kg_task_pairs else 0.0
        ),
        "kg_task_fraction_unique_due_to_kg_candidate_restriction": (
            len(kg_task_pairs - task_pairs) / len(kg_task_pairs) if kg_task_pairs else 0.0
        ),
    }


def source_count(rows: list[dict[str, Any]]) -> int:
    return len({int(row["source_drug_index"]) for row in rows})


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
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
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


def format_value(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return "NA" if not np.isfinite(value) else f"{float(value):.8f}"
    return str(value)


def build_report(diagnostics: dict[str, Any]) -> str:
    graph_rows = diagnostics["graphs"]
    comparison = diagnostics["task_only_vs_kg_task"]
    coverage = diagnostics["coverage"]
    overlap = diagnostics["task_overlap"]
    lines = [
        "BioKORF Step 31B — SMDdatabase Additive Graph Construction",
        "===========================================================",
        "Graph construction only. Original similarity values are preserved and no model was trained/tested.",
        "",
        "Parameters",
        "----------",
        f"k_original = {K_ORIGINAL}",
        f"kg_candidate_k = {KG_CANDIDATE_K}",
        f"add_k = {ADD_K}",
        f"min_task_overlap = {MIN_TASK_OVERLAP}",
        f"seed = {SEED}",
        f"KG/task combined weights = {KG_WEIGHT:.1f} / {TASK_WEIGHT:.1f}",
        "Task frequency centering = {1:-2, 2:-1, 3:0, 4:+1, 5:+2}",
        "Missing/validation/test entries remain unobserved and never enter masked cosine calculations.",
        "",
        "Source view",
        "-----------",
        f"Audit source_file: {diagnostics['source']['audit_source_file']}",
        f"Resolved source: {diagnostics['source']['resolved_source_path']}",
        f"Source shape: {diagnostics['source']['shape']}",
        f"Source SHA-256: {diagnostics['source']['sha256_before']}",
        "The source SMDdatabase matrix was not altered and no KG/task score replaces a source similarity.",
        "",
        "Train-only overlap diagnostics",
        "------------------------------",
        *[f"{key}: {format_value(value)}" for key, value in overlap.items()],
        "",
        "Graph statistics and task alignment",
        "-----------------------------------",
    ]
    for row in graph_rows:
        lines.extend(
            [
                f"{row['graph']}: directed={row['total_directed_edges']}, original={row['original_edges']}, added={row['added_edges']}",
                f"  degree mean/min/max={row['mean_degree']:.4f}/{row['min_degree']}/{row['max_degree']}; isolated={row['isolated_nodes']}",
                f"  reciprocal={row['reciprocal_edge_percentage']:.4f}%; components={row['connected_component_count']}; largest={row['largest_component_size']}",
                f"  retained original similarity mean={format_value(row['mean_original_similarity_of_retained_original_edges'])}",
                f"  added KG mean={format_value(row['added_mean_kg_similarity'])}; added TRAIN-task mean={format_value(row['added_mean_task_similarity'])}; added overlap mean={format_value(row['added_mean_task_overlap_count'])}",
                f"  graph task mean={format_value(row['mean_task_similarity'])}; median={format_value(row['median_task_similarity'])}; top-neighborhood={format_value(row['top_neighborhood_task_similarity'])}",
                f"  random baseline={format_value(row['random_task_baseline'])}; enrichment={format_value(row['task_enrichment'])}; delta-vs-ORIGINAL={format_value(row['delta_mean_task_vs_original'])}",
            ]
        )
    lines.extend(
        [
            "",
            "KG contribution diagnostics",
            "---------------------------",
            *[f"{key}: {format_value(value)}" for key, value in comparison.items()],
            "",
            "Coverage",
            "--------",
            *[f"{key}: {format_value(value)}" for key, value in coverage.items()],
            "",
            "Concise comparison",
            "------------------",
            "Graph | Original edges | Added edges | Mean task similarity | Task enrichment",
        ]
    )
    lines.extend(
        f"{row['graph']} | {row['original_edges']} | {row['added_edges']} | "
        f"{format_value(row['mean_task_similarity'])} | {format_value(row['task_enrichment'])}"
        for row in graph_rows
    )
    lines.extend(
        [
            "",
            f"KG_TASK source-drug coverage: {coverage['kg_task_source_drug_coverage_percentage']:.4f}%",
            f"KG_TASK added-edge count: {coverage['kg_task_added_edge_count']}",
            f"TASK_ONLY vs KG_TASK added-edge Jaccard: {comparison['jaccard_overlap']:.8f}",
            "",
            "Improved TRAIN task alignment for TASK_ONLY/KG_TASK is expected because task similarity participates in selection.",
            "These diagnostics do not establish predictive improvement. Downstream validation is required in Step 31C.",
            "No downstream winner is selected in Step 31B.",
            "",
            f"TASK-AWARE LEAKAGE CHECK: {'PASS' if diagnostics['checks']['task_aware_leakage'] else 'FAIL'}",
            f"SOURCE SIMILARITY PRESERVATION CHECK: {'PASS' if diagnostics['checks']['source_similarity_preservation'] else 'FAIL'}",
            f"KG ARTIFACT SAFETY CHECK: {'PASS' if diagnostics['checks']['kg_artifact_safety'] else 'FAIL'}",
            f"ORIGINAL MODEL SAFETY CHECK: {'PASS' if diagnostics['checks']['original_model_safety'] else 'FAIL'}",
            "Training/testing performed: NO",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    require_files(
        (AUDIT_PATH, SPLIT_PATH, FREQUENCY_PATH, KG_PATH, MSSF_PATH, MODEL_PATH, CLEAN_MODEL_PATH)
    )
    audit_row, source_path = read_audit_source()
    protected_before = {
        "source_similarity": sha256(source_path),
        "drug_side": sha256(FREQUENCY_PATH),
        "fixed_split": sha256(SPLIT_PATH),
        "kg_artifact": sha256(KG_PATH),
        "mssf": sha256(MSSF_PATH),
        "model": sha256(MODEL_PATH),
        "mssf_clean": sha256(CLEAN_MODEL_PATH),
    }
    original = load_source_matrix(source_path)
    observed, centered, _train_labels, leakage = load_train_only_profiles()
    task_similarity, task_overlap, overlap_diagnostics = build_masked_task_similarity(
        observed, centered
    )
    _kg_embeddings, kg_available, kg_similarity = load_kg()
    kg_neighbors = top_neighbors(kg_similarity, KG_CANDIDATE_K, kg_available)

    original_rows, original_neighbor_sets = build_original_edges(
        original, task_similarity, kg_similarity, task_overlap
    )
    kg_added = build_kg_only_added(
        original_neighbor_sets, kg_neighbors, kg_available,
        original, task_similarity, kg_similarity, task_overlap,
    )
    task_added = build_task_only_added(
        original_neighbor_sets, original, task_similarity, kg_similarity, task_overlap
    )
    kg_task_added, source_task_baselines = build_kg_task_added(
        original_neighbor_sets, kg_neighbors, kg_available,
        original, task_similarity, kg_similarity, task_overlap,
    )
    graphs = {
        "ORIGINAL": original_rows,
        "KG_ONLY": combine_edges(original_rows, kg_added),
        "TASK_ONLY": combine_edges(original_rows, task_added),
        "KG_TASK": combine_edges(original_rows, kg_task_added),
    }

    random_baseline, random_pair_count = deterministic_random_task_baseline(task_similarity)
    original_stats = graph_statistics("ORIGINAL", graphs["ORIGINAL"], random_baseline)
    graph_rows = [original_stats]
    for name in ("KG_ONLY", "TASK_ONLY", "KG_TASK"):
        graph_rows.append(
            graph_statistics(
                name,
                graphs[name],
                random_baseline,
                original_mean_task=original_stats["mean_task_similarity"],
            )
        )
    contribution = set_diagnostics(task_added, kg_task_added)
    usable_kg_sources = int(kg_available.sum())
    coverage = {
        "source_drugs_with_usable_kg": usable_kg_sources,
        "source_drugs_without_kg": int((~kg_available).sum()),
        "source_drugs_receiving_at_least_one_kg_only_edge": source_count(kg_added),
        "source_drugs_receiving_at_least_one_kg_task_edge": source_count(kg_task_added),
        "kg_only_mean_added_edges_per_eligible_drug": len(kg_added) / usable_kg_sources,
        "kg_task_mean_added_edges_per_eligible_drug": len(kg_task_added) / usable_kg_sources,
        "kg_task_source_drug_coverage_percentage": 100.0 * source_count(kg_task_added) / DRUG_COUNT,
        "kg_task_added_edge_count": len(kg_task_added),
    }

    protected_after = {
        "source_similarity": sha256(source_path),
        "drug_side": sha256(FREQUENCY_PATH),
        "fixed_split": sha256(SPLIT_PATH),
        "kg_artifact": sha256(KG_PATH),
        "mssf": sha256(MSSF_PATH),
        "model": sha256(MODEL_PATH),
        "mssf_clean": sha256(CLEAN_MODEL_PATH),
    }
    checks = {
        "task_aware_leakage": bool(
            leakage["validation_positions_hidden"] and leakage["test_positions_hidden"]
        ),
        "source_similarity_preservation": (
            protected_before["source_similarity"] == protected_after["source_similarity"]
        ),
        "kg_artifact_safety": protected_before["kg_artifact"] == protected_after["kg_artifact"],
        "original_model_safety": all(
            protected_before[key] == protected_after[key]
            for key in ("mssf", "model", "mssf_clean")
        ),
        "drug_side_safety": protected_before["drug_side"] == protected_after["drug_side"],
        "fixed_split_safety": protected_before["fixed_split"] == protected_after["fixed_split"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"A Step 31B safety check failed: {checks}")

    baseline_values = [value for value in source_task_baselines.values() if value is not None]
    diagnostics = {
        "parameters": {
            "k_original": K_ORIGINAL,
            "kg_candidate_k": KG_CANDIDATE_K,
            "add_k": ADD_K,
            "min_task_overlap": MIN_TASK_OVERLAP,
            "seed": SEED,
            "kg_weight": KG_WEIGHT,
            "task_weight": TASK_WEIGHT,
        },
        "source": {
            "audit_source_file": audit_row["source_file"],
            "resolved_source_path": str(source_path),
            "shape": list(original.shape),
            "sha256_before": protected_before["source_similarity"],
            "sha256_after": protected_after["source_similarity"],
        },
        "train_only_profiles": leakage,
        "task_overlap": overlap_diagnostics,
        "random_baseline_pair_count": random_pair_count,
        "kg_task_source_baseline": {
            "sources_with_available_original_top10_task_baseline": len(baseline_values),
            "sources_without_available_original_top10_task_baseline": DRUG_COUNT - len(baseline_values),
            "mean_available_baseline": finite_mean(baseline_values),
        },
        "graphs": graph_rows,
        "task_only_vs_kg_task": contribution,
        "coverage": coverage,
        "protected_hashes_before": protected_before,
        "protected_hashes_after": protected_after,
        "checks": checks,
        "interpretation_guard": (
            "TRAIN-task alignment is descriptive and selection-inflated for TASK_ONLY/KG_TASK; "
            "downstream validation is required in Step 31C."
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(ORIGINAL_OUTPUT, original_rows, ORIGINAL_COLUMNS)
    write_csv(KG_ONLY_OUTPUT, graphs["KG_ONLY"], DETAILED_EDGE_COLUMNS)
    write_csv(TASK_ONLY_OUTPUT, graphs["TASK_ONLY"], DETAILED_EDGE_COLUMNS)
    write_csv(KG_TASK_OUTPUT, graphs["KG_TASK"], DETAILED_EDGE_COLUMNS)
    write_csv(ADDED_KG_OUTPUT, kg_added, DETAILED_EDGE_COLUMNS)
    write_csv(ADDED_TASK_OUTPUT, task_added, DETAILED_EDGE_COLUMNS)
    write_csv(ADDED_KG_TASK_OUTPUT, kg_task_added, DETAILED_EDGE_COLUMNS)
    comparison_columns = [
        "graph", "original_edges", "added_edges", "total_directed_edges", "mean_degree",
        "min_degree", "max_degree", "isolated_nodes", "reciprocal_edge_percentage",
        "connected_component_count", "largest_component_size",
        "mean_original_similarity_of_retained_original_edges", "added_mean_kg_similarity",
        "added_mean_task_similarity", "added_mean_task_overlap_count",
        "added_evaluable_task_edge_count", "mean_task_similarity", "median_task_similarity",
        "top_neighborhood_task_similarity", "random_task_baseline", "task_enrichment",
        "delta_mean_task_vs_original",
    ]
    write_csv(COMPARISON_OUTPUT, graph_rows, comparison_columns)
    write_json(DIAGNOSTICS_OUTPUT, diagnostics)
    report = build_report(diagnostics)
    temporary_report = REPORT_OUTPUT.with_suffix(REPORT_OUTPUT.suffix + ".tmp")
    temporary_report.write_text(report, encoding="utf-8")
    temporary_report.replace(REPORT_OUTPUT)

    final_hashes = {
        "source_similarity": sha256(source_path),
        "drug_side": sha256(FREQUENCY_PATH),
        "fixed_split": sha256(SPLIT_PATH),
        "kg_artifact": sha256(KG_PATH),
        "mssf": sha256(MSSF_PATH),
        "model": sha256(MODEL_PATH),
        "mssf_clean": sha256(CLEAN_MODEL_PATH),
    }
    if final_hashes != protected_before:
        raise RuntimeError("A protected input changed while Step 31B outputs were written")

    print("Graph | Original edges | Added edges | Mean task similarity | Task enrichment")
    for row in graph_rows:
        print(
            f"{row['graph']} | {row['original_edges']} | {row['added_edges']} | "
            f"{format_value(row['mean_task_similarity'])} | {format_value(row['task_enrichment'])}"
        )
    print(f"KG_TASK source-drug coverage: {coverage['kg_task_source_drug_coverage_percentage']:.4f}%")
    print(f"KG_TASK added-edge count: {len(kg_task_added)}")
    print(f"TASK_ONLY vs KG_TASK added-edge Jaccard: {contribution['jaccard_overlap']:.8f}")
    print(f"TASK-AWARE LEAKAGE CHECK: {'PASS' if checks['task_aware_leakage'] else 'FAIL'}")
    print(f"SOURCE SIMILARITY PRESERVATION CHECK: {'PASS' if checks['source_similarity_preservation'] else 'FAIL'}")
    print(f"KG ARTIFACT SAFETY CHECK: {'PASS' if checks['kg_artifact_safety'] else 'FAIL'}")
    print(f"ORIGINAL MODEL SAFETY CHECK: {'PASS' if checks['original_model_safety'] else 'FAIL'}")
    print("Training/testing performed: NO")


if __name__ == "__main__":
    main()
