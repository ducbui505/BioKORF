"""Audit MSSF similarity views for leakage-safe task/KG-guided rewiring.

This script performs data inspection and statistics only. It never trains or
tests a model and never writes a similarity matrix. Run it explicitly with:

    python scripts/31a_audit_mssf_similarity_graphs.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Datas"
SPLIT_PATH = PROJECT_ROOT / "data_processed" / "experiments" / "pilot_fold1_split.npz"
KG_PATH = PROJECT_ROOT / "data_processed" / "kg_features" / "biokorf_kg_embeddings.pt"
DRUG_ANCHOR_PATH = (
    PROJECT_ROOT / "data_processed" / "optimuskg" / "final_drug_anchor_mapping.csv"
)
SIDE_ANCHOR_PATH = (
    PROJECT_ROOT / "data_processed" / "optimuskg" / "final_side_effect_anchor_mapping.csv"
)
KG_NODES_PATH = PROJECT_ROOT / "data_processed" / "biomedical_kg" / "nodes.csv"
KG_EDGES_PATH = PROJECT_ROOT / "data_processed" / "biomedical_kg" / "edges.csv"
MSSF_PATH = PROJECT_ROOT / "mssf.py"
MODEL_PATH = PROJECT_ROOT / "model.py"
CLEAN_PIPELINE_PATH = PROJECT_ROOT / "scripts" / "26a_pilot_compare_clean_vs_kg.py"
README_PATH = PROJECT_ROOT / "README.md"
OUTPUT_DIR = PROJECT_ROOT / "data_processed" / "rewiring"
AUDIT_PATH = OUTPUT_DIR / "mssf_similarity_audit.csv"
REPORT_PATH = OUTPUT_DIR / "mssf_similarity_audit_report.txt"
RANKING_PATH = OUTPUT_DIR / "rewiring_candidate_ranking.csv"

DRUG_COUNT = 757
SIDE_COUNT = 994
TOP_K_VALUES = (5, 10, 20)
RANDOM_SEED = 42
EPSILON = 1e-12

PROTECTED_PATHS = (
    MSSF_PATH,
    MODEL_PATH,
    PROJECT_ROOT / "models" / "mssf_clean.py",
    *tuple(sorted(DATA_DIR.glob("*"))),
    KG_PATH,
    DRUG_ANCHOR_PATH,
    SIDE_ANCHOR_PATH,
    KG_NODES_PATH,
    KG_EDGES_PATH,
    SPLIT_PATH,
)


@dataclass(frozen=True)
class ViewSpec:
    entity_type: str
    view_index: int
    view_name: str
    code_variable: str
    source_file: str
    transform: str
    provenance_class: str
    semantic_evidence: str
    code_evidence: str
    file_evidence: str
    modification_meaning: str


# Order is transcribed from read_raw_data() append calls and checked against
# source text at runtime. Names use original code comments/variables, not guesses.
VIEW_SPECS = (
    ViewSpec("DRUG", 1, "SMDsimilarity", "drug_Tfeature_one", "Datas/Text_similarity_one.pkl", "DIRECT", "EXTERNAL_STATIC", "mssf.py comment: Load SMDsimilarity; README: STITCH chemical_chemical.links detailed channel.", "mssf.py: drug_features.append(drug_Tfeature_one)", "README.md identifies Text_similarity_one.pkl as a STITCH drug similarity matrix.", "Changing source similarity scores would alter STITCH evidence semantics; preserve values and build a separate task adjacency."),
    ViewSpec("DRUG", 2, "SMDexperimental", "drug_Tfeature_two", "Datas/Text_similarity_two.pkl", "DIRECT", "EXTERNAL_STATIC", "mssf.py comment: Load SMDexperimental; README: STITCH chemical_chemical.links detailed channel.", "mssf.py: drug_features.append(drug_Tfeature_two)", "README.md identifies Text_similarity_two.pkl as a STITCH drug similarity matrix.", "Changing source scores would alter experimental-evidence semantics; preserve values and build a separate adjacency."),
    ViewSpec("DRUG", 3, "SMDdatabase", "drug_Tfeature_three", "Datas/Text_similarity_three.pkl", "DIRECT", "EXTERNAL_STATIC", "mssf.py comment: Load SMDdatabase; README: STITCH chemical_chemical.links detailed channel.", "mssf.py: drug_features.append(drug_Tfeature_three)", "README.md identifies Text_similarity_three.pkl as a STITCH drug similarity matrix.", "Changing source scores would alter database-evidence semantics; preserve values and build a separate adjacency."),
    ViewSpec("DRUG", 4, "SMDtext", "drug_Tfeature_four", "Datas/Text_similarity_four.pkl", "DIRECT", "EXTERNAL_STATIC", "mssf.py comment: Load SMDtext; README: STITCH chemical_chemical.links detailed channel.", "mssf.py: drug_features.append(drug_Tfeature_four)", "README.md identifies Text_similarity_four.pkl as a STITCH drug similarity matrix.", "Changing source scores would alter text-mining evidence semantics; preserve values and build a separate adjacency."),
    ViewSpec("DRUG", 5, "SMDcombined", "drug_Tfeature_five", "Datas/Text_similarity_five.pkl", "DIRECT", "EXTERNAL_STATIC", "mssf.py comment: Load SMDcombined; README: STITCH chemical_chemical.links detailed channel.", "mssf.py: drug_features.append(drug_Tfeature_five)", "README.md identifies Text_similarity_five.pkl as a STITCH drug similarity matrix.", "Changing combined source scores would alter their evidence meaning; preserve values and build a separate adjacency."),
    ViewSpec("DRUG", 6, "Drug_word_sim", "Drug_word_sim", "Datas/drug_mol.pkl", "ROW_COSINE", "EXTERNAL_STATIC", "README: Mol2vec molecular-substructure embeddings.", "mssf.py: Drug_word_sim = cosine_similarity(Drug_word2vec)", "README.md describes drug_mol.pkl as externally learned Mol2vec vectors.", "This is molecular-representation similarity; preserve the similarity feature and derive any task graph separately."),
    ViewSpec("DRUG", 7, "drug_target_sim", "drug_target_sim", "Datas/drug_target.pkl", "ROW_COSINE", "EXTERNAL_STATIC", "README: target-protein information obtained from DrugBank.", "mssf.py: drug_target_sim = cosine_similarity(drug_target)", "README.md identifies drug_target.pkl as DrugBank target information.", "Edges mean shared target profiles; added KG-supported edges must remain distinguishable from original target similarity."),
    ViewSpec("DRUG", 8, "drug_f_sim", "drug_f_sim", "Datas/fingerprint_similarity.pkl", "DIRECT", "EXTERNAL_STATIC", "README: chemical-structure Jaccard similarity.", "mssf.py: drug_f_sim = pickle.load(...fingerprint_similarity.pkl)", "README.md identifies fingerprint_similarity.pkl as structure similarity from Jaccard scores.", "Do not rewrite chemical similarity values; only build a separate task-aware adjacency or prune an adjacency view."),
    ViewSpec("DRUG", 9, "SMD_DIPF", "drug_side_sim", "Datas/drug_side.pkl", "TRAIN_FREQUENCY_ROW_COSINE", "LABEL_DERIVED", "mssf.py: cosine similarity over remaining frequency matrix, SMD_DIPF.", "mssf.py: drug_side_sim = cosine_similarity(drug_side)", "drug_side.pkl contains known drug-side frequency labels; Fold-1 version must be reconstructed from train rows only.", "This already encodes training outcomes; KG/task rewiring risks circular target use and must not modify the feature."),
    ViewSpec("DRUG", 10, "SMD_DIPA", "drug_side_label_sim", "Datas/drug_side.pkl", "TRAIN_BINARY_ROW_COSINE", "LABEL_DERIVED", "mssf.py: cosine similarity over binary association matrix, SMD_DIPA.", "mssf.py: drug_side_label_sim = cosine_similarity(drug_side_label)", "drug_side.pkl supplies known associations; Fold-1 version must be reconstructed from train rows only.", "This already encodes training associations; preserve the safely recomputed feature and do not rewire it."),
    ViewSpec("DRUG", 11, "drug_p_e_sim", "drug_p_e_sim", "Datas/drug_pathway_enzyme_similarity.pkl", "DIRECT", "EXTERNAL_STATIC", "README: DrugBank drug-pathway-enzyme similarity.", "mssf.py: drug_features.append(drug_p_e_sim)", "README.md identifies the matrix as constructed from DrugBank pathway/enzyme data.", "Edges represent pathway/enzyme similarity; keep source values separate from any task/KG adjacency."),
    ViewSpec("SIDE_EFFECT", 1, "effect_side_semantic", "effect_side_semantic", "Datas/side_effect_semantic.pkl", "DIRECT", "EXTERNAL_STATIC", "README: ADR classification-system descriptor semantic similarity.", "mssf.py: side_features.append(effect_side_semantic)", "README.md documents descriptor-based side-effect semantic similarity.", "Changing values changes ontology/descriptor semantics; preserve them and build a separate adjacency."),
    ViewSpec("SIDE_EFFECT", 2, "side_glove_sim", "side_glove_sim", "Datas/glove_wordEmbedding.pkl", "ROW_COSINE", "EXTERNAL_STATIC", "README: Wikipedia-trained GloVe side-effect word embeddings.", "mssf.py: side_glove_sim = cosine_similarity(glove_word)", "README.md describes glove_wordEmbedding.pkl as external GloVe vectors.", "Edges represent lexical distributional similarity; added biomedical edges must remain a separate adjacency."),
    ViewSpec("SIDE_EFFECT", 3, "SME_DIPF", "side_drug_sim", "Datas/drug_side.pkl", "TRAIN_FREQUENCY_COLUMN_COSINE", "LABEL_DERIVED", "mssf.py: side effect frequency-profile similarity, SME_DIPF.", "mssf.py: side_drug_sim = cosine_similarity(drug_side.T)", "drug_side.pkl contains frequency labels; Fold-1 version must be reconstructed from train rows only.", "This already encodes training outcomes and must not be rewired with validation/test or circular task evidence."),
    ViewSpec("SIDE_EFFECT", 4, "SME_DIPA", "side_drug_label_sim", "Datas/drug_side.pkl", "TRAIN_BINARY_COLUMN_COSINE", "LABEL_DERIVED", "mssf.py: side effect binary-association similarity, SME_DIPA.", "mssf.py: side_drug_label_sim = cosine_similarity(drug_side_label.T)", "drug_side.pkl contains known associations; Fold-1 version must be reconstructed from train rows only.", "This already encodes training associations; preserve the safely recomputed feature and do not rewire it."),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protected_hashes() -> dict[Path, str]:
    missing = [path for path in PROTECTED_PATHS if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required protected inputs are missing: {missing}")
    return {path: sha256(path) for path in PROTECTED_PATHS}


def verify_protected(before: dict[Path, str]) -> None:
    after = {path: sha256(path) for path in before}
    if before != after:
        changed = [str(path) for path in before if before[path] != after[path]]
        raise RuntimeError(f"Protected inputs changed during audit: {changed}")


def load_pickle(relative_path: str) -> np.ndarray:
    path = PROJECT_ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Required MSSF input not found: {path}")
    with path.open("rb") as handle:
        return np.asarray(pickle.load(handle))


def cosine_similarity(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Cosine input must be two-dimensional, found {values.shape}")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)
    return normalized @ normalized.T


def verify_code_trace() -> None:
    mssf_source = MSSF_PATH.read_text(encoding="utf-8")
    model_source = MODEL_PATH.read_text(encoding="utf-8")
    clean_source = CLEAN_PIPELINE_PATH.read_text(encoding="utf-8")
    drug_append_order = [
        "drug_features.append(drug_Tfeature_one)",
        "drug_features.append(drug_Tfeature_two)",
        "drug_features.append(drug_Tfeature_three)",
        "drug_features.append(drug_Tfeature_four)",
        "drug_features.append(drug_Tfeature_five)",
        "drug_features.append(Drug_word_sim)",
        "drug_features.append(drug_target_sim)",
        "drug_features.append(drug_f_sim)",
        "drug_features.append(drug_side_sim)",
        "drug_features.append(drug_side_label_sim)",
        "drug_features.append(drug_p_e_sim)",
    ]
    side_append_order = [
        "side_features.append(effect_side_semantic)",
        "side_features.append(side_glove_sim)",
        "side_features.append(side_drug_sim)",
        "side_features.append(side_drug_label_sim)",
    ]
    for ordered_snippets, label in (
        (drug_append_order, "drug"), (side_append_order, "side-effect")
    ):
        positions = [mssf_source.find(snippet) for snippet in ordered_snippets]
        if any(position < 0 for position in positions) or positions != sorted(positions):
            raise RuntimeError(f"Could not verify original MSSF {label} view order")
    if "drug_features.chunk(11, 1)" not in model_source:
        raise RuntimeError("model.py does not expose the expected 11 drug chunks")
    if "side_features.chunk(4, 1)" not in model_source:
        raise RuntimeError("model.py does not expose the expected four side-effect chunks")
    clean_requirements = (
        "masked_frequency[hidden_drug, hidden_side] = 0",
        "cosine_similarity(masked_frequency)",
        "cosine_similarity(binary_matrix)",
        "cosine_similarity(masked_frequency.T)",
        "cosine_similarity(binary_matrix.T)",
    )
    if not all(snippet in clean_source for snippet in clean_requirements):
        raise RuntimeError("Could not verify MSSF-clean validation/test masking logic")


def load_train_only_frequency() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not SPLIT_PATH.is_file():
        raise FileNotFoundError(f"Fixed Fold-1 split not found: {SPLIT_PATH}")
    with np.load(SPLIT_PATH) as split:
        required = {"train_samples", "validation_samples", "test_samples", "seed", "fold"}
        missing = required.difference(split.files)
        if missing:
            raise KeyError(f"Fixed split is missing arrays: {sorted(missing)}")
        if int(split["seed"]) != 42 or int(split["fold"]) != 1:
            raise ValueError("Expected the fixed seed-42 Fold-1 split")
        train = np.asarray(split["train_samples"])
        validation_pairs = np.asarray(split["validation_samples"])[:, :2].astype(np.int64)
        test_pairs = np.asarray(split["test_samples"])[:, :2].astype(np.int64)
    if train.ndim != 2 or train.shape[1] != 3:
        raise ValueError("train_samples must contain drug, side-effect, and frequency")
    train_frequency = np.zeros((DRUG_COUNT, SIDE_COUNT), dtype=np.float64)
    drug_indices = train[:, 0].astype(np.int64)
    side_indices = train[:, 1].astype(np.int64)
    train_frequency[drug_indices, side_indices] = train[:, 2].astype(np.float64)
    if np.any(train_frequency[validation_pairs[:, 0], validation_pairs[:, 1]] != 0):
        raise RuntimeError("Validation positions leaked into the train-only frequency matrix")
    if np.any(train_frequency[test_pairs[:, 0], test_pairs[:, 1]] != 0):
        raise RuntimeError("Test positions leaked into the train-only frequency matrix")
    return train_frequency, validation_pairs, test_pairs


def build_view(spec: ViewSpec, train_frequency: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = load_pickle(spec.source_file)
    binary = (train_frequency > 0).astype(np.float64)
    transforms = {
        "DIRECT": lambda: np.asarray(raw, dtype=np.float64),
        "ROW_COSINE": lambda: cosine_similarity(raw),
        "TRAIN_FREQUENCY_ROW_COSINE": lambda: cosine_similarity(train_frequency),
        "TRAIN_BINARY_ROW_COSINE": lambda: cosine_similarity(binary),
        "TRAIN_FREQUENCY_COLUMN_COSINE": lambda: cosine_similarity(train_frequency.T),
        "TRAIN_BINARY_COLUMN_COSINE": lambda: cosine_similarity(binary.T),
    }
    if spec.transform not in transforms:
        raise ValueError(f"Unknown view transform: {spec.transform}")
    final = transforms[spec.transform]()
    expected = DRUG_COUNT if spec.entity_type == "DRUG" else SIDE_COUNT
    if final.shape != (expected, expected):
        raise ValueError(
            f"{spec.entity_type} view {spec.view_index} has final shape {final.shape}; "
            f"expected {(expected, expected)}"
        )
    return raw, final


def shape_text(shape: Iterable[int]) -> str:
    return "x".join(str(int(value)) for value in shape)


def matrix_statistics(raw: np.ndarray, matrix: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(matrix)
    finite_values = matrix[finite]
    if finite_values.size == 0:
        numerical = {name: math.nan for name in ("min", "max", "mean", "std")}
    else:
        numerical = {
            "min": float(finite_values.min()),
            "max": float(finite_values.max()),
            "mean": float(finite_values.mean()),
            "std": float(finite_values.std()),
        }
    nonzero = int(np.count_nonzero(np.nan_to_num(matrix, nan=0.0)))
    diagonal = np.diag(matrix)
    return {
        "raw_shape": shape_text(raw.shape),
        "final_shape": shape_text(matrix.shape),
        "shape": shape_text(matrix.shape),
        "diagonal_exists": bool(np.any(np.isfinite(diagonal) & (diagonal != 0))),
        "diagonal_nonzero_count": int(np.count_nonzero(np.nan_to_num(diagonal, nan=0.0))),
        "symmetric": bool(np.allclose(matrix, matrix.T, atol=1e-8, rtol=1e-7, equal_nan=True)),
        "numerical_min": numerical["min"],
        "numerical_max": numerical["max"],
        "numerical_mean": numerical["mean"],
        "numerical_std": numerical["std"],
        "nonzero_entries": nonzero,
        "density": nonzero / matrix.size,
        "nan_count": int(np.isnan(matrix).sum()),
        "inf_count": int(np.isinf(matrix).sum()),
    }


def deterministic_topk(matrix: np.ndarray, k: int, allowed: np.ndarray | None = None) -> list[np.ndarray]:
    count = matrix.shape[0]
    permitted = np.ones(count, dtype=bool) if allowed is None else np.asarray(allowed, dtype=bool)
    output: list[np.ndarray] = []
    indices = np.arange(count)
    for node in range(count):
        candidates = permitted.copy()
        candidates[node] = False
        candidate_indices = indices[candidates]
        values = np.nan_to_num(matrix[node, candidate_indices], nan=-np.inf)
        order = np.lexsort((candidate_indices, -values))
        output.append(candidate_indices[order[: min(k, len(order))]])
    return output


def connected_components(adjacency: list[set[int]]) -> tuple[int, int]:
    unseen = set(range(len(adjacency)))
    component_sizes: list[int] = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack = [start]
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            neighbors = adjacency[node].intersection(unseen)
            unseen.difference_update(neighbors)
            stack.extend(sorted(neighbors, reverse=True))
        component_sizes.append(size)
    return len(component_sizes), max(component_sizes, default=0)


def graph_diagnostics(matrix: np.ndarray, k: int) -> dict[str, Any]:
    neighbors = deterministic_topk(matrix, k)
    directed = {(node, int(other)) for node, row in enumerate(neighbors) for other in row}
    adjacency = [set() for _ in range(matrix.shape[0])]
    undirected_edges: set[tuple[int, int]] = set()
    retained: list[float] = []
    for node, other in directed:
        edge = (min(node, other), max(node, other))
        undirected_edges.add(edge)
        adjacency[node].add(other)
        adjacency[other].add(node)
        retained.append(float(matrix[node, other]))
    degrees = np.asarray([len(row) for row in adjacency], dtype=np.int64)
    components, largest = connected_components(adjacency)
    reciprocal = sum((other, node) in directed for node, other in directed)
    prefix = f"top{k}_"
    return {
        prefix + "mean_degree": float(degrees.mean()),
        prefix + "min_degree": int(degrees.min()),
        prefix + "max_degree": int(degrees.max()),
        prefix + "isolated_node_count": int(np.count_nonzero(degrees == 0)),
        prefix + "connected_component_count": int(components),
        prefix + "largest_component_size": int(largest),
        prefix + "edge_count": len(undirected_edges),
        prefix + "reciprocal_edge_percentage": (
            100.0 * reciprocal / len(directed) if directed else 0.0
        ),
        prefix + "average_retained_similarity": (
            float(np.mean(retained)) if retained else math.nan
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
    if valid.sum() < 2:
        return math.nan
    return pearson(average_ranks(left[valid]), average_ranks(right[valid]))


def upper_triangle_pairs(matrix: np.ndarray, available: np.ndarray) -> np.ndarray:
    row, column = np.triu_indices(matrix.shape[0], k=1)
    keep = available[row] & available[column]
    return matrix[row[keep], column[keep]]


def task_alignment(
    matrix: np.ndarray,
    task_similarity: np.ndarray,
    observation_counts: np.ndarray,
    random_seed: int,
) -> dict[str, Any]:
    sufficient = observation_counts > 0
    row, column = np.triu_indices(matrix.shape[0], k=1)
    eligible = sufficient[row] & sufficient[column]
    original_pairs = matrix[row[eligible], column[eligible]]
    task_pairs = task_similarity[row[eligible], column[eligible]]
    correlation = pearson(original_pairs, task_pairs)

    neighbor_means: dict[int, float] = {}
    for k in (5, 10):
        neighbors = deterministic_topk(matrix, k, sufficient)
        values = [
            task_similarity[node, other]
            for node, selected in enumerate(neighbors)
            if sufficient[node]
            for other in selected
        ]
        neighbor_means[k] = float(np.mean(values)) if values else math.nan

    rng = np.random.default_rng(random_seed)
    eligible_indices = np.flatnonzero(eligible)
    sample_size = min(10000, len(eligible_indices))
    chosen = (
        rng.choice(eligible_indices, size=sample_size, replace=False)
        if sample_size else np.asarray([], dtype=np.int64)
    )
    random_mean = float(task_similarity[row[chosen], column[chosen]].mean()) if sample_size else math.nan
    return {
        "task_sufficient_entity_count": int(sufficient.sum()),
        "task_alignment_pair_count": int(eligible.sum()),
        "task_alignment_correlation": correlation,
        "top5_task_alignment": neighbor_means[5],
        "top10_task_alignment": neighbor_means[10],
        "random_task_alignment": random_mean,
        "task_alignment_enrichment": neighbor_means[10] - random_mean,
    }


def load_kg_artifact() -> dict[str, np.ndarray]:
    required_paths = (KG_PATH, DRUG_ANCHOR_PATH, SIDE_ANCHOR_PATH, KG_NODES_PATH, KG_EDGES_PATH)
    missing = [path for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required local KG artifacts are missing: {missing}")
    artifact = torch.load(KG_PATH, map_location="cpu", weights_only=False)
    required_keys = {
        "drug_embeddings", "side_embeddings", "drug_available_mask", "side_available_mask"
    }
    missing_keys = required_keys.difference(artifact)
    if missing_keys:
        raise KeyError(f"KG embedding artifact is missing keys: {sorted(missing_keys)}")
    output = {
        key: artifact[key].detach().cpu().numpy()
        for key in required_keys
    }
    if output["drug_embeddings"].shape[0] != DRUG_COUNT:
        raise ValueError("KG drug embedding order/shape does not match BioKORF")
    if output["side_embeddings"].shape[0] != SIDE_COUNT:
        raise ValueError("KG side-effect embedding order/shape does not match BioKORF")
    return output


def classify_kg_complementarity(
    coverage: float, pearson_value: float, spearman_value: float,
    jaccard5: float, jaccard10: float,
) -> str:
    if coverage < 0.10 or not np.isfinite([pearson_value, spearman_value, jaccard5, jaccard10]).all():
        return "INSUFFICIENT_COVERAGE"
    if spearman_value >= 0.70 and jaccard10 >= 0.50:
        return "HIGHLY_REDUNDANT_WITH_KG"
    if spearman_value >= 0.30 and jaccard10 >= 0.20:
        return "MODERATELY_RELATED"
    if abs(pearson_value) < 0.05 and abs(spearman_value) < 0.05 and jaccard10 < 0.03:
        return "EFFECTIVELY_UNRELATED"
    return "COMPLEMENTARY"


def kg_diagnostics(
    matrix: np.ndarray, embeddings: np.ndarray, available: np.ndarray
) -> dict[str, Any]:
    available = np.asarray(available, dtype=bool)
    kg_similarity = cosine_similarity(embeddings)
    original_pairs = upper_triangle_pairs(matrix, available)
    kg_pairs = upper_triangle_pairs(kg_similarity, available)
    pearson_value = pearson(original_pairs, kg_pairs)
    spearman_value = spearman(original_pairs, kg_pairs)
    overlaps: dict[int, float] = {}
    for k in (5, 10):
        original_neighbors = deterministic_topk(matrix, k, available)
        kg_neighbors = deterministic_topk(kg_similarity, k, available)
        jaccards = []
        for node in np.flatnonzero(available):
            original_set = set(map(int, original_neighbors[node]))
            kg_set = set(map(int, kg_neighbors[node]))
            union = original_set | kg_set
            jaccards.append(len(original_set & kg_set) / len(union) if union else 0.0)
        overlaps[k] = float(np.mean(jaccards)) if jaccards else math.nan
    coverage = float(available.mean())
    return {
        "kg_coverage": coverage,
        "kg_available_entity_count": int(available.sum()),
        "kg_pair_count": int(len(original_pairs)),
        "kg_pearson": pearson_value,
        "kg_spearman": spearman_value,
        "kg_top5_jaccard": overlaps[5],
        "kg_top10_jaccard": overlaps[10],
        "kg_complementarity": classify_kg_complementarity(
            coverage, pearson_value, spearman_value, overlaps[5], overlaps[10]
        ),
    }


def rewiring_decision(spec: ViewSpec, row: dict[str, Any]) -> tuple[str, str, str]:
    if spec.provenance_class in {"LABEL_DERIVED", "HYBRID"}:
        return (
            "DO_NOT_REWIRE",
            "DO_NOT_MODIFY",
            "The view directly encodes Fold-1 training labels; further task-guided rewiring would be circular and must never use validation/test labels.",
        )
    if row["leakage_status"] != "SAFE_EXTERNAL":
        return "DO_NOT_REWIRE", "DO_NOT_MODIFY", "Leakage safety is not established."
    task_correlation = row["task_alignment_correlation"]
    enrichment = row["task_alignment_enrichment"]
    complementarity = row["kg_complementarity"]
    if (
        np.isfinite(task_correlation)
        and np.isfinite(enrichment)
        and abs(task_correlation) < 0.35
        and enrichment < 0.15
        and complementarity == "COMPLEMENTARY"
    ):
        return (
            "HIGH_PRIORITY_REWIRING_CANDIDATE",
            "ADD_EDGES_WITH_KG_SUPPORT",
            "External and leakage-safe, with task-alignment headroom and complementary KG evidence; preserve source values and alter only a derived adjacency.",
        )
    if complementarity in {"COMPLEMENTARY", "MODERATELY_RELATED"}:
        return (
            "MEDIUM_PRIORITY",
            "KEEP_MATRIX_BUILD_TASK_ADJACENCY",
            "Safe external view with potentially useful KG/task information, but evidence is weaker or partially redundant.",
        )
    if complementarity == "HIGHLY_REDUNDANT_WITH_KG" or (
        np.isfinite(task_correlation) and abs(task_correlation) >= 0.50
    ):
        return (
            "LOW_PRIORITY",
            "KEEP_MATRIX_BUILD_TASK_ADJACENCY",
            "The view is already strongly task-aligned or strongly redundant with KG, leaving limited rewiring headroom.",
        )
    return (
        "NEEDS_MORE_EVIDENCE",
        "NEEDS_REVIEW",
        "Available task/KG diagnostics do not provide consistent evidence for safe rewiring.",
    )


def leakage_fields(spec: ViewSpec) -> dict[str, Any]:
    if spec.provenance_class == "EXTERNAL_STATIC":
        return {
            "leakage_status": "SAFE_EXTERNAL",
            "test_edges_hidden": "NOT_APPLICABLE",
            "validation_edges_hidden": "NOT_APPLICABLE",
            "recomputed_from_training_only": False,
            "original_mssf_uses_full_label_information": False,
            "mssf_clean_corrected_behavior": "NOT_APPLICABLE",
            "external_or_label_derived": "EXTERNAL",
            "leakage_explanation": "No drug-side frequency or association labels enter this view.",
        }
    return {
        "leakage_status": "SAFE_TRAIN_ONLY",
        "test_edges_hidden": True,
        "validation_edges_hidden": True,
        "recomputed_from_training_only": True,
        "original_mssf_uses_full_label_information": True,
        "mssf_clean_corrected_behavior": True,
        "external_or_label_derived": "LABEL_DERIVED",
        "leakage_explanation": (
            "The audit reconstructs this view from train_samples only. Original mssf.py masks "
            "only data_test, so a separate model-selection validation subset would remain; "
            "MSSF-clean masks validation and test together."
        ),
    }


def csv_safe(value: Any) -> Any:
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            {column: csv_safe(row.get(column)) for column in columns} for row in rows
        )
    temporary.replace(path)


def ranking_score(row: dict[str, Any]) -> float:
    priority_score = {
        "HIGH_PRIORITY_REWIRING_CANDIDATE": 4.0,
        "MEDIUM_PRIORITY": 3.0,
        "LOW_PRIORITY": 2.0,
        "NEEDS_MORE_EVIDENCE": 1.0,
    }.get(row["rewiring_priority"], 0.0)
    task = row["task_alignment_correlation"]
    enrichment = row["task_alignment_enrichment"]
    jaccard = row["kg_top10_jaccard"]
    headroom = 1.0 - min(1.0, abs(task)) if np.isfinite(task) else 0.0
    enrichment_headroom = 1.0 - min(1.0, max(0.0, enrichment)) if np.isfinite(enrichment) else 0.0
    kg_nonredundancy = 1.0 - min(1.0, jaccard) if np.isfinite(jaccard) else 0.0
    drug_preference = 0.05 if row["entity_type"] == "DRUG" else 0.0
    return priority_score + 0.25 * headroom + 0.15 * enrichment_headroom + 0.10 * kg_nonredundancy + drug_preference


def build_ranking(audit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        row for row in audit_rows
        if row["provenance_class"] == "EXTERNAL_STATIC"
        and row["leakage_status"] == "SAFE_EXTERNAL"
        and row["rewiring_priority"] != "DO_NOT_REWIRE"
    ]
    candidates.sort(
        key=lambda row: (-ranking_score(row), row["entity_type"] != "DRUG", row["view_index"])
    )
    output = []
    for rank, row in enumerate(candidates, start=1):
        output.append(
            {
                "rank": rank,
                "entity_type": row["entity_type"],
                "view_index": row["view_index"],
                "view_name": row["view_name"],
                "rewiring_priority": row["rewiring_priority"],
                "recommended_rewiring_mode": row["recommended_rewiring_mode"],
                "reason_for_rank": (
                    f"score={ranking_score(row):.4f}; task_corr={row['task_alignment_correlation']:.4f}; "
                    f"top10_enrichment={row['task_alignment_enrichment']:.4f}; "
                    f"KG={row['kg_complementarity']}; KG_top10_Jaccard={row['kg_top10_jaccard']:.4f}"
                ),
                "expected_benefit": "Inject complementary KG/task neighborhood evidence while retaining the original similarity feature.",
                "main_risk": row["modification_meaning"],
            }
        )
    return output


def report_view_line(row: dict[str, Any]) -> str:
    return (
        f"{row['entity_type']} {row['view_index']:02d} {row['view_name']} | "
        f"variable={row['code_variable']} | source={row['source_file']} | "
        f"raw={row['raw_shape']} final={row['final_shape']} | "
        f"provenance={row['provenance_class']} leakage={row['leakage_status']} | "
        f"priority={row['rewiring_priority']} mode={row['recommended_rewiring_mode']}"
    )


def format_metric(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return "NA" if not np.isfinite(value) else f"{float(value):.6f}"
    return str(value)


def build_report(
    audit_rows: list[dict[str, Any]], ranking: list[dict[str, Any]], leakage_safe: bool
) -> str:
    lines = [
        "BioKORF Step 31A — MSSF Similarity Graph Audit",
        "================================================",
        "Audit type: code/data/statistics only; no training, testing, or matrix modification.",
        "Task profiles: reconstructed solely from fixed Fold-1 train_samples.",
        "Validation/test arrays are used only for their drug/side index pairs to verify exclusion.",
        "",
        "1. Exact 11 Drug views in original MSSF order",
        "------------------------------------------------",
        *[report_view_line(row) for row in audit_rows if row["entity_type"] == "DRUG"],
        "",
        "2. Exact 4 Side-effect views in original MSSF order",
        "----------------------------------------------------",
        *[report_view_line(row) for row in audit_rows if row["entity_type"] == "SIDE_EFFECT"],
        "",
        "3–4. Provenance and leakage classification",
        "--------------------------------------------",
    ]
    for row in audit_rows:
        lines.extend(
            [
                report_view_line(row),
                f"  Code evidence: {row['code_evidence']}",
                f"  File evidence: {row['file_evidence']}",
                f"  Explanation: {row['leakage_explanation']}",
            ]
        )
    lines.extend(["", "5. Graph statistics", "-------------------"])
    for row in audit_rows:
        lines.append(
            f"{row['entity_type']} {row['view_index']:02d} {row['view_name']}: "
            f"shape={row['shape']} symmetric={row['symmetric']} density={row['density']:.6f} "
            f"range=[{format_metric(row['numerical_min'])}, {format_metric(row['numerical_max'])}] "
            f"mean={format_metric(row['numerical_mean'])} std={format_metric(row['numerical_std'])} "
            f"nonzero={row['nonzero_entries']} NaN={row['nan_count']} inf={row['inf_count']}"
        )
        for k in TOP_K_VALUES:
            prefix = f"top{k}_"
            lines.append(
                f"  k={k}: degree mean/min/max={row[prefix + 'mean_degree']:.3f}/"
                f"{row[prefix + 'min_degree']}/{row[prefix + 'max_degree']}; "
                f"isolated={row[prefix + 'isolated_node_count']}; "
                f"components={row[prefix + 'connected_component_count']}; "
                f"largest={row[prefix + 'largest_component_size']}; "
                f"edges={row[prefix + 'edge_count']}; reciprocal={row[prefix + 'reciprocal_edge_percentage']:.2f}%; "
                f"retained_mean={format_metric(row[prefix + 'average_retained_similarity'])}"
            )
    lines.extend(["", "6. Train-only task-alignment statistics", "---------------------------------------"])
    for row in audit_rows:
        lines.append(
            f"{row['entity_type']} {row['view_index']:02d} {row['view_name']}: "
            f"corr={format_metric(row['task_alignment_correlation'])}; "
            f"top5={format_metric(row['top5_task_alignment'])}; "
            f"top10={format_metric(row['top10_task_alignment'])}; "
            f"random={format_metric(row['random_task_alignment'])}; "
            f"enrichment={format_metric(row['task_alignment_enrichment'])}; "
            f"sufficient_entities={row['task_sufficient_entity_count']}"
        )
    lines.extend(
        [
            "",
            "7. KG redundancy/complementarity statistics",
            "--------------------------------------------",
            "Side-effect coverage is reported explicitly and low-coverage results are descriptive only.",
        ]
    )
    for row in audit_rows:
        lines.append(
            f"{row['entity_type']} {row['view_index']:02d} {row['view_name']}: "
            f"coverage={100 * row['kg_coverage']:.2f}%; Pearson={format_metric(row['kg_pearson'])}; "
            f"Spearman={format_metric(row['kg_spearman'])}; Jaccard@5={format_metric(row['kg_top5_jaccard'])}; "
            f"Jaccard@10={format_metric(row['kg_top10_jaccard'])}; class={row['kg_complementarity']}"
        )
    lines.extend(["", "8. Rewiring suitability", "------------------------"])
    for row in audit_rows:
        lines.extend(
            [
                f"{report_view_line(row)}",
                f"  Rationale: {row['rationale']}",
                f"  Scientific meaning: {row['modification_meaning']}",
            ]
        )
    lines.extend(["", "9. Top three safest candidates", "------------------------------"])
    if ranking:
        for item in ranking[:3]:
            lines.append(
                f"#{item['rank']} {item['entity_type']} {item['view_index']} {item['view_name']} — "
                f"{item['rewiring_priority']} / {item['recommended_rewiring_mode']}: {item['reason_for_rank']}"
            )
    else:
        lines.append("No view has sufficient measured evidence for a safe candidate ranking.")
    lines.extend(["", "10. Views that must NOT be rewired", "-----------------------------------"])
    forbidden = [row for row in audit_rows if row["rewiring_priority"] == "DO_NOT_REWIRE"]
    lines.extend(
        f"- {row['entity_type']} {row['view_index']} {row['view_name']}: {row['rationale']}"
        for row in forbidden
    )
    lines.extend(["", "11. Recommended first Step 31B experiment", "------------------------------------------"])
    clear = next(
        (
            item for item in ranking
            if item["entity_type"] == "DRUG"
            and item["rewiring_priority"] == "HIGH_PRIORITY_REWIRING_CANDIDATE"
        ),
        None,
    )
    if clear is None:
        lines.append("NO SINGLE TARGET RECOMMENDED: measured evidence is not strong enough for Step 31B.")
    else:
        lines.append(
            f"Recommend only DRUG view {clear['view_index']} ({clear['view_name']}) using "
            f"{clear['recommended_rewiring_mode']}. Preserve the original matrix; create one "
            "separate train-only/KG-supported adjacency and compare it against unchanged MSSF-clean."
        )
    lines.extend(
        [
            "",
            f"TASK-AWARE LEAKAGE CHECK: {'PASS' if leakage_safe else 'FAIL'}",
            "PROTECTED INPUT IMMUTABILITY CHECK: PASS",
            "TRAINING/TESTING PERFORMED: NO",
        ]
    )
    return "\n".join(lines) + "\n"


def audit_columns() -> list[str]:
    base = [
        "entity_type", "view_index", "view_name", "code_variable", "source_file",
        "raw_shape", "final_shape", "shape", "diagonal_exists", "diagonal_nonzero_count",
        "symmetric", "numerical_min", "numerical_max", "numerical_mean", "numerical_std",
        "nonzero_entries", "density", "nan_count", "inf_count", "provenance_class",
        "leakage_status", "external_or_label_derived", "test_edges_hidden",
        "validation_edges_hidden", "recomputed_from_training_only",
        "original_mssf_uses_full_label_information", "mssf_clean_corrected_behavior",
        "semantic_evidence", "code_evidence", "file_evidence", "leakage_explanation",
    ]
    graph = [
        f"top{k}_{metric}"
        for k in TOP_K_VALUES
        for metric in (
            "mean_degree", "min_degree", "max_degree", "isolated_node_count",
            "connected_component_count", "largest_component_size", "edge_count",
            "reciprocal_edge_percentage", "average_retained_similarity",
        )
    ]
    tail = [
        "task_sufficient_entity_count", "task_alignment_pair_count",
        "task_alignment_correlation", "top5_task_alignment", "top10_task_alignment",
        "random_task_alignment", "task_alignment_enrichment", "kg_coverage",
        "kg_available_entity_count", "kg_pair_count", "kg_pearson", "kg_spearman",
        "kg_top5_jaccard", "kg_top10_jaccard", "kg_complementarity",
        "rewiring_priority", "recommended_rewiring_mode", "modification_meaning", "rationale",
    ]
    return base + graph + tail


def main() -> None:
    before = protected_hashes()
    verify_code_trace()
    train_frequency, validation_pairs, test_pairs = load_train_only_frequency()
    leakage_safe = bool(
        np.all(train_frequency[validation_pairs[:, 0], validation_pairs[:, 1]] == 0)
        and np.all(train_frequency[test_pairs[:, 0], test_pairs[:, 1]] == 0)
    )
    if not leakage_safe:
        raise RuntimeError("TASK-AWARE LEAKAGE CHECK failed")

    frequency_task_drug = cosine_similarity(train_frequency)
    frequency_task_side = cosine_similarity(train_frequency.T)
    drug_observations = np.count_nonzero(train_frequency, axis=1)
    side_observations = np.count_nonzero(train_frequency, axis=0)
    kg = load_kg_artifact()
    audit_rows: list[dict[str, Any]] = []
    for spec in VIEW_SPECS:
        raw, matrix = build_view(spec, train_frequency)
        task_matrix = frequency_task_drug if spec.entity_type == "DRUG" else frequency_task_side
        observation_counts = drug_observations if spec.entity_type == "DRUG" else side_observations
        embeddings = kg["drug_embeddings"] if spec.entity_type == "DRUG" else kg["side_embeddings"]
        mask = kg["drug_available_mask"] if spec.entity_type == "DRUG" else kg["side_available_mask"]
        row: dict[str, Any] = {
            "entity_type": spec.entity_type,
            "view_index": spec.view_index,
            "view_name": spec.view_name,
            "code_variable": spec.code_variable,
            "source_file": spec.source_file,
            "provenance_class": spec.provenance_class,
            "semantic_evidence": spec.semantic_evidence,
            "code_evidence": spec.code_evidence,
            "file_evidence": spec.file_evidence,
            "modification_meaning": spec.modification_meaning,
            **matrix_statistics(raw, matrix),
            **leakage_fields(spec),
        }
        for k in TOP_K_VALUES:
            row.update(graph_diagnostics(matrix, k))
        row.update(task_alignment(matrix, task_matrix, observation_counts, RANDOM_SEED))
        row.update(kg_diagnostics(matrix, embeddings, mask))
        priority, mode, rationale = rewiring_decision(spec, row)
        row.update(
            {
                "rewiring_priority": priority,
                "recommended_rewiring_mode": mode,
                "rationale": rationale,
            }
        )
        audit_rows.append(row)

    drug_rows = [row for row in audit_rows if row["entity_type"] == "DRUG"]
    side_rows = [row for row in audit_rows if row["entity_type"] == "SIDE_EFFECT"]
    if len(audit_rows) != 15 or len(drug_rows) != 11 or len(side_rows) != 4:
        raise RuntimeError("Audit must contain exactly 11 drug and four side-effect views")
    if [row["view_index"] for row in drug_rows] != list(range(1, 12)):
        raise RuntimeError("Drug view order is not continuous 1..11")
    if [row["view_index"] for row in side_rows] != list(range(1, 5)):
        raise RuntimeError("Side-effect view order is not continuous 1..4")

    ranking = build_ranking(audit_rows)
    verify_protected(before)
    report = build_report(audit_rows, ranking, leakage_safe)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(AUDIT_PATH, audit_rows, audit_columns())
    write_csv(
        RANKING_PATH,
        ranking,
        [
            "rank", "entity_type", "view_index", "view_name", "rewiring_priority",
            "recommended_rewiring_mode", "reason_for_rank", "expected_benefit", "main_risk",
        ],
    )
    temporary_report = REPORT_PATH.with_suffix(REPORT_PATH.suffix + ".tmp")
    temporary_report.write_text(report, encoding="utf-8")
    temporary_report.replace(REPORT_PATH)
    verify_protected(before)

    print(f"Identified MSSF drug views: {len(drug_rows)}")
    print(f"Identified MSSF side-effect views: {len(side_rows)}")
    print(f"Safe ranked candidates: {len(ranking)}")
    print(f"TASK-AWARE LEAKAGE CHECK: {'PASS' if leakage_safe else 'FAIL'}")
    print(f"Audit CSV: {AUDIT_PATH}")
    print(f"Candidate ranking: {RANKING_PATH}")
    print(f"Report: {REPORT_PATH}")
    print("Training/testing performed: NO")


if __name__ == "__main__":
    main()
