"""Fold-1 MSSF-clean experiment with sparse SMDdatabase topology variants.

No new neural architecture is introduced. The selected adjacency changes only
which original SMDdatabase values appear in drug view #3.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PILOT_SCRIPT = PROJECT_ROOT / "scripts" / "26a_pilot_compare_clean_vs_kg.py"
AUDIT_PATH = PROJECT_ROOT / "data_processed" / "rewiring" / "mssf_similarity_audit.csv"
STEP31B_DIR = PROJECT_ROOT / "data_processed" / "rewiring" / "smd_database"
STEP31B_DIAGNOSTICS_PATH = STEP31B_DIR / "rewiring_diagnostics.json"
SPLIT_PATH = PROJECT_ROOT / "data_processed" / "experiments" / "pilot_fold1_split.npz"
DRUG_MAPPING_PATH = PROJECT_ROOT / "data_processed" / "mappings" / "drug_mapping.csv"
CLEAN_RESULT_DIR = (
    PROJECT_ROOT / "data_processed" / "experiments" / "kg_alignment_fold1"
    / "clean_seed42_bs64"
)
OUTPUT_ROOT = PROJECT_ROOT / "data_processed" / "experiments" / "rewiring_fold1"

VARIANT_FILES = {
    "original_top10": STEP31B_DIR / "original_top10_edges.csv",
    "kg_task": STEP31B_DIR / "kg_task_edges.csv",
    "kg_only": STEP31B_DIR / "kg_only_edges.csv",
    "task_only": STEP31B_DIR / "task_only_edges.csv",
}
EXPECTED_ADDED_ORIGIN = {
    "kg_task": "KG_TASK_ADDED",
    "kg_only": "KG_ONLY_ADDED",
    "task_only": "TASK_ONLY_ADDED",
}

DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 64
MAX_EPOCHS = 30
PATIENCE = 7
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
DROPOUT = 0.4
LATENT_DIM = 64
DRUG_COUNT = 757
SIDE_EFFECT_COUNT = 994
DRUG_VIEW_COUNT = 11
SIDE_VIEW_COUNT = 4
CLASS_LABELS = (1, 2, 3, 4, 5)
SMD_VIEW_ZERO_BASED = 2
K_ORIGINAL = 10
ADD_K = 3


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(PROJECT_ROOT))
pilot = load_module("biokorf_pilot_helpers_step31c", PILOT_SCRIPT)
from models.mssf_clean import MSSFClean, MSSFCleanConfig


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_files(paths: list[Path] | tuple[Path, ...]) -> None:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required Step 31C inputs are missing: {missing}")


def output_directory(variant: str, seed: int, batch_size: int) -> Path:
    return OUTPUT_ROOT / f"{variant}_seed{seed}_bs{batch_size}"


def require_new(paths: list[Path], operation: str) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            f"Refusing to {operation}; output already exists: " + ", ".join(existing)
        )


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
        raise ValueError(f"Expected one audited SMDdatabase row; found {len(matches)}")
    source = (PROJECT_ROOT / matches[0]["source_file"]).resolve()
    try:
        source.relative_to((PROJECT_ROOT / "Datas").resolve())
    except ValueError as error:
        raise ValueError(f"Audited SMDdatabase source resolves outside Datas/: {source}") from error
    if not source.is_file():
        raise FileNotFoundError(f"Audited SMDdatabase source not found: {source}")
    return matches[0], source


def load_source_matrix(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        matrix = np.asarray(pickle.load(handle), dtype=np.float64)
    if matrix.shape != (DRUG_COUNT, DRUG_COUNT):
        raise ValueError(f"Expected SMDdatabase shape [757, 757], found {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("SMDdatabase contains non-finite values")
    matrix.setflags(write=False)
    return matrix


def drug_matrix_order_check() -> bool:
    if not DRUG_MAPPING_PATH.is_file():
        raise FileNotFoundError(f"Drug mapping metadata not found: {DRUG_MAPPING_PATH}")
    with DRUG_MAPPING_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != DRUG_COUNT:
        return False
    indices = [int(row["matrix_index"]) for row in rows]
    return bool(
        indices == list(range(DRUG_COUNT))
        and rows[0].get("drug_name", "").strip().casefold() == "lepirudin"
    )


def read_edge_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Step 31B adjacency not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "source_drug_index", "target_drug_index", "original_similarity", "edge_origin"
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise KeyError(f"Adjacency {path.name} is missing columns: {sorted(missing)}")
        for raw in reader:
            rows.append(
                {
                    "source_drug_index": int(raw["source_drug_index"]),
                    "target_drug_index": int(raw["target_drug_index"]),
                    "recorded_original_similarity": float(raw["original_similarity"]),
                    "edge_origin": raw["edge_origin"],
                }
            )
    return rows


def validate_adjacency(
    variant: str,
    rows: list[dict[str, Any]],
    original_rows: list[dict[str, Any]],
    source_matrix: np.ndarray,
) -> None:
    pairs: list[tuple[int, int]] = []
    for row in rows:
        source = row["source_drug_index"]
        target = row["target_drug_index"]
        if not (0 <= source < DRUG_COUNT and 0 <= target < DRUG_COUNT):
            raise IndexError(f"Adjacency index outside 0..756: {(source, target)}")
        if source == target:
            raise ValueError(f"Adjacency contains a self-edge: {(source, target)}")
        if not math.isclose(
            row["recorded_original_similarity"],
            float(source_matrix[source, target]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"Recorded original similarity differs from source matrix at {(source, target)}"
            )
        pairs.append((source, target))
    if len(pairs) != len(set(pairs)):
        raise ValueError(f"{variant} adjacency contains duplicate directed edges")

    original_pairs = {
        (row["source_drug_index"], row["target_drug_index"])
        for row in original_rows
    }
    if len(original_pairs) != DRUG_COUNT * K_ORIGINAL:
        raise ValueError("Original top-10 adjacency must contain exactly 7,570 directed edges")
    source_counts = np.bincount(
        [row["source_drug_index"] for row in original_rows], minlength=DRUG_COUNT
    )
    if not np.all(source_counts == K_ORIGINAL):
        raise ValueError("Original top-10 adjacency must have exactly ten edges per source")
    if any(row["edge_origin"] != "ORIGINAL" for row in original_rows):
        raise ValueError("Original top-10 control contains non-ORIGINAL edge origins")

    selected_original_pairs = {
        (row["source_drug_index"], row["target_drug_index"])
        for row in rows
        if row["edge_origin"] == "ORIGINAL"
    }
    if selected_original_pairs != original_pairs:
        raise ValueError(f"{variant} does not preserve the exact original top-10 edge set")
    added_rows = [row for row in rows if row["edge_origin"] != "ORIGINAL"]
    if variant == "original_top10":
        if added_rows or len(rows) != len(original_rows):
            raise ValueError("original_top10 must contain no added edges")
        return
    expected_origin = EXPECTED_ADDED_ORIGIN[variant]
    if any(row["edge_origin"] != expected_origin for row in added_rows):
        raise ValueError(f"{variant} contains an unexpected added-edge origin")
    added_counts = np.bincount(
        [row["source_drug_index"] for row in added_rows], minlength=DRUG_COUNT
    )
    if np.any(added_counts > ADD_K):
        raise ValueError(f"{variant} adds more than {ADD_K} edges for a source drug")


def construct_rewired_matrix(
    variant: str, source_matrix: np.ndarray
) -> tuple[np.ndarray, dict[str, Any], Path]:
    reference_rows = read_edge_rows(VARIANT_FILES["original_top10"])
    selected_path = VARIANT_FILES[variant]
    selected_rows = (
        reference_rows if variant == "original_top10" else read_edge_rows(selected_path)
    )
    validate_adjacency(variant, selected_rows, reference_rows, source_matrix)
    rewired = np.zeros((DRUG_COUNT, DRUG_COUNT), dtype=np.float64)
    diagonal = np.diag(source_matrix).copy()
    np.fill_diagonal(rewired, diagonal)
    source_indices = np.asarray(
        [row["source_drug_index"] for row in selected_rows], dtype=np.int64
    )
    target_indices = np.asarray(
        [row["target_drug_index"] for row in selected_rows], dtype=np.int64
    )
    rewired[source_indices, target_indices] = source_matrix[source_indices, target_indices]
    selected_values = source_matrix[source_indices, target_indices]
    zero_count = int(np.count_nonzero(selected_values == 0))
    added_rows = [row for row in selected_rows if row["edge_origin"] != "ORIGINAL"]
    added_values = np.asarray(
        [
            source_matrix[row["source_drug_index"], row["target_drug_index"]]
            for row in added_rows
        ],
        dtype=np.float64,
    )
    added_zero_count = int(np.count_nonzero(added_values == 0))
    diagnostics = {
        "variant": variant,
        "source_shape": list(source_matrix.shape),
        "rewired_shape": list(rewired.shape),
        "selected_adjacency_edge_count": len(selected_rows),
        "added_edge_count": len(added_rows),
        "selected_zero_weight_edge_count": zero_count,
        "selected_zero_weight_percentage": 100.0 * zero_count / len(selected_rows),
        "mean_selected_original_similarity": float(selected_values.mean()),
        "median_selected_original_similarity": float(np.median(selected_values)),
        "added_zero_weight_edge_count": added_zero_count,
        "added_zero_weight_percentage": (
            100.0 * added_zero_count / len(added_rows) if added_rows else 0.0
        ),
        "mean_added_original_similarity": (
            float(added_values.mean()) if added_values.size else None
        ),
        "median_added_original_similarity": (
            float(np.median(added_values)) if added_values.size else None
        ),
        "nonzero_rewired_matrix_count": int(np.count_nonzero(rewired)),
        "diagonal_preserved": bool(np.array_equal(np.diag(rewired), diagonal)),
        "feature_values_come_only_from_original_smd_database": True,
        "kg_task_high_zero_weight_warning": bool(
            variant == "kg_task" and added_rows
            and 100.0 * added_zero_count / len(added_rows) > 20.0
        ),
    }
    if not diagnostics["diagonal_preserved"]:
        raise RuntimeError("SMDdatabase diagonal was not preserved")
    return rewired, diagnostics, selected_path


def load_fixed_split(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(SPLIT_PATH) as split:
        required = {
            "train_indices", "validation_indices", "test_indices", "train_samples",
            "validation_samples", "test_samples", "seed", "fold",
        }
        missing = required.difference(split.files)
        if missing:
            raise KeyError(f"Fixed split is missing arrays: {sorted(missing)}")
        if int(split["seed"]) != 42 or int(split["fold"]) != 1:
            raise ValueError("Expected the fixed seed-42 Fold-1 split")
        parts: list[np.ndarray] = []
        all_indices: list[np.ndarray] = []
        for name in ("train", "validation", "test"):
            indices = np.asarray(split[f"{name}_indices"])
            saved = np.asarray(split[f"{name}_samples"])
            if not np.array_equal(samples[indices], saved):
                raise ValueError(f"Saved {name} samples do not match fixed indices")
            parts.append(saved.copy())
            all_indices.append(indices)
    combined = np.concatenate(all_indices)
    if len(combined) != len(samples) or len(np.unique(combined)) != len(samples):
        raise ValueError("Fixed split is not a complete disjoint partition")
    return tuple(parts)


def rewiring_leakage_check(source_hash: str) -> bool:
    if not STEP31B_DIAGNOSTICS_PATH.is_file():
        raise FileNotFoundError(
            f"Step 31B diagnostics not found: {STEP31B_DIAGNOSTICS_PATH}"
        )
    diagnostics = json.loads(STEP31B_DIAGNOSTICS_PATH.read_text(encoding="utf-8"))
    profiles = diagnostics.get("train_only_profiles", {})
    checks = diagnostics.get("checks", {})
    hashes = diagnostics.get("protected_hashes_after", {})
    parameters = diagnostics.get("parameters", {})
    return bool(
        parameters.get("seed") == 42
        and profiles.get("validation_positions_hidden") is True
        and profiles.get("test_positions_hidden") is True
        and checks.get("task_aware_leakage") is True
        and checks.get("source_similarity_preservation") is True
        and checks.get("fixed_split_safety") is True
        and hashes.get("fixed_split") == sha256(SPLIT_PATH)
        and hashes.get("source_similarity") == source_hash
    )


def load_experiment_data(
    variant: str,
) -> tuple[Any, Any, Any, dict[str, Any], dict[str, bool], dict[str, str]]:
    audit_row, source_path = read_audit_source()
    source_hash = sha256(source_path)
    source_matrix = load_source_matrix(source_path)
    rewired, feature_diagnostics, adjacency_path = construct_rewired_matrix(
        variant, source_matrix
    )
    frequency = np.asarray(pilot.load_pickle("drug_side.pkl"))
    samples = pilot.original_positive_sample_order(frequency)
    train_samples, validation_samples, test_samples = load_fixed_split(samples)
    hidden_samples = np.concatenate((validation_samples, test_samples), axis=0)
    drug_features, side_features, label_safe = pilot.build_leakage_safe_features(
        frequency, hidden_samples
    )
    drug_features = drug_features.clone()
    view_start = SMD_VIEW_ZERO_BASED * DRUG_COUNT
    view_stop = view_start + DRUG_COUNT
    dense_view = drug_features[:, view_start:view_stop].cpu().numpy()
    if not np.array_equal(dense_view, source_matrix.astype(np.float32)):
        raise RuntimeError("MSSF-clean drug view #3 does not match audited SMDdatabase")
    drug_features[:, view_start:view_stop] = torch.from_numpy(
        rewired.astype(np.float32, copy=False)
    )
    if tuple(drug_features.shape) != (DRUG_COUNT, DRUG_COUNT * DRUG_VIEW_COUNT):
        raise ValueError(f"Unexpected final drug input matrix shape: {tuple(drug_features.shape)}")
    if tuple(side_features.shape) != (SIDE_EFFECT_COUNT, SIDE_EFFECT_COUNT * SIDE_VIEW_COUNT):
        raise ValueError(f"Unexpected final side-effect matrix shape: {tuple(side_features.shape)}")
    finite = bool(torch.isfinite(drug_features).all() and torch.isfinite(side_features).all())
    order_safe = drug_matrix_order_check()
    task_rewiring_safe = rewiring_leakage_check(source_hash)
    graph_safe = bool(pilot.scan_graph_leakage())
    checks = {
        "label_derived_feature_leakage": bool(label_safe),
        "drug_phenotype_leakage": graph_safe,
        "task_aware_rewiring_leakage": task_rewiring_safe,
        "drug_matrix_order": order_safe,
        "finite_values": finite,
    }
    if not all(checks.values()):
        raise RuntimeError(f"A data safety check failed: {checks}")
    datasets = tuple(
        pilot.IndexedPairDataset(part, drug_features, side_features)
        for part in (train_samples, validation_samples, test_samples)
    )
    feature_diagnostics.update(
        {
            "audit_source_file": audit_row["source_file"],
            "source_path": str(source_path),
            "source_sha256": source_hash,
            "adjacency_path": str(adjacency_path),
            "adjacency_sha256": sha256(adjacency_path),
            "final_drug_input_dimension": int(drug_features.shape[1]),
            "final_side_effect_input_dimension": int(side_features.shape[1]),
            "checks": checks,
        }
    )
    protected = {
        "source": source_hash,
        "adjacency": sha256(adjacency_path),
        "split": sha256(SPLIT_PATH),
        "mssf": sha256(PROJECT_ROOT / "mssf.py"),
        "model": sha256(PROJECT_ROOT / "model.py"),
        "mssf_clean": sha256(PROJECT_ROOT / "models" / "mssf_clean.py"),
    }
    return (*datasets, feature_diagnostics, checks, protected)


def verify_protected(protected: dict[str, str], feature_diagnostics: dict[str, Any]) -> tuple[bool, bool]:
    source_safe = protected["source"] == sha256(Path(feature_diagnostics["source_path"]))
    all_safe = bool(
        source_safe
        and protected["adjacency"] == sha256(Path(feature_diagnostics["adjacency_path"]))
        and protected["split"] == sha256(SPLIT_PATH)
        and protected["mssf"] == sha256(PROJECT_ROOT / "mssf.py")
        and protected["model"] == sha256(PROJECT_ROOT / "model.py")
        and protected["mssf_clean"] == sha256(PROJECT_ROOT / "models" / "mssf_clean.py")
    )
    return source_safe, all_safe


def make_loader(
    dataset: Any, shuffle: bool, seed: int, batch_size: int
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def create_model(seed: int) -> MSSFClean:
    pilot.configure_reproducibility(seed)
    return MSSFClean(MSSFCleanConfig(dropout=DROPOUT, gp=LATENT_DIM))


def fairness_check(seed: int, batch_size: int) -> bool:
    model = create_model(seed)
    return bool(
        type(model) is MSSFClean
        and LEARNING_RATE == pilot.LEARNING_RATE
        and WEIGHT_DECAY == pilot.WEIGHT_DECAY
        and batch_size > 0
        and model.feature_nums == DRUG_VIEW_COUNT * SIDE_VIEW_COUNT
        and not any("kg" in name.casefold() for name, _ in model.named_modules())
    )


def smoke_checks(
    variant: str,
    dataset: Any,
    diagnostics: dict[str, Any],
    device: torch.device,
    seed: int,
) -> bool:
    if len(dataset) < 4:
        raise ValueError("At least four training samples are needed for smoke mode")
    samples = [dataset[index] for index in range(4)]
    drugs = torch.stack([sample[0] for sample in samples]).to(device)
    sides = torch.stack([sample[1] for sample in samples]).to(device)
    model = create_model(seed).to(device).eval()
    with torch.inference_mode():
        outputs = model(drugs, sides, device=device, return_debug=True)
    logits, _rec_con, _rec_add, mu, _logvar, debug = outputs
    contract = bool(
        tuple(debug["H_en_con"].shape) == (4, 128)
        and tuple(debug["H_en_add"].shape) == (4, 128)
        and tuple(debug["H_cnn_im"].shape) == (4, 128)
        and tuple(debug["H_pair"].shape) == (4, 384)
        and tuple(mu.shape) == (4, LATENT_DIM)
        and tuple(debug["latent"].shape) == (4, LATENT_DIM)
        and tuple(logits.shape) == (4, 5)
        and torch.isfinite(logits).all()
    )
    print(f"Variant: {variant}")
    print(f"Source SMDdatabase shape: {diagnostics['source_shape']}")
    print(f"Rewired matrix shape: {diagnostics['rewired_shape']}")
    print(f"Selected adjacency edge count: {diagnostics['selected_adjacency_edge_count']}")
    print(f"Nonzero rewired matrix count: {diagnostics['nonzero_rewired_matrix_count']}")
    print(f"Diagonal preservation: {diagnostics['diagonal_preserved']}")
    print(f"Mean selected source similarity: {diagnostics['mean_selected_original_similarity']:.10f}")
    print(f"Zero-weight selected-edge count: {diagnostics['selected_zero_weight_edge_count']}")
    print(f"Zero-weight selected-edge percentage: {diagnostics['selected_zero_weight_percentage']:.4f}%")
    print(f"Final Drug input dimension: {diagnostics['final_drug_input_dimension']}")
    print(f"Final Side-effect input dimension: {diagnostics['final_side_effect_input_dimension']}")
    if diagnostics["kg_task_high_zero_weight_warning"]:
        print("HIGH ZERO-WEIGHT ADDED-EDGE RATE")
    print(f"REWIRED INPUT CONTRACT CHECK: {'PASS' if contract else 'FAIL'}")
    return contract


def prediction_loss(
    outputs: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    labels: Tensor,
    drugs: Tensor,
    sides: Tensor,
) -> Tensor:
    return pilot.composite_loss(*outputs, labels, drugs, sides)


def train_epoch(
    model: MSSFClean,
    dataset: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    seed: int,
    batch_size: int,
) -> float:
    model.train()
    total_loss, total_count = 0.0, 0
    for drugs, sides, _drug_index, _side_index, labels in make_loader(
        dataset, shuffle=True, seed=seed + epoch, batch_size=batch_size
    ):
        drugs, sides, labels = drugs.to(device), sides.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(drugs, sides, device=device)
        loss = prediction_loss(outputs, labels, drugs, sides)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        count = int(labels.shape[0])
        total_loss += float(loss.detach()) * count
        total_count += count
    return total_loss / total_count


def evaluate(
    model: MSSFClean,
    dataset: Any,
    device: torch.device,
    seed: int,
    batch_size: int,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    total_loss, total_count = 0.0, 0
    labels_all: list[np.ndarray] = []
    probabilities_all: list[np.ndarray] = []
    with torch.inference_mode():
        for drugs, sides, _drug_index, _side_index, labels in make_loader(
            dataset, shuffle=False, seed=seed, batch_size=batch_size
        ):
            drugs, sides = drugs.to(device), sides.to(device)
            outputs = model(drugs, sides, device=device)
            loss = prediction_loss(outputs, labels.to(device), drugs, sides)
            probabilities = torch.softmax(outputs[0], dim=1)
            if not torch.isfinite(loss) or not torch.isfinite(probabilities).all():
                raise FloatingPointError("Non-finite validation/test value")
            count = int(labels.shape[0])
            total_loss += float(loss) * count
            total_count += count
            labels_all.append((labels.numpy() - 1).astype(np.int64))
            probabilities_all.append(probabilities.cpu().numpy())
    metrics = pilot.classification_metrics(
        np.concatenate(labels_all), np.vstack(probabilities_all)
    )
    return total_loss / total_count, metrics


def checkpoint_state(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


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
    raise TypeError(f"Unsupported JSON value type: {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def print_safety(checks: dict[str, bool]) -> None:
    labels = {
        "experiment_fairness": "EXPERIMENT FAIRNESS",
        "label_derived_feature_leakage": "LABEL-DERIVED FEATURE LEAKAGE",
        "drug_phenotype_leakage": "DRUG-PHENOTYPE LEAKAGE",
        "task_aware_rewiring_leakage": "TASK-AWARE REWIRING LEAKAGE",
        "source_smd_database_preservation": "SOURCE SMDDATABASE PRESERVATION",
        "drug_matrix_order": "DRUG MATRIX ORDER",
        "finite_values": "FINITE-VALUE",
    }
    for key, label in labels.items():
        print(f"{label} CHECK: {'PASS' if checks[key] else 'FAIL'}")


def smoke_mode(variant: str, seed: int, batch_size: int) -> None:
    train_data, _validation_data, _test_data, diagnostics, data_checks, protected = (
        load_experiment_data(variant)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contract = smoke_checks(variant, train_data, diagnostics, device, seed)
    source_safe, protected_safe = verify_protected(protected, diagnostics)
    checks = {
        "experiment_fairness": fairness_check(seed, batch_size) and protected_safe,
        **data_checks,
        "source_smd_database_preservation": source_safe,
        "finite_values": data_checks["finite_values"] and contract,
    }
    print_safety(checks)
    if not all(checks.values()):
        raise RuntimeError("Step 31C smoke checks failed")


def train_mode(variant: str, seed: int, batch_size: int) -> dict[str, Any]:
    output_dir = output_directory(variant, seed, batch_size)
    history_path = output_dir / "training_history.csv"
    checkpoint_path = output_dir / "best_checkpoint.pt"
    feature_path = output_dir / "rewiring_feature_diagnostics.json"
    require_new([history_path, checkpoint_path, feature_path], "train")
    train_data, validation_data, _test_data, diagnostics, data_checks, protected = (
        load_experiment_data(variant)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contract = smoke_checks(variant, train_data, diagnostics, device, seed)
    fair = fairness_check(seed, batch_size)
    if not all((fair, contract, *data_checks.values())):
        raise RuntimeError("A required pre-training check failed")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(feature_path, diagnostics)
    model = create_model(seed).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    history: list[dict[str, Any]] = []
    columns = ["epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1", "val_aupr"]
    best_epoch, best_f1, stale = 0, -1.0, 0
    finite = True
    for epoch in range(1, MAX_EPOCHS + 1):
        pilot.configure_reproducibility(seed + epoch)
        started = time.perf_counter()
        train_loss = train_epoch(
            model, train_data, optimizer, device, epoch, seed, batch_size
        )
        val_loss, metrics = evaluate(
            model, validation_data, device, seed, batch_size
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": metrics["accuracy"],
            "val_macro_f1": metrics["macro_f1"],
            "val_aupr": metrics["aupr"],
        }
        finite = finite and all(np.isfinite(value) for value in row.values())
        history.append(row)
        write_csv(history_path, history, columns)
        if metrics["macro_f1"] > best_f1:
            best_epoch, best_f1, stale = epoch, float(metrics["macro_f1"]), 0
            temporary = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(
                {
                    "model": "mssf_clean",
                    "variant": variant,
                    "seed": seed,
                    "batch_size": batch_size,
                    "epoch": epoch,
                    "validation_macro_f1": best_f1,
                    "selection_metric": "validation_macro_f1",
                    "source_smd_sha256": protected["source"],
                    "adjacency_sha256": protected["adjacency"],
                    "split_sha256": protected["split"],
                    "model_state_dict": checkpoint_state(model),
                },
                temporary,
            )
            temporary.replace(checkpoint_path)
        else:
            stale += 1
        print(
            f"variant={variant} epoch={epoch:02d} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} val_macro_f1={metrics['macro_f1']:.6f} "
            f"best={best_epoch} patience={stale}/{PATIENCE} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
        if stale >= PATIENCE:
            break
    source_safe, protected_safe = verify_protected(protected, diagnostics)
    checks = {
        "experiment_fairness": fair and protected_safe,
        **data_checks,
        "source_smd_database_preservation": source_safe,
        "finite_values": finite and data_checks["finite_values"],
    }
    print_safety(checks)
    if not all(checks.values()):
        raise RuntimeError("A required training safety check failed")
    return {"best_epoch": best_epoch, "best_validation_macro_f1": best_f1}


def expected_test_outputs(output_dir: Path) -> list[Path]:
    return [
        output_dir / "test_metrics.json",
        output_dir / "confusion_matrix.csv",
        output_dir / "per_class_metrics.csv",
        output_dir / "report.txt",
    ]


def test_mode(variant: str, seed: int, batch_size: int) -> dict[str, Any]:
    output_dir = output_directory(variant, seed, batch_size)
    checkpoint_path = output_dir / "best_checkpoint.pt"
    feature_path = output_dir / "rewiring_feature_diagnostics.json"
    history_path = output_dir / "training_history.csv"
    require_files((checkpoint_path, feature_path, history_path))
    outputs = expected_test_outputs(output_dir)
    require_new(outputs, "test")
    train_data, _validation_data, test_data, diagnostics, data_checks, protected = (
        load_experiment_data(variant)
    )
    saved_diagnostics = json.loads(feature_path.read_text(encoding="utf-8"))
    if (
        saved_diagnostics.get("source_sha256") != protected["source"]
        or saved_diagnostics.get("adjacency_sha256") != protected["adjacency"]
    ):
        raise ValueError("Saved training feature diagnostics do not match current inputs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contract = smoke_checks(variant, train_data, diagnostics, device, seed)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("model") != "mssf_clean"
        or checkpoint.get("variant") != variant
        or int(checkpoint.get("seed", -1)) != seed
        or int(checkpoint.get("batch_size", -1)) != batch_size
        or checkpoint.get("source_smd_sha256") != protected["source"]
        or checkpoint.get("adjacency_sha256") != protected["adjacency"]
        or checkpoint.get("split_sha256") != protected["split"]
    ):
        raise ValueError("Checkpoint metadata does not match this Step 31C run")
    model = create_model(seed)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)
    test_loss, metrics = evaluate(model, test_data, device, seed, batch_size)
    source_safe, protected_safe = verify_protected(protected, diagnostics)
    finite = bool(
        np.isfinite(test_loss)
        and all(
            np.isfinite(metrics[key])
            for key in (
                "accuracy", "macro_precision", "macro_recall", "macro_f1", "micro_f1", "aupr"
            )
        )
    )
    checks = {
        "experiment_fairness": fairness_check(seed, batch_size) and protected_safe,
        **data_checks,
        "source_smd_database_preservation": source_safe,
        "finite_values": finite and contract and data_checks["finite_values"],
    }
    metrics.update(
        {
            "model": "mssf_clean",
            "variant": variant,
            "seed": seed,
            "batch_size": batch_size,
            "best_epoch": int(checkpoint["epoch"]),
            "best_validation_macro_f1": float(checkpoint["validation_macro_f1"]),
            "test_loss": float(test_loss),
            "checks": checks,
        }
    )
    write_csv(
        output_dir / "confusion_matrix.csv",
        [
            {
                "true_class": label,
                **{f"predicted_{predicted}": row[predicted - 1] for predicted in CLASS_LABELS},
            }
            for label, row in zip(CLASS_LABELS, metrics["confusion_matrix"])
        ],
        ["true_class", *[f"predicted_{label}" for label in CLASS_LABELS]],
    )
    write_csv(
        output_dir / "per_class_metrics.csv",
        [
            {"class": label, **metrics["per_class"][str(label)]}
            for label in CLASS_LABELS
        ],
        ["class", "precision", "recall", "f1", "support"],
    )
    write_json(output_dir / "test_metrics.json", metrics)
    report_lines = [
        "BioKORF Step 31C — Fold-1 Rewired SMDdatabase Experiment",
        "========================================================",
        f"Variant: {variant}",
        f"Seed: {seed}",
        f"Batch size: {batch_size}",
        f"Best epoch: {checkpoint['epoch']}",
        f"Best validation Macro-F1: {checkpoint['validation_macro_f1']:.8f}",
        f"Test Accuracy: {metrics['accuracy']:.8f}",
        f"Test Macro Precision: {metrics['macro_precision']:.8f}",
        f"Test Macro Recall: {metrics['macro_recall']:.8f}",
        f"Test Macro-F1: {metrics['macro_f1']:.8f}",
        f"Test Micro-F1: {metrics['micro_f1']:.8f}",
        f"Test AUPR: {metrics['aupr']:.8f}",
        "Only drug view #3 topology differs; all feature values come from original SMDdatabase.",
        "",
        *[
            f"{name.replace('_', ' ').upper()} CHECK: {'PASS' if value else 'FAIL'}"
            for name, value in checks.items()
        ],
        "Training/testing performed: YES (explicit test mode).",
    ]
    report = "\n".join(report_lines) + "\n"
    (output_dir / "report.txt").write_text(report, encoding="utf-8")
    print(report, end="")
    print_safety(checks)
    if not all(checks.values()):
        raise RuntimeError("A required test safety check failed")
    return metrics


def train_test_mode(variant: str, seed: int, batch_size: int) -> None:
    output_dir = output_directory(variant, seed, batch_size)
    require_new(
        [
            output_dir / "training_history.csv",
            output_dir / "best_checkpoint.pt",
            output_dir / "rewiring_feature_diagnostics.json",
            *expected_test_outputs(output_dir),
        ],
        "train_test",
    )
    training = train_mode(variant, seed, batch_size)
    metrics = test_mode(variant, seed, batch_size)
    print("STEP 31C TRAIN_TEST SUMMARY")
    print(f"Variant: {variant}")
    print(f"Best epoch: {training['best_epoch']}")
    print(f"Best validation Macro-F1: {training['best_validation_macro_f1']:.8f}")
    print(f"Test Accuracy: {metrics['accuracy']:.8f}")
    print(f"Test Macro-F1: {metrics['macro_f1']:.8f}")
    print(f"Test AUPR: {metrics['aupr']:.8f}")


def read_result(path: Path, expected_variant: str | None, seed: int, batch_size: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Completed result not found: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if int(result.get("seed", -1)) != seed or int(result.get("batch_size", -1)) != batch_size:
        raise ValueError(f"Result settings do not match seed={seed}, batch_size={batch_size}: {path}")
    if expected_variant is not None and result.get("variant") != expected_variant:
        raise ValueError(f"Result variant mismatch: {path}")
    return result


def compare_mode(seed: int, batch_size: int) -> None:
    if seed != 42 or batch_size != 64:
        raise ValueError("Stored CLEAN_DENSE comparison is available only for seed 42, batch size 64")
    clean = read_result(CLEAN_RESULT_DIR / "test_metrics.json", None, seed, batch_size)
    original = read_result(
        output_directory("original_top10", seed, batch_size) / "test_metrics.json",
        "original_top10", seed, batch_size,
    )
    kg_task = read_result(
        output_directory("kg_task", seed, batch_size) / "test_metrics.json",
        "kg_task", seed, batch_size,
    )
    models = (("CLEAN_DENSE", clean), ("ORIGINAL_TOP10", original), ("KG_TASK", kg_task))
    print("Model | Best epoch | Accuracy | Macro-F1 | AUPR")
    print("--- | ---: | ---: | ---: | ---:")
    for name, result in models:
        print(
            f"{name} | {result['best_epoch']} | {result['accuracy']:.8f} | "
            f"{result['macro_f1']:.8f} | {result['aupr']:.8f}"
        )
    for label, left, right in (
        ("ORIGINAL_TOP10 - CLEAN_DENSE", original, clean),
        ("KG_TASK - CLEAN_DENSE", kg_task, clean),
        ("KG_TASK - ORIGINAL_TOP10", kg_task, original),
    ):
        print(label)
        for key in ("accuracy", "macro_f1", "aupr"):
            print(f"{key}: {left[key] - right[key]:+.8f}")
    beats_original = kg_task["macro_f1"] > original["macro_f1"]
    beats_dense = kg_task["macro_f1"] > clean["macro_f1"]
    print(f"Does KG_TASK beat ORIGINAL_TOP10 on Macro-F1? {'YES' if beats_original else 'NO'}")
    print(f"Does KG_TASK beat CLEAN_DENSE on Macro-F1? {'YES' if beats_dense else 'NO'}")
    print(
        "REWIRING_TOPOLOGY_SIGNAL = " + ("POSITIVE" if beats_original else "NEGATIVE")
    )
    print("DOWNSTREAM_IMPROVEMENT = " + ("YES" if beats_dense else "NO"))
    print("These are controlled experiment labels, not publication claims.")
    original_checks = original.get("checks", {})
    kg_task_checks = kg_task.get("checks", {})
    _audit_row, source_path = read_audit_source()
    step31b = json.loads(STEP31B_DIAGNOSTICS_PATH.read_text(encoding="utf-8"))
    source_hash = sha256(source_path)
    compare_checks = {
        "experiment_fairness": bool(
            original_checks.get("experiment_fairness")
            and kg_task_checks.get("experiment_fairness")
        ),
        "label_derived_feature_leakage": bool(
            original_checks.get("label_derived_feature_leakage")
            and kg_task_checks.get("label_derived_feature_leakage")
        ),
        "drug_phenotype_leakage": bool(
            original_checks.get("drug_phenotype_leakage")
            and kg_task_checks.get("drug_phenotype_leakage")
        ),
        "task_aware_rewiring_leakage": bool(
            original_checks.get("task_aware_rewiring_leakage")
            and kg_task_checks.get("task_aware_rewiring_leakage")
            and step31b.get("checks", {}).get("task_aware_leakage")
        ),
        "source_smd_database_preservation": bool(
            original_checks.get("source_smd_database_preservation")
            and kg_task_checks.get("source_smd_database_preservation")
            and step31b.get("source", {}).get("sha256_after") == source_hash
        ),
        "drug_matrix_order": bool(
            original_checks.get("drug_matrix_order")
            and kg_task_checks.get("drug_matrix_order")
            and drug_matrix_order_check()
        ),
        "finite_values": bool(
            original_checks.get("finite_values")
            and kg_task_checks.get("finite_values")
            and all(
                np.isfinite(result[key])
                for _name, result in models
                for key in ("accuracy", "macro_f1", "aupr")
            )
        ),
    }
    print_safety(compare_checks)
    if not all(compare_checks.values()):
        raise RuntimeError("Stored results failed a Step 31C compare safety check")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("original_top10", "kg_task", "kg_only", "task_only"),
        default="original_top10",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("smoke", "train", "test", "train_test", "compare"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def main() -> None:
    args = parse_args()
    print(f"Mode: {args.mode}; variant: {args.variant}; seed: {args.seed}; batch size: {args.batch_size}")
    require_files(
        [
            AUDIT_PATH,
            STEP31B_DIAGNOSTICS_PATH,
            SPLIT_PATH,
            DRUG_MAPPING_PATH,
            VARIANT_FILES["original_top10"],
            *([] if args.mode == "compare" else [VARIANT_FILES[args.variant]]),
        ]
    )
    if args.mode == "smoke":
        smoke_mode(args.variant, args.seed, args.batch_size)
    elif args.mode == "train":
        train_mode(args.variant, args.seed, args.batch_size)
    elif args.mode == "test":
        test_mode(args.variant, args.seed, args.batch_size)
    elif args.mode == "train_test":
        train_test_mode(args.variant, args.seed, args.batch_size)
    else:
        compare_mode(args.seed, args.batch_size)


if __name__ == "__main__":
    main()
