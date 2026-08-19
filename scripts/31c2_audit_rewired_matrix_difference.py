"""Read-only audit of Step 31C ORIGINAL_TOP10 versus KG_TASK matrices."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import math
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEP31C_PATH = PROJECT_ROOT / "scripts" / "31c_fold1_rewired_smd_experiment.py"
ORIGINAL_EDGES_PATH = (
    PROJECT_ROOT / "data_processed" / "rewiring" / "smd_database"
    / "original_top10_edges.csv"
)
KG_TASK_EDGES_PATH = (
    PROJECT_ROOT / "data_processed" / "rewiring" / "smd_database"
    / "kg_task_edges.csv"
)
ADDED_KG_TASK_PATH = (
    PROJECT_ROOT / "data_processed" / "rewiring" / "smd_database"
    / "added_edges_kg_task.csv"
)
EXPECTED_ADDED_COUNT = 1911


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import Step 31C module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_added_rows(path: Path) -> list[dict[str, Any]]:
    required = {
        "source_drug_index",
        "target_drug_index",
        "original_similarity",
        "kg_similarity",
        "task_similarity",
        "combined_score",
        "edge_origin",
    }
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise KeyError(f"{path.name} is missing columns: {sorted(missing)}")
        rows = []
        for raw in reader:
            if raw["edge_origin"] != "KG_TASK_ADDED":
                raise ValueError(
                    f"Unexpected edge_origin in {path.name}: {raw['edge_origin']}"
                )
            rows.append(
                {
                    "source": int(raw["source_drug_index"]),
                    "target": int(raw["target_drug_index"]),
                    "recorded_original_similarity": float(raw["original_similarity"]),
                    "kg_similarity": float(raw["kg_similarity"]),
                    "task_similarity": float(raw["task_similarity"]),
                    "combined_score": float(raw["combined_score"]),
                }
            )
    return rows


def format_number(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "N/A"
    return f"{value:.12g}"


def main() -> None:
    required_paths = (
        STEP31C_PATH,
        ORIGINAL_EDGES_PATH,
        KG_TASK_EDGES_PATH,
        ADDED_KG_TASK_PATH,
    )
    missing = [path for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required Step 31C audit inputs are missing: {missing}")

    step31c = load_module("biokorf_step31c_matrix_audit", STEP31C_PATH)
    audit_row, source_path = step31c.read_audit_source()
    source_matrix = step31c.load_source_matrix(source_path)
    input_paths = (*required_paths, step31c.AUDIT_PATH, source_path)
    hashes_before = {path: sha256(path) for path in input_paths}

    original_top10, original_diagnostics, original_route = (
        step31c.construct_rewired_matrix("original_top10", source_matrix)
    )
    kg_task, kg_task_diagnostics, kg_task_route = step31c.construct_rewired_matrix(
        "kg_task", source_matrix
    )
    added_rows = load_added_rows(ADDED_KG_TASK_PATH)

    routing_safe = bool(
        step31c.VARIANT_FILES["original_top10"].resolve()
        == ORIGINAL_EDGES_PATH.resolve()
        and step31c.VARIANT_FILES["kg_task"].resolve()
        == KG_TASK_EDGES_PATH.resolve()
        and original_route.resolve() == ORIGINAL_EDGES_PATH.resolve()
        and kg_task_route.resolve() == KG_TASK_EDGES_PATH.resolve()
    )

    difference = kg_task - original_top10
    absolute_difference = np.abs(difference)
    differing_entries = int(np.count_nonzero(difference != 0))
    array_equal = bool(np.array_equal(original_top10, kg_task))
    allclose = bool(np.allclose(original_top10, kg_task))

    source_values = np.asarray(
        [source_matrix[row["source"], row["target"]] for row in added_rows],
        dtype=np.float64,
    )
    recorded_values = np.asarray(
        [row["recorded_original_similarity"] for row in added_rows],
        dtype=np.float64,
    )
    if not np.array_equal(source_values, recorded_values):
        raise ValueError(
            "An added-edge recorded original_similarity differs from SMDdatabase"
        )
    zero_count = int(np.count_nonzero(source_values == 0))
    positive_values = source_values[source_values > 0]
    positive_count = int(positive_values.size)

    full_kg_task_rows = step31c.read_edge_rows(KG_TASK_EDGES_PATH)
    full_added_pairs = {
        (row["source_drug_index"], row["target_drug_index"])
        for row in full_kg_task_rows
        if row["edge_origin"] == "KG_TASK_ADDED"
    }
    detail_added_pairs = {(row["source"], row["target"]) for row in added_rows}
    if full_added_pairs != detail_added_pairs:
        raise ValueError(
            "added_edges_kg_task.csv does not match KG_TASK_ADDED rows in kg_task_edges.csv"
        )

    hashes_after = {path: sha256(path) for path in input_paths}
    if hashes_before != hashes_after:
        raise RuntimeError("An input file changed during the read-only matrix audit")

    print("STEP 31C2 REWIRED MATRIX DIFFERENCE AUDIT")
    print("==========================================")
    print(f"Audit source_file: {audit_row['source_file']}")
    print(f"Resolved source path: {source_path}")
    print(f"Source matrix shape: {list(source_matrix.shape)}")
    print(f"original_top10 route: {original_route}")
    print(f"kg_task route: {kg_task_route}")
    print(f"VARIANT FILE ROUTING CHECK: {'PASS' if routing_safe else 'FAIL'}")
    print()
    print("MATRIX COMPARISON")
    print(f"np.array_equal(S_original_top10, S_kg_task): {array_equal}")
    print(f"np.allclose(S_original_top10, S_kg_task): {allclose}")
    print(f"Number of matrix entries that differ: {differing_entries}")
    print(f"Maximum absolute difference: {format_number(float(absolute_difference.max()))}")
    print(f"Mean absolute difference: {format_number(float(absolute_difference.mean()))}")
    print(f"L1 difference: {format_number(float(absolute_difference.sum()))}")
    print(f"L2 difference: {format_number(float(np.linalg.norm(difference)))}")
    print(f"Nonzero count in S_original_top10: {int(np.count_nonzero(original_top10))}")
    print(f"Nonzero count in S_kg_task: {int(np.count_nonzero(kg_task))}")
    print(
        "Original selected adjacency edge count: "
        f"{original_diagnostics['selected_adjacency_edge_count']}"
    )
    print(
        "KG_TASK selected adjacency edge count: "
        f"{kg_task_diagnostics['selected_adjacency_edge_count']}"
    )
    print()
    print("KG_TASK ADDED-EDGE SOURCE VALUES")
    print(f"Added edge count: {len(added_rows)}")
    if len(added_rows) != EXPECTED_ADDED_COUNT:
        print(
            f"WARNING: expected {EXPECTED_ADDED_COUNT} KG_TASK added edges, "
            f"found {len(added_rows)}"
        )
    print(f"Count with original similarity == 0: {zero_count}")
    print(
        "Percentage with original similarity == 0: "
        f"{100.0 * zero_count / len(added_rows):.8f}%"
    )
    print(f"Count with original similarity > 0: {positive_count}")
    print(
        "Min positive original similarity: "
        f"{format_number(float(positive_values.min()) if positive_count else None)}"
    )
    print(
        "Median positive original similarity: "
        f"{format_number(float(np.median(positive_values)) if positive_count else None)}"
    )
    print(
        "Mean positive original similarity: "
        f"{format_number(float(positive_values.mean()) if positive_count else None)}"
    )
    print(f"Max original similarity: {format_number(float(source_values.max()))}")
    print()
    print("FIRST 20 KG_TASK ADDED EDGES")
    print("source | target | original_similarity | kg_similarity | task_similarity | combined_score")
    for row in added_rows[:20]:
        print(
            f"{row['source']} | {row['target']} | "
            f"{format_number(row['recorded_original_similarity'])} | "
            f"{format_number(row['kg_similarity'])} | "
            f"{format_number(row['task_similarity'])} | "
            f"{format_number(row['combined_score'])}"
        )
    print()
    print(
        "REWIRED INPUT EFFECT = "
        + ("ZERO" if array_equal else "NONZERO")
    )
    print("READ-ONLY INPUT HASH CHECK: PASS")
    print("Training/testing performed: NO")


if __name__ == "__main__":
    main()
