"""Fuse deterministic name and STITCH/InChIKey drug-mapping evidence."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESOLVED_PATH = (
    PROJECT_ROOT / "data_processed" / "optimuskg" / "drug_mapping_resolved.csv"
)
CANDIDATES_PATH = (
    PROJECT_ROOT / "data_processed" / "optimuskg" / "drug_mapping_candidates.csv"
)
STRUCTURE_MATCHES_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "optimuskg"
    / "drug_stitch_inchikey_matches.csv"
)
STRUCTURE_SUMMARY_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "optimuskg"
    / "drug_structure_mapping_summary.csv"
)
FUSED_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "optimuskg"
    / "drug_mapping_evidence_fused.csv"
)
REVIEW_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "optimuskg"
    / "drug_mapping_review_required.csv"
)

EXPECTED_DRUG_COUNT = 757
EVIDENCE_CLASSES = (
    "confirmed_name_structure",
    "resolved_by_unique_intersection",
    "resolved_by_unique_structure_candidate",
    "supported_canonical_name",
    "name_only_resolved",
    "structure_only_candidate",
    "conflicting_evidence",
    "unresolved_ambiguous",
    "unresolved_unmatched",
)
DETAIL_DRUGS = (
    "lepirudin",
    "bivalirudin",
    "goserelin",
    "erythropoietin",
    "daptomycin",
    "reboxetine",
    "sulfamethoxazole.and.trimethoprim",
    "gadoversetamide",
    "technetium.(99mTc).tetrofosmin",
    "iotrolan",
)
OUTPUT_SCHEMA = {
    "matrix_index": pl.Int64,
    "drug_name": pl.String,
    "stitch_id": pl.String,
    "name_resolution_status": pl.String,
    "name_selected_id": pl.String,
    "name_candidate_ids": pl.String,
    "structure_status": pl.String,
    "structure_match_ids": pl.String,
    "intersection_ids": pl.String,
    "proposed_optimuskg_id": pl.String,
    "evidence_class": pl.String,
    "confidence": pl.String,
    "final_status": pl.String,
}


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found. Expected path: {path}")


def json_ids(values: set[str] | list[str]) -> str:
    return json.dumps(sorted(set(values)), ensure_ascii=False, separators=(",", ":"))


def validate_inputs(
    resolved: pl.DataFrame,
    candidates: pl.DataFrame,
    structure_matches: pl.DataFrame,
    structure_summary: pl.DataFrame,
) -> None:
    required_resolved = {
        "matrix_index",
        "drug_name",
        "stitch_id",
        "optimuskg_id",
        "mapping_method",
        "resolution_status",
    }
    if missing := required_resolved.difference(resolved.columns):
        raise ValueError(f"Resolved mapping is missing columns: {sorted(missing)}")
    if resolved.height != EXPECTED_DRUG_COUNT:
        raise ValueError(f"Resolved mapping must contain {EXPECTED_DRUG_COUNT} rows")
    if resolved["matrix_index"].to_list() != list(range(EXPECTED_DRUG_COUNT)):
        raise ValueError("Resolved mapping matrix_index must be continuous 0..756")

    required_candidates = {"matrix_index", "optimuskg_id"}
    if missing := required_candidates.difference(candidates.columns):
        raise ValueError(f"Name candidates are missing columns: {sorted(missing)}")
    required_matches = {
        "matrix_index",
        "optimuskg_id",
        "was_previous_name_candidate",
    }
    if missing := required_matches.difference(structure_matches.columns):
        raise ValueError(f"Structure matches are missing columns: {sorted(missing)}")
    required_summary = {
        "matrix_index",
        "drug_name",
        "stitch_id",
        "optimuskg_structure_match_count",
        "optimuskg_structure_match_ids",
        "structure_status",
    }
    if missing := required_summary.difference(structure_summary.columns):
        raise ValueError(f"Structure summary is missing columns: {sorted(missing)}")
    if structure_summary.height != EXPECTED_DRUG_COUNT:
        raise ValueError(f"Structure summary must contain {EXPECTED_DRUG_COUNT} rows")
    if not structure_summary.select(["matrix_index", "drug_name", "stitch_id"]).equals(
        resolved.select(["matrix_index", "drug_name", "stitch_id"])
    ):
        raise ValueError("Name and structure summary rows are not aligned")

    actual_structure_ids: dict[int, set[str]] = defaultdict(set)
    for row in structure_matches.iter_rows(named=True):
        actual_structure_ids[row["matrix_index"]].add(row["optimuskg_id"])
    for row in structure_summary.iter_rows(named=True):
        recorded_ids = set(json.loads(row["optimuskg_structure_match_ids"]))
        actual_ids = actual_structure_ids.get(row["matrix_index"], set())
        if recorded_ids != actual_ids:
            raise ValueError(
                "Structure match IDs do not reconcile for matrix_index "
                f"{row['matrix_index']}"
            )
        if row["optimuskg_structure_match_count"] != len(actual_ids):
            raise ValueError(
                "Structure match count does not reconcile for matrix_index "
                f"{row['matrix_index']}"
            )


def classify_evidence(
    name_status: str,
    name_selected_id: str,
    mapping_method: str,
    name_candidate_ids: set[str],
    structure_match_ids: set[str],
) -> tuple[str, str, str, str]:
    """Return proposed ID, evidence class, confidence, and final status."""

    intersection = name_candidate_ids & structure_match_ids
    if name_status == "resolved":
        if name_selected_id in structure_match_ids:
            return (
                name_selected_id,
                "confirmed_name_structure",
                "very_high",
                "resolved",
            )
        if structure_match_ids:
            return "", "conflicting_evidence", "review_required", "review_required"
        if mapping_method == "canonical_name_disambiguation":
            return (
                name_selected_id,
                "supported_canonical_name",
                "high",
                "resolved",
            )
        return name_selected_id, "name_only_resolved", "high", "resolved"

    if name_status == "ambiguous":
        # Rule C is more specific than Rule B, so it is evaluated first.
        if len(structure_match_ids) == 1 and len(intersection) == 1:
            selected = next(iter(intersection))
            return (
                selected,
                "resolved_by_unique_structure_candidate",
                "very_high",
                "resolved",
            )
        if len(intersection) == 1:
            selected = next(iter(intersection))
            return (
                selected,
                "resolved_by_unique_intersection",
                "very_high",
                "resolved",
            )
        return "", "unresolved_ambiguous", "review_required", "unresolved"

    if name_status == "unmatched":
        if len(structure_match_ids) == 1:
            selected = next(iter(structure_match_ids))
            return (
                selected,
                "structure_only_candidate",
                "review_required",
                "review_required",
            )
        if structure_match_ids:
            return "", "unresolved_ambiguous", "review_required", "unresolved"
        return "", "unresolved_unmatched", "unresolved", "unresolved"

    raise ValueError(f"Unexpected name resolution status: {name_status!r}")


def fuse_evidence(
    resolved: pl.DataFrame,
    candidates: pl.DataFrame,
    structure_matches: pl.DataFrame,
    structure_summary: pl.DataFrame,
) -> pl.DataFrame:
    name_candidates: dict[int, set[str]] = defaultdict(set)
    for row in candidates.iter_rows(named=True):
        name_candidates[row["matrix_index"]].add(row["optimuskg_id"])

    structure_ids: dict[int, set[str]] = defaultdict(set)
    previous_name_flags: dict[tuple[int, str], bool] = {}
    for row in structure_matches.iter_rows(named=True):
        matrix_index = row["matrix_index"]
        node_id = row["optimuskg_id"]
        structure_ids[matrix_index].add(node_id)
        previous_name_flags[(matrix_index, node_id)] = row[
            "was_previous_name_candidate"
        ]

    structure_by_index = {
        row["matrix_index"]: row for row in structure_summary.iter_rows(named=True)
    }
    rows: list[dict[str, Any]] = []
    for drug in resolved.iter_rows(named=True):
        matrix_index = drug["matrix_index"]
        candidate_ids = name_candidates.get(matrix_index, set())
        match_ids = structure_ids.get(matrix_index, set())
        intersection_ids = candidate_ids & match_ids
        proposed_id, evidence_class, confidence, final_status = classify_evidence(
            drug["resolution_status"],
            drug["optimuskg_id"] or "",
            drug["mapping_method"] or "",
            candidate_ids,
            match_ids,
        )

        if evidence_class == "resolved_by_unique_structure_candidate":
            if not previous_name_flags.get((matrix_index, proposed_id), False):
                raise ValueError(
                    "Unique structure candidate was not marked as a previous name "
                    f"candidate for matrix_index {matrix_index}"
                )

        structure = structure_by_index[matrix_index]
        rows.append(
            {
                "matrix_index": matrix_index,
                "drug_name": drug["drug_name"],
                "stitch_id": drug["stitch_id"],
                "name_resolution_status": drug["resolution_status"],
                "name_selected_id": drug["optimuskg_id"] or "",
                "name_candidate_ids": json_ids(candidate_ids),
                "structure_status": structure["structure_status"],
                "structure_match_ids": json_ids(match_ids),
                "intersection_ids": json_ids(intersection_ids),
                "proposed_optimuskg_id": proposed_id,
                "evidence_class": evidence_class,
                "confidence": confidence,
                "final_status": final_status,
            }
        )
    return pl.DataFrame(rows, schema=OUTPUT_SCHEMA)


def print_drug_evidence(drug_name: str, fused: pl.DataFrame) -> None:
    selected = fused.filter(pl.col("drug_name") == drug_name)
    if selected.is_empty():
        print(f"{drug_name}: NOT PRESENT IN BIOKORF")
        return
    row = selected.row(0, named=True)
    print(f"matrix_index={row['matrix_index']} | {drug_name} | {row['stitch_id']}")
    print(
        f"  name_status={row['name_resolution_status']} | "
        f"name_selected_id={row['name_selected_id'] or '<none>'}"
    )
    print(f"  name_candidate_ids={row['name_candidate_ids']}")
    print(
        f"  structure_status={row['structure_status']} | "
        f"structure_match_ids={row['structure_match_ids']}"
    )
    print(f"  intersection_ids={row['intersection_ids']}")
    print(
        f"  proposed_id={row['proposed_optimuskg_id'] or '<none>'} | "
        f"class={row['evidence_class']} | confidence={row['confidence']} | "
        f"final_status={row['final_status']}"
    )


def print_results(fused: pl.DataFrame, review: pl.DataFrame) -> None:
    class_counts = dict(fused.group_by("evidence_class").len().iter_rows())
    resolved_count = fused.filter(pl.col("final_status") == "resolved").height

    print("Drug mapping evidence fusion summary")
    print(f"Total drugs: {fused.height}")
    for evidence_class in EVIDENCE_CLASSES:
        print(f"{evidence_class}: {class_counts.get(evidence_class, 0)}")
    print(f"Total resolved after evidence fusion: {resolved_count}")
    print(f"Resolved percentage: {100.0 * resolved_count / fused.height:.2f}%")
    print(f"Total requiring manual review: {review.height}")

    print("\nDetailed evidence for requested drugs:")
    for drug_name in DETAIL_DRUGS:
        print_drug_evidence(drug_name, fused)

    print("\nAll remaining review-required or unresolved drugs:")
    if review.is_empty():
        print("None")
    else:
        for row in review.iter_rows(named=True):
            print(
                f"matrix_index={row['matrix_index']} | {row['drug_name']} | "
                f"class={row['evidence_class']} | final_status={row['final_status']} | "
                f"name_selected={row['name_selected_id'] or '<none>'} | "
                f"name_candidates={row['name_candidate_ids']} | "
                f"structure_matches={row['structure_match_ids']} | "
                f"intersection={row['intersection_ids']} | "
                f"proposed={row['proposed_optimuskg_id'] or '<none>'}"
            )

    print(f"\nFused evidence saved to: {FUSED_PATH}")
    print(f"Review-required evidence saved to: {REVIEW_PATH}")
    print("No fuzzy matching, network access, or existing-file modification was used.")


def main() -> None:
    for path, description in (
        (RESOLVED_PATH, "Resolved name mapping file"),
        (CANDIDATES_PATH, "Name candidate file"),
        (STRUCTURE_MATCHES_PATH, "Structure match file"),
        (STRUCTURE_SUMMARY_PATH, "Structure summary file"),
    ):
        require_file(path, description)

    resolved = pl.read_csv(RESOLVED_PATH)
    candidates = pl.read_csv(CANDIDATES_PATH)
    structure_matches = pl.read_csv(STRUCTURE_MATCHES_PATH)
    structure_summary = pl.read_csv(STRUCTURE_SUMMARY_PATH)
    validate_inputs(resolved, candidates, structure_matches, structure_summary)

    fused = fuse_evidence(resolved, candidates, structure_matches, structure_summary)
    review = fused.filter(pl.col("final_status") != "resolved")
    fused.write_csv(FUSED_PATH)
    review.write_csv(REVIEW_PATH)
    print_results(fused, review)


if __name__ == "__main__":
    main()
