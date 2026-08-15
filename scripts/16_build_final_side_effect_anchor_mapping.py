"""Build stable BioKORF anchors for all 994 side effects."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIDE_EFFECT_PATH = PROJECT_ROOT / "data_processed" / "mappings" / "side_effect_mapping.csv"
NAME_SUMMARY_PATH = PROJECT_ROOT / "data_processed" / "optimuskg" / "side_effect_mapping_summary.csv"
UMLS_SUMMARY_PATH = PROJECT_ROOT / "data_processed" / "optimuskg" / "side_effect_umls_mapping_summary.csv"
MEDDRA_MAPPING_PATH = PROJECT_ROOT / "data_processed" / "optimuskg" / "side_effect_meddra_mapping.csv"
PHENOTYPE_PATH = PROJECT_ROOT / "kg" / "optimuskg" / "nodes" / "phenotype.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data_processed" / "optimuskg" / "final_side_effect_anchor_mapping.csv"

EXPECTED_COUNT = 994
ALLOWED_STATUSES = (
    "canonical_meddra_with_kg", "meddra_with_alias_kg", "external_meddra_only",
    "exact_other_ontology", "no_kg_evidence",
)
SAFE_STEP12_STATUSES = {"unique_other_exact_name", "exact_synonym_only"}
DETAIL_TERMS = (
    "abdominal discomfort", "abdominal distension", "abdominal pain", "nausea",
    "vomiting", "headache", "dizziness", "dry eye", "proteinuria", "wheezing",
)
OUTPUT_COLUMNS = (
    "matrix_index", "biokorf_side_id", "side_effect_name", "sider_umls_cui",
    "meddra_id", "meddra_name", "canonical_optimuskg_id", "alias_optimuskg_ids",
    "optimuskg_node_count", "identity_source", "mapping_confidence",
    "kg_mapping_status", "has_kg_node", "requires_review",
)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_json_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON list, found {value!r}")
    return [str(item) for item in parsed if item is not None and str(item).strip()]


def json_ids(values: list[str] | set[str]) -> str:
    return json.dumps(sorted(set(values)), ensure_ascii=False, separators=(",", ":"))


def validate_aligned_inputs(named_inputs: dict[str, list[dict[str, str]]]) -> None:
    for label, rows in named_inputs.items():
        if len(rows) != EXPECTED_COUNT:
            raise ValueError(f"Expected {EXPECTED_COUNT} rows in {label}; found {len(rows)}")
        if [int(row["matrix_index"]) for row in rows] != list(range(EXPECTED_COUNT)):
            raise ValueError(f"matrix_index in {label} must be exactly 0 through 993")
    for index in range(EXPECTED_COUNT):
        names = {rows[index]["side_effect_name"] for rows in named_inputs.values()}
        if len(names) != 1:
            raise ValueError(f"Side-effect name mismatch at matrix_index {index}: {names}")


def build_anchor_row(
    index: int,
    source: dict[str, str],
    name_summary: dict[str, str],
    meddra: dict[str, str],
) -> dict[str, Any]:
    canonical_id = meddra.get("canonical_optimuskg_id", "").strip()
    aliases = set(parse_json_list(meddra.get("alias_optimuskg_ids", "[]")))
    meddra_id = meddra.get("meddra_id", "").strip()
    meddra_name = meddra.get("meddra_name", "").strip()
    identity_source = meddra.get("mapping_method", "").strip()

    if meddra_id:
        if canonical_id and aliases:
            kg_status = "meddra_with_alias_kg"
        elif canonical_id:
            kg_status = "canonical_meddra_with_kg"
        elif aliases:
            kg_status = "meddra_with_alias_kg"
        else:
            kg_status = "external_meddra_only"
        confidence = "high"
    else:
        candidates = parse_json_list(name_summary.get("candidate_ids", "[]"))
        step12_status = name_summary.get("mapping_status", "")
        if step12_status in SAFE_STEP12_STATUSES and len(candidates) == 1:
            candidate = candidates[0]
            if candidate.casefold().startswith("meddra:"):
                raise ValueError(
                    f"Step 12 fallback at matrix_index {index} unexpectedly selected "
                    f"a MedDRA node without MedDRA identity: {candidate}"
                )
            canonical_id = candidate
            kg_status = "exact_other_ontology"
            identity_source = (
                "step12_unique_exact_name"
                if step12_status == "unique_other_exact_name"
                else "step12_unique_exact_synonym"
            )
            confidence = "high"
        else:
            canonical_id = ""
            aliases.clear()
            kg_status = "no_kg_evidence"
            identity_source = "none"
            confidence = "none"

    all_ids = ({canonical_id} if canonical_id else set()) | aliases
    requires_review = kg_status in {"external_meddra_only", "no_kg_evidence"}
    return {
        "matrix_index": index,
        "biokorf_side_id": f"BIOKORF_SIDE_{index:03d}",
        "side_effect_name": source["side_effect_name"],
        "sider_umls_cui": meddra.get("sider_umls_cui", "").strip(),
        "meddra_id": meddra_id,
        "meddra_name": meddra_name,
        "canonical_optimuskg_id": canonical_id,
        "alias_optimuskg_ids": json_ids(aliases),
        "optimuskg_node_count": len(all_ids),
        "identity_source": identity_source,
        "mapping_confidence": confidence,
        "kg_mapping_status": kg_status,
        "has_kg_node": bool(all_ids),
        "requires_review": requires_review,
    }


def validate_final(rows: list[dict[str, Any]], phenotype_ids: set[str]) -> None:
    if len(rows) != EXPECTED_COUNT:
        raise ValueError(f"Expected {EXPECTED_COUNT} final anchors; found {len(rows)}")
    if [row["matrix_index"] for row in rows] != list(range(EXPECTED_COUNT)):
        raise ValueError("Final matrix_index must be exactly 0 through 993")
    anchor_ids = [row["biokorf_side_id"] for row in rows]
    if len(set(anchor_ids)) != EXPECTED_COUNT:
        raise ValueError("BIOKORF side-effect anchor IDs must be unique")
    if rows[0]["biokorf_side_id"] != "BIOKORF_SIDE_000" or rows[0]["side_effect_name"].casefold() != "abdominal discomfort":
        raise ValueError("BIOKORF_SIDE_000 must correspond to abdominal discomfort")
    if any(row["kg_mapping_status"] not in ALLOWED_STATUSES for row in rows):
        raise ValueError("An invalid kg_mapping_status was generated")
    for row in rows:
        ids = set(parse_json_list(row["alias_optimuskg_ids"]))
        if row["canonical_optimuskg_id"]:
            ids.add(row["canonical_optimuskg_id"])
        missing = ids.difference(phenotype_ids)
        if missing:
            raise ValueError(
                f"Anchor {row['biokorf_side_id']} contains missing OptimusKG IDs: {sorted(missing)}"
            )
        if len(ids) != row["optimuskg_node_count"]:
            raise ValueError(f"Node count mismatch for {row['biokorf_side_id']}")
        if bool(ids) != row["has_kg_node"]:
            raise ValueError(f"has_kg_node mismatch for {row['biokorf_side_id']}")


def print_detail(row: dict[str, Any]) -> None:
    for column in OUTPUT_COLUMNS:
        print(f"  {column}: {row[column]}")


def main() -> None:
    side_effects = read_csv_rows(SIDE_EFFECT_PATH)
    name_summary = read_csv_rows(NAME_SUMMARY_PATH)
    umls_summary = read_csv_rows(UMLS_SUMMARY_PATH)
    meddra_mapping = read_csv_rows(MEDDRA_MAPPING_PATH)
    validate_aligned_inputs({
        "side-effect mapping": side_effects,
        "Step 12 name summary": name_summary,
        "Step 14 UMLS summary": umls_summary,
        "Step 15 MedDRA mapping": meddra_mapping,
    })
    if not PHENOTYPE_PATH.is_file():
        raise FileNotFoundError(f"Required file not found: {PHENOTYPE_PATH}")
    phenotype_ids = set(
        pl.read_parquet(PHENOTYPE_PATH, columns=["id"])["id"].drop_nulls().to_list()
    )

    rows = [
        build_anchor_row(index, side_effects[index], name_summary[index], meddra_mapping[index])
        for index in range(EXPECTED_COUNT)
    ]
    validate_final(rows, phenotype_ids)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["kg_mapping_status"] for row in rows)
    with_kg = sum(row["has_kg_node"] for row in rows)
    with_meddra = sum(bool(row["meddra_id"]) for row in rows)
    review_count = sum(row["requires_review"] for row in rows)
    print(f"Total side-effect anchors: {len(rows)}")
    for status in ALLOWED_STATUSES:
        print(f"{status}: {counts[status]}")
    print(f"Anchors with at least one OptimusKG node: {with_kg}")
    print(f"KG node coverage percentage: {with_kg / EXPECTED_COUNT * 100:.2f}%")
    print(f"Anchors with MedDRA identity: {with_meddra}")
    print(f"MedDRA identity coverage percentage: {with_meddra / EXPECTED_COUNT * 100:.2f}%")
    print(f"requires_review count: {review_count}")

    by_name = {row["side_effect_name"].casefold(): row for row in rows}
    print("\nDetailed requested side effects")
    for term in DETAIL_TERMS:
        print(term)
        print_detail(by_name[term.casefold()])
    print(f"\nFinal anchor mapping: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
