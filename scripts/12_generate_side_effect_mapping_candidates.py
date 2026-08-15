"""Generate conservative exact OptimusKG candidates for BioKORF side effects."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIDE_EFFECT_PATH = (
    PROJECT_ROOT / "data_processed" / "mappings" / "side_effect_mapping.csv"
)
PHENOTYPE_PATH = PROJECT_ROOT / "kg" / "optimuskg" / "nodes" / "phenotype.parquet"
CANDIDATES_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "optimuskg"
    / "side_effect_mapping_candidates.csv"
)
SUMMARY_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "optimuskg"
    / "side_effect_mapping_summary.csv"
)
UNMATCHED_PATH = (
    PROJECT_ROOT / "data_processed" / "optimuskg" / "side_effect_unmatched.csv"
)

EXPECTED_SIDE_EFFECT_COUNT = 994
MATCH_FIELDS = (
    "properties.name",
    "properties.exact_synonyms",
    "properties.concept_names",
    "properties.snomed_full_names",
)
MAPPING_STATUSES = (
    "unique_meddra_exact_name",
    "multiple_meddra_exact_name",
    "meddra_exact_with_other_ontology_candidates",
    "unique_other_exact_name",
    "exact_synonym_only",
    "ambiguous_exact",
    "unmatched",
)
DETAIL_SIDE_EFFECTS = (
    "abdominal discomfort",
    "abdominal distension",
    "abdominal pain",
    "nausea",
    "vomiting",
    "headache",
    "dizziness",
    "dry eye",
    "proteinuria",
    "wheezing",
)
WHITESPACE = re.compile(r"\s+")


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found. Expected path: {path}")


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.strip().casefold()
    return WHITESPACE.sub(" ", value)


def json_list(value: Any) -> str:
    if value is None:
        values: list[str] = []
    elif isinstance(value, list):
        values = value
    else:
        values = [str(value)]
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def validate_side_effects(side_effects: pl.DataFrame) -> None:
    required = {"matrix_index", "side_effect_name"}
    if missing := required.difference(side_effects.columns):
        raise ValueError(f"Side-effect mapping is missing columns: {sorted(missing)}")
    if side_effects.height != EXPECTED_SIDE_EFFECT_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SIDE_EFFECT_COUNT} side effects; "
            f"found {side_effects.height}"
        )
    if side_effects["matrix_index"].to_list() != list(
        range(EXPECTED_SIDE_EFFECT_COUNT)
    ):
        raise ValueError("matrix_index must be exactly 0 through 993")
    first_term = side_effects.item(0, "side_effect_name")
    if not isinstance(first_term, str) or normalize_text(first_term) != (
        "abdominal discomfort"
    ):
        raise ValueError(
            "The first side effect must be 'abdominal discomfort'; "
            f"found {first_term!r}"
        )


def ontology_namespace(node_id: str) -> str:
    lowered = node_id.casefold()
    if lowered.startswith("meddra:"):
        return "MEDDRA"
    if node_id.upper().startswith("HP_"):
        return "HPO"
    return "OTHER"


def build_exact_index(
    phenotypes: pl.DataFrame,
) -> dict[str, dict[str, dict[str, Any]]]:
    if missing := {"id", "properties"}.difference(phenotypes.columns):
        raise ValueError(f"Phenotype table is missing columns: {sorted(missing)}")

    index: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in phenotypes.iter_rows(named=True):
        node_id = row["id"]
        properties = row["properties"] or {}
        if node_id is None:
            continue
        values_by_field = {
            "properties.name": [properties.get("name")],
            "properties.exact_synonyms": properties.get("exact_synonyms") or [],
            "properties.concept_names": properties.get("concept_names") or [],
            "properties.snomed_full_names": properties.get("snomed_full_names") or [],
        }
        node = {
            "optimuskg_id": node_id,
            "optimuskg_name": properties.get("name") or "",
            "ontology_namespace": ontology_namespace(node_id),
            "properties.code": properties.get("code") or "",
            "properties.umls_cui": properties.get("umls_cui") or "",
            "properties.concept_ids": json_list(properties.get("concept_ids")),
            "properties.xrefs": json_list(properties.get("xrefs")),
            "properties.snomed_concept_ids": json_list(
                properties.get("snomed_concept_ids")
            ),
        }
        for matched_field, values in values_by_field.items():
            for value in values:
                if not isinstance(value, str):
                    continue
                normalized = normalize_text(value)
                if not normalized:
                    continue
                candidate = index[normalized].setdefault(
                    node_id, {**node, "matched_fields": set()}
                )
                candidate["matched_fields"].add(matched_field)
    return index


def classify_candidates(
    candidates: list[dict[str, Any]],
) -> tuple[str, str]:
    meddra_canonical = [
        candidate
        for candidate in candidates
        if candidate["ontology_namespace"] == "MEDDRA"
        and "properties.name" in candidate["matched_fields"]
    ]
    non_meddra_canonical = [
        candidate
        for candidate in candidates
        if candidate["ontology_namespace"] != "MEDDRA"
        and "properties.name" in candidate["matched_fields"]
    ]
    has_non_meddra_candidate = any(
        candidate["ontology_namespace"] != "MEDDRA" for candidate in candidates
    )

    if len(meddra_canonical) > 1:
        return "multiple_meddra_exact_name", ""
    if len(meddra_canonical) == 1:
        preferred_id = meddra_canonical[0]["optimuskg_id"]
        if has_non_meddra_candidate:
            return "meddra_exact_with_other_ontology_candidates", preferred_id
        return "unique_meddra_exact_name", preferred_id
    if len(non_meddra_canonical) == 1:
        return "unique_other_exact_name", ""
    if not meddra_canonical and not non_meddra_canonical and candidates:
        return "exact_synonym_only", ""
    if candidates:
        return "ambiguous_exact", ""
    return "unmatched", ""


def generate_outputs(
    side_effects: pl.DataFrame,
    exact_index: dict[str, dict[str, dict[str, Any]]],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    candidate_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for side_effect in side_effects.iter_rows(named=True):
        normalized = normalize_text(side_effect["side_effect_name"])
        candidates = sorted(
            exact_index.get(normalized, {}).values(),
            key=lambda candidate: candidate["optimuskg_id"],
        )
        for candidate in candidates:
            candidate_rows.append(
                {
                    "matrix_index": side_effect["matrix_index"],
                    "side_effect_name": side_effect["side_effect_name"],
                    "optimuskg_id": candidate["optimuskg_id"],
                    "optimuskg_name": candidate["optimuskg_name"],
                    "ontology_namespace": candidate["ontology_namespace"],
                    "matched_field": ";".join(
                        field
                        for field in MATCH_FIELDS
                        if field in candidate["matched_fields"]
                    ),
                    "properties.code": candidate["properties.code"],
                    "properties.umls_cui": candidate["properties.umls_cui"],
                    "properties.concept_ids": candidate["properties.concept_ids"],
                    "properties.xrefs": candidate["properties.xrefs"],
                    "properties.snomed_concept_ids": candidate[
                        "properties.snomed_concept_ids"
                    ],
                }
            )

        mapping_status, preferred_meddra_id = classify_candidates(candidates)
        summary_rows.append(
            {
                "matrix_index": side_effect["matrix_index"],
                "side_effect_name": side_effect["side_effect_name"],
                "candidate_count": len(candidates),
                "meddra_candidate_count": sum(
                    candidate["ontology_namespace"] == "MEDDRA"
                    for candidate in candidates
                ),
                "hpo_candidate_count": sum(
                    candidate["ontology_namespace"] == "HPO"
                    for candidate in candidates
                ),
                "preferred_meddra_id": preferred_meddra_id,
                "candidate_ids": json.dumps(
                    [candidate["optimuskg_id"] for candidate in candidates],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "mapping_status": mapping_status,
            }
        )

    candidate_schema = {
        "matrix_index": pl.Int64,
        "side_effect_name": pl.String,
        "optimuskg_id": pl.String,
        "optimuskg_name": pl.String,
        "ontology_namespace": pl.String,
        "matched_field": pl.String,
        "properties.code": pl.String,
        "properties.umls_cui": pl.String,
        "properties.concept_ids": pl.String,
        "properties.xrefs": pl.String,
        "properties.snomed_concept_ids": pl.String,
    }
    summary_schema = {
        "matrix_index": pl.Int64,
        "side_effect_name": pl.String,
        "candidate_count": pl.Int64,
        "meddra_candidate_count": pl.Int64,
        "hpo_candidate_count": pl.Int64,
        "preferred_meddra_id": pl.String,
        "candidate_ids": pl.String,
        "mapping_status": pl.String,
    }
    return (
        pl.DataFrame(candidate_rows, schema=candidate_schema),
        pl.DataFrame(summary_rows, schema=summary_schema),
    )


def print_detail(
    side_effect_name: str, candidates: pl.DataFrame, summary: pl.DataFrame
) -> None:
    summary_row = summary.filter(pl.col("side_effect_name") == side_effect_name)
    if summary_row.is_empty():
        print(f"{side_effect_name}: NOT PRESENT IN BIOKORF")
        return
    row = summary_row.row(0, named=True)
    print(
        f"matrix_index={row['matrix_index']} | {side_effect_name} | "
        f"status={row['mapping_status']} | candidates={row['candidate_count']} | "
        f"MedDRA={row['meddra_candidate_count']} | HPO={row['hpo_candidate_count']} | "
        f"preferred_meddra_id={row['preferred_meddra_id'] or '<none>'}"
    )
    selected = candidates.filter(pl.col("matrix_index") == row["matrix_index"])
    if selected.is_empty():
        print("  No high-confidence candidates")
    for candidate in selected.iter_rows(named=True):
        print(
            f"  {candidate['optimuskg_id']} | {candidate['optimuskg_name']} | "
            f"namespace={candidate['ontology_namespace']} | "
            f"matched_field={candidate['matched_field']}"
        )


def print_results(candidates: pl.DataFrame, summary: pl.DataFrame) -> None:
    counts = dict(summary.group_by("mapping_status").len().iter_rows())
    with_candidate = summary.filter(pl.col("candidate_count") >= 1).height
    with_meddra_canonical = summary.filter(
        pl.col("mapping_status").is_in(
            [
                "unique_meddra_exact_name",
                "multiple_meddra_exact_name",
                "meddra_exact_with_other_ontology_candidates",
            ]
        )
    ).height

    print("Side-effect exact candidate mapping summary")
    print(f"Total side effects: {summary.height}")
    for status in MAPPING_STATUSES:
        print(f"{status}: {counts.get(status, 0)}")
    print(
        "Percentage having at least one high-confidence candidate: "
        f"{100.0 * with_candidate / summary.height:.2f}%"
    )
    print(
        "Percentage having a MedDRA canonical-name candidate: "
        f"{100.0 * with_meddra_canonical / summary.height:.2f}%"
    )

    print("\nDetailed results for requested side effects:")
    for side_effect_name in DETAIL_SIDE_EFFECTS:
        print_detail(side_effect_name, candidates, summary)

    unmatched = summary.filter(pl.col("mapping_status") == "unmatched")
    print("\nAll unmatched side effects:")
    if unmatched.is_empty():
        print("None")
    else:
        for row in unmatched.iter_rows(named=True):
            print(f"matrix_index={row['matrix_index']} | {row['side_effect_name']}")

    print(f"\nCandidate rows saved to: {CANDIDATES_PATH}")
    print(f"Summary rows saved to: {SUMMARY_PATH}")
    print(f"Unmatched rows saved to: {UNMATCHED_PATH}")
    print("No fuzzy matching or final anchor mapping was performed.")


def main() -> None:
    require_file(SIDE_EFFECT_PATH, "BioKORF side-effect mapping")
    require_file(PHENOTYPE_PATH, "OptimusKG phenotype node table")
    side_effects = pl.read_csv(SIDE_EFFECT_PATH)
    phenotypes = pl.read_parquet(PHENOTYPE_PATH)
    validate_side_effects(side_effects)

    exact_index = build_exact_index(phenotypes)
    candidates, summary = generate_outputs(side_effects, exact_index)
    unmatched = summary.filter(pl.col("mapping_status") == "unmatched")
    candidates.write_csv(CANDIDATES_PATH)
    summary.write_csv(SUMMARY_PATH)
    unmatched.write_csv(UNMATCHED_PATH)
    print_results(candidates, summary)


if __name__ == "__main__":
    main()
