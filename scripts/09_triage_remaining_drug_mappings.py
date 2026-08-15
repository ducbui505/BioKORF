"""Conservatively triage BioKORF drug mappings that remain unresolved."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FUSED_PATH = (
    PROJECT_ROOT / "data_processed" / "optimuskg" / "drug_mapping_evidence_fused.csv"
)
REVIEW_INPUT_PATH = (
    PROJECT_ROOT / "data_processed" / "optimuskg" / "drug_mapping_review_required.csv"
)
NAME_CANDIDATES_PATH = (
    PROJECT_ROOT / "data_processed" / "optimuskg" / "drug_mapping_candidates.csv"
)
STRUCTURE_MATCHES_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "optimuskg"
    / "drug_stitch_inchikey_matches.csv"
)
IDENTITY_CANDIDATES_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "optimuskg"
    / "ambiguous_drug_identity_candidates.csv"
)
DRUG_NODE_PATH = PROJECT_ROOT / "kg" / "optimuskg" / "nodes" / "drug.parquet"
TRIAGE_PATH = (
    PROJECT_ROOT / "data_processed" / "optimuskg" / "drug_mapping_triage.csv"
)
MANUAL_REVIEW_PATH = (
    PROJECT_ROOT / "data_processed" / "optimuskg" / "drug_mapping_manual_review.csv"
)

EXPECTED_REVIEW_COUNT = 89
TRIAGE_CLASSES = (
    "SAFE_MULTI_NODE",
    "SINGLE_HIGH_CONFIDENCE",
    "FORM_OR_SALT_AMBIGUITY",
    "BIOLOGIC_OR_PEPTIDE",
    "COMBINATION_DRUG",
    "TRUE_CONFLICT",
    "NO_OPTIMUSKG_NODE",
)
MANUAL_CLASSES = {
    "FORM_OR_SALT_AMBIGUITY",
    "TRUE_CONFLICT",
    "NO_OPTIMUSKG_NODE",
}
BIOLOGIC_TYPES = {"Antibody", "Enzyme", "Oligosaccharide", "Protein"}
BIOLOGIC_TERMS = (
    "albumin",
    "insulin",
    "interferon",
    "antibody",
    "immunoglobulin",
    "peptide",
    "heparin",
    "corticotropin",
    "romiplostim",
    "ecallantide",
    "exenatide",
    "nesiritide",
    "mipomersen",
    "ustekinumab",
    "ofatumumab",
    "liraglutide",
    "tesamorelin",
    "teduglutide",
)
GENERIC_FORM_TOKENS = {
    "acid",
    "acetate",
    "cation",
    "chloride",
    "dichloride",
    "hydrochloride",
    "hydrate",
    "ion",
    "maleate",
    "phosphate",
    "sodium",
    "sulfate",
    "tartrate",
}
DETAIL_DRUGS = (
    "albumin",
    "lepirudin",
    "insulin.lispro",
    "daptomycin",
    "reboxetine",
    "sulfamethoxazole.and.trimethoprim",
    "gadoversetamide",
    "technetium.(99mTc).tetrofosmin",
    "iotrolan",
    "sodium.bicarbonate",
    "sodium.phosphate",
)
NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found. Expected path: {path}")


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().replace(".", " ")
    return " ".join(NON_ALPHANUMERIC.sub(" ", value).split())


def json_list(values: list[str] | set[str]) -> str:
    return json.dumps(sorted(set(values)), ensure_ascii=False, separators=(",", ":"))


def parse_json_ids(value: str | None) -> set[str]:
    return set(json.loads(value)) if value else set()


def validate_inputs(fused: pl.DataFrame, review: pl.DataFrame) -> None:
    required = {
        "matrix_index",
        "drug_name",
        "stitch_id",
        "name_resolution_status",
        "name_selected_id",
        "name_candidate_ids",
        "structure_status",
        "structure_match_ids",
        "intersection_ids",
        "evidence_class",
        "final_status",
    }
    if missing := required.difference(fused.columns):
        raise ValueError(f"Fused evidence is missing columns: {sorted(missing)}")
    if fused.height != 757:
        raise ValueError(f"Fused evidence must contain 757 rows; found {fused.height}")
    expected_review = fused.filter(pl.col("final_status") != "resolved")
    if not review.equals(expected_review):
        raise ValueError("Review-required input is not the unresolved subset of fused evidence")
    if review.height != EXPECTED_REVIEW_COUNT:
        print(
            f"WARNING: expected {EXPECTED_REVIEW_COUNT} review-required drugs; "
            f"found {review.height}"
        )


def base_candidate(node_id: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "optimuskg_id": node_id,
        "optimuskg_name": properties.get("name") or "",
        "drug_type": properties.get("type") or "",
        "matched_fields": set(),
        "inchi_key": properties.get("inchi_key") or "",
        "canonical_smiles": properties.get("canonical_smiles") or "",
        "source_ids": properties.get("source_ids") or [],
        "accession_numbers": properties.get("accession_numbers") or [],
        "evidence_sources": set(),
    }


def merge_candidate(
    target: dict[str, Any], source: dict[str, Any], evidence_source: str
) -> None:
    target["evidence_sources"].add(evidence_source)
    matched_field = source.get("matched_field")
    if isinstance(matched_field, str):
        target["matched_fields"].update(
            field for field in matched_field.split(";") if field
        )
    for field in (
        "optimuskg_name",
        "inchi_key",
        "canonical_smiles",
        "source_ids",
        "accession_numbers",
    ):
        value = source.get(field)
        if value not in (None, "", "[]", []):
            if field in {"source_ids", "accession_numbers"} and isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    value = [value]
            target[field] = value


def collect_candidates(
    fused_row: dict[str, Any],
    name_candidates: list[dict[str, Any]],
    identity_candidates: list[dict[str, Any]],
    structure_matches: list[dict[str, Any]],
    kg_nodes: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    candidate_ids = (
        parse_json_ids(fused_row["name_candidate_ids"])
        | parse_json_ids(fused_row["structure_match_ids"])
    )
    if fused_row.get("name_selected_id"):
        candidate_ids.add(fused_row["name_selected_id"])
    candidate_ids.update(row["optimuskg_id"] for row in identity_candidates)

    candidates: dict[str, dict[str, Any]] = {}
    for node_id in candidate_ids:
        properties = kg_nodes.get(node_id, {})
        candidates[node_id] = base_candidate(node_id, properties)
    for rows, source_name in (
        (name_candidates, "name_candidate"),
        (identity_candidates, "identity_analysis"),
        (structure_matches, "stitch_structure"),
    ):
        for row in rows:
            node_id = row["optimuskg_id"]
            if node_id not in candidates:
                candidates[node_id] = base_candidate(node_id, kg_nodes.get(node_id, {}))
            merge_candidate(candidates[node_id], row, source_name)
    if fused_row.get("name_selected_id"):
        candidates[fused_row["name_selected_id"]]["evidence_sources"].add(
            "selected_name_mapping"
        )
    return sorted(candidates.values(), key=lambda row: row["optimuskg_id"])


def significant_tokens(value: str) -> set[str]:
    return {
        token
        for token in normalize_name(value).split()
        if len(token) >= 4 and token not in GENERIC_FORM_TOKENS
    }


def names_are_related(drug_name: str, candidates: list[dict[str, Any]]) -> bool:
    drug_normalized = normalize_name(drug_name)
    drug_tokens = significant_tokens(drug_name)
    for candidate in candidates:
        candidate_name = candidate["optimuskg_name"]
        candidate_normalized = normalize_name(candidate_name)
        if not candidate_normalized:
            continue
        if drug_normalized in candidate_normalized or candidate_normalized in drug_normalized:
            return True
        if drug_tokens & significant_tokens(candidate_name):
            return True
    return False


def is_combination(drug_name: str) -> bool:
    normalized = normalize_name(drug_name)
    return " and " in f" {normalized} "


def is_biologic(drug_name: str, candidates: list[dict[str, Any]]) -> bool:
    normalized = normalize_name(drug_name)
    if any(term in normalized for term in BIOLOGIC_TERMS):
        return True
    return any(candidate["drug_type"] in BIOLOGIC_TYPES for candidate in candidates)


def structure_state(candidates: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    inchikeys = {
        candidate["inchi_key"].strip().upper()
        for candidate in candidates
        if isinstance(candidate["inchi_key"], str) and candidate["inchi_key"].strip()
    }
    smiles = {
        candidate["canonical_smiles"].strip()
        for candidate in candidates
        if isinstance(candidate["canonical_smiles"], str)
        and candidate["canonical_smiles"].strip()
    }
    return inchikeys, smiles


def triage_drug(
    fused_row: dict[str, Any], candidates: list[dict[str, Any]]
) -> tuple[str, str, bool]:
    drug_name = fused_row["drug_name"]
    evidence_class = fused_row["evidence_class"]
    identity_classes = {
        row.get("identity_class")
        for row in candidates
        if row.get("identity_class")
    }
    inchikeys, smiles = structure_state(candidates)
    related_names = names_are_related(drug_name, candidates)
    structure_ids = parse_json_ids(fused_row["structure_match_ids"])
    structure_candidates = [
        candidate
        for candidate in candidates
        if candidate["optimuskg_id"] in structure_ids
    ]
    related_structure_names = names_are_related(drug_name, structure_candidates)

    if is_combination(drug_name):
        return (
            "COMBINATION_DRUG",
            "Retain supported combination nodes; do not collapse to one ingredient.",
            False,
        )
    if not candidates:
        return (
            "NO_OPTIMUSKG_NODE",
            "Keep unmapped and perform later curated/manual ontology review.",
            True,
        )
    if evidence_class == "conflicting_evidence":
        if related_structure_names:
            return (
                "FORM_OR_SALT_AMBIGUITY",
                "Review parent/form relationship; do not collapse chemical forms.",
                True,
            )
        return (
            "TRUE_CONFLICT",
            "Manually adjudicate discordant name and structure identities.",
            True,
        )
    if len(candidates) == 1:
        return (
            "SINGLE_HIGH_CONFIDENCE",
            "Retain the single locally supported OptimusKG node.",
            False,
        )
    if (
        "equivalent_by_inchikey" in identity_classes
        or "equivalent_by_smiles" in identity_classes
    ):
        return (
            "SAFE_MULTI_NODE",
            "Retain all equivalent cross-namespace nodes without selecting a namespace.",
            False,
        )
    if is_biologic(drug_name, candidates) and len(inchikeys) <= 1 and len(smiles) <= 1:
        return (
            "BIOLOGIC_OR_PEPTIDE",
            "Retain supported biologic nodes; structure absence is not disqualifying.",
            False,
        )
    if len(inchikeys) >= 2 or len(smiles) >= 2 or "mixed_identity" in identity_classes:
        if related_names:
            return (
                "FORM_OR_SALT_AMBIGUITY",
                "Review salt, form, stereochemistry, isotope, or formulation differences.",
                True,
            )
        return (
            "TRUE_CONFLICT",
            "Manually adjudicate candidates with different identities.",
            True,
        )
    normalized_names = {
        normalize_name(candidate["optimuskg_name"])
        for candidate in candidates
        if candidate["optimuskg_name"]
    }
    if len(normalized_names) == 1 or related_names:
        return (
            "SAFE_MULTI_NODE",
            "Retain all compatible cross-namespace nodes; no chemical conflict is present.",
            False,
        )
    if is_biologic(drug_name, candidates):
        return (
            "BIOLOGIC_OR_PEPTIDE",
            "Retain biologic candidates and review with biologic-specific identifiers later.",
            False,
        )
    return (
        "TRUE_CONFLICT",
        "Evidence is insufficiently consistent; retain for manual adjudication.",
        True,
    )


def candidate_for_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "optimuskg_id": candidate["optimuskg_id"],
        "name": candidate["optimuskg_name"],
        "type": candidate["drug_type"],
        "matched_fields": sorted(candidate["matched_fields"]),
        "inchi_key": candidate["inchi_key"],
        "canonical_smiles": candidate["canonical_smiles"],
        "source_ids": candidate["source_ids"],
        "accession_numbers": candidate["accession_numbers"],
        "evidence_sources": sorted(candidate["evidence_sources"]),
    }


def build_triage(
    review: pl.DataFrame,
    fused: pl.DataFrame,
    name_candidates: pl.DataFrame,
    identity_candidates: pl.DataFrame,
    structure_matches: pl.DataFrame,
    kg_nodes: dict[str, dict[str, Any]],
) -> tuple[pl.DataFrame, dict[int, list[dict[str, Any]]]]:
    name_by_index: dict[int, list[dict[str, Any]]] = defaultdict(list)
    identity_by_index: dict[int, list[dict[str, Any]]] = defaultdict(list)
    structure_by_index: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in name_candidates.iter_rows(named=True):
        name_by_index[row["matrix_index"]].append(row)
    for row in identity_candidates.iter_rows(named=True):
        identity_by_index[row["matrix_index"]].append(row)
    for row in structure_matches.iter_rows(named=True):
        structure_by_index[row["matrix_index"]].append(row)

    rows: list[dict[str, Any]] = []
    candidates_by_index: dict[int, list[dict[str, Any]]] = {}
    for review_row in review.iter_rows(named=True):
        matrix_index = review_row["matrix_index"]
        candidates = collect_candidates(
            review_row,
            name_by_index.get(matrix_index, []),
            identity_by_index.get(matrix_index, []),
            structure_by_index.get(matrix_index, []),
            kg_nodes,
        )
        for candidate in candidates:
            identity_rows = identity_by_index.get(matrix_index, [])
            candidate["identity_class"] = next(
                (
                    row["identity_class"]
                    for row in identity_rows
                    if row["optimuskg_id"] == candidate["optimuskg_id"]
                ),
                "",
            )
        triage_class, action, manual_review = triage_drug(review_row, candidates)
        candidates_by_index[matrix_index] = candidates
        evidence_summary = {
            "fused_evidence_class": review_row["evidence_class"],
            "name_resolution_status": review_row["name_resolution_status"],
            "name_selected_id": review_row["name_selected_id"] or "",
            "name_candidate_ids": sorted(parse_json_ids(review_row["name_candidate_ids"])),
            "structure_status": review_row["structure_status"],
            "structure_match_ids": sorted(
                parse_json_ids(review_row["structure_match_ids"])
            ),
            "intersection_ids": sorted(parse_json_ids(review_row["intersection_ids"])),
            "candidates": [candidate_for_summary(candidate) for candidate in candidates],
        }
        rows.append(
            {
                "matrix_index": matrix_index,
                "drug_name": review_row["drug_name"],
                "stitch_id": review_row["stitch_id"],
                "triage_class": triage_class,
                "supported_optimuskg_ids": json_list(
                    [candidate["optimuskg_id"] for candidate in candidates]
                ),
                "supported_names": json_list(
                    [candidate["optimuskg_name"] for candidate in candidates]
                ),
                "candidate_count": len(candidates),
                "evidence_summary": json.dumps(
                    evidence_summary, ensure_ascii=False, separators=(",", ":")
                ),
                "recommended_action": action,
                "requires_manual_review": manual_review,
            }
        )

    schema = {
        "matrix_index": pl.Int64,
        "drug_name": pl.String,
        "stitch_id": pl.String,
        "triage_class": pl.String,
        "supported_optimuskg_ids": pl.String,
        "supported_names": pl.String,
        "candidate_count": pl.Int64,
        "evidence_summary": pl.String,
        "recommended_action": pl.String,
        "requires_manual_review": pl.Boolean,
    }
    return pl.DataFrame(rows, schema=schema), candidates_by_index


def print_detail(
    drug_name: str,
    triage: pl.DataFrame,
    fused: pl.DataFrame,
    kg_nodes: dict[str, dict[str, Any]],
) -> None:
    row = triage.filter(pl.col("drug_name") == drug_name)
    if row.is_empty():
        fused_row = fused.filter(pl.col("drug_name") == drug_name)
        if fused_row.is_empty():
            print(f"{drug_name}: NOT PRESENT IN BIOKORF")
        else:
            current = fused_row.row(0, named=True)
            print(
                f"{drug_name}: not in remaining 89; final_status="
                f"{current['final_status']}, evidence_class={current['evidence_class']}, "
                f"proposed_id={current['proposed_optimuskg_id'] or '<none>'}"
            )
        return
    record = row.row(0, named=True)
    print(
        f"matrix_index={record['matrix_index']} | {drug_name} | {record['stitch_id']} | "
        f"class={record['triage_class']} | candidates={record['candidate_count']} | "
        f"manual_review={record['requires_manual_review']}"
    )
    print(f"  supported_ids={record['supported_optimuskg_ids']}")
    print(f"  supported_names={record['supported_names']}")
    print(f"  recommended_action={record['recommended_action']}")
    print(f"  evidence_summary={record['evidence_summary']}")


def print_results(
    triage: pl.DataFrame,
    manual: pl.DataFrame,
    fused: pl.DataFrame,
    kg_nodes: dict[str, dict[str, Any]],
) -> None:
    counts = dict(triage.group_by("triage_class").len().iter_rows())
    print("Remaining drug mapping triage summary")
    print(f"Total triaged drugs: {triage.height}")
    for triage_class in TRIAGE_CLASSES:
        print(f"{triage_class}: {counts.get(triage_class, 0)}")
    print(f"Manual-review rows: {manual.height}")

    print("\nFull details for requested drugs:")
    for drug_name in DETAIL_DRUGS:
        print_detail(drug_name, triage, fused, kg_nodes)

    print(f"\nTriage output saved to: {TRIAGE_PATH}")
    print(f"Manual review output saved to: {MANUAL_REVIEW_PATH}")
    print("No node was arbitrarily selected and no fuzzy matching or network call was used.")


def main() -> None:
    for path, description in (
        (FUSED_PATH, "Fused mapping evidence"),
        (REVIEW_INPUT_PATH, "Review-required mapping evidence"),
        (NAME_CANDIDATES_PATH, "Name candidates"),
        (STRUCTURE_MATCHES_PATH, "STITCH structure matches"),
        (IDENTITY_CANDIDATES_PATH, "Ambiguous identity candidates"),
        (DRUG_NODE_PATH, "OptimusKG drug node table"),
    ):
        require_file(path, description)

    fused = pl.read_csv(FUSED_PATH)
    review = pl.read_csv(REVIEW_INPUT_PATH)
    name_candidates = pl.read_csv(NAME_CANDIDATES_PATH)
    structure_matches = pl.read_csv(STRUCTURE_MATCHES_PATH)
    identity_candidates = pl.read_csv(IDENTITY_CANDIDATES_PATH)
    optimuskg_drugs = pl.read_parquet(DRUG_NODE_PATH)
    validate_inputs(fused, review)
    kg_nodes = {
        row["id"]: row["properties"] or {}
        for row in optimuskg_drugs.iter_rows(named=True)
        if row["id"] is not None
    }

    triage, _ = build_triage(
        review,
        fused,
        name_candidates,
        identity_candidates,
        structure_matches,
        kg_nodes,
    )
    manual = triage.filter(pl.col("requires_manual_review"))
    triage.write_csv(TRIAGE_PATH)
    manual.write_csv(MANUAL_REVIEW_PATH)
    print_results(triage, manual, fused, kg_nodes)


if __name__ == "__main__":
    main()
