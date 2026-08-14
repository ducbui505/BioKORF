"""Analyze ambiguous drug candidates using structure identity evidence only."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNRESOLVED_PATH = (
    PROJECT_ROOT / "data_processed" / "optimuskg" / "drug_mapping_unresolved.csv"
)
RESOLVED_PATH = (
    PROJECT_ROOT / "data_processed" / "optimuskg" / "drug_mapping_resolved.csv"
)
CANDIDATES_PATH = (
    PROJECT_ROOT / "data_processed" / "optimuskg" / "drug_mapping_candidates.csv"
)
DRUG_NODE_PATH = PROJECT_ROOT / "kg" / "optimuskg" / "nodes" / "drug.parquet"
ANALYSIS_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "optimuskg"
    / "ambiguous_drug_identity_analysis.csv"
)
DETAIL_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "optimuskg"
    / "ambiguous_drug_identity_candidates.csv"
)

EXPECTED_AMBIGUOUS_COUNT = 110
FIELD_PRIORITY = {
    "properties.name": 3,
    "properties.synonyms": 2,
    "properties.trade_names": 1,
}
DETAIL_DRUGS = (
    "lepirudin",
    "bivalirudin",
    "erythropoietin",
    "daptomycin",
    "metformin",
    "reboxetine",
    "doxycycline",
    "piperacillin",
)
WHITESPACE = re.compile(r"\s+")


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found. Expected path: {path}")


def normalize_dots(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.strip().casefold().replace(".", " ")
    return WHITESPACE.sub(" ", value)


def list_as_json(value: Any) -> str:
    if value is None:
        return "[]"
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def evidence_score(candidate: dict[str, Any]) -> int:
    fields = set(candidate["matched_field"].split(";"))
    return max((FIELD_PRIORITY.get(field, 0) for field in fields), default=0)


def build_dot_index(
    optimuskg_drugs: pl.DataFrame,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Reconstruct candidate evidence for dot-pass ambiguities missing from CSV."""

    index: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in optimuskg_drugs.iter_rows(named=True):
        node_id = row["id"]
        properties = row["properties"] or {}
        if node_id is None:
            continue
        values_by_field = {
            "properties.name": [properties.get("name")],
            "properties.synonyms": properties.get("synonyms") or [],
            "properties.trade_names": properties.get("trade_names") or [],
        }
        node = {
            "optimuskg_id": node_id,
            "optimuskg_name": properties.get("name") or "",
            "inchi_key": properties.get("inchi_key") or "",
            "canonical_smiles": properties.get("canonical_smiles") or "",
            "source_ids": list_as_json(properties.get("source_ids")),
            "accession_numbers": list_as_json(properties.get("accession_numbers")),
        }
        for field, values in values_by_field.items():
            for value in values:
                if not isinstance(value, str):
                    continue
                normalized = normalize_dots(value)
                if not normalized:
                    continue
                candidate = index[normalized].setdefault(
                    node_id, {**node, "matched_fields": set()}
                )
                candidate["matched_fields"].add(field)
    return index


def recovered_candidates(
    dot_index: dict[str, dict[str, dict[str, Any]]], drug_name: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in dot_index.get(normalize_dots(drug_name), {}).values():
        rows.append(
            {
                key: value
                for key, value in candidate.items()
                if key != "matched_fields"
            }
            | {
                "matched_field": ";".join(
                    field
                    for field in FIELD_PRIORITY
                    if field in candidate["matched_fields"]
                )
            }
        )
    return sorted(rows, key=lambda row: row["optimuskg_id"])


def classify_identity(
    candidates: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]], str]:
    """Classify identity using only candidates tied at the top evidence level."""

    if not candidates:
        return "insufficient_structure", [], "No candidate rows were available."

    top_score = max(evidence_score(candidate) for candidate in candidates)
    top_candidates = [
        candidate for candidate in candidates if evidence_score(candidate) == top_score
    ]
    inchikeys = [
        candidate["inchi_key"].strip().upper()
        for candidate in top_candidates
        if isinstance(candidate.get("inchi_key"), str)
        and candidate["inchi_key"].strip()
    ]
    smiles = [
        candidate["canonical_smiles"].strip()
        for candidate in top_candidates
        if isinstance(candidate.get("canonical_smiles"), str)
        and candidate["canonical_smiles"].strip()
    ]
    shared_inchikeys = sorted(
        value for value, count in Counter(inchikeys).items() if count >= 2
    )
    shared_smiles = sorted(
        value for value, count in Counter(smiles).items() if count >= 2
    )

    if shared_inchikeys:
        identity_class = "equivalent_by_inchikey"
        evidence = f"shared InChIKey(s): {shared_inchikeys}"
    elif shared_smiles:
        identity_class = "equivalent_by_smiles"
        evidence = f"shared canonical SMILES count: {len(shared_smiles)}"
    elif len(set(inchikeys)) >= 2 or len(set(smiles)) >= 2:
        identity_class = "mixed_identity"
        evidence = (
            f"distinct non-empty InChIKeys={len(set(inchikeys))}; "
            f"distinct non-empty canonical SMILES={len(set(smiles))}"
        )
    else:
        identity_class = "insufficient_structure"
        evidence = (
            f"non-empty InChIKeys={len(inchikeys)}; "
            f"non-empty canonical SMILES={len(smiles)}"
        )

    notes = (
        f"Compared {len(top_candidates)} top-evidence candidate(s) of "
        f"{len(candidates)} total; {evidence}. No namespace was preferred."
    )
    return identity_class, top_candidates, notes


def analyze_ambiguous(
    ambiguous: pl.DataFrame,
    source_candidates: pl.DataFrame,
    dot_index: dict[str, dict[str, dict[str, Any]]],
) -> tuple[pl.DataFrame, pl.DataFrame, dict[int, list[dict[str, Any]]]]:
    source_by_index: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for candidate in source_candidates.iter_rows(named=True):
        source_by_index[candidate["matrix_index"]].append(candidate)

    analysis_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    all_candidates: dict[int, list[dict[str, Any]]] = {}

    for drug in ambiguous.iter_rows(named=True):
        matrix_index = drug["matrix_index"]
        candidates = sorted(
            source_by_index.get(matrix_index, []),
            key=lambda row: row["optimuskg_id"],
        )
        candidate_source = "original candidate CSV"
        if not candidates:
            candidates = recovered_candidates(dot_index, drug["drug_name"])
            candidate_source = "local Parquet dot-normalized recovery"
        if len(candidates) != drug["candidate_count"]:
            print(
                "WARNING: candidate count mismatch for "
                f"{drug['drug_name']}: unresolved file={drug['candidate_count']}, "
                f"retrieved={len(candidates)}"
            )

        identity_class, top_candidates, notes = classify_identity(candidates)
        notes = f"Candidate source: {candidate_source}. {notes}"
        all_candidates[matrix_index] = candidates

        top_inchikeys = sorted(
            {
                candidate["inchi_key"].strip().upper()
                for candidate in top_candidates
                if isinstance(candidate.get("inchi_key"), str)
                and candidate["inchi_key"].strip()
            }
        )
        top_smiles = {
            candidate["canonical_smiles"].strip()
            for candidate in top_candidates
            if isinstance(candidate.get("canonical_smiles"), str)
            and candidate["canonical_smiles"].strip()
        }
        analysis_rows.append(
            {
                "matrix_index": matrix_index,
                "drug_name": drug["drug_name"],
                "candidate_count": len(candidates),
                "identity_class": identity_class,
                "candidate_ids": list_as_json(
                    [candidate["optimuskg_id"] for candidate in candidates]
                ),
                "candidate_names": list_as_json(
                    [candidate["optimuskg_name"] for candidate in candidates]
                ),
                "inchikey_values": list_as_json(top_inchikeys),
                "smiles_count": len(top_smiles),
                "notes": notes,
            }
        )
        for candidate in candidates:
            detail_rows.append(
                {
                    "matrix_index": matrix_index,
                    "drug_name": drug["drug_name"],
                    "optimuskg_id": candidate["optimuskg_id"],
                    "optimuskg_name": candidate["optimuskg_name"],
                    "matched_field": candidate["matched_field"],
                    "inchi_key": candidate.get("inchi_key", "") or "",
                    "canonical_smiles": candidate.get("canonical_smiles", "") or "",
                    "source_ids": candidate.get("source_ids", "[]") or "[]",
                    "accession_numbers": candidate.get("accession_numbers", "[]") or "[]",
                    "identity_class": identity_class,
                }
            )

    analysis_schema = {
        "matrix_index": pl.Int64,
        "drug_name": pl.String,
        "candidate_count": pl.Int64,
        "identity_class": pl.String,
        "candidate_ids": pl.String,
        "candidate_names": pl.String,
        "inchikey_values": pl.String,
        "smiles_count": pl.Int64,
        "notes": pl.String,
    }
    detail_schema = {
        "matrix_index": pl.Int64,
        "drug_name": pl.String,
        "optimuskg_id": pl.String,
        "optimuskg_name": pl.String,
        "matched_field": pl.String,
        "inchi_key": pl.String,
        "canonical_smiles": pl.String,
        "source_ids": pl.String,
        "accession_numbers": pl.String,
        "identity_class": pl.String,
    }
    return (
        pl.DataFrame(analysis_rows, schema=analysis_schema),
        pl.DataFrame(detail_rows, schema=detail_schema),
        all_candidates,
    )


def print_requested_details(
    resolved: pl.DataFrame,
    source_candidates: pl.DataFrame,
    analysis: pl.DataFrame,
) -> None:
    print("\nDetailed candidate information for requested drugs:")
    for drug_name in DETAIL_DRUGS:
        current = resolved.filter(pl.col("drug_name") == drug_name)
        candidates = source_candidates.filter(pl.col("drug_name") == drug_name)
        identity = analysis.filter(pl.col("drug_name") == drug_name)
        status = current.item(0, "resolution_status") if current.height else "not found"
        identity_class = (
            identity.item(0, "identity_class")
            if identity.height
            else "not currently ambiguous"
        )
        print(
            f"{drug_name}: current_status={status}, identity_class={identity_class}, "
            f"candidate_count={candidates.height}"
        )
        if candidates.is_empty():
            print("  No rows in drug_mapping_candidates.csv")
        for candidate in candidates.iter_rows(named=True):
            print(
                f"  id={candidate['optimuskg_id']} | "
                f"name={candidate['optimuskg_name']} | "
                f"matched_field={candidate['matched_field']}"
            )
            print(f"    inchi_key={candidate['inchi_key'] or '<empty>'}")
            print(
                "    canonical_smiles="
                f"{candidate['canonical_smiles'] or '<empty>'}"
            )
            print(f"    source_ids={candidate['source_ids'] or '[]'}")
            print(
                "    accession_numbers="
                f"{candidate['accession_numbers'] or '[]'}"
            )


def print_summary(
    analysis: pl.DataFrame,
    unresolved: pl.DataFrame,
    resolved: pl.DataFrame,
    source_candidates: pl.DataFrame,
) -> None:
    class_counts = dict(
        analysis.group_by("identity_class").len().iter_rows()
    )
    print("Ambiguous drug identity analysis summary")
    print(f"Total ambiguous drugs: {analysis.height}")
    for identity_class in (
        "equivalent_by_inchikey",
        "equivalent_by_smiles",
        "mixed_identity",
        "insufficient_structure",
    ):
        print(f"{identity_class}: {class_counts.get(identity_class, 0)}")

    print_requested_details(resolved, source_candidates, analysis)

    unmatched = unresolved.filter(pl.col("resolution_status") == "unmatched")
    print("\nCurrent unmatched BioKORF drugs (not mapped):")
    if unmatched.is_empty():
        print("None")
    else:
        for drug in unmatched.iter_rows(named=True):
            print(
                f"matrix_index={drug['matrix_index']} | "
                f"{drug['drug_name']} | {drug['stitch_id']}"
            )

    print(f"\nIdentity analysis saved to: {ANALYSIS_PATH}")
    print(f"Detailed candidates saved to: {DETAIL_PATH}")
    print("No mapping decision, candidate removal, fuzzy matching, or network call was made.")


def main() -> None:
    for path, description in (
        (UNRESOLVED_PATH, "Unresolved drug mapping file"),
        (RESOLVED_PATH, "Resolved drug mapping file"),
        (CANDIDATES_PATH, "Drug candidate file"),
        (DRUG_NODE_PATH, "OptimusKG drug node file"),
    ):
        require_file(path, description)

    unresolved = pl.read_csv(UNRESOLVED_PATH)
    resolved = pl.read_csv(RESOLVED_PATH)
    source_candidates = pl.read_csv(CANDIDATES_PATH)
    optimuskg_drugs = pl.read_parquet(DRUG_NODE_PATH)
    ambiguous = unresolved.filter(pl.col("resolution_status") == "ambiguous")
    if ambiguous.height != EXPECTED_AMBIGUOUS_COUNT:
        print(
            f"WARNING: expected {EXPECTED_AMBIGUOUS_COUNT} ambiguous drugs, "
            f"found {ambiguous.height}"
        )

    dot_index = build_dot_index(optimuskg_drugs)
    analysis, details, _ = analyze_ambiguous(
        ambiguous, source_candidates, dot_index
    )
    analysis.write_csv(ANALYSIS_PATH)
    details.write_csv(DETAIL_PATH)
    print_summary(analysis, unresolved, resolved, source_candidates)


if __name__ == "__main__":
    main()
