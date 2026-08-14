"""Analyze BioKORF-to-OptimusKG structure matches through local STITCH data."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAPPING_PATH = PROJECT_ROOT / "data_processed" / "mappings" / "drug_mapping.csv"
RESOLVED_PATH = (
    PROJECT_ROOT / "data_processed" / "optimuskg" / "drug_mapping_resolved.csv"
)
CANDIDATES_PATH = (
    PROJECT_ROOT / "data_processed" / "optimuskg" / "drug_mapping_candidates.csv"
)
DRUG_NODE_PATH = PROJECT_ROOT / "kg" / "optimuskg" / "nodes" / "drug.parquet"
STITCH_PATH = (
    PROJECT_ROOT
    / "kg"
    / "stitch"
    / "chemicals.inchikeys.v5.0.tsv"
    / "chemicals.inchikeys.v5.0.tsv"
)
STITCH_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "stitch"
    / "biokorf_stitch_inchikeys.csv"
)
MATCH_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "optimuskg"
    / "drug_stitch_inchikey_matches.csv"
)
SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "optimuskg"
    / "drug_structure_mapping_summary.csv"
)

EXPECTED_DRUG_COUNT = 757
REQUIRED_STITCH_COLUMNS = (
    "flat_chemical_id",
    "stereo_chemical_id",
    "source_cid",
    "inchikey",
)
DETAIL_DRUGS = (
    "lepirudin",
    "bivalirudin",
    "leuprorelin",
    "goserelin",
    "erythropoietin",
    "insulin",
    "gadoversetamide",
    "sulfamethoxazole.and.trimethoprim",
    "technetium.(99mTc).tetrofosmin",
    "iotrolan",
)


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found. Expected path: {path}")


def normalize_inchikey(value: str) -> str:
    return value.strip().upper()


def json_list(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def validate_drugs(drugs: pl.DataFrame) -> None:
    required = {"matrix_index", "drug_name", "stitch_id"}
    if missing := required.difference(drugs.columns):
        raise ValueError(f"drug_mapping.csv is missing columns: {sorted(missing)}")
    if drugs.height != EXPECTED_DRUG_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_DRUG_COUNT} BioKORF drugs; found {drugs.height}"
        )
    if drugs["matrix_index"].to_list() != list(range(EXPECTED_DRUG_COUNT)):
        raise ValueError("matrix_index must be continuous from 0 through 756")
    if drugs["stitch_id"].null_count() or drugs["stitch_id"].n_unique() != drugs.height:
        raise ValueError("All BioKORF stitch_id values must be non-null and unique")


def stream_stitch_once(
    path: Path, target_ids: set[str]
) -> tuple[dict[str, int], dict[str, set[str]], int, int]:
    """Stream the complete STITCH TSV once and retain only target-ID evidence."""

    row_counts: dict[str, int] = defaultdict(int)
    inchikey_sets: dict[str, set[str]] = defaultdict(set)
    total_rows = 0
    malformed_rows = 0
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if header is None:
            raise ValueError(f"STITCH file is empty: {path}")
        missing = set(REQUIRED_STITCH_COLUMNS).difference(header)
        if missing:
            raise ValueError(f"STITCH header is missing columns: {sorted(missing)}")
        flat_id_column = header.index("flat_chemical_id")
        inchikey_column = header.index("inchikey")
        expected_columns = len(header)

        for row in reader:
            total_rows += 1
            if len(row) != expected_columns:
                malformed_rows += 1
                continue
            flat_id = row[flat_id_column]
            if flat_id not in target_ids:
                continue
            row_counts[flat_id] += 1
            inchikey = normalize_inchikey(row[inchikey_column])
            if inchikey:
                inchikey_sets[flat_id].add(inchikey)
    return row_counts, inchikey_sets, total_rows, malformed_rows


def build_stitch_output(
    drugs: pl.DataFrame,
    row_counts: dict[str, int],
    inchikey_sets: dict[str, set[str]],
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for drug in drugs.iter_rows(named=True):
        stitch_id = drug["stitch_id"]
        inchikeys = sorted(inchikey_sets.get(stitch_id, set()))
        rows.append(
            {
                "matrix_index": drug["matrix_index"],
                "drug_name": drug["drug_name"],
                "stitch_id": stitch_id,
                "stitch_found": row_counts.get(stitch_id, 0) > 0,
                "stitch_row_count": row_counts.get(stitch_id, 0),
                "distinct_inchikey_count": len(inchikeys),
                "inchikeys": json_list(inchikeys),
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "matrix_index": pl.Int64,
            "drug_name": pl.String,
            "stitch_id": pl.String,
            "stitch_found": pl.Boolean,
            "stitch_row_count": pl.Int64,
            "distinct_inchikey_count": pl.Int64,
            "inchikeys": pl.String,
        },
    )


def build_optimuskg_index(
    optimuskg_drugs: pl.DataFrame,
) -> dict[str, dict[str, dict[str, str]]]:
    if missing := {"id", "properties"}.difference(optimuskg_drugs.columns):
        raise ValueError(f"OptimusKG drug table is missing columns: {sorted(missing)}")
    index: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for node in optimuskg_drugs.iter_rows(named=True):
        properties = node["properties"] or {}
        raw_inchikey = properties.get("inchi_key")
        if node["id"] is None or not isinstance(raw_inchikey, str):
            continue
        inchikey = normalize_inchikey(raw_inchikey)
        if not inchikey:
            continue
        index[inchikey][node["id"]] = {
            "optimuskg_id": node["id"],
            "optimuskg_name": properties.get("name") or "",
            "optimuskg_inchikey": inchikey,
        }
    return index


def build_matches_and_summary(
    stitch_drugs: pl.DataFrame,
    optimuskg_index: dict[str, dict[str, dict[str, str]]],
    previous_candidate_pairs: set[tuple[int, str]],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    match_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for drug in stitch_drugs.iter_rows(named=True):
        stitch_inchikeys: list[str] = json.loads(drug["inchikeys"])
        matched_nodes: dict[str, dict[str, str]] = {}
        for stitch_inchikey in stitch_inchikeys:
            for node_id, node in optimuskg_index.get(stitch_inchikey, {}).items():
                matched_nodes[node_id] = {**node, "stitch_inchikey": stitch_inchikey}

        ordered_matches = sorted(
            matched_nodes.values(), key=lambda node: node["optimuskg_id"]
        )
        for node in ordered_matches:
            match_rows.append(
                {
                    "matrix_index": drug["matrix_index"],
                    "drug_name": drug["drug_name"],
                    "stitch_id": drug["stitch_id"],
                    "stitch_inchikey": node["stitch_inchikey"],
                    "optimuskg_id": node["optimuskg_id"],
                    "optimuskg_name": node["optimuskg_name"],
                    "optimuskg_inchikey": node["optimuskg_inchikey"],
                    "was_previous_name_candidate": (
                        drug["matrix_index"], node["optimuskg_id"]
                    )
                    in previous_candidate_pairs,
                }
            )

        match_count = len(ordered_matches)
        matched_identities = {
            node["stitch_inchikey"] for node in ordered_matches
        }
        if not drug["stitch_found"]:
            structure_status = "stitch_not_found"
        elif match_count == 0:
            structure_status = "stitch_found_no_optimuskg_match"
        elif match_count == 1:
            structure_status = "unique_structure_match"
        elif len(matched_identities) == 1:
            structure_status = "equivalent_structure_group"
        else:
            structure_status = "multiple_structure_matches"

        summary_rows.append(
            {
                "matrix_index": drug["matrix_index"],
                "drug_name": drug["drug_name"],
                "stitch_id": drug["stitch_id"],
                "stitch_found": drug["stitch_found"],
                "distinct_stitch_inchikey_count": drug["distinct_inchikey_count"],
                "optimuskg_structure_match_count": match_count,
                "optimuskg_structure_match_ids": json_list(
                    [node["optimuskg_id"] for node in ordered_matches]
                ),
                "structure_status": structure_status,
            }
        )

    match_schema = {
        "matrix_index": pl.Int64,
        "drug_name": pl.String,
        "stitch_id": pl.String,
        "stitch_inchikey": pl.String,
        "optimuskg_id": pl.String,
        "optimuskg_name": pl.String,
        "optimuskg_inchikey": pl.String,
        "was_previous_name_candidate": pl.Boolean,
    }
    summary_schema = {
        "matrix_index": pl.Int64,
        "drug_name": pl.String,
        "stitch_id": pl.String,
        "stitch_found": pl.Boolean,
        "distinct_stitch_inchikey_count": pl.Int64,
        "optimuskg_structure_match_count": pl.Int64,
        "optimuskg_structure_match_ids": pl.String,
        "structure_status": pl.String,
    }
    return (
        pl.DataFrame(match_rows, schema=match_schema),
        pl.DataFrame(summary_rows, schema=summary_schema),
    )


def print_drug_result(
    drug_name: str, summary: pl.DataFrame, matches: pl.DataFrame
) -> None:
    selected = summary.filter(pl.col("drug_name") == drug_name)
    if selected.is_empty():
        print(f"{drug_name}: NOT PRESENT IN BIOKORF")
        return
    drug = selected.row(0, named=True)
    print(
        f"matrix_index={drug['matrix_index']} | {drug_name} | "
        f"stitch_id={drug['stitch_id']} | stitch_found={drug['stitch_found']} | "
        f"stitch_inchikeys={drug['distinct_stitch_inchikey_count']} | "
        f"structure_matches={drug['optimuskg_structure_match_count']} | "
        f"status={drug['structure_status']}"
    )
    drug_matches = matches.filter(pl.col("matrix_index") == drug["matrix_index"])
    if drug_matches.is_empty():
        print("  No OptimusKG structure matches")
    for match in drug_matches.iter_rows(named=True):
        print(
            f"  STITCH InChIKey={match['stitch_inchikey']} | "
            f"{match['optimuskg_id']} | {match['optimuskg_name']} | "
            f"previous_name_candidate={match['was_previous_name_candidate']}"
        )


def print_diagnostics(
    stitch_drugs: pl.DataFrame,
    matches: pl.DataFrame,
    summary: pl.DataFrame,
    resolved: pl.DataFrame,
    total_stitch_rows: int,
    malformed_stitch_rows: int,
) -> None:
    status_counts = dict(summary.group_by("structure_status").len().iter_rows())
    stitch_found = summary.filter(pl.col("stitch_found")).height
    with_inchikey = summary.filter(
        pl.col("distinct_stitch_inchikey_count") >= 1
    ).height
    with_match = summary.filter(pl.col("optimuskg_structure_match_count") >= 1).height

    print("Drug STITCH-InChIKey structure mapping summary")
    print(f"Total BioKORF drugs: {summary.height}")
    print(f"STITCH IDs found: {stitch_found}")
    print(f"STITCH IDs not found: {summary.height - stitch_found}")
    print(f"Drugs with >=1 STITCH InChIKey: {with_inchikey}")
    print(
        "Unique OptimusKG structure matches: "
        f"{status_counts.get('unique_structure_match', 0)}"
    )
    print(
        "Equivalent structure groups: "
        f"{status_counts.get('equivalent_structure_group', 0)}"
    )
    print(
        "Multiple structure matches: "
        f"{status_counts.get('multiple_structure_matches', 0)}"
    )
    print(
        "STITCH found but no OptimusKG match: "
        f"{status_counts.get('stitch_found_no_optimuskg_match', 0)}"
    )
    print(f"STITCH not found: {status_counts.get('stitch_not_found', 0)}")
    print(
        "Overall percentage with at least one structure match: "
        f"{100.0 * with_match / summary.height:.2f}%"
    )
    print(f"STITCH data rows streamed: {total_stitch_rows:,}")
    print(f"Malformed STITCH rows skipped: {malformed_stitch_rows:,}")

    print("\nDetailed results for requested drugs:")
    for drug_name in DETAIL_DRUGS:
        print_drug_result(drug_name, summary, matches)

    unresolved = resolved.filter(pl.col("resolution_status") != "resolved")
    print("\nStructure results for every currently ambiguous or unmatched drug:")
    for drug in unresolved.iter_rows(named=True):
        print(
            f"Current resolution_status={drug['resolution_status']} | "
            f"candidate_count={drug['candidate_count']}"
        )
        print_drug_result(drug["drug_name"], summary, matches)

    print(f"\nBioKORF STITCH InChIKeys saved to: {STITCH_OUTPUT_PATH}")
    print(f"OptimusKG structure matches saved to: {MATCH_OUTPUT_PATH}")
    print(f"Structure summary saved to: {SUMMARY_OUTPUT_PATH}")
    print("No existing mapping was overwritten; no fuzzy matching or network access was used.")


def main() -> None:
    for path, description in (
        (MAPPING_PATH, "BioKORF drug mapping file"),
        (RESOLVED_PATH, "Resolved drug mapping file"),
        (CANDIDATES_PATH, "Drug mapping candidate file"),
        (DRUG_NODE_PATH, "OptimusKG drug node file"),
        (STITCH_PATH, "Extracted STITCH InChIKey file"),
    ):
        require_file(path, description)

    drugs = pl.read_csv(MAPPING_PATH)
    resolved = pl.read_csv(RESOLVED_PATH)
    previous_candidates = pl.read_csv(CANDIDATES_PATH)
    optimuskg_drugs = pl.read_parquet(DRUG_NODE_PATH)
    validate_drugs(drugs)
    if resolved.height != EXPECTED_DRUG_COUNT:
        raise ValueError("drug_mapping_resolved.csv must contain 757 rows")

    target_ids = set(drugs["stitch_id"].to_list())
    row_counts, inchikey_sets, total_rows, malformed_rows = stream_stitch_once(
        STITCH_PATH, target_ids
    )
    stitch_drugs = build_stitch_output(drugs, row_counts, inchikey_sets)
    optimuskg_index = build_optimuskg_index(optimuskg_drugs)
    previous_candidate_pairs = set(
        zip(
            previous_candidates["matrix_index"].to_list(),
            previous_candidates["optimuskg_id"].to_list(),
            strict=True,
        )
    )
    matches, summary = build_matches_and_summary(
        stitch_drugs, optimuskg_index, previous_candidate_pairs
    )

    STITCH_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MATCH_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    stitch_drugs.write_csv(STITCH_OUTPUT_PATH)
    matches.write_csv(MATCH_OUTPUT_PATH)
    summary.write_csv(SUMMARY_OUTPUT_PATH)
    print_diagnostics(
        stitch_drugs, matches, summary, resolved, total_rows, malformed_rows
    )


if __name__ == "__main__":
    main()
