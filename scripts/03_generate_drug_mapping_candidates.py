"""Generate exact-match OptimusKG drug candidates without selecting mappings."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAPPING_PATH = PROJECT_ROOT / "data_processed" / "mappings" / "drug_mapping.csv"
DRUG_NODE_PATH = PROJECT_ROOT / "kg" / "optimuskg" / "nodes" / "drug.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data_processed" / "optimuskg"
CANDIDATES_PATH = OUTPUT_DIR / "drug_mapping_candidates.csv"
SUMMARY_PATH = OUTPUT_DIR / "drug_mapping_summary.csv"

EXPECTED_DRUG_COUNT = 757
MATCH_FIELDS = (
    "properties.name",
    "properties.synonyms",
    "properties.trade_names",
)
WHITESPACE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    """Apply only the approved normalization operations."""

    value = unicodedata.normalize("NFKC", value)
    value = value.strip().casefold()
    return WHITESPACE.sub(" ", value)


def validate_biokorf_drugs(drugs: pl.DataFrame) -> None:
    required = {"matrix_index", "drug_name", "stitch_id"}
    missing = required.difference(drugs.columns)
    if missing:
        raise ValueError(f"drug_mapping.csv is missing columns: {sorted(missing)}")
    if drugs.height != EXPECTED_DRUG_COUNT:
        raise ValueError(
            f"drug_mapping.csv must have exactly {EXPECTED_DRUG_COUNT} rows; "
            f"found {drugs.height}"
        )
    if drugs["matrix_index"].to_list() != list(range(EXPECTED_DRUG_COUNT)):
        raise ValueError("matrix_index must be continuous from 0 through 756")
    first_drug = drugs.item(0, "drug_name")
    if not isinstance(first_drug, str) or first_drug.strip().casefold() != "lepirudin":
        raise ValueError(f"The first drug must be lepirudin; found {first_drug!r}")


def require_local_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found. Expected path: {path}")


def list_as_json(value: Any) -> str:
    """Serialize list-valued OptimusKG properties without losing boundaries."""

    if value is None:
        return "[]"
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def build_exact_index(
    optimuskg_drugs: pl.DataFrame,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Index exact normalized terms by OptimusKG node and matched field."""

    required_columns = {"id", "properties"}
    missing = required_columns.difference(optimuskg_drugs.columns)
    if missing:
        raise ValueError(f"OptimusKG drug table is missing columns: {sorted(missing)}")

    exact_index: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in optimuskg_drugs.iter_rows(named=True):
        properties = row["properties"] or {}
        node_id = row["id"]
        if node_id is None:
            continue

        field_values = {
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

        for matched_field, values in field_values.items():
            for value in values:
                if not isinstance(value, str):
                    continue
                normalized = normalize_text(value)
                if not normalized:
                    continue
                indexed_node = exact_index[normalized].setdefault(
                    node_id, {**node, "matched_fields": set()}
                )
                indexed_node["matched_fields"].add(matched_field)
    return exact_index


def generate_outputs(
    biokorf_drugs: pl.DataFrame,
    exact_index: dict[str, dict[str, dict[str, Any]]],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    candidate_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for drug in biokorf_drugs.iter_rows(named=True):
        normalized_name = normalize_text(drug["drug_name"])
        matched_nodes = exact_index.get(normalized_name, {})
        ordered_nodes = sorted(matched_nodes.values(), key=lambda node: node["optimuskg_id"])

        for node in ordered_nodes:
            candidate_rows.append(
                {
                    "matrix_index": drug["matrix_index"],
                    "drug_name": drug["drug_name"],
                    "stitch_id": drug["stitch_id"],
                    "optimuskg_id": node["optimuskg_id"],
                    "optimuskg_name": node["optimuskg_name"],
                    "matched_field": ";".join(
                        field for field in MATCH_FIELDS if field in node["matched_fields"]
                    ),
                    "inchi_key": node["inchi_key"],
                    "canonical_smiles": node["canonical_smiles"],
                    "source_ids": node["source_ids"],
                    "accession_numbers": node["accession_numbers"],
                }
            )

        candidate_count = len(ordered_nodes)
        if candidate_count == 1:
            mapping_status = "unique_exact"
        elif candidate_count > 1:
            mapping_status = "ambiguous_exact"
        else:
            mapping_status = "unmatched"
        summary_rows.append(
            {
                "matrix_index": drug["matrix_index"],
                "drug_name": drug["drug_name"],
                "stitch_id": drug["stitch_id"],
                "candidate_count": candidate_count,
                "candidate_ids": ";".join(
                    node["optimuskg_id"] for node in ordered_nodes
                ),
                "mapping_status": mapping_status,
            }
        )

    candidate_schema = {
        "matrix_index": pl.Int64,
        "drug_name": pl.String,
        "stitch_id": pl.String,
        "optimuskg_id": pl.String,
        "optimuskg_name": pl.String,
        "matched_field": pl.String,
        "inchi_key": pl.String,
        "canonical_smiles": pl.String,
        "source_ids": pl.String,
        "accession_numbers": pl.String,
    }
    summary_schema = {
        "matrix_index": pl.Int64,
        "drug_name": pl.String,
        "stitch_id": pl.String,
        "candidate_count": pl.Int64,
        "candidate_ids": pl.String,
        "mapping_status": pl.String,
    }
    return (
        pl.DataFrame(candidate_rows, schema=candidate_schema),
        pl.DataFrame(summary_rows, schema=summary_schema),
    )


def print_summary(candidates: pl.DataFrame, summary: pl.DataFrame) -> None:
    unique_count = summary.filter(pl.col("candidate_count") == 1).height
    ambiguous = summary.filter(pl.col("candidate_count") > 1)
    unmatched = summary.filter(pl.col("candidate_count") == 0)
    coverage = 100.0 * (summary.height - unmatched.height) / summary.height

    print("Drug mapping candidate summary")
    print(f"Total drugs: {summary.height}")
    print(f"Drugs with exactly 1 candidate: {unique_count}")
    print(f"Drugs with >1 candidates: {ambiguous.height}")
    print(f"Drugs with 0 candidates: {unmatched.height}")
    print(f"Exact match coverage: {coverage:.2f}%")

    print("\nAmbiguous matches for the first 20 ambiguous drugs:")
    if ambiguous.is_empty():
        print("None")
    else:
        for drug in ambiguous.head(20).iter_rows(named=True):
            print(
                f"matrix_index={drug['matrix_index']} | {drug['drug_name']} | "
                f"candidate_count={drug['candidate_count']}"
            )
            drug_candidates = candidates.filter(
                pl.col("matrix_index") == drug["matrix_index"]
            )
            for candidate in drug_candidates.iter_rows(named=True):
                print(
                    f"  {candidate['optimuskg_id']} | "
                    f"{candidate['optimuskg_name']} | {candidate['matched_field']}"
                )

    print("\nAll unmatched drugs:")
    if unmatched.is_empty():
        print("None")
    else:
        for drug in unmatched.iter_rows(named=True):
            print(
                f"matrix_index={drug['matrix_index']} | "
                f"{drug['drug_name']} | {drug['stitch_id']}"
            )

    print(f"\nCandidate rows saved to: {CANDIDATES_PATH}")
    print(f"Summary rows saved to: {SUMMARY_PATH}")
    print("No fuzzy matching or final namespace selection was performed.")


def main() -> None:
    require_local_file(MAPPING_PATH, "BioKORF drug mapping file")
    require_local_file(DRUG_NODE_PATH, "OptimusKG drug node file")

    biokorf_drugs = pl.read_csv(MAPPING_PATH)
    validate_biokorf_drugs(biokorf_drugs)
    optimuskg_drugs = pl.read_parquet(DRUG_NODE_PATH)

    exact_index = build_exact_index(optimuskg_drugs)
    candidates, summary = generate_outputs(biokorf_drugs, exact_index)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates.write_csv(CANDIDATES_PATH)
    summary.write_csv(SUMMARY_PATH)
    print_summary(candidates, summary)


if __name__ == "__main__":
    main()
