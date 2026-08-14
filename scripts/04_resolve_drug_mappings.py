"""Resolve deterministic OptimusKG drug mappings without fuzzy matching."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAPPING_PATH = PROJECT_ROOT / "data_processed" / "mappings" / "drug_mapping.csv"
CANDIDATES_PATH = (
    PROJECT_ROOT / "data_processed" / "optimuskg" / "drug_mapping_candidates.csv"
)
SUMMARY_PATH = (
    PROJECT_ROOT / "data_processed" / "optimuskg" / "drug_mapping_summary.csv"
)
DRUG_NODE_PATH = PROJECT_ROOT / "kg" / "optimuskg" / "nodes" / "drug.parquet"
RESOLVED_PATH = (
    PROJECT_ROOT / "data_processed" / "optimuskg" / "drug_mapping_resolved.csv"
)
UNRESOLVED_PATH = (
    PROJECT_ROOT / "data_processed" / "optimuskg" / "drug_mapping_unresolved.csv"
)

EXPECTED_DRUG_COUNT = 757
FIELD_PRIORITY = {
    "properties.name": 3,
    "properties.synonyms": 2,
    "properties.trade_names": 1,
}
WHITESPACE = re.compile(r"\s+")
OUTPUT_SCHEMA = {
    "matrix_index": pl.Int64,
    "drug_name": pl.String,
    "stitch_id": pl.String,
    "optimuskg_id": pl.String,
    "optimuskg_name": pl.String,
    "mapping_method": pl.String,
    "mapping_confidence": pl.String,
    "candidate_count": pl.Int64,
    "resolution_status": pl.String,
}


def normalize_exact(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.strip().casefold()
    return WHITESPACE.sub(" ", value)


def normalize_dots(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.strip().casefold().replace(".", " ")
    return WHITESPACE.sub(" ", value)


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found. Expected path: {path}")


def validate_inputs(
    drugs: pl.DataFrame, candidates: pl.DataFrame, summary: pl.DataFrame
) -> None:
    required_drug_columns = {"matrix_index", "drug_name", "stitch_id"}
    if missing := required_drug_columns.difference(drugs.columns):
        raise ValueError(f"drug_mapping.csv is missing columns: {sorted(missing)}")
    if drugs.height != EXPECTED_DRUG_COUNT:
        raise ValueError(
            f"drug_mapping.csv must contain {EXPECTED_DRUG_COUNT} rows; "
            f"found {drugs.height}"
        )
    if drugs["matrix_index"].to_list() != list(range(EXPECTED_DRUG_COUNT)):
        raise ValueError("matrix_index must be continuous from 0 through 756")
    if normalize_exact(drugs.item(0, "drug_name")) != "lepirudin":
        raise ValueError("The first BioKORF drug must be lepirudin")

    required_candidate_columns = {
        "matrix_index",
        "optimuskg_id",
        "optimuskg_name",
        "matched_field",
    }
    if missing := required_candidate_columns.difference(candidates.columns):
        raise ValueError(
            f"drug_mapping_candidates.csv is missing columns: {sorted(missing)}"
        )
    required_summary_columns = {
        "matrix_index",
        "drug_name",
        "stitch_id",
        "candidate_count",
    }
    if missing := required_summary_columns.difference(summary.columns):
        raise ValueError(
            f"drug_mapping_summary.csv is missing columns: {sorted(missing)}"
        )
    if summary.height != EXPECTED_DRUG_COUNT:
        raise ValueError(
            f"drug_mapping_summary.csv must contain {EXPECTED_DRUG_COUNT} rows"
        )
    if not summary.select(["matrix_index", "drug_name", "stitch_id"]).equals(
        drugs.select(["matrix_index", "drug_name", "stitch_id"])
    ):
        raise ValueError("Drug summary rows do not align with drug_mapping.csv")

    actual_counts = (
        candidates.group_by("matrix_index")
        .len()
        .rename({"len": "actual_candidate_count"})
    )
    count_check = summary.join(actual_counts, on="matrix_index", how="left").with_columns(
        pl.col("actual_candidate_count").fill_null(0)
    )
    if not count_check.filter(
        pl.col("candidate_count") != pl.col("actual_candidate_count")
    ).is_empty():
        raise ValueError("Candidate counts do not reconcile with the candidate table")


def evidence_score(candidate: dict[str, Any]) -> int:
    fields = set(candidate["matched_field"].split(";"))
    return max((FIELD_PRIORITY.get(field, 0) for field in fields), default=0)


def choose_by_evidence(
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    """Return a candidate only when one node has strictly highest evidence."""

    if len(candidates) == 1:
        return candidates[0], "unique_exact"
    if not candidates:
        return None, None

    scores = [evidence_score(candidate) for candidate in candidates]
    top_score = max(scores)
    winners = [
        candidate
        for candidate, score in zip(candidates, scores, strict=True)
        if score == top_score
    ]
    if len(winners) != 1:
        return None, None
    method = {
        3: "canonical_name_disambiguation",
        2: "synonym_disambiguation",
        1: "trade_name_disambiguation",
    }[top_score]
    return winners[0], method


def build_name_index(
    optimuskg_drugs: pl.DataFrame, normalizer: Callable[[str], str]
) -> dict[str, dict[str, dict[str, Any]]]:
    if missing := {"id", "properties"}.difference(optimuskg_drugs.columns):
        raise ValueError(f"OptimusKG drug table is missing columns: {sorted(missing)}")

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
        for field, values in values_by_field.items():
            for value in values:
                if not isinstance(value, str):
                    continue
                normalized = normalizer(value)
                if not normalized:
                    continue
                candidate = index[normalized].setdefault(
                    node_id,
                    {
                        "optimuskg_id": node_id,
                        "optimuskg_name": properties.get("name") or "",
                        "matched_fields": set(),
                    },
                )
                candidate["matched_fields"].add(field)
    return index


def indexed_candidates(
    index: dict[str, dict[str, dict[str, Any]]], normalized_name: str
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for node in index.get(normalized_name, {}).values():
        candidates.append(
            {
                "optimuskg_id": node["optimuskg_id"],
                "optimuskg_name": node["optimuskg_name"],
                "matched_field": ";".join(
                    field
                    for field in FIELD_PRIORITY
                    if field in node["matched_fields"]
                ),
            }
        )
    return sorted(candidates, key=lambda candidate: candidate["optimuskg_id"])


def output_row(
    drug: dict[str, Any],
    candidates: list[dict[str, Any]],
    selected: dict[str, Any] | None,
    method: str | None,
) -> dict[str, Any]:
    if selected is not None:
        status = "resolved"
        confidence = "high"
        optimuskg_id = selected["optimuskg_id"]
        optimuskg_name = selected["optimuskg_name"]
    else:
        status = "ambiguous" if candidates else "unmatched"
        confidence = ""
        optimuskg_id = ""
        optimuskg_name = ""
    return {
        "matrix_index": drug["matrix_index"],
        "drug_name": drug["drug_name"],
        "stitch_id": drug["stitch_id"],
        "optimuskg_id": optimuskg_id,
        "optimuskg_name": optimuskg_name,
        "mapping_method": method or "",
        "mapping_confidence": confidence,
        "candidate_count": len(candidates),
        "resolution_status": status,
    }


def resolve_mappings(
    drugs: pl.DataFrame,
    existing_candidates: pl.DataFrame,
    optimuskg_drugs: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[int, list[dict[str, Any]]]]:
    candidates_by_index: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for candidate in existing_candidates.iter_rows(named=True):
        candidates_by_index[candidate["matrix_index"]].append(candidate)

    dot_index = build_name_index(optimuskg_drugs, normalize_dots)
    resolved_rows: list[dict[str, Any]] = []
    final_candidates: dict[int, list[dict[str, Any]]] = {}

    for drug in drugs.iter_rows(named=True):
        matrix_index = drug["matrix_index"]
        candidates = candidates_by_index.get(matrix_index, [])
        if candidates:
            candidates = sorted(
                candidates, key=lambda candidate: candidate["optimuskg_id"]
            )
            selected, method = choose_by_evidence(candidates)
        else:
            candidates = indexed_candidates(
                dot_index, normalize_dots(drug["drug_name"])
            )
            selected, _ = choose_by_evidence(candidates)
            method = "dot_normalized_exact" if selected is not None else None

        final_candidates[matrix_index] = candidates
        resolved_rows.append(output_row(drug, candidates, selected, method))

    return pl.DataFrame(resolved_rows, schema=OUTPUT_SCHEMA), final_candidates


def print_results(
    resolved: pl.DataFrame,
    final_candidates: dict[int, list[dict[str, Any]]],
) -> None:
    unique_exact = resolved.filter(pl.col("mapping_method") == "unique_exact").height
    canonical = resolved.filter(
        pl.col("mapping_method") == "canonical_name_disambiguation"
    ).height
    synonym_trade = resolved.filter(
        pl.col("mapping_method").is_in(
            ["synonym_disambiguation", "trade_name_disambiguation"]
        )
    ).height
    dot_normalized = resolved.filter(
        pl.col("mapping_method") == "dot_normalized_exact"
    ).height
    ambiguous = resolved.filter(pl.col("resolution_status") == "ambiguous")
    unmatched = resolved.filter(pl.col("resolution_status") == "unmatched")
    resolved_count = resolved.filter(pl.col("resolution_status") == "resolved").height

    print("Drug mapping resolution summary")
    print(f"Total drugs: {resolved.height}")
    print(f"Resolved unique exact: {unique_exact}")
    print(f"Resolved canonical-name disambiguation: {canonical}")
    print(f"Resolved synonym/trade-name disambiguation: {synonym_trade}")
    print(f"Resolved dot-normalized: {dot_normalized}")
    print(f"Remaining ambiguous: {ambiguous.height}")
    print(f"Remaining unmatched: {unmatched.height}")
    print(f"Overall resolved percentage: {100.0 * resolved_count / resolved.height:.2f}%")

    print("\nAll remaining unmatched drugs:")
    if unmatched.is_empty():
        print("None")
    else:
        for drug in unmatched.iter_rows(named=True):
            print(
                f"matrix_index={drug['matrix_index']} | "
                f"{drug['drug_name']} | {drug['stitch_id']}"
            )

    print("\nFirst 30 remaining ambiguous drugs with candidates:")
    if ambiguous.is_empty():
        print("None")
    else:
        for drug in ambiguous.head(30).iter_rows(named=True):
            print(
                f"matrix_index={drug['matrix_index']} | {drug['drug_name']} | "
                f"candidate_count={drug['candidate_count']}"
            )
            for candidate in final_candidates[drug["matrix_index"]]:
                print(
                    f"  {candidate['optimuskg_id']} | "
                    f"{candidate['optimuskg_name']} | {candidate['matched_field']}"
                )

    print(f"\nResolved mappings saved to: {RESOLVED_PATH}")
    print(f"Unresolved mappings saved to: {UNRESOLVED_PATH}")
    print("No fuzzy matching, edges, namespace preference, or network access was used.")


def main() -> None:
    for path, description in (
        (MAPPING_PATH, "BioKORF drug mapping file"),
        (CANDIDATES_PATH, "OptimusKG candidate file"),
        (SUMMARY_PATH, "OptimusKG candidate summary file"),
        (DRUG_NODE_PATH, "OptimusKG drug node file"),
    ):
        require_file(path, description)

    drugs = pl.read_csv(MAPPING_PATH)
    candidates = pl.read_csv(CANDIDATES_PATH)
    summary = pl.read_csv(SUMMARY_PATH)
    optimuskg_drugs = pl.read_parquet(DRUG_NODE_PATH)
    validate_inputs(drugs, candidates, summary)

    resolved, final_candidates = resolve_mappings(drugs, candidates, optimuskg_drugs)
    unresolved = resolved.filter(pl.col("resolution_status") != "resolved")
    resolved.write_csv(RESOLVED_PATH)
    unresolved.write_csv(UNRESOLVED_PATH)
    print_results(resolved, final_candidates)


if __name__ == "__main__":
    main()
