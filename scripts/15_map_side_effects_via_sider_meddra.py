"""Map BioKORF side effects to OptimusKG MedDRA nodes via SIDER."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIDE_EFFECT_PATH = PROJECT_ROOT / "data_processed" / "mappings" / "side_effect_mapping.csv"
SIDER_UMLS_PATH = PROJECT_ROOT / "data_processed" / "sider" / "side_effect_sider_umls_mapping.csv"
UMLS_SUMMARY_PATH = PROJECT_ROOT / "data_processed" / "optimuskg" / "side_effect_umls_mapping_summary.csv"
NAME_CANDIDATES_PATH = PROJECT_ROOT / "data_processed" / "optimuskg" / "side_effect_mapping_candidates.csv"
MEDDRA_REQUESTED_PATH = PROJECT_ROOT / "kg" / "sider" / "meddra.tsv"
MEDDRA_NESTED_PATH = MEDDRA_REQUESTED_PATH / "meddra.tsv"
PHENOTYPE_PATH = PROJECT_ROOT / "kg" / "optimuskg" / "nodes" / "phenotype.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data_processed" / "optimuskg" / "side_effect_meddra_mapping.csv"
REVIEW_PATH = PROJECT_ROOT / "data_processed" / "optimuskg" / "side_effect_meddra_review.csv"

EXPECTED_COUNT = 994
WHITESPACE = re.compile(r"\s+")
MEDDRA_ID_PATTERN = re.compile(r"^\d+$")
DETAIL_TERMS = (
    "abdominal discomfort", "abdominal distension", "abdominal pain", "nausea",
    "vomiting", "headache", "dizziness", "dry eye", "proteinuria", "wheezing",
)
REVIEW_STATUSES = {
    "ambiguous_meddra", "meddra_resolved_but_not_in_optimuskg", "unresolved"
}
OUTPUT_COLUMNS = (
    "matrix_index", "side_effect_name", "sider_umls_cui", "meddra_id",
    "meddra_name", "canonical_optimuskg_id", "alias_optimuskg_ids",
    "mapping_method", "mapping_status", "requires_review",
)


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def resolve_meddra_path() -> Path:
    if MEDDRA_REQUESTED_PATH.is_file():
        return MEDDRA_REQUESTED_PATH
    if MEDDRA_REQUESTED_PATH.is_dir() and MEDDRA_NESTED_PATH.is_file():
        return MEDDRA_NESTED_PATH
    raise FileNotFoundError(
        "SIDER MedDRA dictionary not found. Expected either "
        f"{MEDDRA_REQUESTED_PATH} or {MEDDRA_NESTED_PATH}"
    )


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).strip().casefold()
    return WHITESPACE.sub(" ", value)


def parse_json_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON list, found {value!r}")
    return [str(item) for item in parsed if item is not None and str(item).strip()]


def json_ids(values: list[str] | set[str]) -> str:
    return json.dumps(sorted(set(values)), ensure_ascii=False, separators=(",", ":"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    require_file(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_inputs(
    side_effects: list[dict[str, str]], sider_umls: list[dict[str, str]],
    umls_summary: list[dict[str, str]],
) -> None:
    for label, rows in (
        ("side-effect mapping", side_effects), ("SIDER UMLS mapping", sider_umls),
        ("UMLS OptimusKG summary", umls_summary),
    ):
        if len(rows) != EXPECTED_COUNT:
            raise ValueError(f"Expected {EXPECTED_COUNT} rows in {label}; found {len(rows)}")
        if [int(row["matrix_index"]) for row in rows] != list(range(EXPECTED_COUNT)):
            raise ValueError(f"matrix_index in {label} must be exactly 0 through 993")
    for index in range(EXPECTED_COUNT):
        names = {
            side_effects[index]["side_effect_name"], sider_umls[index]["side_effect_name"],
            umls_summary[index]["side_effect_name"],
        }
        if len(names) != 1:
            raise ValueError(f"Side-effect name mismatch at matrix_index {index}: {names}")


def read_meddra_dictionary(
    path: Path,
) -> tuple[
    dict[str, dict[str, tuple[str, str]]],
    dict[str, dict[str, tuple[str, str]]],
    dict[str, Any],
]:
    by_cui: dict[str, dict[str, tuple[str, str]]] = defaultdict(dict)
    by_name: dict[str, dict[str, tuple[str, str]]] = defaultdict(dict)
    column_counts: Counter[int] = Counter()
    first_row: list[str] | None = None
    total_rows = 0
    pt_rows = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle, delimiter="\t"), start=1):
            if not row:
                continue
            total_rows += 1
            column_counts[len(row)] += 1
            if first_row is None:
                first_row = row
            if len(row) != 4:
                raise ValueError(
                    f"Malformed meddra.tsv row {line_number}: expected 4 columns, found {len(row)}"
                )
            cui, concept_type, meddra_id, name = (value.strip() for value in row)
            if concept_type != "PT":
                continue
            if not cui or not MEDDRA_ID_PATTERN.fullmatch(meddra_id) or not name:
                raise ValueError(f"Invalid PT identity at meddra.tsv row {line_number}: {row!r}")
            pt_rows += 1
            record = (meddra_id, name)
            by_cui[cui.upper()][meddra_id] = record
            by_name[normalize_name(name)][meddra_id] = record
    if first_row is None:
        raise ValueError(f"SIDER MedDRA dictionary is empty: {path}")
    header_tokens = {"umls concept id", "meddra concept type", "meddra id", "side-effect name"}
    has_header = any(normalize_name(value) in header_tokens for value in first_row)
    return by_cui, by_name, {
        "total_rows": total_rows, "pt_rows": pt_rows,
        "column_counts": dict(sorted(column_counts.items())),
        "has_header": has_header, "first_row": first_row,
    }


def build_phenotype_index(frame: pl.DataFrame) -> dict[str, dict[str, Any]]:
    if missing := {"id", "properties"}.difference(frame.columns):
        raise ValueError(f"Phenotype table is missing columns: {sorted(missing)}")
    nodes: dict[str, dict[str, Any]] = {}
    for row in frame.iter_rows(named=True):
        node_id = row["id"]
        if not isinstance(node_id, str):
            continue
        properties = row["properties"] or {}
        nodes[node_id] = {
            "id": node_id, "name": properties.get("name") or "",
            "code": properties.get("code") or "", "xrefs": properties.get("xrefs") or [],
            "exact_synonyms": properties.get("exact_synonyms") or [],
        }
    return nodes


def retained_non_meddra_aliases(row: dict[str, str]) -> list[str]:
    ids = parse_json_list(row["alias_optimuskg_ids"])
    canonical = row.get("canonical_optimuskg_id", "")
    if canonical:
        ids.append(canonical)
    return sorted({node_id for node_id in ids if not node_id.casefold().startswith("meddra:")})


def detail_lines(row: dict[str, Any]) -> list[str]:
    return [
        f"  matrix_index: {row['matrix_index']}",
        f"  sider_umls_cui: {row['sider_umls_cui'] or '<none>'}",
        f"  meddra_id: {row['meddra_id'] or '<none>'}",
        f"  meddra_name: {row['meddra_name'] or '<none>'}",
        f"  canonical_optimuskg_id: {row['canonical_optimuskg_id'] or '<none>'}",
        f"  alias_optimuskg_ids: {row['alias_optimuskg_ids']}",
        f"  mapping_method: {row['mapping_method'] or '<none>'}",
        f"  mapping_status: {row['mapping_status']}",
        f"  requires_review: {row['requires_review']}",
    ]


def main() -> None:
    meddra_path = resolve_meddra_path()
    for path in (PHENOTYPE_PATH, NAME_CANDIDATES_PATH):
        require_file(path)
    side_effects = read_csv_rows(SIDE_EFFECT_PATH)
    sider_umls = read_csv_rows(SIDER_UMLS_PATH)
    umls_summary = read_csv_rows(UMLS_SUMMARY_PATH)
    validate_inputs(side_effects, sider_umls, umls_summary)
    by_cui, by_name, structure = read_meddra_dictionary(meddra_path)
    phenotype_nodes = build_phenotype_index(pl.read_parquet(PHENOTYPE_PATH))

    output_rows: list[dict[str, Any]] = []
    for index, source in enumerate(side_effects):
        cuis = parse_json_list(sider_umls[index]["umls_cuis"])
        sider_cui = cuis[0].upper() if len(cuis) == 1 else ""
        cui_records = by_cui.get(sider_cui, {}) if sider_cui else {}
        name_records = by_name.get(normalize_name(source["side_effect_name"]), {})

        selected: tuple[str, str] | None = None
        mapping_method = ""
        remaining_ids: set[str] = set()
        if len(cui_records) == 1:
            selected = next(iter(cui_records.values()))
            mapping_method = "sider_umls_pt"
        elif len(name_records) == 1:
            selected = next(iter(name_records.values()))
            mapping_method = "exact_meddra_pt_name"
        else:
            remaining_ids = set(cui_records) | set(name_records)

        aliases = retained_non_meddra_aliases(umls_summary[index])
        meddra_id = selected[0] if selected else ""
        meddra_name = selected[1] if selected else ""
        optimuskg_id = f"meddra:{meddra_id}" if meddra_id else ""
        node_exists = bool(optimuskg_id and optimuskg_id in phenotype_nodes)
        if selected and not node_exists:
            status = "meddra_resolved_but_not_in_optimuskg"
        elif selected and aliases:
            status = "meddra_with_hpo_alias"
        elif selected:
            status = "meddra_by_umls" if mapping_method == "sider_umls_pt" else "meddra_by_name"
        elif len(remaining_ids) > 1:
            status = "ambiguous_meddra"
        else:
            status = "unresolved"
        output_rows.append({
            "matrix_index": index,
            "side_effect_name": source["side_effect_name"],
            "sider_umls_cui": sider_cui,
            "meddra_id": meddra_id,
            "meddra_name": meddra_name,
            "canonical_optimuskg_id": optimuskg_id if node_exists else "",
            "alias_optimuskg_ids": json_ids(aliases),
            "mapping_method": mapping_method,
            "mapping_status": status,
            "requires_review": status in REVIEW_STATUSES,
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)
    with REVIEW_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(row for row in output_rows if row["requires_review"])

    status_counts = Counter(row["mapping_status"] for row in output_rows)
    by_umls_count = sum(row["mapping_method"] == "sider_umls_pt" for row in output_rows)
    by_name_count = sum(row["mapping_method"] == "exact_meddra_pt_name" for row in output_rows)
    canonical_count = sum(bool(row["canonical_optimuskg_id"]) for row in output_rows)
    alias_count = sum(bool(parse_json_list(row["alias_optimuskg_ids"])) for row in output_rows)
    usable_count = sum(
        bool(row["canonical_optimuskg_id"]) or bool(parse_json_list(row["alias_optimuskg_ids"]))
        for row in output_rows
    )
    print(f"SIDER MedDRA dictionary: {meddra_path}")
    print(f"Detected column-count frequencies: {structure['column_counts']}")
    print(f"Appears to contain a header: {structure['has_header']}")
    print(f"First row: {structure['first_row']}")
    print(f"Total dictionary rows: {structure['total_rows']}")
    print(f"PT rows: {structure['pt_rows']}")
    print(f"Total side effects: {EXPECTED_COUNT}")
    print(f"Resolved MedDRA IDs by UMLS: {by_umls_count}")
    print(f"Resolved MedDRA IDs by exact name fallback: {by_name_count}")
    print(f"Canonical MedDRA nodes found in OptimusKG: {canonical_count}")
    print(f"Side effects with HPO aliases: {alias_count}")
    print(f"MedDRA resolved but OptimusKG node missing: {status_counts['meddra_resolved_but_not_in_optimuskg']}")
    print(f"Ambiguous: {status_counts['ambiguous_meddra']}")
    print(f"Unresolved: {status_counts['unresolved']}")
    print(f"Final canonical MedDRA coverage percentage: {canonical_count / EXPECTED_COUNT * 100:.2f}%")
    print(f"Overall usable KG phenotype percentage: {usable_count / EXPECTED_COUNT * 100:.2f}%")

    by_term = {row["side_effect_name"].casefold(): row for row in output_rows}
    print("\nDetailed requested terms")
    for term in DETAIL_TERMS:
        row = by_term[term.casefold()]
        print(term)
        print("\n".join(detail_lines(row)))
    print("\nAll unresolved/ambiguous cases")
    selected = [row for row in output_rows if row["mapping_status"] in {"unresolved", "ambiguous_meddra"}]
    if not selected:
        print("  None")
    for row in selected:
        print(row["side_effect_name"])
        print("\n".join(detail_lines(row)))
    print(f"\nMapping CSV: {OUTPUT_PATH}")
    print(f"Review CSV: {REVIEW_PATH}")


if __name__ == "__main__":
    main()
