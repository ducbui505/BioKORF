"""Inspect the local OptimusKG phenotype node table before entity mapping."""

from __future__ import annotations

import sys
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Iterator, TextIO

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHENOTYPE_PATH = PROJECT_ROOT / "kg" / "optimuskg" / "nodes" / "phenotype.parquet"
REPORT_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "optimuskg"
    / "phenotype_node_inspection.txt"
)
EXAMPLE_SIDE_EFFECTS = (
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


class Tee:
    """Write identical output to the console and report file."""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"OptimusKG phenotype node file not found. Expected path: {path}"
        )


def iter_text_values(value: Any, path: str) -> Iterator[tuple[str, str]]:
    """Yield paths and values for every string leaf in nested data."""

    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, nested_value in value.items():
            nested_path = f"{path}.{key}" if path else str(key)
            yield from iter_text_values(nested_value, nested_path)
    elif isinstance(value, (list, tuple)):
        for nested_value in value:
            yield from iter_text_values(nested_value, f"{path}[]")


def identify_likely_fields(textual_paths: set[str]) -> dict[str, list[str]]:
    """Infer semantic field roles from discovered nested field names."""

    roles: dict[str, list[str]] = {}
    roles["canonical phenotype name"] = sorted(
        path for path in textual_paths if path == "properties.name"
    )
    roles["synonyms"] = sorted(
        path
        for path in textual_paths
        if "synonym" in path.casefold()
        or path in {"properties.concept_names[]", "properties.snomed_full_names[]"}
    )
    roles["ontology IDs"] = sorted(
        path
        for path in textual_paths
        if path == "id"
        or path == "properties.code"
        or path.endswith("concept_ids[]")
        or path.endswith("umls_cui")
        or path.endswith("snomed_concept_ids[]")
    )
    source_paths = {
        path for path in textual_paths if path.startswith("properties.sources.")
    }
    if source_paths:
        # Include the schema-declared sibling even when its lists are all empty.
        source_paths.add("properties.sources.indirect[]")
    roles["source IDs"] = sorted(source_paths)
    roles["cross references"] = sorted(
        path for path in textual_paths if "xref" in path.casefold()
    )
    return roles


def inspect_phenotypes() -> None:
    require_file(PHENOTYPE_PATH)
    phenotypes = pl.read_parquet(PHENOTYPE_PATH)

    print("OptimusKG phenotype node inspection")
    print(f"Loaded local resource: {PHENOTYPE_PATH}")
    print(f"Dataframe type: {type(phenotypes)}")
    print(f"Shape: {phenotypes.shape}")
    print(f"All column names: {phenotypes.columns}")
    print("Schema/dtypes:")
    for column, dtype in phenotypes.schema.items():
        print(f"  {column}: {dtype}")

    print("First 10 rows:")
    for row_index, row in enumerate(phenotypes.head(10).iter_rows(named=True)):
        print(f"  row_index={row_index}: {row}")

    textual_paths: set[str] = set()
    targets = {value.casefold(): value for value in EXAMPLE_SIDE_EFFECTS}
    matches: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for row in phenotypes.iter_rows(named=True):
        row_matches: dict[str, set[str]] = defaultdict(set)
        for column, value in row.items():
            for path, text in iter_text_values(value, column):
                textual_paths.add(path)
                target = targets.get(text.casefold())
                if target is not None:
                    row_matches[target].add(path)

        properties = row.get("properties") or {}
        for target, matched_paths in row_matches.items():
            node_id = row.get("id") or "<missing id>"
            matches[target][node_id] = {
                "id": node_id,
                "canonical_name": properties.get("name") or "",
                "matched_fields": sorted(matched_paths),
            }

    print("Textual/nested field paths discovered automatically:")
    for path in sorted(textual_paths):
        print(f"  {path}")

    print("Likely field roles inferred from the actual nested schema:")
    for role, paths in identify_likely_fields(textual_paths).items():
        print(f"  {role}:")
        if paths:
            for path in paths:
                print(f"    {path}")
        else:
            print("    No likely field detected")

    print("Exact case-insensitive side-effect search results:")
    for side_effect in EXAMPLE_SIDE_EFFECTS:
        nodes = sorted(
            matches.get(side_effect, {}).values(), key=lambda node: node["id"]
        )
        status = "FOUND" if nodes else "NOT FOUND"
        print(f"{side_effect}: {status}")
        print(f"  Matching phenotype nodes: {len(nodes)}")
        print(f"  Matching node IDs: {[node['id'] for node in nodes]}")
        print(f"  Canonical names: {[node['canonical_name'] for node in nodes]}")
        print("  Match details:")
        if not nodes:
            print("    None")
        for node in nodes:
            print(
                f"    id={node['id']} | canonical_name={node['canonical_name']!r} | "
                f"matched_fields={node['matched_fields']}"
            )

    print("No fuzzy matching or final mapping was performed.")


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as report:
        with redirect_stdout(Tee(sys.stdout, report)):
            inspect_phenotypes()
            print(f"Inspection report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
