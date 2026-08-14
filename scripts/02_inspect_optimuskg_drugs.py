"""Inspect the OptimusKG drug node table without loading the full graph."""

from __future__ import annotations

from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Iterator, TextIO

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = (
    PROJECT_ROOT / "data_processed" / "optimuskg" / "drug_node_inspection.txt"
)
EXAMPLE_DRUGS = (
    "lepirudin",
    "bivalirudin",
    "leuprorelin",
    "goserelin",
    "erythropoietin",
    "insulin",
    "metformin",
    "aspirin",
)


class Tee:
    """Write identical inspection output to the console and a report file."""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def iter_text_values(value: Any, path: str) -> Iterator[tuple[str, str]]:
    """Yield every string leaf and its path from nested structs/lists."""

    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, nested_value in value.items():
            nested_path = f"{path}.{key}" if path else str(key)
            yield from iter_text_values(nested_value, nested_path)
    elif isinstance(value, (list, tuple)):
        for nested_value in value:
            yield from iter_text_values(nested_value, f"{path}[]")


def infer_column_roles(textual_paths: set[str]) -> dict[str, list[str]]:
    """Identify likely semantic roles from textual field names."""

    role_keywords = {
        "canonical drug name": ("name", "preferred", "title"),
        "synonyms": ("synonym", "alias", "alternative", "alternate"),
        "identifiers": (
            "id",
            "identifier",
            "accession",
            "curie",
            "inchi",
            "smiles",
        ),
        "cross references": ("xref", "cross_ref", "crossref", "external"),
    }
    candidates: dict[str, list[str]] = {}
    for role, keywords in role_keywords.items():
        candidates[role] = sorted(
            path
            for path in textual_paths
            if any(keyword in path.casefold() for keyword in keywords)
        )
    return candidates


def inspect_drug_table() -> None:
    drug_path = PROJECT_ROOT / "kg" / "optimuskg" / "nodes" / "drug.parquet"
    if not drug_path.is_file():
        raise FileNotFoundError(
            f"OptimusKG drug node file not found. Expected path: {drug_path}"
        )
    drugs = pl.read_parquet(drug_path)

    print("OptimusKG drug node inspection")
    print("Loaded resource: nodes/drug.parquet")
    print(f"Dataframe type: {type(drugs)}")
    print(f"Shape: {drugs.shape}")
    print(f"All column names: {drugs.columns}")
    print("Schema/dtypes:")
    for column, dtype in drugs.schema.items():
        print(f"  {column}: {dtype}")
    print("First 10 rows:")
    print(drugs.head(10))

    textual_paths: set[str] = set()
    matches: dict[str, list[tuple[int, set[str], dict[str, Any]]]] = defaultdict(list)
    targets = {drug.casefold(): drug for drug in EXAMPLE_DRUGS}

    for row_index, row in enumerate(drugs.iter_rows(named=True)):
        row_matches: dict[str, set[str]] = defaultdict(set)
        for column, value in row.items():
            for path, text in iter_text_values(value, column):
                textual_paths.add(path)
                target = targets.get(text.casefold())
                if target is not None:
                    row_matches[target].add(path)
        for target, paths in row_matches.items():
            matches[target].append((row_index, paths, row))

    print("Textual columns/field paths discovered automatically:")
    for path in sorted(textual_paths):
        print(f"  {path}")

    print("Likely textual field roles (inferred from field names):")
    for role, paths in infer_column_roles(textual_paths).items():
        print(f"  {role}:")
        if paths:
            for path in paths:
                print(f"    {path}")
        else:
            print("    No likely field detected")

    print("Exact case-insensitive example-drug search results:")
    for drug in EXAMPLE_DRUGS:
        drug_matches = matches.get(drug, [])
        status = "FOUND" if drug_matches else "NOT FOUND"
        print(f"  {drug}: {status} ({len(drug_matches)} matching row(s))")
        for row_index, paths, row in drug_matches:
            row_id = row.get("id", "<no id column>")
            print(
                f"    row_index={row_index}, id={row_id}, "
                f"matched_field_paths={sorted(paths)}"
            )

    print("No fuzzy matching was performed.")


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as report:
        with redirect_stdout(Tee(__import__("sys").stdout, report)):
            inspect_drug_table()
            print(f"Inspection report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
