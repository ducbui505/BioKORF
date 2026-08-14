"""Safely inspect the local STITCH v5.0 chemical ID to InChIKey table."""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from typing import Iterable, TextIO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PATH = PROJECT_ROOT / "kg" / "stitch" / "chemicals.inchikeys.v5.0.tsv"
REPORT_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "stitch"
    / "stitch_inchikey_inspection.txt"
)
SEARCH_IDS = (
    "CIDm16132441",
    "CIDm16129704",
    "CIDm00657180",
    "CIDm00047725",
    "CIDm05288169",
    "CIDm16129672",
    "CIDm00444013",
    "CIDm00358641",
    "CIDm00004274",
    "CIDm00003738",
)

STITCH_ID_PATTERN = re.compile(r"^CID[ms]\d+$")
INCHIKEY_PATTERN = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")


class Tee:
    """Write the same report to the console and a text file."""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def resolve_stitch_file() -> tuple[Path, str | None]:
    """Validate the expected path and accommodate a nested extraction folder."""

    if not EXPECTED_PATH.exists():
        raise FileNotFoundError(
            f"STITCH InChIKey path not found. Expected path: {EXPECTED_PATH}"
        )
    if EXPECTED_PATH.is_file():
        return EXPECTED_PATH, None

    nested_path = EXPECTED_PATH / EXPECTED_PATH.name
    if nested_path.is_file():
        warning = (
            f"WARNING: expected file path is a directory: {EXPECTED_PATH}\n"
            f"Using nested extracted TSV file: {nested_path}"
        )
        return nested_path, warning
    raise FileNotFoundError(
        "The expected STITCH path exists but is not a file, and no nested TSV "
        f"was found. Expected file path: {EXPECTED_PATH}"
    )


def format_size(size_bytes: int) -> str:
    size = float(size_bytes)
    units = ("bytes", "KiB", "MiB", "GiB", "TiB")
    unit = units[0]
    for unit in units:
        if size < 1024 or unit == units[-1]:
            break
        size /= 1024
    return f"{size:.2f} {unit} ({size_bytes:,} bytes)"


def read_raw_sample(path: Path, line_count: int = 10) -> list[str]:
    lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for _ in range(line_count):
            line = handle.readline()
            if not line:
                break
            lines.append(line.rstrip("\r\n"))
    return lines


def looks_like_header(first_row: list[str], later_rows: list[list[str]]) -> bool:
    if not first_row or not later_rows:
        return False
    first_has_data_pattern = any(
        STITCH_ID_PATTERN.fullmatch(value) or INCHIKEY_PATTERN.fullmatch(value)
        for value in first_row
    )
    later_has_data_pattern = any(
        STITCH_ID_PATTERN.fullmatch(value) or INCHIKEY_PATTERN.fullmatch(value)
        for row in later_rows
        for value in row
    )
    header_words = ("id", "cid", "inchikey", "chemical", "source")
    first_has_labels = any(
        any(word in value.strip().casefold() for word in header_words)
        for value in first_row
    )
    return first_has_labels and not first_has_data_pattern and later_has_data_pattern


def detect_columns(
    data_rows: Iterable[list[str]], column_count: int
) -> tuple[int, int, list[int], list[int]]:
    rows = list(data_rows)
    stitch_scores = [0] * column_count
    flat_id_scores = [0] * column_count
    inchikey_scores = [0] * column_count
    for row in rows:
        if len(row) != column_count:
            continue
        for index, value in enumerate(row):
            if STITCH_ID_PATTERN.fullmatch(value):
                stitch_scores[index] += 1
                if value.startswith("CIDm"):
                    flat_id_scores[index] += 1
            if INCHIKEY_PATTERN.fullmatch(value):
                inchikey_scores[index] += 1

    if max(flat_id_scores, default=0) > 0:
        id_column = flat_id_scores.index(max(flat_id_scores))
    elif max(stitch_scores, default=0) > 0:
        id_column = stitch_scores.index(max(stitch_scores))
    else:
        raise ValueError("Could not detect a STITCH chemical ID column from sample rows")
    if max(inchikey_scores, default=0) == 0:
        raise ValueError("Could not detect an InChIKey column from sample rows")
    inchikey_column = inchikey_scores.index(max(inchikey_scores))
    return id_column, inchikey_column, stitch_scores, inchikey_scores


def row_as_record(row: list[str], column_names: list[str]) -> dict[str, str]:
    return {
        column_names[index]: value
        for index, value in enumerate(row)
        if index < len(column_names)
    }


def inspect_file(path: Path, path_warning: str | None) -> None:
    if path_warning:
        print(path_warning)
    print("STITCH v5.0 chemical ID to InChIKey inspection")
    print(f"Expected path: {EXPECTED_PATH}")
    print(f"Inspected file: {path}")
    print(f"File size: {format_size(path.stat().st_size)}")

    raw_lines = read_raw_sample(path, 10)
    if not raw_lines:
        raise ValueError(f"STITCH file is empty: {path}")
    print("\nFirst 10 raw lines:")
    for line_number, line in enumerate(raw_lines, start=1):
        print(f"{line_number}: {line}")

    parsed_sample = list(csv.reader(raw_lines, delimiter="\t"))
    column_counts = Counter(len(row) for row in parsed_sample)
    detected_column_count = column_counts.most_common(1)[0][0]
    header_present = looks_like_header(parsed_sample[0], parsed_sample[1:])
    if header_present:
        column_names = parsed_sample[0]
        sample_data_rows = parsed_sample[1:]
    else:
        column_names = [f"column_{index + 1}" for index in range(detected_column_count)]
        sample_data_rows = parsed_sample

    id_column, inchikey_column, stitch_scores, inchikey_scores = detect_columns(
        sample_data_rows, detected_column_count
    )
    print("\nDetected file structure:")
    print(f"Detected number of columns: {detected_column_count}")
    print(f"Observed sample column counts: {dict(sorted(column_counts.items()))}")
    print(f"Header appears present: {'yes' if header_present else 'no'}")
    print(f"Detected column labels: {column_names}")
    print(
        "Detected STITCH chemical ID column: "
        f"index {id_column} ({column_names[id_column]!r})"
    )
    print(
        "Detected InChIKey column: "
        f"index {inchikey_column} ({column_names[inchikey_column]!r})"
    )
    print(f"STITCH-pattern scores by column: {stitch_scores}")
    print(f"InChIKey-pattern scores by column: {inchikey_scores}")

    print("\nExample parsed rows:")
    for row in sample_data_rows[:5]:
        print(row_as_record(row, column_names))

    targets = set(SEARCH_IDS)
    matches: dict[str, list[list[str]]] = defaultdict(list)
    malformed_rows = 0
    rows_scanned = 0
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        if header_present:
            next(reader, None)
        for row in reader:
            rows_scanned += 1
            if len(row) != detected_column_count:
                malformed_rows += 1
                continue
            stitch_id = row[id_column]
            if stitch_id in targets:
                matches[stitch_id].append(row)

    print("\nSearch results for the 10 BioKORF STITCH IDs:")
    for stitch_id in SEARCH_IDS:
        rows = matches.get(stitch_id, [])
        print(f"{stitch_id}: {'FOUND' if rows else 'NOT FOUND'}")
        print(f"  Matching rows: {len(rows)}")
        for row_number, row in enumerate(rows, start=1):
            print(f"  Complete row {row_number}: {'<TAB>'.join(row)}")
            print(f"  Parsed row {row_number}: {row_as_record(row, column_names)}")
        distinct_inchikeys = sorted(
            {row[inchikey_column] for row in rows if row[inchikey_column].strip()}
        )
        print(f"  Distinct non-empty InChIKeys: {len(distinct_inchikeys)}")
        print(f"  InChIKey values: {distinct_inchikeys}")
        print(
            "  Maps to multiple different InChIKeys: "
            f"{'yes' if len(distinct_inchikeys) > 1 else 'no'}"
        )

    print("\nStreaming scan diagnostics:")
    print(f"Data rows scanned: {rows_scanned:,}")
    print(f"Rows with unexpected column count: {malformed_rows:,}")
    multiple_ids = [
        stitch_id
        for stitch_id in SEARCH_IDS
        if len(
            {
                row[inchikey_column]
                for row in matches.get(stitch_id, [])
                if row[inchikey_column].strip()
            }
        )
        > 1
    ]
    print(f"Searched IDs mapping to multiple different InChIKeys: {multiple_ids}")
    print("No mapping, fuzzy matching, network access, or source modification was performed.")


def main() -> None:
    stitch_path, path_warning = resolve_stitch_file()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as report:
        with redirect_stdout(Tee(sys.stdout, report)):
            inspect_file(stitch_path, path_warning)
            print(f"Inspection report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
