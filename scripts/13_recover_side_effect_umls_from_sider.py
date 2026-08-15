"""Recover SIDER 4.1 MedDRA PT UMLS CUIs for BioKORF side effects."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIDER_REQUESTED_PATH = PROJECT_ROOT / "kg" / "sider" / "meddra_freq.tsv"
SIDER_NESTED_PATH = SIDER_REQUESTED_PATH / "meddra_freq.tsv"
SIDE_EFFECT_PATH = (
    PROJECT_ROOT / "data_processed" / "mappings" / "side_effect_mapping.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "data_processed" / "sider"
MAPPING_PATH = OUTPUT_DIR / "side_effect_sider_umls_mapping.csv"
REPORT_PATH = OUTPUT_DIR / "side_effect_sider_umls_report.txt"

EXPECTED_COUNT = 994
EXPECTED_COLUMNS = 10
FLAT_CHEMICAL_ID_COL = 0
STEREO_CHEMICAL_ID_COL = 1
MEDDRA_CONCEPT_TYPE_COL = 7
MEDDRA_UMLS_CUI_COL = 8
MEDDRA_TERM_NAME_COL = 9
WHITESPACE = re.compile(r"\s+")
DETAIL_TERMS = (
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


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).strip().casefold()
    return WHITESPACE.sub(" ", value)


def resolve_sider_path() -> Path:
    if SIDER_REQUESTED_PATH.is_file():
        path = SIDER_REQUESTED_PATH
    elif SIDER_REQUESTED_PATH.is_dir() and SIDER_NESTED_PATH.is_file():
        path = SIDER_NESTED_PATH
    else:
        raise FileNotFoundError(
            "Plain SIDER meddra_freq.tsv not found. Expected either "
            f"{SIDER_REQUESTED_PATH} or {SIDER_NESTED_PATH}"
        )
    if path.suffix.casefold() == ".gz":
        raise ValueError(f"Expected plain TSV input, not gzip: {path}")
    return path


def read_side_effects() -> list[dict[str, str]]:
    if not SIDE_EFFECT_PATH.is_file():
        raise FileNotFoundError(
            f"BioKORF side-effect mapping not found: {SIDE_EFFECT_PATH}"
        )
    with SIDE_EFFECT_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_COUNT:
        raise ValueError(f"Expected {EXPECTED_COUNT} side effects, found {len(rows)}")
    indices = [int(row["matrix_index"]) for row in rows]
    if indices != list(range(EXPECTED_COUNT)):
        raise ValueError("matrix_index must be exactly continuous from 0 through 993")
    return rows


def inspect_and_match_sider(
    sider_path: Path, target_names: set[str]
) -> tuple[dict[str, list[tuple[str, str, str, str]]], dict[str, object]]:
    matches: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    column_counts: Counter[int] = Counter()
    first_row: list[str] | None = None
    total_rows = 0
    pt_rows = 0

    with sider_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for line_number, row in enumerate(reader, start=1):
            if not row:
                continue
            total_rows += 1
            column_counts[len(row)] += 1
            if first_row is None:
                first_row = row
            if len(row) < EXPECTED_COLUMNS:
                raise ValueError(
                    f"Malformed SIDER row {line_number}: expected at least "
                    f"{EXPECTED_COLUMNS} columns, found {len(row)}"
                )
            if row[MEDDRA_CONCEPT_TYPE_COL].strip() != "PT":
                continue
            pt_rows += 1
            normalized = normalize_name(row[MEDDRA_TERM_NAME_COL])
            if normalized in target_names:
                matches[normalized].append(
                    (
                        row[MEDDRA_TERM_NAME_COL].strip(),
                        row[MEDDRA_UMLS_CUI_COL].strip(),
                        row[FLAT_CHEMICAL_ID_COL].strip(),
                        row[STEREO_CHEMICAL_ID_COL].strip(),
                    )
                )

    if first_row is None:
        raise ValueError(f"SIDER file is empty: {sider_path}")
    header_tokens = {
        "flat_chemical_id", "stereo_chemical_id", "concept_type", "umls_cui"
    }
    appears_to_have_header = any(
        normalize_name(value).replace(" ", "_") in header_tokens
        for value in first_row
    )
    structure = {
        "total_rows": total_rows,
        "pt_rows": pt_rows,
        "column_counts": dict(sorted(column_counts.items())),
        "appears_to_have_header": appears_to_have_header,
        "first_row": first_row,
    }
    return matches, structure


def format_detail(row: dict[str, object]) -> list[str]:
    return [
        f"  matrix_index: {row['matrix_index']}",
        f"  side_effect_name: {row['side_effect_name']}",
        f"  sider_found: {row['sider_found']}",
        f"  sider_pt_row_count: {row['sider_pt_row_count']}",
        f"  umls_cui_count: {row['umls_cui_count']}",
        f"  umls_cuis: {row['umls_cuis']}",
        f"  matched_meddra_names: {row['matched_meddra_names']}",
        f"  mapping_status: {row['mapping_status']}",
    ]


def main() -> None:
    side_effects = read_side_effects()
    target_names = {normalize_name(row["side_effect_name"]) for row in side_effects}
    sider_path = resolve_sider_path()
    matched_rows, structure = inspect_and_match_sider(sider_path, target_names)

    output_rows: list[dict[str, object]] = []
    for source in side_effects:
        records = matched_rows.get(normalize_name(source["side_effect_name"]), [])
        cuis = sorted({record[1] for record in records if record[1]})
        names = sorted({record[0] for record in records if record[0]}, key=str.casefold)
        status = (
            "unmatched" if not cuis else "unique_umls" if len(cuis) == 1 else "multiple_umls"
        )
        output_rows.append(
            {
                "matrix_index": int(source["matrix_index"]),
                "side_effect_name": source["side_effect_name"],
                "sider_found": bool(records),
                "sider_pt_row_count": len(records),
                "umls_cui_count": len(cuis),
                "umls_cuis": json.dumps(cuis, ensure_ascii=False),
                "matched_meddra_names": json.dumps(names, ensure_ascii=False),
                "mapping_status": status,
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "matrix_index", "side_effect_name", "sider_found", "sider_pt_row_count",
        "umls_cui_count", "umls_cuis", "matched_meddra_names", "mapping_status",
    ]
    with MAPPING_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    counts = Counter(str(row["mapping_status"]) for row in output_rows)
    covered = counts["unique_umls"] + counts["multiple_umls"]
    report: list[str] = [
        "BioKORF side-effect SIDER/MedDRA PT UMLS recovery",
        "=" * 55,
        f"SIDER input: {sider_path}",
        "Input type: plain tab-separated text (not gzip)",
        f"Detected column-count frequencies: {structure['column_counts']}",
        f"Appears to contain a header: {structure['appears_to_have_header']}",
        f"First row: {structure['first_row']}",
        f"Total SIDER rows: {structure['total_rows']}",
        f"MedDRA PT rows: {structure['pt_rows']}",
        "Observed positional fields used (zero-based): 0=STITCH flat chemical ID, "
        "1=STITCH stereo chemical ID, 7=MedDRA concept type, "
        "8=MedDRA-term UMLS CUI, 9=MedDRA term name",
        "",
        f"Total BioKORF side effects: {EXPECTED_COUNT}",
        f"unique_umls: {counts['unique_umls']}",
        f"multiple_umls: {counts['multiple_umls']}",
        f"unmatched: {counts['unmatched']}",
        f"SIDER/UMLS coverage percentage: {covered / EXPECTED_COUNT * 100:.2f}%",
        "",
        "Detailed requested terms",
        "-" * 24,
    ]
    by_name = {normalize_name(str(row["side_effect_name"])): row for row in output_rows}
    for term in DETAIL_TERMS:
        report.append(term)
        report.extend(format_detail(by_name[normalize_name(term)]))
        report.append("")

    for status, title in (("multiple_umls", "All multiple-UMLS terms"),
                          ("unmatched", "All unmatched terms")):
        report.extend([title, "-" * len(title)])
        selected = [row for row in output_rows if row["mapping_status"] == status]
        if not selected:
            report.append("  None")
        for row in selected:
            report.extend(format_detail(row))
            report.append("")

    report_text = "\n".join(report).rstrip() + "\n"
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(report_text, end="")
    print(f"Mapping CSV: {MAPPING_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
