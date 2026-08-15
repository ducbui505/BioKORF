"""Build stable BioKORF drug anchors with provisional OptimusKG mappings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MAPPING_PATH = (
    PROJECT_ROOT / "data_processed" / "mappings" / "drug_mapping.csv"
)
FUSED_PATH = (
    PROJECT_ROOT / "data_processed" / "optimuskg" / "drug_mapping_evidence_fused.csv"
)
TRIAGE_PATH = (
    PROJECT_ROOT / "data_processed" / "optimuskg" / "drug_mapping_triage.csv"
)
MANUAL_REVIEW_PATH = (
    PROJECT_ROOT / "data_processed" / "optimuskg" / "drug_mapping_manual_review.csv"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "data_processed" / "optimuskg" / "final_drug_anchor_mapping.csv"
)

EXPECTED_DRUG_COUNT = 757
RETAINING_TRIAGE_CLASSES = {
    "SAFE_MULTI_NODE",
    "SINGLE_HIGH_CONFIDENCE",
    "BIOLOGIC_OR_PEPTIDE",
    "COMBINATION_DRUG",
}
REVIEW_TRIAGE_CLASSES = {
    "FORM_OR_SALT_AMBIGUITY",
    "TRUE_CONFLICT",
    "NO_OPTIMUSKG_NODE",
}
ALLOWED_KG_STATUSES = {
    "mapped_single",
    "mapped_multi",
    "review_required",
    "unmapped",
}
OUTPUT_SCHEMA = {
    "matrix_index": pl.Int64,
    "biokorf_drug_id": pl.String,
    "drug_name": pl.String,
    "stitch_id": pl.String,
    "entity_type": pl.String,
    "optimuskg_ids": pl.String,
    "optimuskg_node_count": pl.Int64,
    "mapping_source": pl.String,
    "mapping_confidence": pl.String,
    "kg_mapping_status": pl.String,
    "requires_manual_review": pl.Boolean,
}


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found. Expected path: {path}")


def parse_ids(value: str | None) -> list[str]:
    if not value:
        return []
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError(f"Expected a JSON list of string IDs; found {value!r}")
    return sorted(set(parsed))


def dump_ids(values: list[str]) -> str:
    return json.dumps(sorted(set(values)), ensure_ascii=False)


def validate_inputs(
    source: pl.DataFrame,
    fused: pl.DataFrame,
    triage: pl.DataFrame,
    manual_review: pl.DataFrame,
) -> None:
    source_columns = {"matrix_index", "drug_name", "stitch_id"}
    if missing := source_columns.difference(source.columns):
        raise ValueError(f"Source drug mapping is missing columns: {sorted(missing)}")
    if source.height != EXPECTED_DRUG_COUNT:
        raise ValueError(f"Source drug mapping must contain {EXPECTED_DRUG_COUNT} rows")
    if source["matrix_index"].to_list() != list(range(EXPECTED_DRUG_COUNT)):
        raise ValueError("Source matrix_index must be continuous from 0 through 756")

    fused_columns = {
        "matrix_index",
        "drug_name",
        "stitch_id",
        "proposed_optimuskg_id",
        "evidence_class",
        "confidence",
        "final_status",
    }
    if missing := fused_columns.difference(fused.columns):
        raise ValueError(f"Fused mapping is missing columns: {sorted(missing)}")
    if fused.height != EXPECTED_DRUG_COUNT:
        raise ValueError(f"Fused mapping must contain {EXPECTED_DRUG_COUNT} rows")
    if not fused.select(["matrix_index", "drug_name", "stitch_id"]).equals(
        source.select(["matrix_index", "drug_name", "stitch_id"])
    ):
        raise ValueError("Fused mapping rows do not align with source drug mapping")

    triage_columns = {
        "matrix_index",
        "drug_name",
        "stitch_id",
        "triage_class",
        "supported_optimuskg_ids",
        "requires_manual_review",
    }
    if missing := triage_columns.difference(triage.columns):
        raise ValueError(f"Triage mapping is missing columns: {sorted(missing)}")
    unresolved_fused = fused.filter(pl.col("final_status") != "resolved")
    if not triage.select(["matrix_index", "drug_name", "stitch_id"]).equals(
        unresolved_fused.select(["matrix_index", "drug_name", "stitch_id"])
    ):
        raise ValueError("Triage rows do not align with unresolved fused rows")
    expected_manual = triage.filter(pl.col("requires_manual_review"))
    if not manual_review.equals(expected_manual):
        raise ValueError("Manual-review file is not the manual subset of triage")


def entity_type_for_triage(triage_class: str) -> str:
    if triage_class == "COMBINATION_DRUG":
        return "combination_drug"
    if triage_class == "BIOLOGIC_OR_PEPTIDE":
        return "biologic_or_peptide"
    return "drug"


def status_for_ids(ids: list[str], requires_manual_review: bool) -> str:
    if requires_manual_review:
        return "review_required"
    if len(ids) == 1:
        return "mapped_single"
    if len(ids) > 1:
        return "mapped_multi"
    return "unmapped"


def build_anchor_mapping(
    source: pl.DataFrame, fused: pl.DataFrame, triage: pl.DataFrame
) -> pl.DataFrame:
    fused_by_index = {
        row["matrix_index"]: row for row in fused.iter_rows(named=True)
    }
    triage_by_index = {
        row["matrix_index"]: row for row in triage.iter_rows(named=True)
    }
    rows: list[dict[str, Any]] = []

    for drug in source.iter_rows(named=True):
        matrix_index = drug["matrix_index"]
        fused_row = fused_by_index[matrix_index]
        entity_type = "drug"

        if fused_row["final_status"] == "resolved":
            selected_id = fused_row["proposed_optimuskg_id"] or ""
            if not selected_id:
                raise ValueError(
                    f"Resolved fused row lacks proposed ID at matrix_index {matrix_index}"
                )
            optimuskg_ids = [selected_id]
            mapping_source = f"evidence_fusion:{fused_row['evidence_class']}"
            mapping_confidence = fused_row["confidence"] or "high"
            requires_manual_review = False
        else:
            if matrix_index not in triage_by_index:
                raise ValueError(f"Missing triage row for matrix_index {matrix_index}")
            triage_row = triage_by_index[matrix_index]
            triage_class = triage_row["triage_class"]
            entity_type = entity_type_for_triage(triage_class)
            mapping_source = f"triage:{triage_class}"

            if triage_class in RETAINING_TRIAGE_CLASSES:
                optimuskg_ids = parse_ids(triage_row["supported_optimuskg_ids"])
                if not optimuskg_ids:
                    raise ValueError(
                        f"Retaining triage class {triage_class} has no supported IDs "
                        f"at matrix_index {matrix_index}"
                    )
                mapping_confidence = (
                    "very_high"
                    if triage_class == "SINGLE_HIGH_CONFIDENCE"
                    else "high"
                )
                requires_manual_review = False
            elif triage_class in REVIEW_TRIAGE_CLASSES:
                # Preserve the anchor but deliberately do not select disputed nodes.
                optimuskg_ids = []
                mapping_confidence = "review_required"
                requires_manual_review = True
            else:
                raise ValueError(
                    f"Unexpected triage class {triage_class!r} at matrix_index "
                    f"{matrix_index}"
                )

        kg_mapping_status = status_for_ids(
            optimuskg_ids, requires_manual_review
        )
        rows.append(
            {
                "matrix_index": matrix_index,
                "biokorf_drug_id": f"BIOKORF_DRUG_{matrix_index:03d}",
                "drug_name": drug["drug_name"],
                "stitch_id": drug["stitch_id"],
                "entity_type": entity_type,
                "optimuskg_ids": dump_ids(optimuskg_ids),
                "optimuskg_node_count": len(optimuskg_ids),
                "mapping_source": mapping_source,
                "mapping_confidence": mapping_confidence,
                "kg_mapping_status": kg_mapping_status,
                "requires_manual_review": requires_manual_review,
            }
        )
    return pl.DataFrame(rows, schema=OUTPUT_SCHEMA)


def validate_output(output: pl.DataFrame, source: pl.DataFrame) -> None:
    if output.height != EXPECTED_DRUG_COUNT:
        raise ValueError(f"Output must contain {EXPECTED_DRUG_COUNT} rows")
    if output["matrix_index"].to_list() != list(range(EXPECTED_DRUG_COUNT)):
        raise ValueError("Output matrix_index must be exactly 0 through 756")
    expected_anchors = [
        f"BIOKORF_DRUG_{index:03d}" for index in range(EXPECTED_DRUG_COUNT)
    ]
    if output["biokorf_drug_id"].to_list() != expected_anchors:
        raise ValueError("Anchor IDs do not correspond exactly to matrix_index")
    if output["biokorf_drug_id"].n_unique() != EXPECTED_DRUG_COUNT:
        raise ValueError("Anchor IDs are not unique")
    if not output.select(["matrix_index", "drug_name", "stitch_id"]).equals(
        source.select(["matrix_index", "drug_name", "stitch_id"])
    ):
        raise ValueError("A BioKORF source drug is missing or reordered")
    if output.item(0, "biokorf_drug_id") != "BIOKORF_DRUG_000":
        raise ValueError("The first anchor must be BIOKORF_DRUG_000")
    if output.item(0, "drug_name").strip().casefold() != "lepirudin":
        raise ValueError("BIOKORF_DRUG_000 must correspond to lepirudin")
    if not set(output["kg_mapping_status"]).issubset(ALLOWED_KG_STATUSES):
        raise ValueError("Output contains a disallowed kg_mapping_status")

    for row in output.iter_rows(named=True):
        ids = parse_ids(row["optimuskg_ids"])
        if len(ids) != row["optimuskg_node_count"]:
            raise ValueError(
                f"Node count mismatch at matrix_index {row['matrix_index']}"
            )
        if row["requires_manual_review"] != (
            row["kg_mapping_status"] == "review_required"
        ):
            raise ValueError(
                f"Manual-review flag mismatch at matrix_index {row['matrix_index']}"
            )


def print_summary(output: pl.DataFrame) -> None:
    counts = dict(output.group_by("kg_mapping_status").len().iter_rows())
    usable = output.filter(
        pl.col("kg_mapping_status").is_in(["mapped_single", "mapped_multi"])
    ).height
    print("Final BioKORF drug anchor mapping summary")
    print(f"mapped_single count: {counts.get('mapped_single', 0)}")
    print(f"mapped_multi count: {counts.get('mapped_multi', 0)}")
    print(f"review_required count: {counts.get('review_required', 0)}")
    print(f"unmapped count: {counts.get('unmapped', 0)}")
    print(f"usable KG anchor count: {usable}")
    print(f"KG usable percentage: {100.0 * usable / output.height:.2f}%")
    print(f"Final anchor mapping saved to: {OUTPUT_PATH}")
    print("No previous mapping file was modified.")


def main() -> None:
    for path, description in (
        (SOURCE_MAPPING_PATH, "Source BioKORF drug mapping"),
        (FUSED_PATH, "Fused drug mapping evidence"),
        (TRIAGE_PATH, "Drug mapping triage"),
        (MANUAL_REVIEW_PATH, "Drug mapping manual-review subset"),
    ):
        require_file(path, description)

    source = pl.read_csv(SOURCE_MAPPING_PATH)
    fused = pl.read_csv(FUSED_PATH)
    triage = pl.read_csv(TRIAGE_PATH)
    manual_review = pl.read_csv(MANUAL_REVIEW_PATH)
    validate_inputs(source, fused, triage, manual_review)

    output = build_anchor_mapping(source, fused, triage)
    validate_output(output, source)
    output.write_csv(OUTPUT_PATH)
    print_summary(output)


if __name__ == "__main__":
    main()
