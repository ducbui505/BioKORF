"""Map BioKORF side effects to OptimusKG phenotypes by exact UMLS CUI."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIDER_MAPPING_PATH = (
    PROJECT_ROOT / "data_processed" / "sider" / "side_effect_sider_umls_mapping.csv"
)
PHENOTYPE_PATH = PROJECT_ROOT / "kg" / "optimuskg" / "nodes" / "phenotype.parquet"
NAME_CANDIDATES_PATH = (
    PROJECT_ROOT / "data_processed" / "optimuskg" / "side_effect_mapping_candidates.csv"
)
NAME_SUMMARY_PATH = (
    PROJECT_ROOT / "data_processed" / "optimuskg" / "side_effect_mapping_summary.csv"
)
MATCHES_PATH = (
    PROJECT_ROOT / "data_processed" / "optimuskg" / "side_effect_umls_optimuskg_matches.csv"
)
SUMMARY_PATH = (
    PROJECT_ROOT / "data_processed" / "optimuskg" / "side_effect_umls_mapping_summary.csv"
)
EXPECTED_COUNT = 994
CUI_PATTERN = re.compile(r"^(?:UMLS:)?(C\d{7})$", re.IGNORECASE)
DETAIL_TERMS = (
    "abdominal discomfort", "abdominal distension", "abdominal pain", "nausea",
    "vomiting", "headache", "dizziness", "dry eye", "proteinuria", "wheezing",
)
STATUSES = (
    "umls_unique_meddra", "umls_meddra_with_aliases", "umls_unique_other",
    "umls_multiple_no_unique_meddra", "umls_no_optimuskg_match", "no_sider_umls",
)
CROSSCHECK_CLASSES = (
    "name_and_umls_agree", "name_and_umls_conflict", "umls_only", "name_only",
)


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def parse_json_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a JSON list, found: {value!r}")
    return [str(item) for item in parsed if item is not None and str(item).strip()]


def json_ids(values: list[str] | set[str]) -> str:
    return json.dumps(sorted(set(values)), ensure_ascii=False, separators=(",", ":"))


def exact_cui(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = CUI_PATTERN.fullmatch(value.strip())
    return match.group(1).upper() if match else None


def ontology_namespace(node_id: str, properties: dict[str, Any]) -> str:
    if node_id.casefold().startswith("meddra:"):
        return "MEDDRA"
    if node_id.upper().startswith("HP_"):
        return "HPO"
    ontology = properties.get("ontology") or {}
    title = ontology.get("title") if isinstance(ontology, dict) else None
    return str(title).strip() if title else "OTHER"


def validate_sider_mapping(frame: pl.DataFrame) -> None:
    required = {"matrix_index", "side_effect_name", "umls_cuis", "mapping_status"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"SIDER mapping is missing columns: {sorted(missing)}")
    if frame.height != EXPECTED_COUNT:
        raise ValueError(f"Expected {EXPECTED_COUNT} rows; found {frame.height}")
    if frame["matrix_index"].to_list() != list(range(EXPECTED_COUNT)):
        raise ValueError("matrix_index must be exactly 0 through 993")


def build_umls_index(
    phenotypes: pl.DataFrame,
) -> dict[str, dict[str, dict[str, Any]]]:
    if missing := {"id", "properties"}.difference(phenotypes.columns):
        raise ValueError(f"Phenotype table is missing columns: {sorted(missing)}")
    index: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in phenotypes.iter_rows(named=True):
        node_id = row["id"]
        properties = row["properties"] or {}
        if not isinstance(node_id, str) or not node_id:
            continue
        values: list[Any] = [properties.get("umls_cui")]
        values.extend(properties.get("concept_ids") or [])
        values.extend(properties.get("xrefs") or [])
        cuis = {cui for value in values if (cui := exact_cui(value))}
        if not cuis:
            continue
        node = {
            "optimuskg_id": node_id,
            "optimuskg_name": properties.get("name") or "",
            "ontology_namespace": ontology_namespace(node_id, properties),
            "optimuskg_umls_cui": properties.get("umls_cui") or "",
            "optimuskg_code": properties.get("code") or "",
        }
        for cui in cuis:
            index[cui][node_id] = node
    return index


def classify(nodes: list[dict[str, Any]], has_sider_cui: bool) -> tuple[str, str, list[str]]:
    if not has_sider_cui:
        return "no_sider_umls", "", []
    if not nodes:
        return "umls_no_optimuskg_match", "", []
    meddra = [node for node in nodes if node["ontology_namespace"] == "MEDDRA"]
    if len(nodes) == 1 and len(meddra) == 1:
        return "umls_unique_meddra", meddra[0]["optimuskg_id"], []
    if len(meddra) == 1:
        canonical = meddra[0]["optimuskg_id"]
        aliases = [node["optimuskg_id"] for node in nodes if node["optimuskg_id"] != canonical]
        return "umls_meddra_with_aliases", canonical, aliases
    if len(nodes) == 1:
        return "umls_unique_other", nodes[0]["optimuskg_id"], []
    return "umls_multiple_no_unique_meddra", "", [node["optimuskg_id"] for node in nodes]


def detail_lines(row: dict[str, Any], crosscheck: str) -> list[str]:
    return [
        f"  matrix_index: {row['matrix_index']}",
        f"  side_effect_name: {row['side_effect_name']}",
        f"  sider_umls_cui: {row['sider_umls_cui'] or '<none>'}",
        f"  optimuskg_match_count: {row['optimuskg_match_count']}",
        f"  meddra_match_count: {row['meddra_match_count']}",
        f"  hpo_match_count: {row['hpo_match_count']}",
        f"  canonical_optimuskg_id: {row['canonical_optimuskg_id'] or '<none>'}",
        f"  alias_optimuskg_ids: {row['alias_optimuskg_ids']}",
        f"  mapping_status: {row['mapping_status']}",
        f"  name/UMLS cross-check: {crosscheck or '<neither>'}",
    ]


def main() -> None:
    for path in (SIDER_MAPPING_PATH, PHENOTYPE_PATH, NAME_CANDIDATES_PATH, NAME_SUMMARY_PATH):
        require_file(path)
    sider = pl.read_csv(SIDER_MAPPING_PATH)
    validate_sider_mapping(sider)
    phenotypes = pl.read_parquet(PHENOTYPE_PATH)
    name_candidates = pl.read_csv(NAME_CANDIDATES_PATH)
    name_summary = pl.read_csv(NAME_SUMMARY_PATH)

    previous_candidates: dict[int, set[str]] = defaultdict(set)
    for row in name_candidates.iter_rows(named=True):
        previous_candidates[int(row["matrix_index"])].add(str(row["optimuskg_id"]))
    preferred_name_id = {
        int(row["matrix_index"]): str(row.get("preferred_meddra_id") or "")
        for row in name_summary.iter_rows(named=True)
    }
    umls_index = build_umls_index(phenotypes)

    match_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    crosschecks: dict[int, str] = {}
    for source in sider.iter_rows(named=True):
        matrix_index = int(source["matrix_index"])
        cuis = parse_json_list(source["umls_cuis"])
        if source["mapping_status"] == "unique_umls" and len(cuis) != 1:
            raise ValueError(f"Row {matrix_index} is unique_umls but has CUIs {cuis}")
        sider_cui = cuis[0].upper() if len(cuis) == 1 else ""
        node_map = umls_index.get(sider_cui, {}) if sider_cui else {}
        nodes = sorted(node_map.values(), key=lambda node: node["optimuskg_id"])
        status, canonical_id, alias_ids = classify(nodes, bool(sider_cui))
        meddra_count = sum(node["ontology_namespace"] == "MEDDRA" for node in nodes)
        hpo_count = sum(node["ontology_namespace"] == "HPO" for node in nodes)

        for node in nodes:
            match_rows.append({
                "matrix_index": matrix_index,
                "side_effect_name": source["side_effect_name"],
                "sider_umls_cui": sider_cui,
                **node,
                "is_meddra": node["ontology_namespace"] == "MEDDRA",
                "was_previous_name_candidate": node["optimuskg_id"] in previous_candidates[matrix_index],
            })
        summary_row = {
            "matrix_index": matrix_index,
            "side_effect_name": source["side_effect_name"],
            "sider_umls_cui": sider_cui,
            "optimuskg_match_count": len(nodes),
            "meddra_match_count": meddra_count,
            "hpo_match_count": hpo_count,
            "canonical_optimuskg_id": canonical_id,
            "alias_optimuskg_ids": json_ids(alias_ids),
            "mapping_status": status,
        }
        summary_rows.append(summary_row)

        name_id = preferred_name_id.get(matrix_index, "")
        umls_meddra_id = canonical_id if meddra_count == 1 else ""
        if name_id and umls_meddra_id:
            crosschecks[matrix_index] = (
                "name_and_umls_agree"
                if name_id == umls_meddra_id
                else "name_and_umls_conflict"
            )
        elif umls_meddra_id:
            crosschecks[matrix_index] = "umls_only"
        elif name_id:
            crosschecks[matrix_index] = "name_only"
        else:
            crosschecks[matrix_index] = ""

    MATCHES_PATH.parent.mkdir(parents=True, exist_ok=True)
    match_schema = {
        "matrix_index": pl.Int64, "side_effect_name": pl.String,
        "sider_umls_cui": pl.String, "optimuskg_id": pl.String,
        "optimuskg_name": pl.String, "ontology_namespace": pl.String,
        "optimuskg_umls_cui": pl.String, "optimuskg_code": pl.String,
        "is_meddra": pl.Boolean, "was_previous_name_candidate": pl.Boolean,
    }
    pl.DataFrame(match_rows, schema=match_schema).write_csv(MATCHES_PATH)
    pl.DataFrame(summary_rows).select(list({
        "matrix_index": None, "side_effect_name": None, "sider_umls_cui": None,
        "optimuskg_match_count": None, "meddra_match_count": None,
        "hpo_match_count": None, "canonical_optimuskg_id": None,
        "alias_optimuskg_ids": None, "mapping_status": None,
    })).write_csv(SUMMARY_PATH)

    status_counts = Counter(row["mapping_status"] for row in summary_rows)
    cross_counts = Counter(value for value in crosschecks.values() if value)
    sider_count = sum(bool(row["sider_umls_cui"]) for row in summary_rows)
    matched_count = sum(row["optimuskg_match_count"] > 0 for row in summary_rows)
    canonical_meddra_count = sum(row["meddra_match_count"] == 1 for row in summary_rows)
    print(f"Total side effects: {EXPECTED_COUNT}")
    print(f"Side effects with SIDER UMLS: {sider_count}")
    print(f"UMLS CUI matched to >=1 OptimusKG phenotype: {matched_count}")
    for status in STATUSES:
        print(f"{status}: {status_counts[status]}")
    print(f"OptimusKG UMLS coverage percentage: {matched_count / EXPECTED_COUNT * 100:.2f}%")
    print(f"MedDRA canonical coverage percentage: {canonical_meddra_count / EXPECTED_COUNT * 100:.2f}%")
    print("\nCross-check against Step 12 preferred MedDRA nodes")
    for label in CROSSCHECK_CLASSES:
        print(f"{label}: {cross_counts[label]}")

    by_name = {row["side_effect_name"].casefold(): row for row in summary_rows}
    print("\nDetailed requested terms")
    for term in DETAIL_TERMS:
        row = by_name[term.casefold()]
        print(term)
        print("\n".join(detail_lines(row, crosschecks[row["matrix_index"]])))

    for crosscheck, title in (
        ("name_and_umls_conflict", "All name_and_umls_conflict terms"),
        ("umls_no_optimuskg_match", "All umls_no_optimuskg_match terms"),
    ):
        print(f"\n{title}")
        if crosscheck == "name_and_umls_conflict":
            selected = [row for row in summary_rows if crosschecks[row["matrix_index"]] == crosscheck]
        else:
            selected = [row for row in summary_rows if row["mapping_status"] == crosscheck]
        if not selected:
            print("  None")
        for row in selected:
            print("\n".join(detail_lines(row, crosschecks[row["matrix_index"]])))
    print(f"\nMatches CSV: {MATCHES_PATH}")
    print(f"Summary CSV: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
