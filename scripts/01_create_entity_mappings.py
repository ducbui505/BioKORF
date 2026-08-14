"""Create ordered drug and side-effect mapping templates from MSSF data."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Datas"
OUTPUT_DIR = PROJECT_ROOT / "data_processed" / "mappings"

EXPECTED_DRUGS = 757
EXPECTED_SIDE_EFFECTS = 994


def _validate_continuous_index(frame: pd.DataFrame, expected_count: int) -> None:
    expected = list(range(expected_count))
    if frame["matrix_index"].tolist() != expected:
        raise ValueError("matrix_index is not continuous from 0 to the final row")


def create_drug_mapping() -> pd.DataFrame:
    source = pd.read_excel(DATA_DIR / "drug_name.xlsx", header=None, dtype=str)
    if source.shape != (EXPECTED_DRUGS, 2):
        raise ValueError(
            "drug_name.xlsx must contain exactly "
            f"{EXPECTED_DRUGS} rows and 2 columns; found {source.shape}"
        )

    original_drug_order = source.iloc[:, 0].tolist()
    if original_drug_order[0].strip().lower() != "lepirudin":
        raise ValueError(
            "The first drug must be lepirudin; "
            f"found {original_drug_order[0]!r}"
        )

    mapping = pd.DataFrame(
        {
            "matrix_index": range(EXPECTED_DRUGS),
            "source_row": range(1, EXPECTED_DRUGS + 1),
            "drug_name": original_drug_order,
            "stitch_id": source.iloc[:, 1].tolist(),
            "optimuskg_id": "",
            "mapping_method": "",
            "mapping_status": "unmapped",
        }
    )

    _validate_continuous_index(mapping, EXPECTED_DRUGS)
    if mapping["drug_name"].tolist() != original_drug_order:
        raise ValueError("Drug rows were reordered while creating the mapping")
    return mapping


def create_side_effect_mapping() -> pd.DataFrame:
    source = pd.read_excel(
        DATA_DIR / "Side_effect_name.xlsx", header=None, dtype=str
    )
    if source.shape != (EXPECTED_SIDE_EFFECTS, 1):
        raise ValueError(
            "Side_effect_name.xlsx must contain exactly "
            f"{EXPECTED_SIDE_EFFECTS} rows and 1 column; found {source.shape}"
        )

    original_side_effect_order = source.iloc[:, 0].tolist()
    mapping = pd.DataFrame(
        {
            "matrix_index": range(EXPECTED_SIDE_EFFECTS),
            "source_row": range(1, EXPECTED_SIDE_EFFECTS + 1),
            "side_effect_name": original_side_effect_order,
            "optimuskg_id": "",
            "mapping_method": "",
            "mapping_status": "unmapped",
        }
    )

    _validate_continuous_index(mapping, EXPECTED_SIDE_EFFECTS)
    if mapping["side_effect_name"].tolist() != original_side_effect_order:
        raise ValueError("Side-effect rows were reordered while creating the mapping")
    return mapping


def _print_summary(label: str, frame: pd.DataFrame) -> None:
    print(f"Number of {label}: {len(frame)}")
    print(f"First 5 {label} rows:")
    print(frame.head(5).to_string(index=False))
    print(f"Last 5 {label} rows:")
    print(frame.tail(5).to_string(index=False))


def main() -> None:
    drug_mapping = create_drug_mapping()
    side_effect_mapping = create_side_effect_mapping()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    drug_mapping.to_csv(OUTPUT_DIR / "drug_mapping.csv", index=False)
    side_effect_mapping.to_csv(OUTPUT_DIR / "side_effect_mapping.csv", index=False)

    _print_summary("drugs", drug_mapping)
    _print_summary("side effects", side_effect_mapping)


if __name__ == "__main__":
    main()
