"""Load raw scraped case JSON files and concatenate them into a single DataFrame per case type.

Ported from `raw_processed_data_git/Msc-Data-Science-Dissertation-Data/scripts/data_aggregation.py`.
Correction: the original hardcoded 15 specific October-2025 filenames (one per court, per case
type). This version globs `data/raw_data/*.json` and filters by a case-type keyword found in every
scraper-produced filename, so it does not go stale as new scrapes are added.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR: Path = _REPO_ROOT / "data" / "raw_data"

# Substring that every raw JSON filename for a given case type contains, regardless of
# source (csm/dgsi) or court. Used to filter the glob in `find_raw_files`.
CASE_TYPE_FILENAME_KEYWORDS: dict[str, str] = {
    "dv": "violencia_domestica",
    "boc": "incumprimento_contratos",
}


def find_raw_files(case_type: str, raw_dir: Path = RAW_DATA_DIR) -> list[Path]:
    """Find every raw scrape JSON file for a given case type.

    Args:
        case_type: Either "dv" (domestic violence) or "boc" (breach of contract).
        raw_dir: Directory containing the raw scrape JSON files.

    Returns:
        Sorted list of matching file paths.

    Raises:
        ValueError: If `case_type` is not "dv" or "boc".
    """
    if case_type not in CASE_TYPE_FILENAME_KEYWORDS:
        raise ValueError(
            f"Unknown case_type '{case_type}'. Expected one of "
            f"{sorted(CASE_TYPE_FILENAME_KEYWORDS)}."
        )
    keyword = CASE_TYPE_FILENAME_KEYWORDS[case_type]
    return sorted(p for p in raw_dir.glob("*.json") if keyword in p.name)


def load_case_dataframes(paths: list[Path]) -> list[pd.DataFrame]:
    """Load and flatten a list of raw scrape JSON files into DataFrames.

    Args:
        paths: File paths to raw scrape JSON files, each containing a list of case records.

    Returns:
        One DataFrame per input path, in the same order, with nested fields (e.g.
        `metadata_decisao`) flattened using an underscore separator.
    """
    dataframes = []
    for path in paths:
        if not path.exists():
            print(f"Error: file not found at {path.resolve()}")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        dataframes.append(pd.json_normalize(data, sep="_"))
    return dataframes


def aggregate_case_type(case_type: str, raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Load and concatenate every raw file for one case type into a single DataFrame.

    Args:
        case_type: Either "dv" (domestic violence) or "boc" (breach of contract).
        raw_dir: Directory containing the raw scrape JSON files.

    Returns:
        A single concatenated DataFrame covering every court found for `case_type`.

    Raises:
        ValueError: If no raw files are found for `case_type`.
    """
    paths = find_raw_files(case_type, raw_dir=raw_dir)
    if not paths:
        raise ValueError(
            f"No raw files found for case_type='{case_type}' under {raw_dir.resolve()}."
        )

    dataframes = load_case_dataframes(paths)
    df_all = pd.concat(dataframes, ignore_index=True)

    print(f"{case_type.upper()}: {len(df_all)} total cases across {len(paths)} files")
    for path, df in zip(paths, dataframes):
        print(f"  - {path.name}: {len(df)} cases")

    return df_all
