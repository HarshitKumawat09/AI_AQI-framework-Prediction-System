"""Utility to append newer AQI measurements to the core dataset.

Usage
-----
python scripts/extend_dataset.py --new-data path/to/new_measurements.csv \
    [--existing notebooks/aqi_dataset.csv] [--backup]

The script will:
1. Load the existing dataset (defaults to notebooks/aqi_dataset.csv)
2. Load the new measurements (must contain a superset of the required columns)
3. Harmonise column names and dtypes
4. Concatenate, drop duplicates, sort chronologically
5. Optionally create a timestamped backup of the original dataset
6. Persist the extended dataset back to disk
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

REQUIRED_COLUMNS: List[str] = [
    "Date", "City", "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO",
    "SO2", "O3", "Benzene", "Toluene", "Xylene", "AQI"
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append new AQI measurements to the core dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--new-data",
        required=True,
        type=Path,
        help="Path to CSV file containing newer AQI measurements",
    )
    parser.add_argument(
        "--existing",
        type=Path,
        default=Path("notebooks/aqi_dataset.csv"),
        help="Path to the existing combined dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the merged dataset. Defaults to overwriting --existing",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create a timestamped backup of the existing dataset before overwriting",
    )
    return parser.parse_args()


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure column casing and expected names align with REQUIRED_COLUMNS."""
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]
    rename_map = {col.lower(): col for col in REQUIRED_COLUMNS}
    harmonised = {}
    for col in df.columns:
        key = col.lower()
        if key in rename_map:
            harmonised[col] = rename_map[key]
    df = df.rename(columns=harmonised)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "New data is missing required columns: " + ", ".join(missing)
        )

    # Coerce numeric columns
    numeric_cols = [col for col in REQUIRED_COLUMNS if col not in {"Date", "City"}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "City", "AQI"])
    return df


def create_backup(path: Path) -> None:
    if not path.exists():
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.stem}_backup_{timestamp}{path.suffix}")
    shutil.copy2(path, backup_path)
    print(f"Backup created at {backup_path}")


def main() -> None:
    args = parse_args()

    if not args.new_data.exists():
        raise FileNotFoundError(f"New data file not found: {args.new_data}")

    if not args.existing.exists():
        raise FileNotFoundError(f"Existing dataset not found: {args.existing}")

    output_path = args.output or args.existing

    print("Loading datasets...")
    existing_df = normalise_columns(pd.read_csv(args.existing))
    new_df = normalise_columns(pd.read_csv(args.new_data))

    print(
        f"Existing records: {len(existing_df):,}, "
        f"new records: {len(new_df):,} (unique cities: {new_df['City'].nunique()})"
    )

    combined = pd.concat([existing_df, new_df], ignore_index=True)
    combined = combined.sort_values("Date").drop_duplicates(subset=["Date", "City"], keep="last")

    print(f"Combined dataset records: {len(combined):,}")
    print(f"Date range: {combined['Date'].min().date()} → {combined['Date'].max().date()}")

    if args.backup:
        create_backup(output_path)

    combined.to_csv(output_path, index=False, date_format="%Y-%m-%d")
    print(f"Extended dataset saved to {output_path}")


if __name__ == "__main__":
    main()
