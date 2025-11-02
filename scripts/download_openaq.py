"""Download and aggregate AQI measurements from the OpenAQ API.

Example usage
-------------
python scripts/download_openaq.py \
    --cities Delhi Gurugram Mumbai \
    --start-date 2021-01-01 \
    --end-date 2024-12-31 \
    --output notebooks/aqi_updates_2021_2024.csv

This script will:
  • query OpenAQ measurements for each city in the date window
  • aggregate hourly readings to daily averages per pollutant
  • compute an Indian AQI value using CPCB breakpoints
  • emit a CSV compatible with `scripts/extend_dataset.py`
"""

from __future__ import annotations

import argparse
import math
import time
from datetime import datetime
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests

BASE_URL = "https://api.openaq.org/v2/measurements"
SUPPORTED_PARAMETERS = {
    "pm25": "PM2.5",
    "pm10": "PM10",
    "no": "NO",
    "no2": "NO2",
    "nox": "NOx",
    "nh3": "NH3",
    "co": "CO",
    "so2": "SO2",
    "o3": "O3",
    "benzene": "Benzene",
    "toluene": "Toluene",
    "xylene": "Xylene",
}

# CPCB breakpoints (concentration, AQI) for pollutants used in India AQI formulation
AQI_BREAKPOINTS = {
    "PM2.5": [
        (0, 30, 0, 50),
        (31, 60, 51, 100),
        (61, 90, 101, 200),
        (91, 120, 201, 300),
        (121, 250, 301, 400),
        (251, 500, 401, 500),
    ],
    "PM10": [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 250, 101, 200),
        (251, 350, 201, 300),
        (351, 430, 301, 400),
        (431, 1000, 401, 500),
    ],
    "NO2": [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 180, 101, 200),
        (181, 280, 201, 300),
        (281, 400, 301, 400),
        (401, 1000, 401, 500),
    ],
    "SO2": [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 380, 101, 200),
        (381, 800, 201, 300),
        (801, 1600, 301, 400),
        (1601, 10000, 401, 500),
    ],
    "CO": [
        (0.0, 1.0, 0, 50),
        (1.1, 2.0, 51, 100),
        (2.1, 10.0, 101, 200),
        (10.1, 17.0, 201, 300),
        (17.1, 34.0, 301, 400),
        (34.1, 100.0, 401, 500),
    ],
    "O3": [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 168, 101, 200),
        (169, 208, 201, 300),
        (209, 748, 301, 400),
        (749, 1000, 401, 500),
    ],
    "NH3": [
        (0, 200, 0, 50),
        (201, 400, 51, 100),
        (401, 800, 101, 200),
        (801, 1200, 201, 300),
        (1201, 1800, 301, 400),
        (1801, 2000, 401, 500),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download OpenAQ data and aggregate to daily AQI")
    parser.add_argument("--cities", nargs="+", required=True, help="One or more city names recognised by OpenAQ")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", default="openaq_output.csv", help="Path to save the aggregated CSV")
    parser.add_argument("--page-size", type=int, default=10000, help="Records per API page (max 10000)")
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep between API requests (rate limiting)")
    return parser.parse_args()


def fetch_city_measurements(city: str, start: str, end: str, page_size: int, sleep: float) -> List[Dict]:
    """Fetch all measurements for a city in the date window."""
    records: List[Dict] = []
    page = 1

    while True:
        params = {
            "city": city,
            "country": "IN",
            "date_from": f"{start}T00:00:00Z",
            "date_to": f"{end}T23:59:59Z",
            "limit": page_size,
            "page": page,
            "sort": "asc",
            "order_by": "datetime",
            "offset": (page - 1) * page_size,
        }
        resp = requests.get(BASE_URL, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            break
        records.extend(results)
        meta = data.get("meta", {})
        found = meta.get("found", len(records))
        print(f"{city}: fetched page {page}, cumulative records {len(records)}/{found}")
        page += 1
        if len(records) >= found:
            break
        time.sleep(sleep)

    return records


def records_to_dataframe(records: Iterable[Dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.json_normalize(records)
    # Flatten nested date fields
    if "date.utc" in df.columns:
        df["datetime_utc"] = pd.to_datetime(df["date.utc"], errors="coerce")
    if "date.local" in df.columns:
        df["datetime_local"] = pd.to_datetime(df["date.local"], errors="coerce")
    return df


def compute_cpcb_aqi(row: pd.Series) -> Optional[float]:
    sub_indices = []
    for pollutant, breakpoints in AQI_BREAKPOINTS.items():
        value = row.get(pollutant)
        if value is None or math.isnan(value):
            continue
        for c_low, c_high, a_low, a_high in breakpoints:
            if c_low <= value <= c_high:
                # Linear interpolation within the segment
                sub_index = ((a_high - a_low) / (c_high - c_low)) * (value - c_low) + a_low
                sub_indices.append(sub_index)
                break
    if not sub_indices:
        return None
    return max(sub_indices)


def aggregate_daily(city: str, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df = df[df["parameter"].isin(SUPPORTED_PARAMETERS.keys())].copy()
    if df.empty:
        return pd.DataFrame()

    df["Date"] = df["datetime_local"].dt.date
    df = df.dropna(subset=["Date"])

    # Average measurements per day
    grouped = (
        df.groupby(["City", "Date", "parameter"], as_index=False)["value"].mean()
    )
    pivot = grouped.pivot_table(
        index=["City", "Date"],
        columns="parameter",
        values="value"
    ).reset_index()

    # Rename columns to match training dataset
    rename_map = {param: SUPPORTED_PARAMETERS[param] for param in SUPPORTED_PARAMETERS}
    pivot = pivot.rename(columns=rename_map)

    # Ensure all required pollutant columns exist
    for col in SUPPORTED_PARAMETERS.values():
        if col not in pivot.columns:
            pivot[col] = pd.NA

    # Compute AQI
    pivot["AQI"] = pivot.apply(compute_cpcb_aqi, axis=1)

    # Reorder columns
    ordered_cols = [
        "Date", "City",
        "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
        "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene", "AQI"
    ]
    pivot = pivot[ordered_cols]
    return pivot


def main() -> None:
    args = parse_args()
    all_frames: List[pd.DataFrame] = []

    for city in args.cities:
        print(f"Fetching measurements for {city}...")
        records = fetch_city_measurements(city, args.start_date, args.end_date, args.page_size, args.sleep)
        df = records_to_dataframe(records)
        if df.empty:
            print(f"No data returned for {city} in the specified window.")
            continue
        df["City"] = city
        aggregated = aggregate_daily(city, df)
        if aggregated.empty:
            print(f"No supported parameters found for {city}; skipping.")
            continue
        all_frames.append(aggregated)

    if not all_frames:
        raise SystemExit("No data fetched for any city. Check city names/date range.")

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.sort_values(["City", "Date"])
    combined["Date"] = combined["Date"].astype(str)

    combined.to_csv(args.output, index=False)
    print(f"Saved aggregated data to {args.output} ({len(combined):,} rows)")


if __name__ == "__main__":
    main()
