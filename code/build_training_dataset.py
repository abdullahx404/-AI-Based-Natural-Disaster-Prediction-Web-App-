"""Combine Meteostat, NASA POWER, and NDMA flood reports into a modeling-ready table.

Example:

    python -m code.build_training_dataset \
        --meteostat data/raw/weather_combined.csv \
        --nasa data/raw/nasa_power_combined.csv \
        --ndma data/raw/ndma_flood_reports.csv \
        --output data/processed/flood_weather_dataset.csv

The script fills Meteostat gaps with NASA POWER values, adds humidity/solar radiation,
encodes the location, and attaches manual flood-event labels derived from NDMA reports.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

DATE_FMT = "%Y-%m-%d"
DEFAULT_START = None  # Entire available range
DEFAULT_END = None
LOCATION_ENCODINGS: Dict[str, int] = {
    "swat": 0,
    "upper_dir": 1,
}
META_COLUMNS = ["location_name", "latitude", "longitude", "elevation_m"]
FILL_MAP = {
    "tavg": "nasa_tavg",
    "tmin": "nasa_tmin",
    "tmax": "nasa_tmax",
    "prcp": "nasa_prcp",
    "wspd": "nasa_wspd",
    "wpgt": "nasa_wpgt",
    "pres": "nasa_pres",
}
OUTPUT_COLUMNS = [
    "date",
    "location_key",
    "location_id",
    "location_name",
    "latitude",
    "longitude",
    "elevation_m",
    "tavg",
    "tmin",
    "tmax",
    "prcp",
    "snow",
    "wspd",
    "wpgt",
    "pres",
    "tsun",
    "humidity",
    "solar_radiation",
    "flood_event",
    "flood_severity",
    "damages_inr_crore",
    "warnings",
    "source_url",
    "notes",
]


def _parse_date(value: str | None, label: str) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.strptime(value, DATE_FMT)
    except ValueError as exc:
        raise ValueError(f"{label} must follow YYYY-MM-DD format (got {value})") from exc


def _standardize_location_keys(df: pd.DataFrame, column: str = "location_key") -> pd.DataFrame:
    if column in df.columns:
        df[column] = df[column].astype(str).str.strip().str.lower()
    return df


def _convert_nasa_units(df: pd.DataFrame) -> pd.DataFrame:
    # NASA wind speed/gust are reported in m/s; convert to km/h to match Meteostat.
    for wind_col in ["nasa_wspd", "nasa_wpgt"]:
        if wind_col in df.columns:
            df[wind_col] = df[wind_col] * 3.6
    # NASA surface pressure (PS) is in kPa; convert to hPa.
    if "nasa_pres" in df.columns:
        df["nasa_pres"] = df["nasa_pres"] * 10.0
    return df


def _load_csv(path: Path, parse_dates: Iterable[str]) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=list(parse_dates))
    return df


def _load_meteostat(path: Path) -> pd.DataFrame:
    df = _load_csv(path, parse_dates=["date"])
    df = _standardize_location_keys(df)
    return df


def _load_nasa(path: Path) -> pd.DataFrame:
    df = _load_csv(path, parse_dates=["date"])
    df = _standardize_location_keys(df)
    df = _convert_nasa_units(df)
    # Preserve metadata by renaming so we can combine-first later.
    rename_map = {col: f"{col}_nasa" for col in META_COLUMNS if col in df.columns}
    df = df.rename(columns=rename_map)
    return df


def _load_ndma(path: Path) -> pd.DataFrame:
    df = _load_csv(path, parse_dates=["date"])
    df = _standardize_location_keys(df)
    if "flood_event" not in df.columns:
        raise ValueError("NDMA CSV must include a 'flood_event' column")
    df["flood_event"] = df["flood_event"].fillna(0).astype(int).clip(0, 1)
    return df


def _ensure_location_ids(df: pd.DataFrame) -> pd.DataFrame:
    df["location_id"] = df["location_key"].map(LOCATION_ENCODINGS)
    if df["location_id"].isna().any():
        missing_keys = sorted(df.loc[df["location_id"].isna(), "location_key"].unique())
        next_idx = max(LOCATION_ENCODINGS.values(), default=-1) + 1
        for key in missing_keys:
            LOCATION_ENCODINGS[key] = next_idx
            df.loc[df["location_key"] == key, "location_id"] = next_idx
            next_idx += 1
        logging.warning("Encountered new location keys %s; assigned incremental IDs", missing_keys)
    df["location_id"] = df["location_id"].astype(int)
    return df


def build_dataset(
    meteostat_path: Path,
    nasa_path: Path,
    ndma_path: Path,
    output_path: Path,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> pd.DataFrame:
    logging.info("Loading Meteostat data from %s", meteostat_path)
    meteo = _load_meteostat(meteostat_path)
    logging.info("Loading NASA POWER data from %s", nasa_path)
    nasa = _load_nasa(nasa_path)
    logging.info("Loading NDMA flood reports from %s", ndma_path)
    ndma = _load_ndma(ndma_path)

    merged = pd.merge(
        meteo,
        nasa,
        on=["date", "location_key"],
        how="outer",
        suffixes=("", "_y"),
    )

    # Fill metadata columns with NASA fallbacks where Meteostat is missing.
    for col in META_COLUMNS:
        nasa_col = f"{col}_nasa"
        if nasa_col in merged.columns:
            merged[col] = merged[col].combine_first(merged[nasa_col])
            merged.drop(columns=[nasa_col], inplace=True)

    # Fill Meteostat gaps with NASA data.
    for target_col, fallback_col in FILL_MAP.items():
        if target_col in merged.columns and fallback_col in merged.columns:
            merged[target_col] = merged[target_col].fillna(merged[fallback_col])
        elif fallback_col in merged.columns:
            merged[target_col] = merged[fallback_col]
        if fallback_col in merged.columns:
            merged.drop(columns=[fallback_col], inplace=True)
    # Keep NASA-only humidity/solar columns (already named correctly).

    # Merge NDMA labels and metadata.
    merged = merged.merge(
        ndma,
        on=["date", "location_key"],
        how="left",
        suffixes=("", "_ndma"),
    )

    merged["flood_event"] = merged["flood_event"].fillna(0).astype(int)
    for text_col in ["flood_severity", "warnings", "source_url", "notes"]:
        if text_col not in merged.columns:
            merged[text_col] = pd.NA

    merged = _ensure_location_ids(merged)

    if start_date is not None:
        merged = merged[merged["date"] >= start_date]
    if end_date is not None:
        merged = merged[merged["date"] <= end_date]

    merged = merged.sort_values(["date", "location_key"]).reset_index(drop=True)

    for col in OUTPUT_COLUMNS:
        if col not in merged.columns:
            merged[col] = pd.NA

    merged = merged[OUTPUT_COLUMNS]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    logging.info("Wrote %s (%d rows)", output_path, len(merged))
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meteostat", default="data/raw/weather_combined.csv", help="Path to Meteostat CSV")
    parser.add_argument("--nasa", default="data/raw/nasa_power_combined.csv", help="Path to NASA POWER CSV")
    parser.add_argument("--ndma", default="data/raw/ndma_flood_reports.csv", help="Path to NDMA labels CSV")
    parser.add_argument(
        "--output",
        default="data/processed/flood_weather_dataset.csv",
        help="Destination CSV for the merged dataset",
    )
    parser.add_argument("--start-date", default=None, help="Optional start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None, help="Optional end date (YYYY-MM-DD)")
    parser.add_argument("--log-level", default="INFO", help="Logging verbosity (default: INFO)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s: %(message)s")

    start = _parse_date(args.start_date, "--start-date") if args.start_date else None
    end = _parse_date(args.end_date, "--end-date") if args.end_date else None

    build_dataset(
        meteostat_path=Path(args.meteostat),
        nasa_path=Path(args.nasa),
        ndma_path=Path(args.ndma),
        output_path=Path(args.output),
        start_date=start,
        end_date=end,
    )


if __name__ == "__main__":
    main()
