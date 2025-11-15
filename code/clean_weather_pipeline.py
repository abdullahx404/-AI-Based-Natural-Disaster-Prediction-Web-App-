"""Clean and merge Meteostat + NASA POWER weather feeds into daily location datasets.

Running the script will:
1. Load per-location Meteostat/NASA CSVs from the raw directory.
2. Standardize schema, parse dates, and merge both sources per location.
3. Reindex to a continuous daily timeline, interpolate short gaps, and fill precipitation zeros.
4. Write `cleaned_<location>.csv` files along with a combined `kp_cleaned.csv` table.

Example:

	python -m code.clean_weather_pipeline \
		--raw-dir data/raw \
		--processed-dir data/processed
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

STANDARD_COLUMNS = [
	"date",
	"tavg",
	"tmin",
	"tmax",
	"prcp",
	"snow",
	"humidity",
	"wdir",
	"wspd",
	"wpgt",
	"pres",
	"tsun",
	"location_name",
	"location_key",
	"latitude",
	"longitude",
	"elevation_m",
]

NUMERIC_COLUMNS = [
	"tavg",
	"tmin",
	"tmax",
	"prcp",
	"snow",
	"humidity",
	"wdir",
	"wspd",
	"wpgt",
	"pres",
	"tsun",
]

META_COLUMNS = ["location_name", "location_key", "latitude", "longitude", "elevation_m"]

COLUMN_SYNONYMS = {
	"date": "date",
	"time": "date",
	"datetime": "date",
	"tavg": "tavg",
	"temp_mean": "tavg",
	"temperature_mean": "tavg",
	"mean_temp": "tavg",
	"tmin": "tmin",
	"temp_min": "tmin",
	"temperature_min": "tmin",
	"tmax": "tmax",
	"temp_max": "tmax",
	"temperature_max": "tmax",
	"prcp": "prcp",
	"precipitation": "prcp",
	"precipitation_mm": "prcp",
	"rain_mm": "prcp",
	"rainfall": "prcp",
	"nasa_prcp": "prcp",
	"snow": "snow",
	"humidity": "humidity",
	"relative_humidity": "humidity",
	"rh": "humidity",
	"nasa_tavg": "tavg",
	"nasa_tmin": "tmin",
	"nasa_tmax": "tmax",
	"nasa_wspd": "wspd",
	"nasa_wpgt": "wpgt",
	"nasa_pres": "pres",
	"wdir": "wdir",
	"wind_direction": "wdir",
	"wspd": "wspd",
	"wind_speed": "wspd",
	"ws2m": "wspd",
	"wpgt": "wpgt",
	"wind_gust": "wpgt",
	"ws2m_max": "wpgt",
	"pres": "pres",
	"pressure": "pres",
	"ps": "pres",
	"tsun": "tsun",
	"sunshine_duration": "tsun",
	"solar_minutes": "tsun",
	"location_name": "location_name",
	"location": "location_name",
	"location_key": "location_key",
	"district": "location_name",
	"latitude": "latitude",
	"lat": "latitude",
	"longitude": "longitude",
	"lon": "longitude",
	"lng": "longitude",
	"elevation": "elevation_m",
	"elevation_m": "elevation_m",
}


@dataclass(frozen=True)
class LocationConfig:
	"""Static metadata for a district."""

	key: str
	name: str
	latitude: float
	longitude: float
	elevation_m: float


DEFAULT_LOCATIONS: Dict[str, LocationConfig] = {
	"swat": LocationConfig(
		key="swat",
		name="Swat District, Khyber Pakhtunkhwa",
		latitude=34.8091,
		longitude=72.3617,
		elevation_m=980,
	),
	"upper_dir": LocationConfig(
		key="upper_dir",
		name="Upper Dir District, Khyber Pakhtunkhwa",
		latitude=35.3350,
		longitude=71.8760,
		elevation_m=1420,
	),
}


def _normalize_column(name: str) -> str:
	return name.strip().lower().replace(" ", "_")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
	"""Rename columns using the synonym map, coerce numeric types, and ensure schema."""

	rename_map = {}
	for column in df.columns:
		normalized = _normalize_column(str(column))
		target = COLUMN_SYNONYMS.get(normalized)
		if target:
			rename_map[column] = target
	df = df.rename(columns=rename_map)

	# Keep only the relevant columns while preserving existing data.
	for column in STANDARD_COLUMNS:
		if column not in df.columns:
			df[column] = pd.NA

	numeric_cols = [col for col in NUMERIC_COLUMNS if col in df.columns]
	for col in numeric_cols:
		df[col] = pd.to_numeric(df[col], errors="coerce")

	# Return the columns in a canonical order for downstream merges.
	ordered_cols = [col for col in STANDARD_COLUMNS if col in df.columns]
	return df[ordered_cols]


def parse_date_column(df: pd.DataFrame, column: str = "date") -> pd.DataFrame:
	"""Convert the date column to datetime, floor to daily, and drop invalid entries."""

	if column not in df.columns:
		raise ValueError(f"Missing required '{column}' column after standardization")

	df[column] = pd.to_datetime(df[column], errors="coerce")
	df[column] = df[column].dt.floor("D")
	df = df.dropna(subset=[column]).reset_index(drop=True)
	return df


def merge_sources(meteostat_df: pd.DataFrame, nasa_df: pd.DataFrame) -> pd.DataFrame:
	"""Outer-join Meteostat and NASA data, preferring Meteostat when overlaps exist."""

	# Deduplicate before adding suffixes to keep one row per day per source.
	meteostat_df = meteostat_df.drop_duplicates(subset=["date"]).sort_values("date")
	nasa_df = nasa_df.drop_duplicates(subset=["date"]).sort_values("date")

	met_prefixed = meteostat_df.add_suffix("_met").rename(columns={"date_met": "date"})
	nasa_prefixed = nasa_df.add_suffix("_nasa").rename(columns={"date_nasa": "date"})

	# Harmonize NASA units to match Meteostat (m/s -> km/h for wind, kPa -> hPa for pressure).
	for col in list(nasa_prefixed.columns):
		if not col.endswith("_nasa"):
			continue
		base = col.replace("_nasa", "")
		if base in {"wspd", "wpgt"}:
			nasa_prefixed[col] = pd.to_numeric(nasa_prefixed[col], errors="coerce") * 3.6
		elif base == "pres":
			nasa_prefixed[col] = pd.to_numeric(nasa_prefixed[col], errors="coerce") * 10.0

	merged = pd.merge(met_prefixed, nasa_prefixed, on="date", how="outer")
	merged = merged.sort_values("date").reset_index(drop=True)

	for column in STANDARD_COLUMNS:
		if column == "date":
			continue
		met_col = f"{column}_met"
		nasa_col = f"{column}_nasa"
		if met_col in merged.columns and nasa_col in merged.columns:
			merged[column] = merged[met_col].combine_first(merged[nasa_col])
		elif met_col in merged.columns:
			merged[column] = merged[met_col]
		elif nasa_col in merged.columns:
			merged[column] = merged[nasa_col]

	# Drop the temporary suffixed columns to keep the frame tidy.
	drop_cols = [col for col in merged.columns if col.endswith("_met") or col.endswith("_nasa")]
	merged = merged.drop(columns=drop_cols)

	return merged


def forward_fill_small_gaps(df: pd.DataFrame, columns: Iterable[str], limit: int = 2) -> pd.DataFrame:
	"""Interpolate short gaps (<=limit days) for numeric columns and zero-fill precipitation."""

	columns = [col for col in columns if col in df.columns]
	if columns:
		df[columns] = df[columns].interpolate(method="linear", limit=limit, limit_direction="both")
	if "prcp" in df.columns:
		df["prcp"] = df["prcp"].fillna(0)
	return df


def _build_continuous_index(df: pd.DataFrame) -> pd.DataFrame:
	"""Ensure the dataframe has one row per day between the min and max dates."""

	if df.empty:
		return df
	start_date = df["date"].min()
	end_date = df["date"].max()
	full_range = pd.date_range(start=start_date, end=end_date, freq="D")
	df = df.set_index("date").reindex(full_range)
	df.index.name = "date"
	return df.reset_index().rename(columns={"index": "date"})


def clean_location_dataset(
	location: LocationConfig,
	meteostat_path: Path,
	nasa_path: Path,
) -> pd.DataFrame:
	"""Execute the full cleaning workflow for a single location."""

	if not meteostat_path.exists():
		raise FileNotFoundError(f"Missing Meteostat file: {meteostat_path}")
	if not nasa_path.exists():
		raise FileNotFoundError(f"Missing NASA file: {nasa_path}")

	# Step 1: Load both raw files.
	met_df = pd.read_csv(meteostat_path)
	nasa_df = pd.read_csv(nasa_path)

	# Step 2: Standardize column names and ensure consistent schema.
	met_df = standardize_columns(met_df)
	nasa_df = standardize_columns(nasa_df)

	# Step 3: Parse dates and drop invalid rows.
	met_df = parse_date_column(met_df)
	nasa_df = parse_date_column(nasa_df)

	# Step 4: Merge NASA/Meteostat and prefer Meteostat readings.
	merged = merge_sources(met_df, nasa_df)

	# Step 5: Reindex to continuous daily coverage.
	merged = _build_continuous_index(merged)

	# Step 6: Attach location metadata and ensure static columns.
	merged["location_key"] = location.key
	merged["location_name"] = location.name
	merged["latitude"] = merged["latitude"].fillna(location.latitude)
	merged["longitude"] = merged["longitude"].fillna(location.longitude)
	merged["elevation_m"] = merged["elevation_m"].fillna(location.elevation_m)

	# Step 7: Interpolate short gaps and enforce precipitation rules.
	merged = forward_fill_small_gaps(merged, NUMERIC_COLUMNS, limit=2)

	# Step 8: Drop rows that remain entirely empty across measurements.
	numeric_cols = [col for col in NUMERIC_COLUMNS if col in merged.columns]
	if numeric_cols:
		all_missing = merged[numeric_cols].isna().all(axis=1)
		merged = merged.loc[~all_missing]

	# Step 9: Remove duplicate days and sort chronologically.
	merged = merged.drop_duplicates(subset=["date", "location_key"]).sort_values("date").reset_index(drop=True)

	return merged


def run_pipeline(
	raw_dir: Path,
	processed_dir: Path,
	locations: Dict[str, LocationConfig] | None = None,
) -> None:
	"""Clean every configured location and write per-location plus combined CSVs."""

	locations = locations or DEFAULT_LOCATIONS
	processed_dir.mkdir(parents=True, exist_ok=True)
	cleaned_frames: List[pd.DataFrame] = []

	for key, config in locations.items():
		meteostat_path = raw_dir / f"{key}_meteostat.csv"
		nasa_path = raw_dir / f"{key}_nasa.csv"
		logging.info("Cleaning %s using %s and %s", key, meteostat_path.name, nasa_path.name)
		cleaned = clean_location_dataset(config, meteostat_path, nasa_path)
		output_path = processed_dir / f"cleaned_{key}.csv"
		cleaned.to_csv(output_path, index=False)
		logging.info("Wrote %s (%d rows)", output_path, len(cleaned))
		cleaned_frames.append(cleaned)

	if cleaned_frames:
		combined = pd.concat(cleaned_frames, ignore_index=True).sort_values(["date", "location_key"]).reset_index(drop=True)
		combined_path = processed_dir / "kp_cleaned.csv"
		combined.to_csv(combined_path, index=False)
		logging.info("Wrote %s (%d rows)", combined_path, len(combined))


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--raw-dir", default="data/raw", help="Directory containing <location>_<source>.csv files")
	parser.add_argument("--processed-dir", default="data/processed", help="Destination directory for cleaned CSVs")
	parser.add_argument("--log-level", default="INFO", help="Python logging level (default: INFO)")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s: %(message)s")

	raw_dir = Path(args.raw_dir)
	processed_dir = Path(args.processed_dir)
	run_pipeline(raw_dir=raw_dir, processed_dir=processed_dir)


if __name__ == "__main__":
	main()
