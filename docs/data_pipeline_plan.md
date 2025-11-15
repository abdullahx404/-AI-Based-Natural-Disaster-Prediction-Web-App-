# Multi-Source Weather & Flood Data Pipeline

## Goals

- Fill Meteostat gaps using NASA POWER daily reanalysis metrics (temperature, precipitation, wind, humidity, pressure, solar radiation).
- Incorporate NDMA flood situation reports to derive a supervised `flood_event` label per district-date.
- Emit a modeling-ready table with the agreed feature set plus the binary target column.

## Sources

1. **Meteostat** (`data/raw/weather_*.csv`) – high-quality station observations but with coverage holes (notably Swat before 2021).
2. **NASA POWER** (`data/raw/nasa_power_*.csv`) – spatially complete, lower-resolution reanalysis data used to backfill missing Meteostat columns and add humidity/solar radiation.
3. **NDMA Flood Reports** (`data/raw/ndma_flood_reports.csv`) – manually curated CSV summarizing official situation reports with severity, damage notes, and the derived `flood_event` label.

## Integration Strategy

1. **Normalize Schema**
   - Parse all date columns to `datetime64[ns]`.
   - Lowercase `location_key` and ensure it matches between sources.
2. **NASA Backfill**
   - Outer-join Meteostat and NASA on (`date`, `location_key`).
   - For each Meteostat weather column, prefer station data; fill remaining gaps with the NASA counterpart (`nasa_tavg` → `tavg`, etc.).
   - Carry forward NASA-only metrics (`humidity`, `solar_radiation`) as new feature columns.
3. **NDMA Labels**
   - Read `ndma_flood_reports.csv` containing (`date`, `location_key`, `flood_event`, `flood_severity`, `damages_inr_crore`, `warnings`, `source_url`).
   - Merge onto the weather table; default `flood_event` to 0 when no report exists for that day/location.
4. **Feature Engineering**
   - Encode `location_key` → integer `location_id` (Swat=0, Upper Dir=1, extendable).
   - Retain metadata columns for audit (`location_name`, `latitude`, `longitude`, `elevation_m`).
5. **Output**
   - Save to `data/processed/flood_weather_dataset.csv` with the agreed feature list plus `flood_event`.
   - Version outputs with date suffixes when re-running (`flood_weather_dataset_YYYYMMDD.csv`).

## Implementation Plan

1. Run existing fetchers:
   ```powershell
   python -m code.fetch_meteostat_weather --start-date 2018-01-01 --end-date 2025-11-15 --combine
   python -m code.fetch_nasa_power --start-date 2018-01-01 --end-date 2025-11-15 --combine
   ```
2. Create/maintain `data/raw/ndma_flood_reports.csv` by transcribing NDMA situation reports (include at least the 2022 monsoon floods as positives).
3. Add a new script `code/build_training_dataset.py` which performs the joins, fills, and exports.
4. Clean Meteostat + NASA raw feeds into per-location daily tables:
   ```powershell
   python -m code.clean_weather_pipeline --raw-dir data/raw --processed-dir data/processed
   ```
   This emits `cleaned_swat.csv`, `cleaned_upper_dir.csv`, and `kp_cleaned.csv` with standardized schema, continuous dates, and short-gap interpolation.
5. Document the workflow in `docs/data_report.md` (sources, assumptions, and QA checks).
6. Commit datasets + scripts once per refresh cycle to keep the repo reproducible.
