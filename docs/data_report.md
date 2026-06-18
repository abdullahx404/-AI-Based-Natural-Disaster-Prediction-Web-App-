# Meteostat Weather Data Export

## Overview

- **Source:** [Meteostat Daily API](https://meteostat.net/) via the official Python client (v1.6.7 from `requirements.txt`).
- **Extraction Script:** `python -m code.fetch_meteostat_weather --combine`
- **Default Window:** 2018-01-01 through the current day at run time (2025-11-15 for this export).
- **Output Directory:** `data/raw/`

Running the script downloads per-location CSVs and, when `--combine` is set, a merged `weather_combined.csv`. Each file follows the same schema and includes station metadata columns for traceability.

## Locations

| Key         | Human Name                             | Latitude   | Longitude  | Elevation (m) | Notes                                                                                          |
| ----------- | -------------------------------------- | ---------- | ---------- | ------------- | ---------------------------------------------------------------------------------------------- |
| `swat`      | Swat District, Khyber Pakhtunkhwa      | 34.8091° N | 72.3617° E | 980           | Represents Mingora valley conditions; Meteostat station IDs 41523/41501 supply most rows.      |
| `upper_dir` | Upper Dir District, Khyber Pakhtunkhwa | 35.3350° N | 71.8760° E | 1420          | Captures Hindu Kush foothills climate; Meteostat station IDs 41508/41505 dominate the dataset. |

> Coordinates/elevations are approximations based on district centroids; adjust in `code/fetch_meteostat_weather.py` if higher-fidelity station metadata becomes available.

## Files Created

| File                                                   | Rows  | Description                                                                                        |
| ------------------------------------------------------ | ----- | -------------------------------------------------------------------------------------------------- |
| `data/raw/weather_swat_2018-01-01_2025-11-15.csv`      | 1,042 | Daily summary for Swat; Meteostat did not provide coverage before 2021, hence warnings in the log. |
| `data/raw/weather_upper_dir_2018-01-01_2025-11-15.csv` | 2,872 | Daily summary for Upper Dir; mostly complete for 2018 onward.                                      |
| `data/raw/weather_combined.csv`                        | 3,914 | Concatenation of the above with identical schema, sorted by `date`.                                |

All files share the following columns (after resetting the index returned by Meteostat):

| Column                                                                  | Meaning                                                           |
| ----------------------------------------------------------------------- | ----------------------------------------------------------------- |
| `date`                                                                  | ISO date of measurement (local time).                             |
| `tavg`, `tmin`, `tmax`                                                  | Daily average/min/max temperature in °C.                          |
| `prcp`, `snow`                                                          | Precipitation depth (mm) and snowfall (mm).                       |
| `wdir`, `wspd`, `wpgt`                                                  | Mean wind direction (°), mean speed (km/h), and peak gust (km/h). |
| `pres`                                                                  | Air pressure (hPa).                                               |
| `tsun`                                                                  | Sunshine duration (minutes).                                      |
| `location_key`, `location_name`, `latitude`, `longitude`, `elevation_m` | Metadata injected by the script for downstream joins/filters.     |

Missing measurements are left blank by Meteostat and therefore appear as empty cells in the CSVs. Pandas will interpret them as `NaN` automatically when loading the files.

## Reproducibility Checklist

1. Activate the project virtual environment: `source .venv/Scripts/activate` (PowerShell automatically activates via `.venv`).
2. From the repo root, run `python -m code.fetch_meteostat_weather --start-date YYYY-MM-DD --end-date YYYY-MM-DD --combine`.
3. Inspect `data/raw/` for fresh CSVs and review the INFO/Warning log output for coverage issues.

## Source 2 – NASA POWER Gap Fill

- **Source:** [NASA POWER Daily Point API](https://power.larc.nasa.gov/) via `code/fetch_nasa_power.py`.
- **Usage:**
  ```powershell
  python -m code.fetch_nasa_power --start-date 2018-01-01 --end-date 2025-11-15 --combine
  ```
- **Outputs:** `data/raw/nasa_power_<location>_YYYY-MM-DD_YYYY-MM-DD.csv` plus `data/raw/nasa_power_combined.csv` (5,752 rows for the 2018-01-01→2025-11-15 window).
- **Parameters collected:** `T2M`, `T2M_MIN`, `T2M_MAX`, `PRECTOTCORR`, `WS2M`, `WS2M_MAX`, `RH2M`, `PS`, `ALLSKY_SFC_SW_DWN`.
- **Unit normalization:** the dataset builder converts NASA wind speeds from m/s → km/h and surface pressure from kPa → hPa to align with Meteostat.

NASA rows are merged with Meteostat by (`date`, `location_key`) so that any blank Meteostat fields can be backfilled with NASA values while preserving the higher-quality station readings where available.

## Source 3 – NDMA Flood Reports

- **Source:** NDMA Situation Reports (SITREP) and GLOF advisories published via ReliefWeb/NDMA portal.
- **Storage:** `data/raw/ndma_flood_reports.csv` with columns:
  - `date`, `location_key`, `flood_event` (binary label), `flood_severity`, `damages_inr_crore`, `warnings`, `source_url`, `notes`.
- **Current coverage:** Positive flood labels for key 2022–2024 events affecting Swat and Upper Dir (e.g., SitRep #089 on 27-Aug-2022). Additional rows can be appended as more reports are transcribed.
- **Labeling rule:** For days not present in the CSV, the pipeline assumes `flood_event = 0`.

## Final Training Dataset

- **Builder:** `python -m code.build_training_dataset --start-date 2018-01-01 --end-date 2025-11-15`.
- **Inputs:** Meteostat combined CSV, NASA POWER combined CSV, NDMA flood report CSV.
- **Output:** `data/processed/flood_weather_dataset.csv` (5,752 rows, two per date – one per district) with the feature/label schema required for modeling.
- **Feature handling:**
  - Meteostat columns (`tavg`, `tmin`, `tmax`, `prcp`, `wspd`, `wpgt`, `pres`, `snow`, `tsun`) are preferred; missing entries fall back to NASA counterparts where available.
  - NASA-only metrics `humidity` and `solar_radiation` are appended.
  - `location_key` is integer-encoded into `location_id` (Swat=0, Upper Dir=1, auto-extends for additional districts).
  - NDMA metadata (`flood_severity`, `damages_inr_crore`, `warnings`, `source_url`, `notes`) accompany the binary `flood_event` label.

| Column                                                                                 | Description                                                                      |
| -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `date`                                                                                 | Observation day (UTC ISO date).                                                  |
| `location_key`, `location_id`, `location_name`, `latitude`, `longitude`, `elevation_m` | Spatial metadata / encoded key.                                                  |
| `tavg`, `tmin`, `tmax`                                                                 | Temperature metrics in °C after gap filling.                                     |
| `prcp`, `snow`                                                                         | Precipitation & snowfall depth (mm).                                             |
| `wspd`, `wpgt`                                                                         | Wind speed / gust in km/h (NASA data converted from m/s when used).              |
| `pres`                                                                                 | Surface pressure in hPa (NASA data converted from kPa when used).                |
| `tsun`                                                                                 | Sunshine duration in minutes (Meteostat only).                                   |
| `humidity`                                                                             | Relative humidity (%) from NASA POWER.                                           |
| `solar_radiation`                                                                      | Daily all-sky surface shortwave downward radiation (kWh/m²/day) from NASA POWER. |
| `flood_event`                                                                          | Target label (1 if NDMA reported flooding, else 0).                              |
| `flood_severity`, `damages_inr_crore`, `warnings`, `source_url`, `notes`               | NDMA context for auditing/interpretability.                                      |

## Next Steps

- Expand NDMA labeling to cover additional districts/seasons and to include "no flood" confirmations if available.
- Pull additional Meteostat stations nearer to flood-prone tehsils if higher spatial resolution is required.
- Enrich files with auxiliary socio-economic indicators (population density, river discharge, GLOF catalogues) for downstream models.
- Automate a QA notebook in `notebooks/` to visualize coverage, missingness, and label distribution before training.
