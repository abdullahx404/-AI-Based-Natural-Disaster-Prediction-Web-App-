"""Integration test for weather data merge functionality.

This test verifies that the merge_weather_data.py script correctly:
1. Combines Meteostat and NASA POWER data
2. Fills missing values
3. Preserves original Meteostat data where available
4. Produces a valid merged dataset

Usage:
    python tests/test_merge_integration.py
"""

import sys
from pathlib import Path

import pandas as pd


def test_required_files_exist():
    """Test that all required data files exist."""
    print("[Test 1] Checking required files...")
    required_files = [
        "data/raw/weather_combined.csv",
        "data/raw/nasa_power_combined.csv",
        "data/processed/weather_merged.csv"
    ]
    
    for file_path in required_files:
        exists = Path(file_path).exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {file_path}")
        assert exists, f"Required file not found: {file_path}"
    
    print("✓ All required files exist\n")


def test_merged_data_structure():
    """Test that merged data has expected structure."""
    print("[Test 2] Verifying merged data structure...")
    merged_df = pd.read_csv("data/processed/weather_merged.csv")
    
    expected_cols = [
        "date", "tavg", "tmin", "tmax", "prcp", "wspd", "pres", 
        "humidity", "solar_radiation", "location_key"
    ]
    
    for col in expected_cols:
        assert col in merged_df.columns, f"Missing expected column: {col}"
    
    print(f"✓ All expected columns present ({len(merged_df.columns)} total columns)\n")


def test_no_missing_values():
    """Test that key features have no missing values."""
    print("[Test 3] Checking for missing values in key features...")
    merged_df = pd.read_csv("data/processed/weather_merged.csv")
    
    key_features = [
        "tavg", "tmin", "tmax", "prcp", "wspd", "pres", 
        "humidity", "solar_radiation"
    ]
    
    missing_counts = merged_df[key_features].isnull().sum()
    
    for col in key_features:
        count = missing_counts[col]
        assert count == 0, f"Found {count} missing values in {col}"
    
    print("✓ No missing values in key features\n")


def test_data_ranges():
    """Test that data values are within reasonable ranges."""
    print("[Test 4] Verifying data ranges...")
    merged_df = pd.read_csv("data/processed/weather_merged.csv")
    
    checks = [
        ("tavg", -20, 50, "Temperature avg"),
        ("tmin", -30, 45, "Temperature min"),
        ("tmax", -15, 55, "Temperature max"),
        ("prcp", 0, 500, "Precipitation"),
        ("pres", 700, 1100, "Pressure"),
        ("humidity", 0, 100, "Humidity"),
        ("solar_radiation", 0, 400, "Solar radiation"),
    ]
    
    for col, min_val, max_val, name in checks:
        col_min = merged_df[col].min()
        col_max = merged_df[col].max()
        
        assert col_min >= min_val, f"{name} below expected range: {col_min}"
        assert col_max <= max_val, f"{name} above expected range: {col_max}"
        
        print(f"✓ {name} range OK: {col_min:.2f} to {col_max:.2f}")
    
    print()


def test_row_count():
    """Test that merged dataset has expected row count."""
    print("[Test 5] Verifying row count...")
    meteo_df = pd.read_csv("data/raw/weather_combined.csv")
    merged_df = pd.read_csv("data/processed/weather_merged.csv")
    
    print(f"  Meteostat rows: {len(meteo_df)}")
    print(f"  Merged rows: {len(merged_df)}")
    
    assert len(merged_df) == len(meteo_df), (
        f"Row count mismatch: expected {len(meteo_df)}, got {len(merged_df)}"
    )
    
    print("✓ Row count matches\n")


def test_meteostat_data_preservation():
    """Test that original Meteostat data is preserved where available."""
    print("[Test 6] Verifying Meteostat data preservation...")
    meteo_df = pd.read_csv("data/raw/weather_combined.csv")
    merged_df = pd.read_csv("data/processed/weather_merged.csv")
    
    meteo_df['date'] = pd.to_datetime(meteo_df['date'])
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    
    # Sample check: find rows with Meteostat data and verify preservation
    sample_meteo = meteo_df[meteo_df['tavg'].notna()].head(5)
    
    for _, row in sample_meteo.iterrows():
        merged_sample = merged_df[
            (merged_df['date'] == row['date']) & 
            (merged_df['location_key'] == row['location_key'])
        ]
        
        if not merged_sample.empty:
            original_tavg = row['tavg']
            merged_tavg = merged_sample.iloc[0]['tavg']
            
            # Allow small floating point differences
            assert abs(original_tavg - merged_tavg) < 0.01, (
                f"Meteostat data changed: original={original_tavg}, merged={merged_tavg}"
            )
    
    print("✓ Meteostat data preserved\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Integration Test: Weather Data Merge")
    print("=" * 60)
    print()
    
    tests = [
        test_required_files_exist,
        test_merged_data_structure,
        test_no_missing_values,
        test_data_ranges,
        test_row_count,
        test_meteostat_data_preservation,
    ]
    
    try:
        for test_func in tests:
            test_func()
        
        print("=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        
        # Print summary
        merged_df = pd.read_csv("data/processed/weather_merged.csv")
        print("\nMerged dataset summary:")
        print(f"  - Rows: {len(merged_df)}")
        print(f"  - Columns: {len(merged_df.columns)}")
        print(f"  - Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
        print(f"  - Locations: {', '.join(merged_df['location_key'].unique())}")
        print(f"  - Missing values in key features: 0")
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
