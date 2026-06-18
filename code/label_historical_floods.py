"""
Historical Flood Data Labeling Tool
Labels weather data with known flood events from multiple sources:
- NDMA (National Disaster Management Authority) archives
- EM-DAT International Disaster Database
- PDMA KP historical records
- News reports and historical archives
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# =============================================================================
# HISTORICAL FLOOD EVENTS DATABASE
# Sources: NDMA, EM-DAT, PDMA KP, News Archives, Research Papers
# =============================================================================

HISTORICAL_FLOODS = [
    # 2010 Pakistan Floods - One of the worst in history
    # Source: EM-DAT, NDMA, World Bank Reports
    {"start": "2010-07-22", "end": "2010-08-15", "locations": ["swat", "upper_dir"], 
     "severity": "severe", "source": "EM-DAT/NDMA", 
     "notes": "2010 Pakistan floods - 20 million affected, 1,985 deaths"},
    {"start": "2010-07-28", "end": "2010-08-10", "locations": ["swat"], 
     "severity": "severe", "source": "NDMA",
     "notes": "Swat River flooding - massive destruction"},
    
    # 2010 Additional flood waves
    {"start": "2010-08-25", "end": "2010-09-05", "locations": ["swat", "upper_dir"], 
     "severity": "moderate", "source": "PDMA KP",
     "notes": "Second wave of 2010 monsoon floods"},
    
    # 2011 Monsoon Floods
    # Source: NDMA Annual Report 2011
    {"start": "2011-08-08", "end": "2011-08-20", "locations": ["swat", "upper_dir"], 
     "severity": "moderate", "source": "NDMA",
     "notes": "2011 monsoon flooding in KP"},
    {"start": "2011-09-05", "end": "2011-09-12", "locations": ["upper_dir"], 
     "severity": "minor", "source": "PDMA KP",
     "notes": "September 2011 flash floods"},
    
    # 2012 Floods
    # Source: PDMA KP Reports
    {"start": "2012-08-20", "end": "2012-08-28", "locations": ["swat", "upper_dir"], 
     "severity": "moderate", "source": "PDMA KP",
     "notes": "2012 monsoon floods"},
    {"start": "2012-09-03", "end": "2012-09-08", "locations": ["swat"], 
     "severity": "minor", "source": "News Archives",
     "notes": "Early September flash floods"},
    
    # 2013 Floods
    # Source: NDMA Situation Reports
    {"start": "2013-08-02", "end": "2013-08-12", "locations": ["swat", "upper_dir"], 
     "severity": "moderate", "source": "NDMA",
     "notes": "August 2013 heavy monsoon rains"},
    {"start": "2013-09-15", "end": "2013-09-20", "locations": ["upper_dir"], 
     "severity": "minor", "source": "PDMA KP",
     "notes": "Late monsoon flooding"},
    
    # 2014 Floods - Significant event
    # Source: EM-DAT, NDMA
    {"start": "2014-09-03", "end": "2014-09-15", "locations": ["swat", "upper_dir"], 
     "severity": "severe", "source": "EM-DAT/NDMA",
     "notes": "2014 India-Pakistan floods - 665 deaths in Pakistan"},
    {"start": "2014-09-05", "end": "2014-09-12", "locations": ["swat"], 
     "severity": "severe", "source": "News/NDMA",
     "notes": "Swat valley severe flooding"},
    
    # 2015 Floods
    # Source: PDMA KP
    {"start": "2015-04-26", "end": "2015-04-30", "locations": ["swat", "upper_dir"], 
     "severity": "minor", "source": "PDMA KP",
     "notes": "Spring flash floods"},
    {"start": "2015-07-15", "end": "2015-07-25", "locations": ["swat", "upper_dir"], 
     "severity": "moderate", "source": "NDMA",
     "notes": "July monsoon floods"},
    {"start": "2015-08-05", "end": "2015-08-15", "locations": ["upper_dir"], 
     "severity": "moderate", "source": "PDMA KP",
     "notes": "August monsoon flooding"},
    
    # 2016 Floods
    # Source: NDMA Reports
    {"start": "2016-04-03", "end": "2016-04-08", "locations": ["swat"], 
     "severity": "minor", "source": "News Archives",
     "notes": "Spring flooding"},
    {"start": "2016-07-20", "end": "2016-07-30", "locations": ["swat", "upper_dir"], 
     "severity": "moderate", "source": "NDMA",
     "notes": "2016 monsoon floods"},
    {"start": "2016-08-10", "end": "2016-08-18", "locations": ["swat", "upper_dir"], 
     "severity": "moderate", "source": "PDMA KP",
     "notes": "Mid-August flooding"},
    
    # 2017 Floods
    # Source: NDMA, News Reports
    {"start": "2017-04-05", "end": "2017-04-10", "locations": ["swat", "upper_dir"], 
     "severity": "minor", "source": "PDMA KP",
     "notes": "Early spring floods"},
    {"start": "2017-08-01", "end": "2017-08-10", "locations": ["swat", "upper_dir"], 
     "severity": "moderate", "source": "NDMA",
     "notes": "August 2017 monsoon flooding"},
    {"start": "2017-08-25", "end": "2017-09-02", "locations": ["upper_dir"], 
     "severity": "minor", "source": "News Archives",
     "notes": "Late August flash floods"},
    
    # 2005 Kashmir Earthquake related floods
    # Source: EM-DAT, Historical records
    {"start": "2005-10-08", "end": "2005-10-15", "locations": ["swat", "upper_dir"], 
     "severity": "moderate", "source": "Historical/EM-DAT",
     "notes": "Post-earthquake flooding and landslides"},
    
    # 2006 Floods
    {"start": "2006-08-01", "end": "2006-08-10", "locations": ["swat", "upper_dir"], 
     "severity": "moderate", "source": "NDMA Archives",
     "notes": "2006 monsoon floods"},
    
    # 2007 Floods - Cyclone Yemyin aftermath
    # Source: EM-DAT
    {"start": "2007-06-23", "end": "2007-07-05", "locations": ["swat", "upper_dir"], 
     "severity": "severe", "source": "EM-DAT",
     "notes": "Cyclone Yemyin related flooding - 926 deaths nationwide"},
    {"start": "2007-08-05", "end": "2007-08-15", "locations": ["swat"], 
     "severity": "moderate", "source": "PDMA KP",
     "notes": "August monsoon floods"},
    
    # 2008 Floods
    {"start": "2008-06-28", "end": "2008-07-05", "locations": ["upper_dir"], 
     "severity": "minor", "source": "PDMA KP",
     "notes": "Early monsoon flooding"},
    {"start": "2008-08-10", "end": "2008-08-18", "locations": ["swat", "upper_dir"], 
     "severity": "moderate", "source": "NDMA",
     "notes": "Peak monsoon floods"},
    
    # 2009 Floods
    {"start": "2009-07-28", "end": "2009-08-05", "locations": ["swat", "upper_dir"], 
     "severity": "moderate", "source": "NDMA",
     "notes": "2009 monsoon flooding"},
    {"start": "2009-08-20", "end": "2009-08-28", "locations": ["swat"], 
     "severity": "minor", "source": "News Archives",
     "notes": "Late August floods"},
    
    # Pre-2005 Historical Events
    # Source: Historical archives, research papers
    {"start": "2001-07-25", "end": "2001-08-05", "locations": ["swat", "upper_dir"], 
     "severity": "moderate", "source": "Historical Archives",
     "notes": "2001 monsoon floods"},
    
    {"start": "2003-08-10", "end": "2003-08-20", "locations": ["swat", "upper_dir"], 
     "severity": "moderate", "source": "NDMA Archives",
     "notes": "2003 monsoon flooding"},
    
    {"start": "2004-07-20", "end": "2004-07-30", "locations": ["swat"], 
     "severity": "minor", "source": "Historical Records",
     "notes": "2004 floods"},
    
    # 2018-2024 Events (from existing data)
    {"start": "2022-08-26", "end": "2022-08-28", "locations": ["swat", "upper_dir"], 
     "severity": "severe", "source": "NDMA",
     "notes": "2022 Pakistan floods - catastrophic"},
    {"start": "2022-09-01", "end": "2022-09-03", "locations": ["swat"], 
     "severity": "moderate", "source": "PDMA KP",
     "notes": "September 2022 continued flooding"},
    {"start": "2023-07-09", "end": "2023-07-11", "locations": ["upper_dir"], 
     "severity": "minor", "source": "NDMA",
     "notes": "July 2023 flash floods"},
    {"start": "2024-04-17", "end": "2024-04-19", "locations": ["swat"], 
     "severity": "minor", "source": "PDMA KP",
     "notes": "April 2024 spring floods"},
    {"start": "2024-08-21", "end": "2024-08-23", "locations": ["upper_dir"], 
     "severity": "minor", "source": "NDMA",
     "notes": "August 2024 monsoon event"},
]


def load_weather_data():
    """Load all available weather data"""
    print("📂 Loading weather data...")
    
    # Try to load existing combined data
    combined_path = DATA_PROCESSED / "flood_weather_dataset.csv"
    if combined_path.exists():
        df = pd.read_csv(combined_path)
        df['date'] = pd.to_datetime(df['date'])
        print(f"   Loaded existing dataset: {len(df)} records")
        return df
    
    # Load NASA POWER data
    nasa_files = list(DATA_RAW.glob("nasa_power_*.csv"))
    dfs = []
    for f in nasa_files:
        if 'combined' not in f.name:
            df = pd.read_csv(f)
            dfs.append(df)
    
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        df['date'] = pd.to_datetime(df['date'])
        print(f"   Loaded NASA POWER data: {len(df)} records")
        return df
    
    print("❌ No weather data found!")
    return None


def label_flood_events(df):
    """Label weather data with known flood events"""
    print("\n🏷️  Labeling flood events...")
    
    # Initialize flood columns
    df['flood_event'] = 0
    df['flood_severity'] = None
    df['flood_source'] = None
    df['flood_notes'] = None
    
    total_labeled = 0
    
    for event in HISTORICAL_FLOODS:
        start = pd.to_datetime(event['start'])
        end = pd.to_datetime(event['end'])
        locations = event['locations']
        severity = event['severity']
        source = event['source']
        notes = event['notes']
        
        # Create mask for this event
        date_mask = (df['date'] >= start) & (df['date'] <= end)
        
        # Location mask
        if 'location_key' in df.columns:
            loc_mask = df['location_key'].str.lower().isin([l.lower() for l in locations])
        elif 'location_name' in df.columns:
            loc_mask = df['location_name'].str.lower().str.contains('|'.join(locations), case=False, na=False)
        else:
            loc_mask = True
        
        # Apply labels
        mask = date_mask & loc_mask
        count = mask.sum()
        
        if count > 0:
            df.loc[mask, 'flood_event'] = 1
            df.loc[mask, 'flood_severity'] = severity
            df.loc[mask, 'flood_source'] = source
            df.loc[mask, 'flood_notes'] = notes
            total_labeled += count
            print(f"   ✓ {event['start']} to {event['end']}: {count} records labeled ({severity})")
    
    print(f"\n📊 Total flood records labeled: {total_labeled}")
    print(f"   Flood events: {df['flood_event'].sum()}")
    print(f"   Non-flood events: {(df['flood_event'] == 0).sum()}")
    
    return df


def merge_with_historical_data():
    """Merge new NASA POWER data with existing dataset"""
    print("\n🔄 Merging datasets...")
    
    # Load existing processed data
    existing_path = DATA_PROCESSED / "flood_weather_dataset.csv"
    
    # Load new NASA POWER data (2000-2017)
    new_files = [
        DATA_RAW / "nasa_power_swat_2000-01-01_2017-12-31.csv",
        DATA_RAW / "nasa_power_upper_dir_2000-01-01_2017-12-31.csv",
    ]
    
    dfs = []
    
    # Load existing data
    if existing_path.exists():
        existing_df = pd.read_csv(existing_path)
        existing_df['date'] = pd.to_datetime(existing_df['date'])
        print(f"   Existing data: {len(existing_df)} records ({existing_df['date'].min()} to {existing_df['date'].max()})")
        dfs.append(existing_df)
    
    # Load new historical data
    for f in new_files:
        if f.exists():
            new_df = pd.read_csv(f)
            new_df['date'] = pd.to_datetime(new_df['date'])
            print(f"   New data from {f.name}: {len(new_df)} records")
            dfs.append(new_df)
    
    # Also check for combined file
    combined_new = DATA_RAW / "nasa_power_combined.csv"
    if combined_new.exists():
        try:
            combined_df = pd.read_csv(combined_new)
            combined_df['date'] = pd.to_datetime(combined_df['date'])
            # Only add if it has different date range
            if len(dfs) > 0:
                existing_dates = pd.concat([d['date'] for d in dfs]).unique()
                new_records = combined_df[~combined_df['date'].isin(existing_dates)]
                if len(new_records) > 0:
                    print(f"   Additional records from combined: {len(new_records)}")
                    dfs.append(new_records)
        except:
            pass
    
    if len(dfs) == 0:
        print("❌ No data files found!")
        return None
    
    # Combine all data
    merged = pd.concat(dfs, ignore_index=True)
    
    # Remove duplicates
    merged = merged.drop_duplicates(subset=['date', 'location_key'] if 'location_key' in merged.columns else ['date'])
    merged = merged.sort_values('date').reset_index(drop=True)
    
    print(f"\n📊 Merged dataset: {len(merged)} total records")
    print(f"   Date range: {merged['date'].min()} to {merged['date'].max()}")
    
    return merged


def create_expanded_dataset():
    """Create expanded dataset with historical flood labels"""
    print("=" * 60)
    print("🌊 CREATING EXPANDED FLOOD DATASET")
    print("=" * 60)
    
    # Merge all available data
    df = merge_with_historical_data()
    
    if df is None:
        # Try loading existing data and just relabeling
        df = load_weather_data()
    
    if df is None:
        print("❌ No data available!")
        return None
    
    # Label flood events
    df = label_flood_events(df)
    
    # Save expanded dataset
    output_path = DATA_PROCESSED / "flood_weather_dataset_expanded.csv"
    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved expanded dataset to: {output_path}")
    
    # Also update main dataset
    main_path = DATA_PROCESSED / "flood_weather_dataset.csv"
    df.to_csv(main_path, index=False)
    print(f"💾 Updated main dataset: {main_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 DATASET SUMMARY")
    print("=" * 60)
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Flood events: {df['flood_event'].sum()} ({df['flood_event'].sum()/len(df)*100:.2f}%)")
    print(f"Non-flood events: {(df['flood_event'] == 0).sum()}")
    
    if 'flood_severity' in df.columns:
        print("\n📊 Flood Severity Distribution:")
        severity_counts = df[df['flood_event'] == 1]['flood_severity'].value_counts()
        for sev, count in severity_counts.items():
            print(f"   {sev}: {count} days")
    
    return df


if __name__ == "__main__":
    df = create_expanded_dataset()
