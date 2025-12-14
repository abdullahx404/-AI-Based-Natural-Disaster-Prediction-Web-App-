"""
Data Preprocessing Module
Handles data cleaning, feature engineering, scaling, and train-test split
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create results directory if it doesn't exist
RESULTS_DIR.mkdir(exist_ok=True)


class DataPreprocessor:
    """Handles all preprocessing tasks for flood prediction dataset"""
    
    def __init__(self, data_path):
        """Initialize preprocessor with data path"""
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_data(self):
        """Load dataset from CSV"""
        print("📂 Loading data...")
        self.df = pd.read_csv(self.data_path)
        print(f"✅ Dataset shape: {self.df.shape}")
        print(f"📊 Columns: {list(self.df.columns)}")
        return self.df
    
    def explore_data(self):
        """Explore dataset structure and statistics"""
        print("\n" + "="*60)
        print("📊 DATA EXPLORATION")
        print("="*60)
        
        print(f"\n🔍 Dataset Info:")
        print(f"  • Total rows: {len(self.df)}")
        print(f"  • Total columns: {len(self.df.columns)}")
        print(f"  • Date range: {self.df['date'].min()} to {self.df['date'].max()}")
        
        print(f"\n🎯 Target Variable Distribution:")
        target_dist = self.df['flood_event'].value_counts()
        print(target_dist)
        print(f"  • No Flood: {target_dist.get(0, 0)} ({target_dist.get(0, 0)/len(self.df)*100:.1f}%)")
        print(f"  • Flood: {target_dist.get(1, 0)} ({target_dist.get(1, 0)/len(self.df)*100:.1f}%)")
        
        print(f"\n📍 Locations:")
        print(f"  • Unique locations: {self.df['location_name'].nunique()}")
        for loc in self.df['location_name'].unique():
            count = len(self.df[self.df['location_name'] == loc])
            print(f"    - {loc}: {count} records")
        
        print(f"\n❌ Missing Values:")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        for col in self.df.columns:
            if missing[col] > 0:
                print(f"  • {col}: {missing[col]} ({missing_pct[col]:.2f}%)")
        
        print(f"\n📈 Feature Statistics:")
        print(self.df[['tavg', 'tmin', 'tmax', 'prcp', 'wspd', 'pres', 'humidity', 'solar_radiation']].describe())
    
    def handle_missing_values(self):
        """Handle missing values in the dataset"""
        print("\n" + "="*60)
        print("🧹 HANDLING MISSING VALUES")
        print("="*60)
        
        # First, merge Meteostat and NASA data (they cover different time periods)
        # Meteostat: 2018-2025, NASA POWER: 2000-2017
        nasa_mapping = {
            'tavg': 'nasa_tavg',
            'tmin': 'nasa_tmin', 
            'tmax': 'nasa_tmax',
            'prcp': 'nasa_prcp',
            'wspd': 'nasa_wspd',
            'wpgt': 'nasa_wpgt',
            'pres': 'nasa_pres'
        }
        
        print("🔀 Merging Meteostat and NASA POWER data...")
        for meteo_col, nasa_col in nasa_mapping.items():
            if meteo_col in self.df.columns and nasa_col in self.df.columns:
                # Fill Meteostat missing values with NASA data
                before_missing = self.df[meteo_col].isnull().sum()
                self.df[meteo_col] = self.df[meteo_col].fillna(self.df[nasa_col])
                after_missing = self.df[meteo_col].isnull().sum()
                print(f"   {meteo_col}: {before_missing} → {after_missing} missing (filled {before_missing - after_missing})")
        
        # Fill NaN values in weather features with forward fill then backward fill
        weather_cols = ['tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wspd', 'wpgt', 'pres', 'tsun', 'humidity', 'solar_radiation']
        
        for col in weather_cols:
            if col in self.df.columns:
                # Group by location to avoid data leakage between regions
                self.df[col] = self.df.groupby('location_key')[col].transform(
                    lambda x: x.ffill().bfill()
                )
                # Fill remaining NaN with median by location
                self.df[col] = self.df.groupby('location_key')[col].transform(
                    lambda x: x.fillna(x.median())
                )
        
        print("✅ Missing values handled!")
        print(f"   Remaining NaN count: {self.df.isnull().sum().sum()}")
    
    def feature_engineering(self):
        """Create new features for improved prediction"""
        print("\n" + "="*60)
        print("🔧 FEATURE ENGINEERING")
        print("="*60)
        
        # Convert date to datetime
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Temporal features
        self.df['month'] = self.df['date'].dt.month
        self.df['day_of_year'] = self.df['date'].dt.dayofyear
        self.df['quarter'] = self.df['date'].dt.quarter
        
        # Monsoon season flag (June-September is peak flood season in Pakistan)
        self.df['is_monsoon'] = self.df['month'].isin([6, 7, 8, 9]).astype(int)
        
        # Temperature-based features
        if 'tmax' in self.df.columns and 'tmin' in self.df.columns:
            self.df['temp_range'] = self.df['tmax'] - self.df['tmin']
        
        # Humidity-based features (if available)
        if 'humidity' in self.df.columns:
            self.df['high_humidity'] = (self.df['humidity'] > 70).astype(int)
        
        # Pressure-based features (low pressure often indicates storms)
        if 'pres' in self.df.columns:
            location_pres_mean = self.df.groupby('location_key')['pres'].transform('mean')
            self.df['pressure_anomaly'] = self.df['pres'] - location_pres_mean
        
        # Rolling features (7-day rolling averages)
        for col in ['prcp', 'tavg', 'wspd']:
            if col in self.df.columns:
                self.df[f'{col}_7day_avg'] = self.df.groupby('location_key')[col].transform(
                    lambda x: x.rolling(window=7, min_periods=1).mean()
                )
        
        # CRITICAL: Cumulative precipitation (floods often from multi-day rain)
        if 'prcp' in self.df.columns:
            # 3-day cumulative precipitation
            self.df['prcp_3day_sum'] = self.df.groupby('location_key')['prcp'].transform(
                lambda x: x.rolling(window=3, min_periods=1).sum()
            )
            # 7-day cumulative precipitation
            self.df['prcp_7day_sum'] = self.df.groupby('location_key')['prcp'].transform(
                lambda x: x.rolling(window=7, min_periods=1).sum()
            )
            # Heavy rain flag (above 10mm in a day)
            self.df['heavy_rain'] = (self.df['prcp'] > 10).astype(int)
            # Extreme rain flag (above 50mm in a day)
            self.df['extreme_rain'] = (self.df['prcp'] > 50).astype(int)
        
        # Location encoding
        location_map = {loc: idx for idx, loc in enumerate(self.df['location_key'].unique())}
        self.df['location_encoded'] = self.df['location_key'].map(location_map)
        
        print("✅ Features engineered!")
        print(f"   New feature count: {self.df.shape[1]}")
        print(f"   New features: month, day_of_year, quarter, is_monsoon, temp_range, high_humidity,")
        print(f"                pressure_anomaly, prcp_7day_avg, prcp_3day_sum, prcp_7day_sum,")
        print(f"                heavy_rain, extreme_rain, tavg_7day_avg, wspd_7day_avg, location_encoded")
    
    def select_features(self):
        """Select relevant features for modeling"""
        print("\n" + "="*60)
        print("🎯 FEATURE SELECTION")
        print("="*60)
        
        # Features to use in the model (including new cumulative features)
        feature_columns = [
            'tavg', 'tmin', 'tmax', 'prcp', 'wspd', 'wpgt', 'pres', 'humidity', 'solar_radiation',
            'month', 'day_of_year', 'quarter', 'is_monsoon', 'temp_range', 'high_humidity',
            'pressure_anomaly', 'prcp_7day_avg', 'prcp_3day_sum', 'prcp_7day_sum',
            'heavy_rain', 'extreme_rain', 'tavg_7day_avg', 'wspd_7day_avg', 'location_encoded'
        ]
        
        # Filter only existing columns
        self.feature_names = [col for col in feature_columns if col in self.df.columns]
        
        print(f"✅ Selected {len(self.feature_names)} features:")
        for i, feat in enumerate(self.feature_names, 1):
            print(f"   {i:2d}. {feat}")
        
        return self.feature_names
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """Prepare features and target, then split into train/test"""
        print("\n" + "="*60)
        print("📊 DATA PREPARATION & SPLIT")
        print("="*60)
        
        # Prepare X and y
        X = self.df[self.feature_names].copy()
        y = self.df['flood_event'].copy()
        
        print(f"\n📦 Dataset composition:")
        print(f"   • Features shape: {X.shape}")
        print(f"   • Target shape: {y.shape}")
        print(f"   • Target distribution:\n{y.value_counts()}")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\n✅ Train-Test Split (80-20):")
        print(f"   • Training set: {self.X_train.shape[0]} samples")
        print(f"   • Test set: {self.X_test.shape[0]} samples")
        print(f"   • Train flood ratio: {self.y_train.sum()/len(self.y_train)*100:.2f}%")
        print(f"   • Test flood ratio: {self.y_test.sum()/len(self.y_test)*100:.2f}%")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def scale_features(self):
        """Scale features using StandardScaler"""
        print("\n" + "="*60)
        print("⚖️  FEATURE SCALING")
        print("="*60)
        
        # Fit scaler on training data
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print("✅ Features scaled using StandardScaler!")
        print(f"   • Mean: {self.X_train.mean():.4f}")
        print(f"   • Std Dev: {self.X_train.std():.4f}")
        
        return self.X_train, self.X_test
    
    def save_preprocessed_data(self):
        """Save preprocessed data to CSV"""
        print("\n" + "="*60)
        print("💾 SAVING PREPROCESSED DATA")
        print("="*60)
        
        # Create DataFrames with scaled features
        train_data = pd.DataFrame(
            self.X_train,
            columns=self.feature_names
        )
        train_data['flood_event'] = self.y_train.values
        
        test_data = pd.DataFrame(
            self.X_test,
            columns=self.feature_names
        )
        test_data['flood_event'] = self.y_test.values
        
        # Save
        train_path = RESULTS_DIR / "training_data.csv"
        test_path = RESULTS_DIR / "test_data.csv"
        
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        
        print(f"✅ Training data saved: {train_path}")
        print(f"✅ Test data saved: {test_path}")
    
    def run_full_pipeline(self):
        """Execute complete preprocessing pipeline"""
        print("\n" + "🚀 "*30)
        print("STARTING PREPROCESSING PIPELINE")
        print("🚀 "*30 + "\n")
        
        self.load_data()
        self.explore_data()
        self.handle_missing_values()
        self.feature_engineering()
        self.select_features()
        self.prepare_data()
        self.scale_features()
        self.save_preprocessed_data()
        
        print("\n" + "✅ "*30)
        print("PREPROCESSING COMPLETE!")
        print("✅ "*30)
        
        return {
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'feature_names': self.feature_names,
            'scaler': self.scaler
        }


if __name__ == "__main__":
    # Load and preprocess data
    data_file = DATA_PROCESSED / "flood_weather_dataset.csv"
    
    preprocessor = DataPreprocessor(data_file)
    preprocessor.run_full_pipeline()
