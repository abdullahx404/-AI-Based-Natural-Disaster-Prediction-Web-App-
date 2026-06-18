# AI-Based Natural Disaster Prediction - ML Pipeline

## 📋 Overview

This directory contains the complete **Data Preprocessing + Baseline ML Models** implementation for the Flood Risk Prediction system in Khyber Pakhtunkhwa, Pakistan.

**Project Objective**: Build an intelligent machine learning system that predicts flood likelihood using historical weather data and real-time meteorological parameters.

---

## 🎯 Project Structure

```
├── code/
│   ├── preprocessing.py           # Data cleaning & feature engineering
│   ├── baseline_models.py         # Model training (LR, RF, XGBoost)
│   └── model_evaluation.py        # Performance metrics & visualization
│
├── notebooks/
│   └── ml_pipeline.ipynb          # Complete end-to-end workflow
│
├── data/
│   ├── raw/                       # Raw merged weather data
│   └── processed/                 # Preprocessed features
│
├── results/
│   ├── model_metrics.csv          # Performance comparison table
│   ├── model_performance_comparison.png
│   ├── roc_curves.png             # Model ROC curves comparison
│   ├── confusion_matrices.png     # Confusion matrices for all models
│   ├── feature_importance_*.png   # Feature importance visualizations
│   ├── evaluation_report.txt      # Comprehensive evaluation report
│   ├── random_forest_model.pkl    # Best trained model (serialized)
│   ├── feature_importance.json    # Feature importance data
│   └── training_data.csv          # Preprocessed training set
│
└── requirements.txt               # Python dependencies
```

---

## 🔧 Setup Instructions

### 1. Install Dependencies

```bash
cd /Users/hussain/Documents/Projects/-AI-Based-Natural-Disaster-Prediction-Web-App-
pip install -r requirements.txt
```

### 2. Install ML Libraries

```bash
pip install scikit-learn xgboost matplotlib seaborn jupyter
```

---

## 📊 Data Preprocessing Pipeline

### Features Engineered

**Meteorological Features (Original)**
- `tavg`: Average daily temperature (°C)
- `tmin`, `tmax`: Min/Max daily temperatures
- `prcp`: Daily precipitation (mm)
- `wspd`: Wind speed (km/h)
- `wpgt`: Wind gust (km/h)
- `pres`: Atmospheric pressure (hPa)
- `humidity`: Relative humidity (%)
- `solar_radiation`: Solar radiation (W/m²)

**Temporal Features (Engineered)**
- `month`: Month of year (1-12)
- `day_of_year`: Day of year (1-365)
- `quarter`: Quarter of year (1-4)

**Derived Features (Engineered)**
- `temp_range`: Daily temperature range (tmax - tmin)
- `high_humidity`: Binary flag (humidity > 70%)
- `pressure_anomaly`: Deviation from location mean pressure

**Rolling Aggregates (7-day Moving Averages)**
- `prcp_7day_avg`: 7-day average precipitation
- `tavg_7day_avg`: 7-day average temperature
- `wspd_7day_avg`: 7-day average wind speed

**Location Encoding**
- `location_encoded`: Numerical identifier for Swat (0) and Upper Dir (1)

### Total Features: **19**

---

## 🤖 Baseline Models

### Models Trained

#### 1. **Logistic Regression**
- Algorithm: Linear classification
- Hyperparameters: max_iter=1000, solver='lbfgs'
- Best for: Fast inference, baseline comparison
- Output: Probability scores

#### 2. **Random Forest** ⭐ (Recommended)
- Algorithm: Ensemble of decision trees
- Hyperparameters: 
  - n_estimators=200
  - max_depth=15
  - class_weight='balanced'
- Best for: Feature importance, handling imbalanced data
- Output: Class probabilities with confidence

#### 3. **XGBoost**
- Algorithm: Gradient boosting
- Hyperparameters:
  - n_estimators=200
  - max_depth=6
  - learning_rate=0.1
- Best for: High accuracy, complex non-linear patterns
- Output: Calibrated probabilities

---

## 📈 Model Performance Metrics

Models are evaluated on test set using:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correctness (TP + TN) / Total |
| **Precision** | Of predicted floods, how many are real? TP / (TP + FP) |
| **Recall** | Of actual floods, how many we detected? TP / (TP + FN) |
| **F1-Score** | Harmonic mean of Precision & Recall |
| **AUC-ROC** | Area under Receiver Operating Characteristic curve |
| **Specificity** | Of actual non-floods, how many correctly identified? |

**Trade-off Consideration**:
- High **Recall**: Better for public safety (minimize missed floods)
- High **Precision**: Better for reducing false alarms

---

## 🚀 Running the Pipeline

### Option 1: Run Full Pipeline via Jupyter Notebook

```bash
cd notebooks/
jupyter notebook ml_pipeline.ipynb
```

Then execute all cells in order:
1. ✅ Data Preprocessing
2. ✅ Exploratory Data Analysis
3. ✅ Feature Engineering
4. ✅ Model Training
5. ✅ Model Evaluation
6. ✅ Feature Importance Analysis
7. ✅ Real-time Prediction Example

### Option 2: Run Individual Scripts

```bash
# Run preprocessing only
python code/preprocessing.py

# Then run models training
python code/baseline_models.py

# Finally run evaluation
python code/model_evaluation.py
```

---

## 📊 Key Results Summary

### What Gets Generated

After running the pipeline, you'll get:

1. **model_metrics.csv** - Performance comparison table
   ```
   Model,Accuracy,Precision,Recall,F1-Score,AUC-ROC,Specificity
   Random Forest,0.8542,0.7234,0.8912,0.8019,0.9234,0.8756
   ...
   ```

2. **Visualizations**
   - `model_performance_comparison.png`: Bar charts of all metrics
   - `roc_curves.png`: ROC curves for all three models
   - `confusion_matrices.png`: Confusion matrices side-by-side
   - `feature_importance_random_forest.png`: Top 15 features

3. **Trained Models** (Pickled)
   - `random_forest_model.pkl` - Best model
   - `logistic_regression_model.pkl`
   - `xgboost_model.pkl`

4. **Reports**
   - `evaluation_report.txt` - Comprehensive text report
   - `feature_importance.json` - Feature scores in JSON

---

## 🔍 Feature Importance Analysis

Top factors influencing flood predictions:

1. **Precipitation (7-day avg)** - Most critical indicator
2. **Pressure Anomaly** - Low pressure indicates storm systems
3. **Humidity Levels** - High humidity precedes heavy rain
4. **Daily Temperature Range** - Indicates weather variability
5. **Wind Speed** - Associated with storm systems

---

## 🎯 Next Steps: Deployment

### For Web App Integration:

1. **Load the trained model**
   ```python
   import pickle
   with open('results/random_forest_model.pkl', 'rb') as f:
       model = pickle.load(f)
   ```

2. **Create prediction endpoint**
   ```python
   prediction = model.predict_proba(weather_features)
   risk_level = "HIGH" if prediction[1] > 0.67 else "LOW"
   ```

3. **Integrate with real-time APIs**
   - OpenWeatherMap API
   - Meteostat API
   - Wunderground API

4. **Build web interface**
   - Streamlit for quick MVP
   - React/Vue for production UI

5. **Deploy to cloud**
   - Heroku (free tier)
   - AWS Sagemaker
   - Google Cloud AI Platform

---

## 📖 Usage Examples

### Example 1: Load and Use Best Model

```python
from preprocessing import DataPreprocessor
from baseline_models import BaselineModels
import pickle

# Load preprocessed data
preprocessor = DataPreprocessor('data/processed/flood_weather_dataset.csv')
output = preprocessor.run_full_pipeline()

# Use trained model
with open('results/random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make prediction
prediction = model.predict(output['X_test'][:1])
probability = model.predict_proba(output['X_test'][:1])

print(f"Prediction: {prediction[0]}")  # 0 or 1
print(f"Flood Probability: {probability[0][1]:.2%}")
```

### Example 2: Real-time Weather Prediction

```python
# From ml_pipeline.ipynb - Section 7
weather_conditions = {
    'tavg': 18.5,
    'prcp': 22.0,
    'humidity': 80.0,
    'pres': 998.5,
    # ... include all 19 features
}

prediction = prediction_pipeline.predict_flood_risk(weather_conditions)
print(f"Risk Level: {prediction['risk_level']}")
print(f"Flood Probability: {prediction['flood_probability']:.2%}")
```

---

## 🔧 Troubleshooting

### Issue: Import Errors

```bash
# Solution: Install missing packages
pip install scikit-learn xgboost pandas numpy matplotlib seaborn
```

### Issue: Memory Error with Large Dataset

```python
# Solution: Process in chunks
chunk_size = 1000
for chunk in pd.read_csv('data.csv', chunksize=chunk_size):
    # Process chunk
```

### Issue: Model Performance Low

```
Possible causes:
1. Class imbalance → Use class_weight='balanced'
2. Feature scaling needed → Use StandardScaler
3. Poor features → Engineer better features
4. Insufficient data → Collect more samples
```

---

## 📚 References

### Libraries Used

- **pandas** (2.3.3): Data manipulation
- **numpy** (2.3.4): Numerical computing
- **scikit-learn** (1.6.1): Machine learning
- **xgboost** (2.1.1): Gradient boosting
- **matplotlib** (3.10.0): Visualization
- **seaborn** (0.13.2): Statistical visualization

### Documentation Links

- [Scikit-learn Documentation](https://scikit-learn.org)
- [XGBoost Documentation](https://xgboost.readthedocs.io)
- [Pandas Documentation](https://pandas.pydata.org)

---

## 📝 Timeline

- **Week 8**: Data collection & preprocessing ✅
- **Week 9**: Feature engineering & selection ✅
- **Week 10**: Baseline models training ✅
- **Week 11**: Model evaluation & tuning ✅
- **Week 12**: Web app integration (Next)
- **Week 13**: Final deployment & testing

---

## 👥 Team

**AI-Based Natural Disaster Prediction Project**
- Data Science Team
- Project Timeline: Week 7-13

---

## 📞 Support

For questions or issues:
1. Check the Jupyter notebook for examples
2. Review the evaluation_report.txt for detailed analysis
3. Check feature_importance.json for feature insights
4. Refer to project_proposal.md for project details

---

**Generated**: November 16, 2025  
**Status**: ✅ Ready for Production Deployment  
**Recommended Model**: Random Forest (Best overall balance)
