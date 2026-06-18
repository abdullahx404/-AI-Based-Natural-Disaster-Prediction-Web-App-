# 🌊 AI-Based Natural Disaster Prediction Web App

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An AI-powered web application for predicting floods in Khyber Pakhtunkhwa, Pakistan using machine learning and multiple AI techniques.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [AI Techniques Implemented](#-ai-techniques-implemented)
- [Installation](#-installation)
- [How to Run](#️-how-to-run)
- [Project Structure](#-project-structure)
- [How It Works](#️-how-it-works)
- [Dataset](#-dataset)
- [Model Performance](#-model-performance)
- [API Keys](#-api-keys)
- [Docker Deployment](#-docker-deployment)
- [Technologies Used](#️-technologies-used)

---

## 🎯 Overview

This project is a comprehensive **AI-based flood prediction system** for high-risk districts in Pakistan (Swat and Upper Dir). It combines:

- **Real-time weather data** from OpenWeatherMap API
- **Historical weather data** from NASA POWER and Meteostat (2000-2025)
- **Machine learning models** trained on 18,902 weather observations
- **Multiple AI techniques** including Search Algorithms, CSP, Neural Networks, Clustering, and Reinforcement Learning

### Why This Project?

Pakistan faces devastating floods every year, especially during monsoon season. This system aims to:

- Predict flood risk based on weather conditions
- Help authorities make informed evacuation decisions
- Provide early warnings to save lives

---

## ✨ Features

### Main Application Features

| Feature                  | Description                                           |
| ------------------------ | ----------------------------------------------------- |
| 🏠 **Dashboard**         | Real-time flood risk prediction with weather data     |
| 🔮 **Custom Prediction** | Enter manual weather parameters for prediction        |
| 📊 **Historical Data**   | Explore 25 years of weather and flood data            |
| 🤖 **Model Info**        | View model performance metrics and feature importance |
| ℹ️ **About**             | Project documentation and credits                     |

### AI Techniques (Interactive Demos)

| Technique                     | Application                                 |
| ----------------------------- | ------------------------------------------- |
| 🔍 **Search Algorithms**      | A\*, BFS, DFS for evacuation route planning |
| 🧩 **CSP**                    | Resource allocation for emergency response  |
| 🧬 **Neural Network**         | LSTM for time-series flood prediction       |
| 📈 **K-Means Clustering**     | Weather pattern analysis                    |
| 🎮 **Reinforcement Learning** | Q-Learning for evacuation decisions         |
| 🔬 **SHAP/LIME**              | Model explainability                        |

---

## 🧠 AI Techniques Implemented

### 1. Search Algorithms (Week 8)

**File:** `code/search_algorithms.py`

Finds optimal evacuation routes from flooded areas to safe zones.

```python
# Algorithms implemented:
- A* Search (informed, optimal)
- Breadth-First Search (optimal for unweighted)
- Depth-First Search (memory efficient)
```

**How it works:** Creates a grid-based flood scenario where some cells are flooded (obstacles). The algorithms find the shortest path from a start position to the nearest safe zone.

---

### 2. Constraint Satisfaction Problem (Week 9)

**File:** `code/csp_resource_allocation.py`

Allocates emergency resources (medical teams, rescue boats, supplies) to evacuation shelters.

```python
# Techniques used:
- AC-3 Arc Consistency (preprocessing)
- Backtracking Search
- MRV Heuristic (Minimum Remaining Values)
- LCV Heuristic (Least Constraining Value)
```

**How it works:** Given shelters with different populations and resource requirements, and limited resources, finds an optimal allocation that satisfies all constraints.

---

### 3. LSTM Neural Network (Week 11)

**File:** `code/neural_network.py`

Time-series prediction using Long Short-Term Memory networks.

```
Architecture:
Input (7 days × 5 features) → LSTM (64 units) → Dense (1, sigmoid)
```

**How it works:** Looks at the past 7 days of weather data to predict if a flood will occur. The LSTM can capture patterns like gradual rainfall buildup.

---

### 4. K-Means Clustering (Week 12)

**File:** `code/clustering.py`

Groups weather conditions into risk categories.

```
Clusters identified:
- Monsoon Pattern (HIGH RISK)
- Flash Flood Conditions (HIGH RISK)
- Moderate Rain (MODERATE RISK)
- Dry Conditions (LOW RISK)
```

**How it works:** Uses K-Means++ initialization to group similar weather patterns. Automatically labels clusters based on their characteristics.

---

### 5. Q-Learning / Reinforcement Learning (Week 12)

**File:** `code/reinforcement_learning.py`

Learns optimal evacuation decisions through trial and error.

```
Environment:
- States: (flood_level, population_at_risk, resources, time)
- Actions: Wait, Warn, Voluntary Evac, Mandatory Evac, Deploy Resources
- Rewards: +100/person saved, -500/casualty
```

**How it works:** Simulates thousands of flood scenarios. The agent learns when to issue warnings, start evacuations, and deploy resources to maximize lives saved.

---

### 6. SHAP & LIME Explainability (Bonus)

**File:** `code/explainability.py`

Explains why the model made a specific prediction.

```
Example output:
"Flood risk is 85% because:
 - Heavy rainfall (+40%)
 - High humidity (+25%)
 - Monsoon season (+15%)"
```

---

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/zohaibkhan745/-AI-Based-Natural-Disaster-Prediction-Web-App-.git
cd -AI-Based-Natural-Disaster-Prediction-Web-App-
```

### Step 2: Create Virtual Environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/Mac:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up API Key (Optional but Recommended)

Create `.streamlit/secrets.toml`:

```toml
OPENWEATHER_API_KEY = "your_api_key_here"
```

Get a free API key from [OpenWeatherMap](https://openweathermap.org/api).

---

## ▶️ How to Run

### Option 1: Run the Web App (Recommended)

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### Option 2: Run with Docker

```bash
docker-compose up --build
```

### Option 3: Run Individual Components

| Command                        | Description                   |
| ------------------------------ | ----------------------------- |
| `streamlit run app.py`         | Start web application         |
| `python run_pipeline.py`       | Run full ML training pipeline |
| `python test_model.py`         | Test model predictions        |
| `python verify_predictions.py` | Verify model outputs          |

### Run AI Technique Demos

```bash
# Search Algorithms Demo
python code/search_algorithms.py

# CSP Demo
python code/csp_resource_allocation.py

# Neural Network Demo
python code/neural_network.py

# Clustering Demo
python code/clustering.py

# Reinforcement Learning Demo
python code/reinforcement_learning.py

# Explainability Demo
python code/explainability.py
```

---

## 📁 Project Structure

```
AI-Based-Natural-Disaster/
│
├── 📱 app.py                          # Main Streamlit web application
│
├── 📂 code/                           # Source code modules
│   ├── search_algorithms.py           # A*, BFS, DFS (Week 8)
│   ├── csp_resource_allocation.py     # CSP (Week 9)
│   ├── neural_network.py              # LSTM (Week 11)
│   ├── clustering.py                  # K-Means (Week 12)
│   ├── reinforcement_learning.py      # Q-Learning (Week 12)
│   ├── explainability.py              # SHAP/LIME (Bonus)
│   ├── improved_models.py             # ML model training
│   ├── preprocessing.py               # Data preprocessing
│   ├── baseline_models.py             # Baseline ML models
│   ├── model_evaluation.py            # Evaluation metrics
│   ├── fetch_nasa_power.py            # NASA POWER API
│   ├── fetch_meteostat_weather.py     # Meteostat API
│   ├── merge_weather_data.py          # Data merging
│   └── label_historical_floods.py     # Flood labeling
│
├── 📂 data/
│   ├── raw/                           # Raw API data
│   │   ├── nasa_power_*.csv
│   │   ├── weather_*.csv
│   │   └── ndma_flood_reports.csv
│   └── processed/                     # Cleaned datasets
│       ├── flood_weather_dataset.csv  # Main training data (18,902 records)
│       ├── cleaned_swat.csv
│       └── cleaned_upper_dir.csv
│
├── 📂 results/                        # Model outputs
│   ├── best_flood_model.pkl           # Trained model
│   ├── model_metrics.csv              # Performance metrics
│   ├── feature_importance.json        # Feature rankings
│   └── evaluation_report.txt          # Detailed report
│
├── 📂 docs/                           # Documentation
├── 📂 notebooks/                      # Jupyter notebooks
├── 📂 .streamlit/                     # Streamlit config
├── 📂 .github/workflows/              # CI/CD
│
├── 🐳 Dockerfile                      # Docker config
├── 🐳 docker-compose.yml              # Docker Compose
├── 📋 requirements.txt                # Python dependencies
├── 📖 README.md                       # This file
└── 📖 AI_TECHNIQUES_SUMMARY.md        # AI techniques documentation
```

---

## ⚙️ How It Works

### Data Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   NASA POWER    │────▶│   Data Merge    │────▶│   Preprocessing │
│   (2000-2025)   │     │   & Cleaning    │     │   24 Features   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                                              │
         │              ┌─────────────────┐            ▼
         └─────────────▶│   Fill Missing  │     ┌─────────────────┐
                        │   Values        │     │   ML Training   │
┌─────────────────┐     └─────────────────┘     │   (3 Models)    │
│   Meteostat     │────────────────────────────▶└─────────────────┘
│   (2018-2025)   │                                    │
└─────────────────┘                                    ▼
                                                ┌─────────────────┐
┌─────────────────┐                             │   Best Model    │
│   NDMA Reports  │────▶ Flood Labels ─────────▶│   (60% Recall)  │
│   + Historical  │      (517 events)           └─────────────────┘
└─────────────────┘
```

### Prediction Flow

```
User Input          ──▶  Feature Engineering  ──▶  Model Prediction
(Weather Data)           (24 features)             (Flood Probability)
                                                          │
                                                          ▼
                                                   Risk Assessment
                                                   LOW / MODERATE / HIGH
```

### 24 Engineered Features

| Category          | Features                                                                    |
| ----------------- | --------------------------------------------------------------------------- |
| **Temperature**   | tavg, tmin, tmax, temp_range, tavg_7day_avg                                 |
| **Precipitation** | prcp, prcp_7day_avg, prcp_3day_sum, prcp_7day_sum, heavy_rain, extreme_rain |
| **Atmospheric**   | pres, humidity, pressure_anomaly, high_humidity                             |
| **Wind**          | wspd, wpgt, wspd_7day_avg                                                   |
| **Solar**         | solar_radiation                                                             |
| **Temporal**      | month, day_of_year, quarter, is_monsoon                                     |
| **Location**      | location_encoded                                                            |

---

## 📊 Dataset

### Statistics

| Metric            | Value                          |
| ----------------- | ------------------------------ |
| **Total Records** | 18,902                         |
| **Time Range**    | January 2000 - November 2025   |
| **Flood Events**  | 517 (2.74%)                    |
| **Features**      | 24 engineered                  |
| **Locations**     | Swat, Upper Dir (KP, Pakistan) |

### Data Sources

1. **NASA POWER API** - Satellite-derived meteorological data (2000-2025)
2. **Meteostat API** - Ground station weather data (2018-2025)
3. **NDMA Reports** - Historical flood event records
4. **Historical Archives** - Major flood events database

---

## 📈 Model Performance

### Best Model: Logistic Regression (Class Weighted)

| Metric        | Score  |
| ------------- | ------ |
| **Recall**    | 60% ⭐ |
| **Precision** | 45%    |
| **F1 Score**  | 51%    |
| **Accuracy**  | 97%    |

### Why Recall Matters

In flood prediction, **missing a real flood is worse than a false alarm**:

- ✅ 60% of actual floods are detected
- ⚠️ Some false alarms (acceptable trade-off for safety)

### Model Comparison

| Model                   | Recall  | Precision | F1  |
| ----------------------- | ------- | --------- | --- |
| **Logistic Regression** | **60%** | 45%       | 51% |
| Random Forest           | 53%     | 52%       | 52% |
| Gradient Boosting       | 43%     | 58%       | 49% |

---

## 🔑 API Keys

### OpenWeatherMap (For Real-time Weather)

1. Sign up at [OpenWeatherMap](https://openweathermap.org/api)
2. Get your free API key
3. Create `.streamlit/secrets.toml`:

```toml
OPENWEATHER_API_KEY = "your_api_key_here"
```

**Without API key:** The app uses demo/simulated weather data.

---

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
docker-compose up --build
```

### Manual Docker Build

```bash
# Build the image
docker build -t flood-prediction-app .

# Run the container
docker run -p 8501:8501 flood-prediction-app
```

Access the app at `http://localhost:8501`

---

## 🛠️ Technologies Used

| Category            | Technologies                          |
| ------------------- | ------------------------------------- |
| **Frontend**        | Streamlit, Plotly                     |
| **ML/AI**           | scikit-learn, NumPy, Pandas           |
| **Neural Network**  | Custom LSTM implementation            |
| **APIs**            | OpenWeatherMap, NASA POWER, Meteostat |
| **Deployment**      | Docker, GitHub Actions                |
| **Version Control** | Git, GitHub                           |

---

## 📚 Requirements

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.18.0
requests>=2.31.0
python-dateutil>=2.8.2
```

Full list in `requirements.txt`

---

## 👨‍💻 Author

**CS351 - Artificial Intelligence Project**  
Semester 5

---

## ⚠️ Disclaimer

This is an **educational project** demonstrating AI techniques for disaster prediction. For actual emergency situations, please refer to:

- [NDMA Pakistan](https://ndma.gov.pk/)
- [PMD Pakistan](https://www.pmd.gov.pk/)
- Local emergency services

---

## 🙏 Acknowledgments

- NASA POWER for satellite data
- Meteostat for weather data
- NDMA Pakistan for flood reports
- Streamlit for the web framework
- scikit-learn for ML tools

---

<p align="center">
  Made with ❤️ for CS351 - Artificial Intelligence
</p>
