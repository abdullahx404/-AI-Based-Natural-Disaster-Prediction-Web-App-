# 📚 Project Stages Explanation

## AI-Based Natural Disaster Prediction Web App

This document explains the project step-by-step, from data gathering to the final web application. Each stage is explained in simple words.

---

## 🔄 Project Flow Overview

```
Stage 1: Data Gathering
    ↓
Stage 2: Data Cleaning & Preparation  
    ↓
Stage 3: Feature Engineering
    ↓
Stage 4: Model Training
    ↓
Stage 5: AI Techniques Implementation
    ↓
Stage 6: Web Application Development
    ↓
Stage 7: Deployment
```

---

# Stage 1: Data Gathering

## What is Data Gathering?
Data gathering is the process of collecting information needed to train our AI models. For flood prediction, we need weather data and records of past floods.

## What Data Was Collected?

### 1. Weather Data
| Source | What It Provides | Time Period |
|--------|------------------|-------------|
| NASA POWER | Satellite weather data (from space) | 2000-2025 |
| Meteostat | Ground station data (from sensors on Earth) | 2018-2025 |
| OpenWeatherMap | Current live weather | Real-time |

### 2. Flood Event Data
- Historical flood records from NDMA (Pakistan's disaster agency)
- News reports of major floods
- Government flood reports

## How Was Data Collected?

### NASA POWER API
```
Simple Explanation:
- We send a request to NASA's website
- We ask for weather data for Swat and Upper Dir
- NASA sends back daily weather readings
- We save this data as CSV files
```

**File**: `code/fetch_nasa_power.py`
- Connects to NASA's servers
- Downloads weather data for each location
- Saves data in `data/raw/` folder

### Meteostat API
```
Simple Explanation:
- Similar to NASA, but from ground weather stations
- More accurate for recent years (2018-2025)
- Provides data like temperature, rain, wind
```

**File**: `code/fetch_meteostat_weather.py`
- Connects to Meteostat servers
- Downloads weather station readings
- Fills gaps in NASA data

## Result of Stage 1
- 13 CSV files in `data/raw/` folder
- ~18,000+ weather records
- Data from 2000 to 2025

---

# Stage 2: Data Cleaning & Preparation

## What is Data Cleaning?
Raw data often has problems:
- Missing values (empty cells)
- Duplicate rows
- Wrong formats
- Outliers (unusual values)

Data cleaning fixes these problems.

## What Problems Were Found?

### 1. Missing Values
```
Problem: Some days had no temperature or rainfall readings
Solution: 
- First, try to fill from NASA data (if Meteostat missing)
- Then use "forward fill" (use yesterday's value)
- Finally, use average value for that location
```

### 2. Duplicate Records
```
Problem: Same date appeared twice for same location
Solution: Keep only one record per date per location
```

### 3. Different Date Formats
```
Problem: Some dates were "2024-01-15", others were "15/01/2024"
Solution: Convert all dates to same format (YYYY-MM-DD)
```

## How Data Was Cleaned

**File**: `code/preprocessing.py` and `code/clean_weather_pipeline.py`

```python
# Simple flow:
1. Load raw CSV files
2. Find missing values
3. Fill missing values using smart techniques
4. Remove duplicates
5. Check for outliers
6. Save cleaned data
```

## Result of Stage 2
- Clean data in `data/processed/` folder
- `flood_weather_dataset.csv` - main clean dataset
- No missing values
- Consistent date formats

---

# Stage 3: Feature Engineering

## What is Feature Engineering?
Feature engineering means creating new useful information from existing data. It helps the AI model learn better.

## Example:
```
Original data: Just the date "2024-07-15"

Engineered features:
- month = 7 (July)
- is_monsoon = 1 (Yes, monsoon season)
- quarter = 3 (Q3 of year)
```

## Features Created (24 Total)

### Temperature Features
| Feature | What It Means |
|---------|---------------|
| tavg | Average temperature of the day |
| tmin | Minimum (lowest) temperature |
| tmax | Maximum (highest) temperature |
| temp_range | Difference between max and min |
| tavg_7day_avg | Average temperature over past 7 days |

### Rainfall Features
| Feature | What It Means |
|---------|---------------|
| prcp | Rainfall amount today |
| prcp_7day_avg | Average rainfall over past 7 days |
| prcp_3day_sum | Total rainfall in last 3 days |
| prcp_7day_sum | Total rainfall in last 7 days |
| heavy_rain | Is rainfall > 50mm? (Yes=1, No=0) |
| extreme_rain | Is rainfall > 100mm? (Yes=1, No=0) |

### Time Features
| Feature | What It Means |
|---------|---------------|
| month | Which month (1-12) |
| day_of_year | Which day of year (1-365) |
| quarter | Which quarter (1-4) |
| is_monsoon | Is it monsoon season? (June-Sept = 1) |

### Why These Features Matter
```
Example: Flood prediction

Monsoon season (June-September) + Heavy rain + High humidity
    = HIGH flood risk

Winter season (December-February) + Low rain + Low humidity
    = LOW flood risk
```

## Result of Stage 3
- 24 features ready for model training
- Each feature helps predict floods
- Data saved in `data/processed/flood_weather_dataset_cleaned.csv`

---

# Stage 4: Model Training

## What is Model Training?
Model training teaches the AI to recognize patterns. We show it examples of weather conditions and tell it whether a flood happened or not.

## How Training Works (Simple Explanation)
```
1. Show the model: "This weather = Flood"
2. Show the model: "This weather = No Flood"
3. Repeat thousands of times
4. Model learns the patterns
5. Now model can predict for new weather
```

## Models Trained

### 1. Logistic Regression
```
Simple Explanation:
- Like drawing a line to separate "flood" from "no flood"
- Simple but effective
- Best model for this project (60% recall)
```

### 2. Random Forest
```
Simple Explanation:
- Builds many "decision trees" (like a flowchart)
- Each tree votes: "flood" or "no flood"
- Final answer = majority vote
```

### 3. Gradient Boosting
```
Simple Explanation:
- Builds trees one at a time
- Each new tree fixes mistakes of previous trees
- Gets better with each tree
```

## Training Process

**File**: `code/improved_models.py`

```
Steps:
1. Split data: 80% for training, 20% for testing
2. Balance classes (floods are rare, only 2.7%)
3. Train each model
4. Find best threshold for predictions
5. Save best model
```

## Model Performance

| Model | Recall | What It Means |
|-------|--------|---------------|
| Logistic Regression | 60% | Catches 60% of real floods |
| Random Forest | 53% | Catches 53% of real floods |
| Gradient Boosting | 43% | Catches 43% of real floods |

### Why Recall Matters
```
In flood prediction:
- Missing a real flood = DANGEROUS (people not warned)
- False alarm = ANNOYING but SAFE

So we prioritize catching real floods (high recall)
even if we have some false alarms.
```

## Result of Stage 4
- Trained models saved in `results/` folder
- `best_flood_model.pkl` - best performing model
- Performance metrics saved

---

# Stage 5: AI Techniques Implementation

## What AI Techniques Were Added?
Beyond basic prediction, the project implements 6 AI techniques learned in the course.

## 1. Search Algorithms (Week 8)

### What They Do
Find the best evacuation route during a flood.

### How They Work
```
Imagine a grid map:
- Some cells are flooded (can't pass)
- Person needs to reach safe zone
- Algorithms find shortest safe path

A* Search: Smart search, uses distance estimate
BFS: Searches all nearby cells first
DFS: Goes deep first, then backtracks
```

**File**: `code/search_algorithms.py`

---

## 2. CSP - Constraint Satisfaction (Week 9)

### What It Does
Allocates emergency resources (boats, medical teams) to shelters.

### How It Works
```
Problem:
- 5 shelters need help
- 10 resources available
- Each shelter has requirements
- Resources can only be in one place

Solution:
- CSP finds the best assignment
- Satisfies all constraints
- Uses smart techniques (MRV, LCV)
```

**File**: `code/csp_resource_allocation.py`

---

## 3. Neural Network - LSTM (Week 11)

### What It Does
Predicts floods using past 7 days of weather data.

### How It Works
```
LSTM (Long Short-Term Memory):
- Looks at weather patterns over time
- Remembers important past information
- Forgets irrelevant information
- Predicts future flood probability

Example:
Day 1: Light rain
Day 2: Medium rain
Day 3: Heavy rain
Day 4: Heavy rain
Day 5: Very heavy rain
Day 6: Extreme rain
Day 7: Extreme rain
    → LSTM predicts: HIGH flood risk
```

**File**: `code/neural_network.py`

---

## 4. Clustering - K-Means (Week 12)

### What It Does
Groups similar weather patterns together.

### How It Works
```
K-Means finds groups (clusters) in data:

Cluster 1: Hot + Dry = Low risk
Cluster 2: Monsoon + Wet = High risk
Cluster 3: Normal rain = Medium risk

New weather → Assigned to nearest cluster → Risk level
```

**File**: `code/clustering.py`

---

## 5. Reinforcement Learning (Week 12)

### What It Does
Learns when to issue flood warnings and evacuations.

### How It Works
```
Agent learns by trial and error:
1. Sees flood situation
2. Takes action (wait, warn, evacuate)
3. Gets reward/punishment
4. Learns better actions

Rewards:
+100: Person saved
-500: Person harmed
-50: False alarm
```

**File**: `code/reinforcement_learning.py`

---

## 6. Explainability - SHAP/LIME (Bonus)

### What It Does
Explains WHY the model made a prediction.

### How It Works
```
Instead of just "85% flood risk"
It says:
- Heavy rainfall: +40% to risk
- High humidity: +25% to risk  
- Monsoon season: +15% to risk
- Low pressure: +5% to risk
```

**File**: `code/explainability.py`

---

# Stage 6: Web Application Development

## What is the Web Application?
A user-friendly website where people can:
- Check current flood risk
- Enter weather data for prediction
- View historical data
- See AI technique demos

## Technology Used

### Streamlit
```
Simple Explanation:
- Python library for building web apps
- Write Python code → Get website
- Easy to use, looks professional
```

## App Features

**File**: `app.py`

### 1. Dashboard
- Shows current weather
- Displays flood risk level
- Uses live weather API

### 2. Custom Prediction
- User enters weather values
- Model predicts flood risk
- Shows risk factors

### 3. Historical Data
- Explore 25 years of data
- Filter by location, date
- View charts and statistics

### 4. AI Techniques Demo
- Interactive demos for each technique
- See search algorithms in action
- Try CSP resource allocation

### 5. Model Info
- View model performance
- See feature importance
- Understand predictions

## Result of Stage 6
- Complete web application
- Modern dark theme UI
- Responsive design

---

# Stage 7: Deployment

## What is Deployment?
Making the application available for others to use.

## Deployment Options

### 1. Local Run
```bash
# Install requirements
pip install -r requirements.txt

# Run app
streamlit run app.py

# Open browser: http://localhost:8501
```

### 2. Docker
```bash
# Build and run with Docker
docker-compose up --build

# Open browser: http://localhost:8501
```

### 3. CI/CD (Automatic Deployment)
- GitHub Actions configured
- Automatic testing on code push
- Can deploy to cloud services

## Files for Deployment
- `requirements.txt` - Python packages needed
- `Dockerfile` - Docker configuration
- `docker-compose.yml` - Docker compose settings
- `.github/workflows/` - CI/CD configuration

---

# Summary: Complete Project Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA GATHERING                            │
│  NASA POWER + Meteostat + NDMA → Raw data files             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    DATA CLEANING                             │
│  Missing values + Duplicates + Format → Clean data          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 FEATURE ENGINEERING                          │
│  24 features created → Better predictions                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   MODEL TRAINING                             │
│  3 models trained → Best model saved                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              AI TECHNIQUES ADDED                             │
│  Search + CSP + LSTM + Clustering + RL + Explainability     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   WEB APPLICATION                            │
│  Streamlit app with all features integrated                 │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT                                │
│  Docker + CI/CD → Production ready                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Takeaways

1. **Data is Foundation**: Quality data (25 years) makes good predictions
2. **Feature Engineering Matters**: Raw data → Useful features → Better AI
3. **Multiple Models**: Try several models, pick the best
4. **AI Techniques**: Course concepts applied to real problem
5. **User Interface**: Makes AI accessible to everyone
6. **Deployment**: Makes project usable in real world
