# 📁 Project Structure Documentation

## AI-Based Natural Disaster Prediction Web App

This document explains the project's file structure, what each file does, and how they connect together.

---

## 🗂️ Folder Structure Overview

```
AI-Based-Natural-Disaster-Prediction-Web-App/
│
├── 📱 app.py                      # Main web application
├── 📂 code/                       # All Python modules
├── 📂 data/                       # Data files (raw & processed)
├── 📂 results/                    # Trained models & outputs
├── 📂 docs/                       # Documentation files
├── 📂 notebooks/                  # Jupyter notebooks
├── 📂 tests/                      # Test files
├── 📂 .streamlit/                 # Streamlit configuration
├── 📂 .github/                    # GitHub CI/CD workflows
├── 🐳 Dockerfile                  # Docker configuration
├── 📋 requirements.txt            # Python dependencies
└── 📖 README.md                   # Project documentation
```

---

## 📱 Main Application Files

### app.py (Main File - 924 lines)
**What it does**: The main web application that users see and interact with.

**Key Parts**:
| Section | What It Does |
|---------|--------------|
| Configuration | Sets up page title, colors, API keys |
| Styling | Defines the visual look (CSS) |
| Weather Functions | Gets live weather from API |
| Prediction Functions | Uses trained model to predict flood |
| Dashboard Page | Shows current risk and weather |
| Custom Prediction Page | Lets users enter manual values |
| Historical Data Page | Shows past weather and floods |
| AI Techniques Page | Interactive demos of AI methods |

**How to run**:
```bash
streamlit run app.py
```

### app_simple.py
**What it does**: A simpler version of the app (backup/testing).

### app_backup_20251215_232703.py
**What it does**: Backup of app.py before major changes.

---

## 📂 code/ Folder (Main Logic)

This folder contains all the Python modules that do the actual work.

### Data Collection Files

#### fetch_nasa_power.py (231 lines)
**What it does**: Downloads weather data from NASA's satellites.

**Key Functions**:
| Function | What It Does |
|----------|--------------|
| `_request_payload()` | Creates the API request |
| `fetch_location()` | Downloads data for one location |
| `main()` | Downloads data for all locations |

**Used By**: Run manually to collect data.

---

#### fetch_meteostat_weather.py
**What it does**: Downloads weather data from ground stations.

**Key Functions**:
| Function | What It Does |
|----------|--------------|
| `get_station_data()` | Gets data from weather station |
| `download_weather()` | Downloads for date range |
| `save_to_csv()` | Saves data to file |

---

#### merge_weather_data.py
**What it does**: Combines NASA and Meteostat data into one dataset.

**Key Functions**:
| Function | What It Does |
|----------|--------------|
| `load_data()` | Loads both datasets |
| `merge_datasets()` | Joins data by date and location |
| `save_merged()` | Saves combined data |

---

#### label_historical_floods.py
**What it does**: Labels each day as "flood" or "no flood" using historical records.

**Key Functions**:
| Function | What It Does |
|----------|--------------|
| `load_flood_records()` | Loads known flood dates |
| `label_data()` | Marks flood days in dataset |

---

### Data Preprocessing Files

#### preprocessing.py (315 lines)
**What it does**: Cleans data and creates features for model training.

**Key Class**: `DataPreprocessor`

| Method | What It Does |
|--------|--------------|
| `load_data()` | Loads CSV file |
| `explore_data()` | Shows statistics about data |
| `handle_missing_values()` | Fills empty cells |
| `feature_engineering()` | Creates 24 new features |
| `prepare_for_training()` | Splits into train/test |

**Technical Terms**:
- **Feature Engineering**: Creating new columns from existing data
- **Train-Test Split**: Dividing data for training (80%) and testing (20%)
- **Scaling**: Making all numbers similar size (helps AI learn)

---

#### clean_weather_pipeline.py
**What it does**: Additional cleaning steps for weather data.

---

#### clean_missing_values.py
**What it does**: Specialized script for handling missing data.

---

#### build_training_dataset.py
**What it does**: Creates the final dataset for model training.

---

### Model Training Files

#### baseline_models.py
**What it does**: Trains simple models as a starting point.

---

#### improved_models.py (297 lines)
**What it does**: Trains better models with class balancing.

**Key Class**: `ImprovedFloodModels`

| Method | What It Does |
|--------|--------------|
| `train_logistic_regression()` | Trains LR model |
| `train_random_forest()` | Trains RF model |
| `train_gradient_boosting()` | Trains GB model |
| `find_optimal_threshold()` | Finds best cutoff for predictions |
| `evaluate_all_models()` | Compares all models |
| `save_best_model()` | Saves the winner |

**Technical Terms**:
- **Class Imbalance**: When one class (flood) is rare (2.7%) compared to other (no flood: 97.3%)
- **Class Weights**: Giving more importance to rare class during training
- **Threshold**: The probability cutoff to decide "flood" vs "no flood" (default 0.5)

---

#### model_evaluation.py
**What it does**: Evaluates how good the models are.

**Metrics Calculated**:
| Metric | What It Measures |
|--------|-----------------|
| Accuracy | Overall correctness |
| Precision | Of predicted floods, how many were real? |
| Recall | Of real floods, how many did we catch? |
| F1 Score | Balance of precision and recall |

---

### AI Technique Files

#### search_algorithms.py (379 lines)
**What it does**: Implements A*, BFS, DFS for evacuation routes.

**Key Class**: `FloodEvacuationGrid`

| Method | What It Does |
|--------|--------------|
| `generate_flood_scenario()` | Creates a grid with flooded cells |
| `a_star_search()` | Finds optimal path using A* |
| `bfs_search()` | Finds path using BFS |
| `dfs_search()` | Finds path using DFS |
| `get_neighbors()` | Gets valid adjacent cells |
| `visualize_path()` | Shows the path on grid |

**Technical Terms**:
- **Heuristic**: An estimate of distance to goal (A* uses this)
- **Open Set**: Cells we still need to explore
- **Closed Set**: Cells we've already explored

---

#### csp_resource_allocation.py (377 lines)
**What it does**: Allocates emergency resources using CSP.

**Key Class**: `FloodResourceAllocationCSP`

| Method | What It Does |
|--------|--------------|
| `add_shelter()` | Adds an evacuation shelter |
| `add_resource()` | Adds available resource |
| `solve_backtracking()` | Finds solution using backtracking |
| `ac3_arc_consistency()` | Reduces possible values |
| `is_consistent()` | Checks if assignment is valid |
| `select_unassigned_variable()` | Uses MRV heuristic |

**Technical Terms**:
- **CSP**: Problem where we need to assign values satisfying constraints
- **Backtracking**: Try, fail, undo, try again
- **MRV (Minimum Remaining Values)**: Pick variable with fewest choices first
- **AC-3**: Algorithm to remove impossible values

---

#### neural_network.py (522 lines)
**What it does**: LSTM neural network for time-series prediction.

**Key Class**: `FloodLSTM`

| Method | What It Does |
|--------|--------------|
| `_initialize_weights()` | Sets up network weights |
| `_lstm_cell_forward()` | One step of LSTM |
| `_forward_pass()` | Full forward computation |
| `_backward_pass()` | Computes gradients |
| `train()` | Trains the network |
| `predict()` | Makes predictions |

**Technical Terms**:
- **LSTM**: Long Short-Term Memory (remembers past information)
- **Hidden State**: Network's "memory"
- **Cell State**: Long-term memory
- **Gates**: Control what to remember/forget
  - Forget Gate: What to throw away
  - Input Gate: What new info to add
  - Output Gate: What to output

---

#### clustering.py (501 lines)
**What it does**: K-Means clustering for weather patterns.

**Key Class**: `FloodPatternKMeans`

| Method | What It Does |
|--------|--------------|
| `_initialize_centroids()` | K-Means++ initialization |
| `fit()` | Trains clustering model |
| `predict()` | Assigns new data to clusters |
| `_assign_clusters()` | Finds nearest centroid |
| `_update_centroids()` | Moves centroids to mean |
| `_calculate_inertia()` | Measures cluster tightness |

**Technical Terms**:
- **Centroid**: Center point of a cluster
- **K-Means++**: Smart way to pick initial centroids
- **Inertia**: How spread out the clusters are (lower = tighter)
- **Elbow Method**: Finding optimal number of clusters

---

#### reinforcement_learning.py (570 lines)
**What it does**: Q-Learning for evacuation decisions.

**Key Classes**:

**FloodEvacuationEnvironment**:
| Method | What It Does |
|--------|--------------|
| `reset()` | Starts new flood scenario |
| `step()` | Takes action, returns result |
| `_update_flood_level()` | Simulates flood progression |

**QLearningAgent**:
| Method | What It Does |
|--------|--------------|
| `choose_action()` | Picks action (explore vs exploit) |
| `update_q_table()` | Updates Q-values |
| `train()` | Runs training episodes |

**Technical Terms**:
- **Q-Value**: Expected reward for action in state
- **Epsilon-Greedy**: Sometimes explore random actions
- **Discount Factor (γ)**: How much to value future rewards
- **Learning Rate (α)**: How fast to update Q-values

---

#### explainability.py (649 lines)
**What it does**: Explains model predictions using SHAP and LIME.

**Key Classes**:

**SHAPExplainer**:
| Method | What It Does |
|--------|--------------|
| `fit()` | Learns background distribution |
| `_kernel_shap()` | Computes Shapley values |
| `explain()` | Explains single prediction |

**LIMEExplainer**:
| Method | What It Does |
|--------|--------------|
| `explain()` | Creates local explanation |
| `_perturb_instance()` | Creates similar samples |
| `_fit_interpretable_model()` | Trains simple model locally |

**Technical Terms**:
- **SHAP**: Shapley Additive Explanations (fair division of contribution)
- **LIME**: Local Interpretable Model-agnostic Explanations
- **Feature Contribution**: How much each feature affects prediction

---

## 📂 data/ Folder

### data/raw/ (Raw, unprocessed data)
| File | Contents |
|------|----------|
| `nasa_power_combined.csv` | All NASA data combined |
| `nasa_power_swat_*.csv` | Swat district NASA data |
| `nasa_power_upper_dir_*.csv` | Upper Dir NASA data |
| `weather_swat_*.csv` | Swat Meteostat data |
| `weather_upper_dir_*.csv` | Upper Dir Meteostat data |
| `ndma_flood_reports.csv` | Flood event records |

### data/processed/ (Clean, ready-to-use data)
| File | Contents |
|------|----------|
| `flood_weather_dataset.csv` | Main training dataset |
| `flood_weather_dataset_cleaned.csv` | Cleaned version |
| `cleaned_swat.csv` | Swat cleaned data |
| `cleaned_upper_dir.csv` | Upper Dir cleaned data |

---

## 📂 results/ Folder

### Model Files
| File | What It Contains |
|------|-----------------|
| `best_flood_model.pkl` | Best trained model |
| `logistic_regression_model.pkl` | LR model |
| `random_forest_model.pkl` | RF model |

### Metrics & Reports
| File | What It Contains |
|------|-----------------|
| `model_metrics.csv` | Performance numbers |
| `improved_model_metrics.csv` | Better model metrics |
| `feature_importance.json` | Which features matter most |
| `evaluation_report.txt` | Detailed evaluation |
| `optimal_thresholds.json` | Best cutoff values |

### Visualizations
| File | What It Shows |
|------|--------------|
| `confusion_matrices.png` | Prediction accuracy grid |
| `roc_curves.png` | Model comparison curves |
| `feature_importance_*.png` | Feature importance charts |

---

## 📂 Other Important Files

### Configuration Files
| File | Purpose |
|------|---------|
| `requirements.txt` | Python packages to install |
| `Dockerfile` | Docker container setup |
| `docker-compose.yml` | Docker compose configuration |
| `.streamlit/secrets.toml` | API keys (not in repo) |

### Documentation Files
| File | Purpose |
|------|---------|
| `README.md` | Main project documentation |
| `AI_TECHNIQUES_SUMMARY.md` | AI techniques details |
| `ML_PIPELINE_README.md` | ML pipeline explanation |
| `ENVIRONMENT_SETUP.md` | Setup instructions |
| `QUICK_START.md` | Quick start guide |

### Run Scripts
| File | Purpose |
|------|---------|
| `run_pipeline.py` | Runs full training pipeline |
| `test_model.py` | Tests model predictions |
| `verify_predictions.py` | Verifies model outputs |
| `train_improved_models.py` | Trains improved models |

---

## 🔗 How Files Connect

```
Data Flow:
──────────────────────────────────────────────────────────────────
fetch_nasa_power.py ─┐
                     ├─→ merge_weather_data.py ─→ preprocessing.py
fetch_meteostat.py ──┘                                    │
                                                          ↓
                                              improved_models.py
                                                          │
                                                          ↓
                                              results/best_model.pkl
                                                          │
                                                          ↓
                                                      app.py
──────────────────────────────────────────────────────────────────

AI Techniques (Independent):
──────────────────────────────────────────────────────────────────
search_algorithms.py     ─┐
csp_resource_allocation.py├─→ app.py (AI Demos tab)
neural_network.py        ─┤
clustering.py            ─┤
reinforcement_learning.py─┤
explainability.py        ─┘
──────────────────────────────────────────────────────────────────
```

---

## 💡 Quick Reference

### To collect new data:
```bash
python -m code.fetch_nasa_power
python -m code.fetch_meteostat_weather
```

### To preprocess data:
```bash
python -m code.preprocessing
```

### To train models:
```bash
python train_improved_models.py
```

### To run the app:
```bash
streamlit run app.py
```

### To run with Docker:
```bash
docker-compose up --build
```

---

## ❓ Common Questions

**Q: Where is the main prediction logic?**
A: In `app.py` (functions like `predict_flood()`) and `code/improved_models.py`.

**Q: Where are the trained models?**
A: In the `results/` folder as `.pkl` files.

**Q: How do I add a new AI technique?**
A: Create a new file in `code/`, then import and add a demo in `app.py`.

**Q: Where is the weather data?**
A: Raw data in `data/raw/`, processed data in `data/processed/`.

**Q: How do I change the UI look?**
A: Edit the CSS in the `GLOBAL_CSS` variable in `app.py`.
